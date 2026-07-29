import json
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from skyrl.train.utils.tracking import Tracking

import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorOutput,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input,
)
from skyrl.train.utils import Timer
from skyrl.train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output,
)
from skyrl.train.utils.trajectory_logging import TrajectoryLogger, pretty_print_example

if TYPE_CHECKING:
    from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper


def _maybe_redel_reaggregate_rollout_metrics(
    generator: GeneratorInterface, concat_generator_outputs: GeneratorOutput
) -> None:
    """If ``generator`` is a ReDelGenerator, recompute env-aware rollout metrics
    over the concatenated generator outputs and write them back in-place.

    ``concatenate_generator_outputs`` re-runs the base (env-blind)
    ``get_rollout_metrics``, which drops per-env metrics. The ReDel generator
    exposes ``env_metrics``/``env_classes`` as list-valued keys that the base
    concat propagates; we just re-run the generator's own aggregator here.
    """
    try:
        from redel_rl.redel_generator import ReDelGenerator
    except ImportError:
        return
    if not isinstance(generator, ReDelGenerator):
        return
    # env_metrics/env_classes aren't part of the GeneratorOutput TypedDict — ReDelGenerator
    # adds them as extra list-valued keys that concatenate_generator_outputs propagates.
    env_metrics = concat_generator_outputs.get("env_metrics")
    env_classes = concat_generator_outputs.get("env_classes")
    if env_metrics is None or env_classes is None:
        return
    concat_generator_outputs["rollout_metrics"] = generator.get_rollout_metrics(
        responses=concat_generator_outputs["response_ids"],
        rewards=concat_generator_outputs["rewards"],
        stop_reasons=concat_generator_outputs["stop_reasons"],
        env_metrics=env_metrics,
        env_classes=env_classes,
    )


def _maybe_redel_dump_subagent_trajectories(
    generator: GeneratorInterface,
    concat_generator_outputs: GeneratorOutput,
    tokenizer: AutoTokenizer,
    dump_dir_path: Path,
) -> None:
    """If ``generator`` is a ReDelGenerator, dump the per-rollout fork/join tree to
    ``subagent_trajectories.jsonl`` alongside the standard per-dataset eval dumps.

    Reads the ``redel_step_records`` extra key the generator stashes on its output (see the
    CONTRACT comment in redel_generator.py). The standard ``dump_per_dataset_eval_results``
    only decodes the flat per-step response ids and can't tell root from subagent steps;
    this reconstructs, per eval sample (``instance_id``), the set of agents and each agent's
    steps. No-op for non-ReDel generators or if the key is absent.
    """
    try:
        from redel_rl.redel_generator import ReDelGenerator
    except ImportError:
        return
    if not isinstance(generator, ReDelGenerator):
        return
    records = concat_generator_outputs.get("redel_step_records")
    if not records:
        return
    response_ids = concat_generator_outputs["response_ids"]
    # Authoritative per-step score: matches what dump_per_dataset_eval_results writes, and
    # reflects generator post-processing (zero_reward_on_non_stop / overlong filtering) that
    # the record's raw token_rewards predate. Token-level rewards are summed to a scalar.
    rewards = concat_generator_outputs["rewards"]

    def _scalar_reward(idx: int):
        r = rewards[idx]
        return float(sum(r)) if isinstance(r, list) else float(r)

    # Group records by rollout, then by agent. A rollout is keyed by the FULL trajectory id
    # (instance_id, repetition_id), NOT instance_id alone: at eval all steps of a rollout —
    # root and every subagent — share the root's trajectory_id (run_agent_loop bakes it in via
    # functools.partial and fork reuses it), and instance_id is the *group* id shared across
    # GRPO repetitions. Keying on instance_id alone would merge the N>1 samples-per-prompt
    # rollouts together and collide every rollout's agent_id="root". Agents within one rollout
    # are distinguished by agent_id. Each agent spans multiple step records; we keep every
    # step's decoded response and take agent-level fields from the latest record (subagent
    # bookkeeping is only complete on the agent's last step).
    by_rollout: "OrderedDict[tuple, OrderedDict[str, dict]]" = OrderedDict()
    for i, rec in enumerate(records):
        rollout_key = (rec["instance_id"], rec["repetition_id"])
        agents = by_rollout.setdefault(rollout_key, OrderedDict())
        agent = agents.setdefault(rec["agent_id"], {"meta": rec, "steps": []})
        agent["meta"] = rec  # latest wins
        agent["steps"].append(
            {
                "step_idx": len(agent["steps"]),
                "is_last": rec["is_last"],
                "reward": _scalar_reward(i),
                "response": tokenizer.decode(response_ids[i]),
            }
        )

    filename = dump_dir_path / "subagent_trajectories.jsonl"
    with open(filename, "w") as f:
        for (instance_id, repetition_id), agents in by_rollout.items():
            agent_list = []
            for agent_id, agent in agents.items():
                m = agent["meta"]
                agent_list.append(
                    {
                        "agent_id": agent_id,
                        "depth": m["depth"],
                        "num_turns": m["num_turns"],
                        "stop_reason": m["stop_reason"],
                        # agent's terminal score (its last step); see _scalar_reward note above
                        "reward": agent["steps"][-1]["reward"],
                        "subagent_ids": m["subagent_ids"],
                        "subagent_join_turns": m["subagent_join_turns"],
                        "n_children": m["n_children"],
                        "n_children_recursive": m["n_children_recursive"],
                        "max_depth": m["max_depth"],
                        "flags_for_reward": m["flags_for_reward"],
                        "steps": agent["steps"],
                    }
                )
            # root (depth 0) first, then by depth/id so the tree reads top-down
            agent_list.sort(key=lambda a: (a["depth"], a["agent_id"]))
            root = next((a for a in agent_list if a["depth"] == 0), None)
            entry = {
                "instance_id": instance_id,
                "repetition_id": repetition_id,
                "num_agents": len(agent_list),
                "max_depth": max((a["depth"] for a in agent_list), default=0),
                "root_reward": root["reward"] if root else None,
                "agents": agent_list,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Dumped redel subagent trajectories ({len(by_rollout)} rollouts) to {filename}")


@torch.no_grad()
async def evaluate(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    tracker: Optional["Tracking"] = None,
    vllm_metrics_scraper: Optional["VLLMMetricsScraper"] = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        vllm_metrics_scraper: when set, the open ``vllm/eval`` window is resumed
            around each generation and paused after, so only generation time
            counts toward eval throughput.

    Returns:
        Dict[str, float]: evaluation metrics
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    concat_prompts: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    eval_generate_time = 0.0
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        gen_start = time.monotonic()
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.resume()
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.pause()
        eval_generate_time += time.monotonic() - gen_start
        validate_generator_output(len(generator_input["prompts"]), generator_output)
        generator_outputs.append(generator_output)
        concat_all_envs.extend(generator_input["env_classes"])
        concat_env_extras.extend(generator_input["env_extras"])
        concat_uids.extend(uids)
        concat_prompts.extend(generator_input["prompts"])
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)
    _maybe_redel_reaggregate_rollout_metrics(generator, concat_generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]

    if cfg.trainer.print_example_interval > 0:
        vis = tokenizer.decode(generator_output["response_ids"][0])
        pretty_print_example(
            logger,
            prompt=generator_input["prompts"][0],
            response=vis,
            reward=generator_output["rewards"][0],
        )

    # Optionally upload up to `num_logger_eval_samples` samples to tracker (wandb)
    if trajectory_logger is not None:
        with Timer("log_eval_results"):
            trajectory_logger.log(
                tracker=tracker,
                num_samples=cfg.trainer.num_logger_eval_samples,
                prompts=concat_prompts,
                generator_output=concat_generator_outputs,
                tokenizer=tokenizer,
                global_step=global_step,
                wandb_key="trajectories/eval",
            )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        concat_generator_outputs, concat_uids, concat_data_sources, cfg.generator.eval_n_samples_per_prompt
    )

    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )

    for key, value in concat_generator_outputs["rollout_metrics"].items():
        eval_metrics[f"eval/all/{key}"] = value

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )

    eval_metrics["timing/eval_generate"] = eval_generate_time
    return eval_metrics


@torch.no_grad()
async def evaluate_step_wise(
    eval_dataloader: StatefulDataLoader,
    generator: GeneratorInterface,
    cfg: SkyRLTrainConfig,
    global_step: int | None,
    tokenizer: AutoTokenizer,
    trajectory_logger: Optional[TrajectoryLogger] = None,
    tracker: Optional["Tracking"] = None,
    vllm_metrics_scraper: Optional["VLLMMetricsScraper"] = None,
) -> Dict[str, float]:
    """Runs generation and evaluation of trajectories for step-wise training.

    Currently assumes that the rewards are assigned to the last step of each trajectory.

    Args:
        eval_dataloader (StatefulDataLoader): dataloader of the eval dataset
        generator (GeneratorInterface): generator to use
        cfg (SkyRLTrainConfig): config
        global_step (int | None): current global step, or
            `None` to indicate a non-training context (e.g., eval-only)
        tokenizer (AutoTokenizer): tokenizer to use
        vllm_metrics_scraper: when set, the open ``vllm/eval`` window is resumed
            around each generation and paused after, so only generation time
            counts toward eval throughput.

    Returns:
        Dict[str, float]: evaluation metrics
    """

    # 1. Get all generator outputs
    generator_outputs: List[GeneratorOutput] = []
    concat_all_envs: List[str] = []
    concat_env_extras: List[Dict[str, Any]] = []
    concat_uids: List[str] = []
    concat_prompts: List[str] = []
    sampling_params = cfg.generator.eval_sampling_params
    eval_generate_time = 0.0
    pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
    for _, prompts in enumerate(eval_dataloader):
        pbar.update(1)
        generator_input, uids = prepare_generator_input(
            prompts,
            cfg.generator.eval_n_samples_per_prompt,
            get_sampling_params_for_backend(cfg.generator.inference_engine.backend, sampling_params),
            cfg.environment.env_class,
            "eval",
            global_step,
        )
        gen_start = time.monotonic()
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.resume()
        generator_output: GeneratorOutput = await generator.generate(generator_input)
        if vllm_metrics_scraper is not None:
            vllm_metrics_scraper.pause()
        eval_generate_time += time.monotonic() - gen_start
        traj_id_to_input = {
            traj_id.instance_id: {
                "env_class": env_class,
                "env_extras": env_extra,
                "prompt": prompt,
            }
            for traj_id, env_class, env_extra, prompt in zip(
                generator_input["trajectory_ids"],
                generator_input["env_classes"],
                generator_input["env_extras"],
                generator_input["prompts"],
            )
        }
        for traj_id in generator_output["trajectory_ids"]:
            assert traj_id.instance_id in traj_id_to_input, f"Trajectory ID {traj_id.instance_id} not found in input"
            concat_all_envs.append(traj_id_to_input[traj_id.instance_id]["env_class"])
            concat_env_extras.append(traj_id_to_input[traj_id.instance_id]["env_extras"])
            concat_uids.append(traj_id.instance_id)
            concat_prompts.append(traj_id_to_input[traj_id.instance_id]["prompt"])
        validate_generator_output(generator_input, generator_output, step_wise=True)
        generator_outputs.append(generator_output)
    concat_generator_outputs: GeneratorOutput = concatenate_generator_outputs(generator_outputs)
    _maybe_redel_reaggregate_rollout_metrics(generator, concat_generator_outputs)

    # Extract data_sources from env_extras
    concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]

    if cfg.trainer.print_example_interval > 0:
        vis = tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Eval output example: {vis}")

    # Only use the final step metrics
    generator_output_last_step = defaultdict(list)
    is_last_step_mask = concat_generator_outputs["is_last_step"]
    for key in concat_generator_outputs:
        if isinstance(concat_generator_outputs[key], list):
            assert len(concat_generator_outputs[key]) == len(
                is_last_step_mask
            ), f"Length mismatch: {len(concat_generator_outputs[key])} != {len(is_last_step_mask)} for key {key}"
            generator_output_last_step[key] = [
                val for val, is_last_step in zip(concat_generator_outputs[key], is_last_step_mask) if is_last_step
            ]
    uids_last_step = [uid for uid, is_last_step in zip(concat_uids, is_last_step_mask) if is_last_step]
    data_sources_last_step = [
        data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
    ]
    prompts_last_step = [prompt for prompt, is_last_step in zip(concat_prompts, is_last_step_mask) if is_last_step]

    # Optionally upload up to `num_logger_eval_samples` samples to wandb.
    # For step-wise we override the logger's default loss-mask-based
    # num_turns with the total step count per trajectory (counted *before*
    # the last-step filter).
    if trajectory_logger is not None:
        trajectory_step_counts = Counter(concat_uids)
        trajectory_logger.log(
            tracker=tracker,
            num_samples=cfg.trainer.num_logger_eval_samples,
            prompts=prompts_last_step,
            generator_output=generator_output_last_step,
            tokenizer=tokenizer,
            global_step=global_step,
            num_turns_list=[trajectory_step_counts[uid] for uid in uids_last_step],
            wandb_key="trajectories/eval",
        )

    # 2. Group data by data source and calculate per-dataset metrics
    eval_metrics = calculate_per_dataset_metrics(
        generator_output_last_step, uids_last_step, data_sources_last_step, cfg.generator.eval_n_samples_per_prompt
    )
    # 3. Calculate overall metrics across all datasets
    overall_metrics = get_metrics_from_generator_output(generator_output_last_step, uids_last_step)
    eval_metrics.update(
        {
            "eval/all/avg_score": overall_metrics["avg_score"],
            f"eval/all/pass_at_{cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
            "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
        }
    )

    for key, value in concat_generator_outputs["rollout_metrics"].items():
        # ignore multi-step-all/ and redel-agent/ keys
        if "multi-step-all/" in key or "redel-agent/" in key:
            continue
        eval_metrics[f"eval/{key}"] = value

    # 4. Prepare dumping data
    # TODO[Ben] update this to be cloud-compatible
    if cfg.trainer.dump_eval_results:
        with Timer("dump_eval_results"):
            data_save_dir = (
                Path(cfg.trainer.export_path)
                / "dumped_evals"
                / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
            )
            data_save_dir.mkdir(parents=True, exist_ok=True)
            dump_per_dataset_eval_results(
                data_save_dir,
                tokenizer,
                concat_generator_outputs,
                concat_data_sources,
                concat_all_envs,
                concat_env_extras,
                eval_metrics,
            )
            # additive ReDel-only dump of the fork/join tree; no-op for other generators
            _maybe_redel_dump_subagent_trajectories(generator, concat_generator_outputs, tokenizer, data_save_dir)

    eval_metrics["timing/eval_generate"] = eval_generate_time
    return eval_metrics
