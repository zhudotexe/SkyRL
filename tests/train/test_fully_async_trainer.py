"""
CPU unit tests for fully-async trainer building blocks that back `sample_full_batch`:
the staleness manager's filtered-rollout accounting, the dataloader's trained-vs-filtered
UID tracking, the consumer's exhaustion-aware buffer drain, and the training loop's carry of
an epoch's partial mini-batch into the next epoch (including across a crash and resume).
"""

import asyncio
import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from torchdata.stateful_dataloader import StatefulDataLoader

import skyrl.train.fully_async_trainer as fully_async_trainer
from skyrl.train.fully_async_trainer import (
    FullyAsyncRayPPOTrainer,
    GeneratedOutputGroup,
    _AsyncDataloader,
    _AsyncStalenessManager,
)
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import Timer
from skyrl.train.utils.trainer_utils import ResumeMode


def _make_async_dataloader(num_prompts: int, mini_batch_size: int) -> _AsyncDataloader:
    """Build an _AsyncDataloader over a trivial dataset of `num_prompts` single-prompt batches."""
    dataset = [[{"uid": str(i)}] for i in range(num_prompts)]
    # batch_size=1 (one prompt per draw) and identity collate so each batch is a list with one dict.
    loader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    return _AsyncDataloader(loader, mini_batch_size)


# --------------------------------------------------------------------------------------
# _AsyncStalenessManager
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staleness_manager_filter_restores_capacity():
    """Dropping an accepted group via on_rollout_filtered must give producer capacity back.

    This is the deadlock regression: without reclassifying accepted -> filtered, dropped groups
    keep `accepted` climbing against a fixed staleness ceiling and starve the producers.
    """
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # consumer_capacity = (max_staleness_steps + current_global_step) * mini_batch = (1 + 1) * 2 = 4.
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()

    # At the staleness ceiling: accepted == 4 == ceiling, so no producer capacity remains.
    assert mgr._compute_capacity_unlocked() == 0

    await mgr.on_rollout_filtered()

    # Capacity restored by exactly one slot, and accounting is consistent.
    assert mgr._compute_capacity_unlocked() == 1
    assert mgr._stat.accepted == 3
    assert mgr._stat.filtered == 1
    assert mgr._stat.running == 0
    assert mgr._stat.submitted == 4


@pytest.mark.asyncio
async def test_staleness_manager_validate_epoch_end_with_filtered():
    """At epoch end, submitted == accepted + filtered and accepted == trained steps * mini_batch."""
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # Submit and finish 4 groups; drop 2 of them, train on the remaining 2 (one step).
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()
    for _ in range(2):
        await mgr.on_rollout_filtered()

    # One training step completed -> we are now working on global_step 2.
    await mgr.notify_capacity_change(2)
    await mgr.validate_state_at_epoch_end(global_step=2, num_carried=0)  # must not raise


@pytest.mark.asyncio
async def test_staleness_manager_validate_epoch_end_with_carried():
    """Groups carried into the next epoch stay accepted: accepted == trained + carried at epoch end."""
    mgr = _AsyncStalenessManager(max_concurrent_generation_groups=4, mini_batch_size=2, max_staleness_steps=1)
    # 4 groups finish; 1 is dropped, 2 train (one step), 1 is left over when the epoch runs out of prompts.
    for _ in range(4):
        await mgr.acquire_submission_slot()
    for _ in range(4):
        await mgr.on_rollout_accepted()
    await mgr.on_rollout_filtered()
    await mgr.notify_capacity_change(2)

    await mgr.validate_state_at_epoch_end(global_step=2, num_carried=1)  # must not raise
    with pytest.raises(AssertionError):
        await mgr.validate_state_at_epoch_end(global_step=2, num_carried=0)


# --------------------------------------------------------------------------------------
# _AsyncDataloader
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_dataloader_filtered_uids_tracking():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)

    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}
    assert set(adl.get_consumed_uids_list()) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_async_dataloader_skips_filtered_uids():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    await adl.mark_consumed_uids(["0", "1"])
    await adl.mark_filtered_uids(["2"])

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])

    # Trained (0, 1) and filtered (2) are all skipped; only the rest are drawn.
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_restores_filtered():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    adl.load_state_from_checkpoint({"0", "1", "2"}, {"2"})

    assert adl.num_trained() == 2
    assert set(adl.get_filtered_uids_list()) == {"2"}

    seen = []
    while True:
        prompts = await adl.get_next_non_consumed_data()
        if prompts is None:
            break
        seen.append(prompts[0]["uid"])
    assert seen == ["3", "4", "5"]


@pytest.mark.asyncio
async def test_async_dataloader_load_state_without_filtered_is_backward_compatible():
    adl = _make_async_dataloader(num_prompts=6, mini_batch_size=2)
    # Old checkpoints have no filtered set; default treats everything consumed as trained.
    adl.load_state_from_checkpoint({"0", "1"})
    assert adl.num_trained() == 2
    assert adl.get_filtered_uids_list() == []


# --------------------------------------------------------------------------------------
# _drain_next_group
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_next_group_returns_buffered_items_then_exhaustion():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    buffer.put_nowait("b")

    # _drain_next_group uses no instance state, so a bare object stands in for `self`.
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    assert await drain(dummy, buffer, done) == "a"
    assert await drain(dummy, buffer, done) == "b"

    # Buffer empty and generators done -> exhausted.
    done.set()
    assert await drain(dummy, buffer, done) is None


@pytest.mark.asyncio
async def test_drain_next_group_drains_remaining_before_exhaustion():
    """If generators finish while items remain, those items are returned before None."""
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    buffer.put_nowait("a")
    done.set()  # generators done, but a real item is still buffered

    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()
    assert await drain(dummy, buffer, done) == "a"
    assert await drain(dummy, buffer, done) is None


@pytest.mark.asyncio
async def test_drain_next_group_blocks_until_item_arrives():
    buffer: asyncio.Queue = asyncio.Queue()
    done = asyncio.Event()
    drain = FullyAsyncRayPPOTrainer._drain_next_group
    dummy = object()

    async def delayed_put():
        await asyncio.sleep(0.05)
        buffer.put_nowait("x")

    producer = asyncio.create_task(delayed_put())
    assert await drain(dummy, buffer, done) == "x"
    await producer


# --------------------------------------------------------------------------------------
# _should_keep_group
# --------------------------------------------------------------------------------------


def _trainer_with_tol(tol: float):
    """A stand-in for `self` exposing just the cfg field _should_keep_group reads."""
    return SimpleNamespace(
        cfg=SimpleNamespace(trainer=SimpleNamespace(algorithm=SimpleNamespace(zero_variance_filter_tol=tol)))
    )


def _group(rewards, loss_masks, uid="u"):
    return GeneratedOutputGroup(
        generator_output={"rewards": rewards, "loss_masks": loss_masks},
        uid=uid,
        global_step_when_scheduled=0,
    )


def test_should_keep_group():
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Zero-variance group -> drop.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0], [[1], [1]])) is False
    # Group with reward spread -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 0.0], [[1], [1]])) is True
    # Singleton -> keep.
    assert keep(_trainer_with_tol(0.0), _group([1.0], [[1]])) is True
    # Masked trajectories are ignored: two equal live rewards + one masked -> still zero-variance.
    assert keep(_trainer_with_tol(0.0), _group([1.0, 1.0, 0.0], [[1], [1], [0]])) is False
    # Near-equal float rewards within tol -> drop.
    assert keep(_trainer_with_tol(1e-6), _group([0.6667, 0.66670001], [[1], [1]])) is False


def test_reprefix_metrics():
    """generate/X -> generate_<suffix>/X, preserving the leading namespace for tracker grouping."""
    reprefix = FullyAsyncRayPPOTrainer._reprefix_metrics
    out = reprefix(
        {"generate/avg_num_tokens": 10.0, "environment/score": 0.5, "bare": 1},
        "dropped",
    )
    assert out == {
        "generate_dropped/avg_num_tokens": 10.0,
        "environment_dropped/score": 0.5,
        "dropped/bare": 1,
    }


def test_should_keep_group_token_level_rewards():
    """Token-level rewards are collapsed to per-trajectory sequence rewards for the variance check."""
    keep = FullyAsyncRayPPOTrainer._should_keep_group

    # Two trajectories, both summing to 1.0 -> zero variance -> drop.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [1.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is False
    )
    # One trajectory sums to 1.0, the other to 0.0 -> variance -> keep.
    assert (
        keep(
            _trainer_with_tol(0.0),
            _group([[0.0, 1.0], [0.0, 0.0]], [[1, 1], [1, 1]]),
        )
        is True
    )


# --------------------------------------------------------------------------------------
# train(): carrying an epoch's partial mini-batch into the next epoch
# --------------------------------------------------------------------------------------


class _NoopPhaseGauge:
    """TrainingPhaseGauge without the Ray metrics backend."""

    @contextmanager
    def timed_phase(self, name, timings):
        with Timer(name, timings):
            yield


class _Crash(Exception):
    """A simulated failure in a training step."""


def _base_save_checkpoints(self):
    """RayPPOTrainer.save_checkpoints without the model: just the step folder and the latest-step pointer."""
    folder = os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{self.global_step}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt"), "w") as f:
        f.write(str(self.global_step))
    return folder


def _base_load_checkpoints(self):
    """RayPPOTrainer.load_checkpoints without the model: the latest step and its folder."""
    with open(os.path.join(self.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")) as f:
        global_step = int(f.read())
    return global_step, os.path.join(self.cfg.trainer.ckpt_path, f"global_step_{global_step}")


@pytest.fixture
def stub_backend(monkeypatch):
    """Stub out the sampling-param lookup and the model half of checkpointing; the fully-async half (consumed /
    filtered uids, epoch) runs for real."""
    monkeypatch.setattr(fully_async_trainer, "get_sampling_params_for_backend", lambda backend, params: {})
    monkeypatch.setattr(RayPPOTrainer, "save_checkpoints", _base_save_checkpoints)
    monkeypatch.setattr(RayPPOTrainer, "load_checkpoints", _base_load_checkpoints)


def _carry_test_trainer(
    num_prompts,
    mini_batch_size,
    epochs,
    kept_uids,
    ckpt_path="",
    ckpt_interval=0,
    save_on_epoch_end=False,
    max_training_steps=None,
    resume=False,
    crash_at_step=None,
):
    """A FullyAsyncRayPPOTrainer running the real train loop, staleness manager, dataloader, mini-batch conversion
    (including the step-wise concatenation and its validation) and fully-async checkpoint state, with stubbed
    generation (instant; a group has reward variance iff its prompt uid is in ``kept_uids``), training (raises
    ``_Crash`` at ``crash_at_step``) and model checkpointing.

    Returns the trainer and the list it appends ``(epoch, kept uids, async/num_groups_dropped)`` to for each trained
    step.
    """
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            epochs=epochs,
            eval_interval=0,
            eval_before_train=False,
            save_on_epoch_end=save_on_epoch_end,
            ckpt_interval=ckpt_interval,
            ckpt_path=ckpt_path,
            hf_save_interval=0,
            max_training_steps=max_training_steps,
            update_ref_every_epoch=False,
            algorithm=SimpleNamespace(zero_variance_filter_tol=0.0),
            critic=SimpleNamespace(model=SimpleNamespace(path=None)),
        ),
        generator=SimpleNamespace(
            n_samples_per_prompt=2,
            step_wise_trajectories=True,
            inference_engine=SimpleNamespace(backend="vllm"),
            sampling_params=None,
        ),
        environment=SimpleNamespace(env_class="env"),
    )
    trainer.mini_batch_size = mini_batch_size
    trainer.max_staleness_steps = 1
    trainer.num_parallel_generation_workers = mini_batch_size
    trainer.sample_full_batch = True
    trainer._carried_groups = []
    trainer._num_carried_dropped = 0
    trainer._gen_buffer_maxsize = mini_batch_size * (trainer.max_staleness_steps + 1)
    trainer._phase_gauge = _NoopPhaseGauge()
    trainer._loop_gauges = SimpleNamespace(set=lambda *args, **kwargs: None)
    trainer.all_metrics, trainer.all_timings = {}, {}
    trainer.resume_mode = ResumeMode.LATEST if resume else ResumeMode.NONE
    trainer._ray_gpu_monitor = trainer._vllm_metrics_scraper = trainer.ref_model = None
    trainer.trajectory_logger = None
    trainer.tokenizer = SimpleNamespace(decode=lambda ids: "")
    trainer.tracker = SimpleNamespace(log=lambda *args, **kwargs: None, finish=lambda: None)
    trainer.dispatch = SimpleNamespace(
        save_weights_for_sampler=_async_noop, get_timing_metrics=dict, finalize_pending_saves=lambda name: None
    )
    trainer.init_weight_sync_state = trainer._profiler_start = trainer._profiler_step = lambda: None
    trainer._profiler_stop = lambda: None

    dataset = [[{"prompt": f"p{i}", "env_class": None, "env_extras": {}, "uid": str(i)}] for i in range(num_prompts)]
    trainer.train_dataloader = StatefulDataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])
    trainer.async_train_dataloader = _AsyncDataloader(trainer.train_dataloader, mini_batch_size)
    trainer._staleness_manager = _AsyncStalenessManager(
        max_concurrent_generation_groups=trainer.num_parallel_generation_workers,
        mini_batch_size=mini_batch_size,
        max_staleness_steps=trainer.max_staleness_steps,
    )
    trainer.num_steps_per_epoch = num_prompts // mini_batch_size
    trainer.total_training_steps = trainer.num_steps_per_epoch * epochs

    async def generate(generator_input):
        await asyncio.sleep(0)
        trajectory_ids = generator_input["trajectory_ids"]
        n = len(trajectory_ids)
        has_variance = trajectory_ids[0].instance_id in kept_uids
        return {
            "prompt_token_ids": [[0]] * n,
            "response_ids": [[1]] * n,
            "rewards": [1.0] + [0.0] * (n - 1) if has_variance else [0.0] * n,
            "loss_masks": [[1]] * n,
            "stop_reasons": ["stop"] * n,
            "rollout_logprobs": None,
            "is_last_step": [True] * n,
            "trajectory_ids": trajectory_ids,
            "rollout_metrics": {},
        }

    trainer.generator = SimpleNamespace(generate=generate)
    trainer.postprocess_generator_output = lambda generator_output, uids, **kwargs: (generator_output, uids)
    trainer.convert_to_training_input = lambda generator_output, uids: None

    convert = trainer.convert_generation_group_mini_batch_to_training_input

    def convert_and_record(kept, dropped):
        convert(kept, dropped)
        return trainer.epoch, [g.uid for g in kept], trainer.all_metrics["async/num_groups_dropped"]

    trainer.convert_generation_group_mini_batch_to_training_input = convert_and_record

    trained = []

    async def run_training(step):
        if trainer.global_step == crash_at_step:
            raise _Crash
        trained.append(step)
        return {}

    trainer._run_training = run_training
    return trainer, trained


async def _async_noop(*args, **kwargs):
    return None


async def _cancel_leftover_tasks():
    """Cancel the generation workers a crashed train() left running."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _drawn_in(uid, step_epoch):
    """(prompt uid, epoch the group was drawn in) for a group trained in ``step_epoch``: a carried group's first
    ``@carry<epoch>`` suffix names the epoch it was drawn in."""
    prompt_uid, *carried_out_of = uid.split("@carry")
    return prompt_uid, int(carried_out_of[0]) if carried_out_of else step_epoch


def _assert_each_kept_draw_trains_once(trained, carried, num_prompts, mini_batch_size, epochs, kept_uids):
    """Every kept group of every epoch trains exactly once, in a full mini-batch, except the last epoch's carry."""
    stream = [str(i) for i in range(num_prompts // mini_batch_size * mini_batch_size)]
    expected = sorted((epoch, uid) for epoch in range(epochs) for uid in stream if uid in kept_uids)
    assert all(len(kept) == mini_batch_size for _, kept, _ in trained)
    drawn = [_drawn_in(uid, epoch) for epoch, kept, _ in trained for uid in kept]
    drawn += [_drawn_in(g.uid, epochs - 1) for g in carried]
    assert sorted((epoch, uid) for uid, epoch in drawn) == expected
    assert len(carried) == len(expected) % mini_batch_size


def _assert_each_drop_counted_once(trained, trainer, num_dropped_draws):
    """Every dropped group counts toward exactly one step's async/num_groups_dropped, except those dropped after the
    last step, which are still carried as a count."""
    assert sum(num_dropped for _, _, num_dropped in trained) + trainer._num_carried_dropped == num_dropped_draws


@pytest.mark.asyncio
async def test_train_carries_partial_mini_batch_across_epochs(stub_backend):
    """An epoch that runs out of prompts mid mini-batch carries its kept groups into the next epoch.

    3 of each epoch's 8 drawn prompts have reward variance and the mini-batch is 4, so no single epoch fills a
    mini-batch. Over 3 epochs the 9 kept groups make 2 steps, and the last 1 is left untrained.
    """
    trainer, trained = _carry_test_trainer(num_prompts=10, mini_batch_size=4, epochs=3, kept_uids={"1", "2", "5"})

    await trainer.train()  # runs the epoch-end staleness / trained-count checks along the way

    assert len(trained) == 2
    # Each step starts from the previous epoch's kept groups, re-keyed by epoch so they cannot collide with this
    # epoch's own draw of the same prompt ("1@carry0" and "1" train in one mini-batch, through the step-wise
    # validation of the concatenated batch).
    (epoch_1, kept_1, num_dropped_1), (epoch_2, kept_2, _) = trained
    assert (epoch_1, epoch_2) == (1, 2)
    assert kept_1 == ["1@carry0", "2@carry0", "5@carry0", "1"]
    assert kept_2[:2] == ["2@carry1", "5@carry1"] and sorted(kept_2[2:]) == ["1", "2"]
    # The step that trains carried groups also counts the groups dropped while collecting them: epoch 0's 5, plus
    # its own "0".
    assert num_dropped_1 == 6
    assert trainer.global_step == 3
    assert [g.uid for g in trainer._carried_groups] == ["5@carry2"]
    _assert_each_kept_draw_trains_once(trained, trainer._carried_groups, 10, 4, 3, {"1", "2", "5"})
    _assert_each_drop_counted_once(trained, trainer, num_dropped_draws=3 * 5)


@pytest.mark.asyncio
async def test_train_carry_that_covers_an_epochs_drops(stub_backend):
    """A carried-in partial batch can fill the gap an epoch's drops leave, so the epoch trains a full
    num_steps_per_epoch steps with kept groups still left over; those are carried on, not lost.

    7 of 8 prompts are kept and the mini-batch is 4: epoch 0 trains 1 step and carries 3; epoch 1 trains 2 steps
    (its full count) from 3 + 7 kept groups and carries the other 2.
    """
    kept_uids = {str(i) for i in range(1, 8)}
    trainer, trained = _carry_test_trainer(num_prompts=8, mini_batch_size=4, epochs=3, kept_uids=kept_uids)

    await trainer.train()

    assert [epoch for epoch, _, _ in trained] == [0, 1, 1, 2, 2]
    _assert_each_kept_draw_trains_once(trained, trainer._carried_groups, 8, 4, 3, kept_uids)
    _assert_each_drop_counted_once(trained, trainer, num_dropped_draws=3 * 1)


@pytest.mark.asyncio
async def test_train_epochs_without_a_kept_group_carry_only_a_count(stub_backend):
    """Epochs that keep no group take no step, and the carried state stays O(1) however many of them pass: the
    trainer holds no generated group, only the number dropped."""
    trainer, trained = _carry_test_trainer(num_prompts=8, mini_batch_size=4, epochs=10, kept_uids=set())

    await trainer.train()

    assert trained == []
    held = [v for value in vars(trainer).values() if isinstance(value, list) for v in value]
    assert not any(isinstance(v, GeneratedOutputGroup) for v in held)
    assert trainer._num_carried_dropped == 10 * 8


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ckpt_interval, save_on_epoch_end, crash_at_step",
    [
        # Periodic checkpoints after a step that trained carried groups: the checkpoint's consumed set holds their
        # re-keyed uids, which count as trained but never match a drawn prompt.
        (1, False, 2),
        (1, False, 3),
        # The end-of-epoch checkpoint, written while a carry is pending: the carried prompts are unconsumed in it,
        # so the resumed run regenerates them.
        (1000, True, 1),
        (1000, True, 2),
    ],
)
async def test_train_resume_with_carry(stub_backend, tmp_path, ckpt_interval, save_on_epoch_end, crash_at_step):
    """A run that crashes after a carry and resumes from its latest checkpoint trains every kept draw exactly
    once across the two runs, the same steps as an uninterrupted run."""
    setup = dict(
        num_prompts=10,
        mini_batch_size=4,
        epochs=4,
        kept_uids={"1", "2", "5"},
        ckpt_path=str(tmp_path),
        ckpt_interval=ckpt_interval,
        save_on_epoch_end=save_on_epoch_end,
    )
    uninterrupted, uninterrupted_trained = _carry_test_trainer(**{**setup, "ckpt_path": str(tmp_path / "ref")})
    await uninterrupted.train()

    crashed, crashed_trained = _carry_test_trainer(**setup, crash_at_step=crash_at_step)
    with pytest.raises(_Crash):
        await crashed.train()
    await _cancel_leftover_tasks()
    # Every step before the crash is in the checkpoint the resume loads.
    assert len(crashed_trained) == crash_at_step - 1

    resumed, resumed_trained = _carry_test_trainer(**setup, resume=True)
    await resumed.train()

    trained = crashed_trained + resumed_trained
    assert sorted(kept for _, kept, _ in trained) == sorted(kept for _, kept, _ in uninterrupted_trained)
    _assert_each_kept_draw_trains_once(trained, resumed._carried_groups, 10, 4, 4, {"1", "2", "5"})


@pytest.mark.asyncio
async def test_train_resume_from_max_training_steps_checkpoint(stub_backend, tmp_path):
    """Resuming from the checkpoint a max_training_steps stop writes, which is labelled one step past the last
    trained step, still drains the epoch it stopped in.

    The stop lands in epoch 0 after its first step, with the epoch's last kept groups still to come.
    """
    setup = dict(num_prompts=8, mini_batch_size=4, epochs=3, kept_uids={str(i) for i in range(1, 8)})
    stopped, stopped_trained = _carry_test_trainer(
        **setup, ckpt_path=str(tmp_path), ckpt_interval=1000, max_training_steps=1
    )
    await stopped.train()
    assert len(stopped_trained) == 1

    resumed, resumed_trained = _carry_test_trainer(**setup, ckpt_path=str(tmp_path), ckpt_interval=1000, resume=True)
    await resumed.train()

    trained = stopped_trained + resumed_trained
    assert [epoch for epoch, _, _ in trained] == [0, 1, 1, 2, 2]
    _assert_each_kept_draw_trains_once(trained, resumed._carried_groups, 8, 4, 3, setup["kept_uids"])
