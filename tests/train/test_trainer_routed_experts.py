"""``RayPPOTrainer.convert_to_training_input`` with rollout routes (R3). Megatron gets a dense route tensor padded like
``sequences`` plus its router padding mask. FSDP gets one ``[captured_i, layers, topk]`` tensor per row, so the batch
holds only the captured tokens' routes, and the rows the DP padding adds copy row 0 together with its routes."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.hf_router_replay import token_routes
from skyrl.train.trainer import RayPPOTrainer
from tests.train.util import example_dummy_config

NUM_LAYERS = 3
TOPK = 2
PROMPTS = [[1, 2], [3, 4, 5], [6]]
RESPONSES = [[7, 8, 9], [10], [11, 12, 13, 14]]
DP_SIZE = 4  # 3 rows -> one padding row


class DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def make_trainer(strategy: str) -> RayPPOTrainer:
    cfg = example_dummy_config()
    cfg.trainer.strategy = strategy
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.step_wise_trajectories = True
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=SimpleNamespace(pad_token_id=0),
        train_dataset=DummyDataset(),
        eval_dataset=DummyDataset(),
        inference_engine_client=None,
        generator=MagicMock(),
    )
    trainer.dispatch = SimpleNamespace(get_lcm_dp_size=lambda: DP_SIZE)
    return trainer


def make_generator_output() -> dict:
    rng = np.random.default_rng(0)
    # a route for every token but each row's last, which vLLM never forwards; int64 as a client could decode them
    routes = [rng.integers(0, 8, (len(p) + len(r) - 1, NUM_LAYERS, TOPK)) for p, r in zip(PROMPTS, RESPONSES)]
    return {
        "prompt_token_ids": PROMPTS,
        "response_ids": RESPONSES,
        "rewards": [[0.0] * len(r) for r in RESPONSES],
        "loss_masks": [[1] * len(r) for r in RESPONSES],
        "rollout_logprobs": None,
        "rollout_expert_indices": routes,
    }


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_fsdp_batch_holds_per_row_routes(strategy):
    generator_output = make_generator_output()
    batch = make_trainer(strategy).convert_to_training_input(generator_output, ["p0", "p0", "p1"])
    routes = batch["rollout_expert_indices"]
    assert isinstance(routes, TensorList)
    assert batch["router_padding_mask"] is None
    expected = [torch.from_numpy(r.astype(np.uint8)) for r in generator_output["rollout_expert_indices"]]
    assert routes == TensorList(expected + expected[:1])  # the padding row copies row 0
    # and HFModelWrapper's packed layout puts each row's routes on the row's first tokens
    layout = token_routes(routes, batch["attention_mask"], packed=True)
    lengths = batch["attention_mask"].sum(dim=1).tolist()
    starts = np.cumsum([0, *lengths[:-1]])
    for start, row_routes in zip(starts, routes.tensors):
        assert torch.equal(layout[start : start + len(row_routes)], row_routes)


def test_megatron_batch_holds_dense_routes_and_the_padding_mask():
    batch = make_trainer("megatron").convert_to_training_input(make_generator_output(), ["p0", "p0", "p1"])
    max_total = max(len(p) + len(r) for p, r in zip(PROMPTS, RESPONSES))
    assert batch["rollout_expert_indices"].shape == (DP_SIZE, max_total, NUM_LAYERS, TOPK)
    assert batch["router_padding_mask"].shape == (DP_SIZE, max_total)
