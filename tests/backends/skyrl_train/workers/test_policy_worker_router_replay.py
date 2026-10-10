"""CPU tests that the FSDP policy worker hands the rollout's routed experts (R3) to the model, both in the training
forward and in the forward that computes the old logprobs, as the per-row ``TensorList`` the trainer builds."""

from unittest.mock import MagicMock, patch

import torch

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.backends.skyrl_train.utils.hf_router_replay import token_routes
from skyrl.backends.skyrl_train.workers.worker_utils import TokenBasedBatchIterator
from tests.backends.skyrl_train.workers.test_sft_loss_fn_outputs_gate import (
    BATCH_SIZE,
    NUM_ACTIONS,
    SEQ_LEN,
    _make_cpu_policy_worker,
    _make_experience,
)

NUM_LAYERS = 3
TOPK = 2


def make_routes() -> TensorList:
    """Per-row routes for every token of a SEQ_LEN row but the last."""
    return TensorList(
        [torch.randint(0, 8, (SEQ_LEN - 1, NUM_LAYERS, TOPK), dtype=torch.uint8) for _ in range(BATCH_SIZE)]
    )


def make_batch(routes: TensorList) -> TrainingInputBatch:
    batch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)),
            "attention_mask": torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
            "rollout_expert_indices": routes,
        }
    )
    batch.metadata = {"response_length": NUM_ACTIONS}
    return batch


def test_training_forward_replays_the_experience_routes():
    worker = _make_cpu_policy_worker()
    worker.cfg.algorithm.policy_loss_type = "regular"
    worker.cfg.algorithm.use_kl_loss = False
    worker.cfg.algorithm.use_entropy_loss = False
    log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    worker.model = MagicMock(return_value=(log_probs, {"entropy": torch.zeros(BATCH_SIZE, SEQ_LEN)}))
    experience = _make_experience()
    experience.action_log_probs = log_probs.clone()
    routes = make_routes()
    experience.rollout_expert_indices = routes
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", MagicMock()):
        worker._forward_backward_micro(experience, microbatch_weight=1.0, loss_fn="regular")
    assert worker.model.call_args.kwargs["rollout_expert_indices"] == routes


def test_old_logprob_forward_replays_the_batch_routes():
    worker = _make_cpu_policy_worker()
    worker.model = MagicMock(return_value=torch.zeros(BATCH_SIZE, NUM_ACTIONS))
    routes = make_routes()
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", MagicMock()):
        worker._forward_micro_batch(make_batch(routes))
    assert worker.model.call_args.kwargs["rollout_expert_indices"] == routes


def test_experience_moves_per_row_routes_to_the_device():
    experience = _make_experience()
    experience.rollout_expert_indices = make_routes()
    experience.to_device(torch.device("meta"))
    assert experience.rollout_expert_indices.device == torch.device("meta")


def test_padding_microbatch_carries_no_captured_routes():
    """A DP rank with fewer microbatches pads with a one-token row, which then gets a single dummy route."""
    iterator = TokenBasedBatchIterator(make_batch(make_routes()), max_tokens_per_microbatch=SEQ_LEN)
    padding = iterator._create_padding_microbatch()
    routes = padding["rollout_expert_indices"]
    assert [tuple(t.shape) for t in routes.tensors] == [(0, NUM_LAYERS, TOPK)]
    assert token_routes(routes, padding["attention_mask"], packed=True).shape == (1, NUM_LAYERS, TOPK)
