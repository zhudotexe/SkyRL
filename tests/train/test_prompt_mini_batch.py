"""
Tests for prompt-based mini-batching.

uv run --isolated --extra dev pytest tests/train/test_prompt_mini_batch.py -v
"""

from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import MeshDispatch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import compute_prompt_end_indices, compute_prompt_mini_batch_boundaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_uids_stepwise(
    prompts: List[Tuple[str, int, List[int]]],
) -> List[str]:
    """Build uid list for a step-wise batch.

    Args:
        prompts: List of (instance_id, n_samples, turns_per_sample) tuples.
            ``turns_per_sample`` is a list of length ``n_samples`` giving
            the number of turns for each trajectory of that prompt.

    Returns:
        Flat uid list — same uid for all sequences of the same prompt.
    """
    uids: List[str] = []
    for instance_id, _, turns_list in prompts:
        for num_turns in turns_list:
            for _ in range(num_turns):
                uids.append(instance_id)
    return uids


def _make_uids_fixed(num_prompts: int, n_samples: int) -> List[str]:
    """Build uid list for a non-step-wise batch (fixed n_samples per prompt)."""
    return [f"p{i}" for i in range(num_prompts) for _ in range(n_samples)]


def _make_batch(num_sequences: int, seq_len: int = 8) -> TrainingInputBatch:
    """Create a minimal TrainingInputBatch with the given number of sequences."""
    batch = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (num_sequences, seq_len)),
            "attention_mask": torch.ones(num_sequences, seq_len, dtype=torch.long),
            "response_mask": torch.ones(num_sequences, seq_len, dtype=torch.long),
            "advantages": torch.randn(num_sequences, seq_len),
            "loss_mask": torch.ones(num_sequences, seq_len, dtype=torch.float),
            "is_last_step": torch.zeros(num_sequences, dtype=torch.bool),
        }
    )
    batch.metadata = {}
    return batch


# ---------------------------------------------------------------------------
# Tests for compute_prompt_end_indices
# ---------------------------------------------------------------------------


class TestComputePromptEndIndices:
    def test_non_stepwise(self):
        """Each prompt repeated n_samples times."""
        uids = _make_uids_fixed(num_prompts=3, n_samples=4)
        end_indices = compute_prompt_end_indices(uids)
        assert end_indices == [4, 8, 12]

    def test_stepwise_variable_turns(self):
        uids = _make_uids_stepwise([
            ("p0", 2, [3, 2]),  # 5 sequences
            ("p1", 2, [1, 4]),  # 5 sequences
        ])
        end_indices = compute_prompt_end_indices(uids)
        assert end_indices == [5, 10]

    def test_single_sequence_per_prompt(self):
        uids = ["a", "b", "c"]
        assert compute_prompt_end_indices(uids) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Tests for compute_prompt_mini_batch_boundaries
# ---------------------------------------------------------------------------


class TestComputePromptMiniBatchBoundaries:
    def test_uniform_turns_single_sample(self):
        """All prompts have 1 sample with 1 turn => same as fixed-size."""
        uids = _make_uids_fixed(num_prompts=4, n_samples=1)
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=2)
        assert boundaries == [(0, 2), (2, 4)]

    def test_uniform_turns_multiple_samples(self):
        """All prompts have n_samples=2, each with 1 turn."""
        uids = _make_uids_fixed(num_prompts=4, n_samples=2)
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=2)
        assert boundaries == [(0, 4), (4, 8)]

    def test_variable_turns(self):
        """Step-wise: prompts have variable numbers of turns."""
        uids = _make_uids_stepwise([
            ("p0", 2, [3, 2]),  # 5 seqs
            ("p1", 2, [1, 4]),  # 5 seqs
            ("p2", 2, [2, 1]),  # 3 seqs
            ("p3", 2, [1, 1]),  # 2 seqs
        ])
        assert len(uids) == 15
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=2)
        assert boundaries == [(0, 10), (10, 15)]

    def test_single_prompt_per_minibatch(self):
        uids = _make_uids_stepwise([
            ("p0", 3, [2, 1, 3]),  # 6 seqs
            ("p1", 3, [1, 1, 1]),  # 3 seqs
        ])
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=1)
        assert boundaries == [(0, 6), (6, 9)]

    def test_all_prompts_in_one_minibatch(self):
        uids = _make_uids_stepwise([
            ("p0", 1, [2]),
            ("p1", 1, [3]),
        ])
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=4)
        assert boundaries == [(0, 5)]

    @pytest.mark.parametrize(
        "n_prompts, n_samples, mb_prompts",
        [
            (8, 4, 4),
            (256, 5, 128),
            (16, 1, 4),
            (32, 8, 32),
            (128, 5, 64),
        ],
    )
    def test_non_stepwise_boundaries_are_uniform(self, n_prompts, n_samples, mb_prompts):
        """For non-step-wise, every boundary must be [i*mb_size, (i+1)*mb_size)
        where mb_size = mb_prompts * n_samples — identical to the old fixed-size slicing."""
        uids = _make_uids_fixed(n_prompts, n_samples)
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=mb_prompts)

        expected_mb_size = mb_prompts * n_samples
        assert len(boundaries) == n_prompts // mb_prompts
        for i, (start, end) in enumerate(boundaries):
            assert start == i * expected_mb_size
            assert end - start == expected_mb_size


# ---------------------------------------------------------------------------
# Tests for MeshDispatch.stage_chunks
# ---------------------------------------------------------------------------


class TestStageChunksVariable:
    def test_uniform_minibatches_dp1(self):
        """All mini-batches same size, dp_size=1 => no padding needed."""
        batch = _make_batch(10)
        batch.metadata["pad_size"] = 0
        boundaries = [(0, 5), (5, 10)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            mock_ray.put.side_effect = lambda x: x
            result = MeshDispatch.stage_chunks(dp_size=1, data=batch, mini_batch_boundaries=boundaries)

        assert len(result) == 2
        assert len(result[0]) == 1
        assert len(result[1]) == 1

    def test_variable_minibatches_dp2_padding(self):
        """Variable sizes with dp_size=2 => odd-sized mini-batches get padded."""
        batch = _make_batch(7)
        batch.metadata["pad_size"] = 0
        boundaries = [(0, 3), (3, 7)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            result = MeshDispatch.stage_chunks(dp_size=2, data=batch, mini_batch_boundaries=boundaries)

        assert len(result) == 2
        assert len(result[0]) == 2  # 3->4, split into 2
        assert len(result[1]) == 2  # 4, split into 2
        assert len(chunks_put[0]) + len(chunks_put[1]) == 4  # padded from 3

    def test_dp_size_4_heavy_padding(self):
        """dp_size=4, mini-batch of 5 => padded to 8."""
        batch = _make_batch(5)
        batch.metadata["pad_size"] = 0
        boundaries = [(0, 5)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            result = MeshDispatch.stage_chunks(dp_size=4, data=batch, mini_batch_boundaries=boundaries)

        assert len(result) == 1
        assert len(result[0]) == 4
        for chunk in chunks_put:
            assert len(chunk) == 2

    def test_loss_mask_zero_for_padding(self):
        """Padding entries should have loss_mask=0."""
        batch = _make_batch(3, seq_len=4)
        batch.metadata = {"pad_size": 0}
        boundaries = [(0, 3)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            MeshDispatch.stage_chunks(dp_size=2, data=batch, mini_batch_boundaries=boundaries)

        all_loss_masks = torch.cat([c["loss_mask"] for c in chunks_put], dim=0)
        assert all_loss_masks[:3].sum() == 3 * 4
        assert all_loss_masks[3].sum() == 0

    def test_is_last_step_true_for_padding(self):
        """Padding entries should have is_last_step=True."""
        batch = _make_batch(3)
        batch.metadata = {"pad_size": 0}
        boundaries = [(0, 3)]

        with patch("skyrl.backends.skyrl_train.distributed.dispatch.ray") as mock_ray:
            chunks_put = []
            mock_ray.put.side_effect = lambda x: (chunks_put.append(x), len(chunks_put) - 1)[1]
            MeshDispatch.stage_chunks(dp_size=2, data=batch, mini_batch_boundaries=boundaries)

        all_is_last = torch.cat([c["is_last_step"] for c in chunks_put], dim=0)
        assert all_is_last[3].item() is True


# ---------------------------------------------------------------------------
# Tests for optimizer step count invariance
# ---------------------------------------------------------------------------


class TestOptimizerStepCount:
    def test_num_minibatches_equals_train_over_policy(self):
        """Number of mini-batches = train_batch_size / policy_mini_batch_size."""
        uids = _make_uids_stepwise([
            ("p0", 2, [3, 2]),
            ("p1", 2, [1, 4]),
            ("p2", 2, [2, 1]),
            ("p3", 2, [1, 1]),
        ])
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=2)
        assert len(boundaries) == 4 // 2

    def test_step_count_with_epochs(self):
        """Total optimizer steps = num_mini_batches * update_epochs_per_batch."""
        uids = _make_uids_stepwise([
            ("p0", 5, [3, 2, 1, 4, 2]),
            ("p1", 5, [1, 1, 1, 1, 1]),
            ("p2", 5, [2, 3, 1, 1, 2]),
            ("p3", 5, [1, 2, 3, 2, 1]),
        ])
        end_indices = compute_prompt_end_indices(uids)
        boundaries = compute_prompt_mini_batch_boundaries(end_indices, mini_batch_size_in_prompts=2)
        update_epochs = 3
        assert len(boundaries) * update_epochs == (4 // 2) * update_epochs

    def test_same_step_count_as_non_stepwise(self):
        """Step-wise and non-step-wise produce the same number of mini-batches."""
        train_batch_size = 256
        policy_mini_batch_size = 128
        n_samples = 5

        # Non-step-wise
        non_stepwise_uids = _make_uids_fixed(train_batch_size, n_samples)
        non_stepwise_ends = compute_prompt_end_indices(non_stepwise_uids)
        non_stepwise_bounds = compute_prompt_mini_batch_boundaries(non_stepwise_ends, policy_mini_batch_size)

        # Step-wise with variable turns
        prompts = []
        for i in range(train_batch_size):
            turns = [1 + (i * j) % 4 for j in range(n_samples)]
            prompts.append((f"p{i}", n_samples, turns))
        stepwise_uids = _make_uids_stepwise(prompts)
        stepwise_ends = compute_prompt_end_indices(stepwise_uids)
        stepwise_bounds = compute_prompt_mini_batch_boundaries(stepwise_ends, policy_mini_batch_size)

        assert len(stepwise_bounds) == len(non_stepwise_bounds) == 2

        # Non-step-wise boundaries should be uniform
        assert non_stepwise_bounds == [(0, 640), (640, 1280)]
