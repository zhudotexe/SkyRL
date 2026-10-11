"""Rollout routing replay (R3) for HF Qwen3-MoE models on the FSDP backend.

vLLM records the experts it routed each token to (``generator.inference_engine.enable_return_routed_experts``).
Replaying those ids in the trainer's forward makes every MoE layer use the experts the sampler used, so the
train/rollout logprob gap is not widened by tokens whose top-k flips between the two engines. The router still
computes its softmax, and the replayed experts' probabilities are gathered from it, so the router weight keeps its
gradient (as in Megatron's R3).

Kept free of flash_attn so it can be tested on CPU.
"""

import functools
import re
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.replay_utils import make_replay_padding_indices

# a router's module name ends in `layers.<i>.mlp.gate`; <i> is the layer id vLLM records routes under
_ROUTER_LAYER = re.compile(r"layers\.(\d+)\.mlp\.gate$")


class RouterReplay:
    """The expert ids every patched router of one model replays.

    ``routes`` is ``(tokens, layers, topk)``, row-aligned with the tokens the routers see (``token_routes`` lays a
    microbatch out), or None to route by top-k. ``set`` is called before each forward, and the routes stay in place
    after it: gradient checkpointing re-runs the routers during backward, and that recompute must route exactly as
    the forward did.
    """

    def __init__(self, model: nn.Module):
        self.routes: Optional[torch.Tensor] = None
        self.num_routers = 0
        self.num_replayed = 0
        for name, module in model.named_modules():
            if isinstance(module, Qwen3MoeTopKRouter):
                layer = int(_ROUTER_LAYER.search(name).group(1))
                module.forward = functools.partial(_replay_forward, module, self, layer)
                self.num_routers += 1

    def set(self, routes: Optional[torch.Tensor]) -> None:
        if routes is not None and self.num_routers == 0:
            raise ValueError("rollout_expert_indices were passed, but the model has no Qwen3MoeTopKRouter to replay")
        self.routes = routes
        self.num_replayed = 0

    def set_rows(self, routes: Optional[TensorList], attention_mask: torch.Tensor, packed: bool) -> None:
        """``set`` one microbatch's per-row routes, laid out by ``token_routes``, or None to route by top-k."""
        self.set(None if routes is None else token_routes(routes, attention_mask, packed))

    def check_replayed(self) -> None:
        """Raise unless every patched router replayed the routes in the forward since ``set``. A MoE block that routes
        without calling its router module (a fused kernel, say) would otherwise ignore the routes silently."""
        if self.routes is not None and self.num_replayed != self.num_routers:
            raise RuntimeError(
                f"{self.num_replayed} of {self.num_routers} MoE routers replayed the rollout routes in this forward"
            )


def token_routes(routes: TensorList, attention_mask: torch.Tensor, packed: bool) -> torch.Tensor:
    """Lay one microbatch's per-row routes out in the order its routers see the tokens.

    ``routes[i]`` is ``(captured_i, layers, topk)``: the routes of row i's first ``captured_i`` real tokens. vLLM never
    forwards a row's last generated token, nor an EOS appended after it, so those have none. Rows are left-padded, as
    ``attention_mask`` marks. The routers see the flattened ``(batch, seq)`` positions or, when ``packed``, only the
    real tokens, row by row (``unpad_input``'s order). Padding and uncaptured tokens get dummy routes of distinct
    experts: their outputs feed no logprob, or only a loss-masked one (the appended EOS).
    """
    seq_len = attention_mask.shape[1]
    chunks = []
    for row_routes, num_tokens in zip(routes.tensors, attention_mask.sum(dim=1).tolist(), strict=True):
        num_uncaptured = num_tokens - len(row_routes)
        if num_uncaptured < 0:
            raise ValueError(f"{len(row_routes)} rollout routes for a row of {num_tokens} tokens")
        num_padding = 0 if packed else seq_len - num_tokens
        chunks += [_dummy_routes(num_padding, row_routes), row_routes, _dummy_routes(num_uncaptured, row_routes)]
    return torch.cat(chunks)


def _dummy_routes(num_tokens: int, like: torch.Tensor) -> torch.Tensor:
    return make_replay_padding_indices((num_tokens, *like.shape[1:]), dtype=like.dtype, device=like.device)


def _replay_forward(router: Qwen3MoeTopKRouter, replay: RouterReplay, layer: int, hidden_states: torch.Tensor):
    """``Qwen3MoeTopKRouter.forward`` with the experts taken from ``replay.routes`` instead of the top-k."""
    if replay.routes is None:
        return Qwen3MoeTopKRouter.forward(router, hidden_states)
    hidden_states = hidden_states.reshape(-1, router.hidden_dim)
    if replay.routes.shape[0] != hidden_states.shape[0]:
        raise ValueError(f"{replay.routes.shape[0]} replayed routes for {hidden_states.shape[0]} router tokens")
    if replay.routes.shape[-1] != router.top_k:
        raise ValueError(f"replayed routes hold {replay.routes.shape[-1]} experts per token, the router {router.top_k}")
    router_logits = F.linear(hidden_states, router.weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    router_indices = replay.routes[:, layer].long()
    router_top_value = router_probs.gather(-1, router_indices)
    if router.norm_topk_prob:
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
    replay.num_replayed += 1
    return router_logits, router_top_value.to(router_logits.dtype), router_indices
