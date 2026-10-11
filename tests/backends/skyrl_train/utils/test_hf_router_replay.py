"""CPU tests for rollout routing replay (R3) on HF Qwen3-MoE routers, using a tiny random model."""

import pytest
import torch
from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from skyrl.backends.skyrl_train.training_batch import TensorList
from skyrl.backends.skyrl_train.utils.hf_router_replay import RouterReplay, token_routes

NUM_LAYERS = 3
NUM_EXPERTS = 8
TOPK = 2
VOCAB = 64


def tiny_moe() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3MoeConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        norm_topk_prob=True,
        max_position_embeddings=128,
    )
    return Qwen3MoeForCausalLM(cfg).float()


def topk_routes(model, ids: torch.Tensor) -> torch.Tensor:
    """The ``(tokens, layers, topk)`` experts the model's own top-k routes ``ids`` to."""
    captured = {}
    routers = [m for m in model.modules() if isinstance(m, Qwen3MoeTopKRouter)]
    hooks = [
        router.register_forward_hook(lambda _m, _i, out, layer=layer: captured.__setitem__(layer, out[2]))
        for layer, router in enumerate(routers)
    ]
    with torch.no_grad():
        model(ids)
    for hook in hooks:
        hook.remove()
    return torch.stack([captured[layer] for layer in range(len(routers))], dim=1)


def other_routes(routes: torch.Tensor) -> torch.Tensor:
    """Valid routes that differ from ``routes`` in every row: shift each expert id by one."""
    return (routes + 1) % NUM_EXPERTS


@pytest.fixture
def model_and_replay():
    model = tiny_moe()
    return model, RouterReplay(model)


@pytest.fixture
def ids():
    return torch.randint(0, VOCAB, (2, 10), generator=torch.Generator().manual_seed(1))


def test_replay_patches_every_router(model_and_replay):
    _, replay = model_and_replay
    assert replay.num_routers == NUM_LAYERS


def test_replaying_own_routes_reproduces_topk(model_and_replay, ids):
    model, replay = model_and_replay
    with torch.no_grad():
        reference = model(ids).logits
    # uint8, as the rollout engine returns them for up to 256 experts
    replay.set(topk_routes(model, ids).to(torch.uint8))
    with torch.no_grad():
        replayed = model(ids).logits
    assert torch.allclose(replayed, reference, atol=1e-6)


def test_replaying_other_routes_changes_logits(model_and_replay, ids):
    model, replay = model_and_replay
    with torch.no_grad():
        reference = model(ids).logits
    replay.set(other_routes(topk_routes(model, ids)))
    with torch.no_grad():
        replayed = model(ids).logits
    assert not torch.allclose(replayed, reference, atol=1e-4)


def test_replay_reads_each_layer_its_own_routes(model_and_replay, ids):
    """Changing only layer 1's routes changes the output: the routers index the layer dim, not just row 0."""
    model, replay = model_and_replay
    routes = topk_routes(model, ids)
    replay.set(routes)
    with torch.no_grad():
        reference = model(ids).logits
    changed = routes.clone()
    changed[:, 1] = other_routes(routes[:, 1])
    replay.set(changed)
    with torch.no_grad():
        replayed = model(ids).logits
    assert not torch.allclose(replayed, reference, atol=1e-4)


def test_unset_replay_routes_by_topk(model_and_replay, ids):
    model, replay = model_and_replay
    with torch.no_grad():
        reference = model(ids).logits
    replay.set(other_routes(topk_routes(model, ids)))
    with torch.no_grad():
        model(ids)
    replay.set(None)
    with torch.no_grad():
        assert torch.allclose(model(ids).logits, reference, atol=1e-6)


def _grads(model, replay, ids, routes, checkpointing: bool) -> dict[str, torch.Tensor]:
    model.zero_grad()
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    model.train()
    replay.set(routes)
    model(ids).logits.square().mean().backward()
    model.eval()
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_router_weight_gets_a_gradient_under_replay(model_and_replay, ids):
    model, replay = model_and_replay
    grads = _grads(model, replay, ids, other_routes(topk_routes(model, ids)), checkpointing=False)
    for layer in range(NUM_LAYERS):
        assert grads[f"model.layers.{layer}.mlp.gate.weight"].abs().sum() > 0


def test_checkpointed_grads_match_plain_grads_under_replay(model_and_replay, ids):
    """The routes stay installed through backward, so the checkpoint recompute replays them too."""
    model, replay = model_and_replay
    routes = other_routes(topk_routes(model, ids))
    plain = _grads(model, replay, ids, routes, checkpointing=False)
    checkpointed = _grads(model, replay, ids, routes, checkpointing=True)
    assert plain.keys() == checkpointed.keys()
    for name in plain:
        assert torch.allclose(plain[name], checkpointed[name], atol=1e-6), name


def test_route_count_must_match_router_tokens(model_and_replay, ids):
    model, replay = model_and_replay
    replay.set(topk_routes(model, ids)[:-1])
    with pytest.raises(ValueError, match="replayed routes for"):
        model(ids)


def test_route_topk_must_match_the_router(model_and_replay, ids):
    model, replay = model_and_replay
    replay.set(topk_routes(model, ids)[..., :1])
    with pytest.raises(ValueError, match="1 experts per token, the router 2"):
        model(ids)


def test_check_replayed_counts_the_routers_that_replayed(model_and_replay, ids):
    model, replay = model_and_replay
    replay.set(topk_routes(model, ids))
    with pytest.raises(RuntimeError, match="0 of 3 MoE routers"):
        replay.check_replayed()
    with torch.no_grad():
        model(ids)
    replay.check_replayed()
    replay.set(None)
    replay.check_replayed()


def test_token_routes_lays_rows_out_as_the_routers_see_them():
    """Row 0 has 3 real tokens and 2 captured routes, row 1 has 2 and 1; every other position gets a dummy route of
    distinct experts. Packed, the routers see only the real tokens, row by row; padded, every position."""
    a0, a1, b0, dummy = [[5, 6]], [[7, 1]], [[3, 4]], [[0, 1]]  # [layers=1, topk=2] each
    routes = TensorList([torch.tensor([a0, a1], dtype=torch.uint8), torch.tensor([b0], dtype=torch.uint8)])
    attention_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    assert token_routes(routes, attention_mask, packed=True).tolist() == [a0, a1, dummy, b0, dummy]
    assert token_routes(routes, attention_mask, packed=False).tolist() == [
        *[dummy, a0, a1, dummy],
        *[dummy, dummy, b0, dummy],
    ]


def test_routes_for_a_model_without_routers_raise():
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    replay = RouterReplay(Qwen3ForCausalLM(cfg))
    assert replay.num_routers == 0
    replay.set(None)
    with pytest.raises(ValueError, match="no Qwen3MoeTopKRouter"):
        replay.set(torch.zeros((4, 2, 2), dtype=torch.uint8))
