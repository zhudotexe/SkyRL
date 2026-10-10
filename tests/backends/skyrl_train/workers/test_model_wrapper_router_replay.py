"""CPU tests for HFModelWrapper's rollout routing replay (R3): each row's routes reach the routers at the positions of
the row's own tokens, both padded and packed (``remove_microbatch_padding``), and stay in place through a
gradient-checkpointed backward.

HFModelWrapper imports ``flash_attn.bert_padding``; without flash_attn (CPU dev boxes) a pure-torch stand-in for
its ``unpad_input`` / ``pad_input`` is installed for the duration of each test. The packed forward runs on SDPA,
whose mask creation recovers the sequence boundaries from the packed position ids, so packed and padded logprobs
agree.
"""

import importlib
import sys
import types

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from skyrl.backends.skyrl_train.training_batch import TensorList

NUM_LAYERS = 2
NUM_EXPERTS = 8
TOPK = 2
VOCAB = 64


def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return hidden_states.flatten(0, 1)[indices], indices, cu_seqlens, int(seqlens.max()), seqlens


def _pad_input(hidden_states, indices, batch, seqlen):
    out = torch.zeros(batch * seqlen, *hidden_states.shape[1:], dtype=hidden_states.dtype)
    out[indices] = hidden_states
    return out.view(batch, seqlen, *hidden_states.shape[1:])


@pytest.fixture
def HFModelWrapper(monkeypatch):
    try:
        import flash_attn.bert_padding  # noqa: F401
    except ImportError:
        bert_padding = types.ModuleType("flash_attn.bert_padding")
        bert_padding.unpad_input = _unpad_input
        bert_padding.pad_input = _pad_input
        flash_attn = types.ModuleType("flash_attn")
        flash_attn.bert_padding = bert_padding
        monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
        monkeypatch.setitem(sys.modules, "flash_attn.bert_padding", bert_padding)
    # import under the stub, and drop the stub-bound module again on teardown
    name = "skyrl.backends.skyrl_train.workers.model_wrapper"
    monkeypatch.setitem(sys.modules, name, None)
    del sys.modules[name]
    return importlib.import_module(name).HFModelWrapper


@pytest.fixture
def batch():
    """Two left-padded rows, as convert_prompts_responses_to_batch_tensors lays them out."""
    gen = torch.Generator().manual_seed(1)
    attention_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    sequences = torch.randint(1, VOCAB, attention_mask.shape, generator=gen) * attention_mask
    return sequences, attention_mask


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
    model = Qwen3MoeForCausalLM(cfg).float()
    model.config._attn_implementation = "sdpa"
    # as HFModelWrapper sets it for a model it loads; a KV cache would turn off the packed-sequence mask
    model.config.use_cache = False
    return model


def own_routes(wrapper, sequences, attention_mask) -> TensorList:
    """The wrapper's own top-k routes, laid out as the trainer's batch carries them on FSDP: per row, the routes of
    every real token but the last, which vLLM never forwards."""
    captured = {}
    routers = [m for m in wrapper.model.modules() if isinstance(m, Qwen3MoeTopKRouter)]
    hooks = [
        router.register_forward_hook(lambda _m, _i, out, layer=layer: captured.__setitem__(layer, out[2]))
        for layer, router in enumerate(routers)
    ]
    with torch.no_grad():
        wrapper(sequences, sequences.shape[1] - 1, attention_mask=attention_mask)
    for hook in hooks:
        hook.remove()
    routes = torch.stack([captured[layer] for layer in range(NUM_LAYERS)], dim=1).to(torch.uint8)
    routes = routes.view(*sequences.shape, NUM_LAYERS, TOPK)
    return TensorList([row[mask.bool()][:-1].clone() for row, mask in zip(routes, attention_mask)])


def other_routes(routes: torch.Tensor) -> torch.Tensor:
    """Valid routes that differ from ``routes`` in every row: shift each expert id by one."""
    return (routes + 1) % NUM_EXPERTS


def logprobs(wrapper, sequences, attention_mask, routes=None) -> torch.Tensor:
    with torch.no_grad():
        out = wrapper(sequences, sequences.shape[1] - 1, attention_mask=attention_mask, rollout_expert_indices=routes)
    # compare the logprobs at real positions only; the packed path fills padding with zeros
    return out * attention_mask[:, :-1]


@pytest.mark.parametrize("packed", [False, True])
def test_own_routes_reproduce_topk_logprobs(HFModelWrapper, batch, packed):
    """Every logprob comes from a position with a captured route (the last token's output predicts no token), so
    replaying the rows' own routes reproduces the top-k forward exactly, but only if each route lands on its token."""
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    reference = logprobs(wrapper, sequences, attention_mask)
    routes = own_routes(wrapper, sequences, attention_mask)
    # the packed forward runs on SDPA here; production requires flash attention for it
    wrapper.remove_microbatch_padding = packed
    assert torch.allclose(logprobs(wrapper, sequences, attention_mask, routes), reference, atol=1e-5)


@pytest.mark.parametrize("packed", [False, True])
def test_each_row_replays_its_own_routes(HFModelWrapper, batch, packed):
    """Changing only row 1's routes changes only row 1's logprobs."""
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    reference = logprobs(wrapper, sequences, attention_mask)
    routes = own_routes(wrapper, sequences, attention_mask)
    routes = TensorList([routes[0], other_routes(routes[1])])
    wrapper.remove_microbatch_padding = packed
    replayed = logprobs(wrapper, sequences, attention_mask, routes)
    assert torch.allclose(replayed[0], reference[0], atol=1e-5)
    assert not torch.allclose(replayed[1], reference[1], atol=1e-4)


def test_forward_without_routes_clears_the_previous_routes(HFModelWrapper, batch):
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    reference = logprobs(wrapper, sequences, attention_mask)
    routes = own_routes(wrapper, sequences, attention_mask)
    logprobs(wrapper, sequences, attention_mask, TensorList([other_routes(r) for r in routes.tensors]))
    assert torch.allclose(logprobs(wrapper, sequences, attention_mask), reference, atol=1e-6)


def _grads(wrapper, sequences, attention_mask, routes, checkpointing: bool) -> dict[str, torch.Tensor]:
    wrapper.model.zero_grad()
    if checkpointing:
        wrapper.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        wrapper.model.gradient_checkpointing_disable()
    wrapper.model.train()
    out = wrapper(sequences, sequences.shape[1] - 1, attention_mask=attention_mask, rollout_expert_indices=routes)
    (out * attention_mask[:, :-1]).square().sum().backward()
    wrapper.model.eval()
    return {name: p.grad.clone() for name, p in wrapper.model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("packed", [False, True])
def test_checkpointed_grads_match_plain_grads_under_replay(HFModelWrapper, batch, packed):
    """The routes the wrapper sets stay installed until backward, so the checkpoint recompute replays them too (the
    routes differ from the top-k, so a recompute that re-routed would change the gradients)."""
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    routes = TensorList([other_routes(r) for r in own_routes(wrapper, sequences, attention_mask).tensors])
    wrapper.remove_microbatch_padding = packed
    plain = _grads(wrapper, sequences, attention_mask, routes, checkpointing=False)
    checkpointed = _grads(wrapper, sequences, attention_mask, routes, checkpointing=True)
    assert plain.keys() == checkpointed.keys()
    for name in plain:
        assert torch.allclose(plain[name], checkpointed[name], atol=1e-6), name


def test_a_router_the_moe_block_bypasses_fails_loudly(HFModelWrapper, batch):
    """A MoE block that routes without the patched router (as a fused kernel might) would ignore the routes."""
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    routes = own_routes(wrapper, sequences, attention_mask)
    gate = wrapper.model.model.layers[1].mlp.gate
    gate.forward = lambda hidden_states: Qwen3MoeTopKRouter.forward(gate, hidden_states)
    with pytest.raises(RuntimeError, match="1 of 2 MoE routers replayed"):
        logprobs(wrapper, sequences, attention_mask, routes)


def test_more_routes_than_row_tokens_raise(HFModelWrapper, batch):
    sequences, attention_mask = batch
    wrapper = HFModelWrapper(tiny_moe())
    routes = own_routes(wrapper, sequences, attention_mask)
    routes = TensorList([torch.cat([routes[0], routes[0][:2]]), routes[1]])  # row 0 has 7 tokens
    with pytest.raises(ValueError, match="8 rollout routes for a row of 7 tokens"):
        logprobs(wrapper, sequences, attention_mask, routes)
