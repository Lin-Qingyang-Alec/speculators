"""Offline CPU checks for the experimental native-initialized parallel head."""

import json

import pytest
import torch
from torch import nn
from transformers import AutoConfig

from speculators import (
    PMTPDraftModel,
    PMTPSpeculatorConfig,
    SpeculatorModel,
    SpeculatorModelConfig,
    SpeculatorsConfig,
    VerifierConfig,
)
from speculators.cli.stitch import stitch
from speculators.convert.mtp.converter import MTPConverter
from speculators.models.mtp.data import shift_batch_mtp
from speculators.proposals import GreedyTokenProposalConfig
from speculators.train.config import TrainConfig
from speculators.train.data import CollateFn


def make_model(model_type="qwen3", steps=4):
    torch.manual_seed(42)
    rope_parameters = {"rope_type": "default", "rope_theta": 10000.0}
    if model_type == "qwen3_5_moe_text":
        rope_parameters.update(partial_rotary_factor=1.0, mrope_section=[2, 3, 3])
    decoder = AutoConfig.for_model(
        model_type,
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        layer_types=["full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        rope_parameters=rope_parameters,
    )
    config = PMTPSpeculatorConfig(
        transformer_layer_config=decoder,
        pmtp_max_anchors=3,
        pmtp_logit_chunk_size=2,
        speculators_config=SpeculatorsConfig(
            algorithm="pmtp",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=steps)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path=None, architectures=["DummyVerifier"]),
        ),
    )
    model = PMTPDraftModel(config)
    # Standalone decoder construction does not initialize raw MoE expert/router
    # Parameters. Production loads them from the native checkpoint; tests must
    # initialize them explicitly rather than reading torch.empty() storage.
    for parameter in model.mtp_layers.parameters():
        if parameter.ndim > 1:
            nn.init.normal_(parameter, std=0.02)
    nn.init.normal_(model.embed_tokens.weight, std=0.02)
    nn.init.normal_(model.lm_head.weight, std=0.02)
    del model.verifier_lm_head  # Matches native initialization's frozen-weight loading.
    return model.eval()


def inputs(length=10):
    return torch.arange(length)[None] % 32, torch.randn(1, length, 32)


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5_moe_text"])
def test_one_decoder_call_and_placeholder_gradients(model_type):
    model = make_model(model_type).train()
    ids, hidden = inputs()
    calls = []
    hook = model.mtp_layers[0].register_forward_hook(lambda *args: calls.append(1))
    predicted, loss, metrics = model(ids, hidden)
    loss.backward()
    hook.remove()
    assert len(calls) == 1
    assert predicted.shape == (3, 4)
    assert torch.isfinite(loss)
    assert model.mask_hidden.grad.abs().sum() > 0
    assert model.mask_embedding.grad.abs().sum() > 0
    assert model.mtp_layers[0].input_proj.weight.grad.abs().sum() > 0
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert "reference_prefix_length_sum" in metrics


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5_moe_text"])
def test_no_future_leakage_and_prefix_inference_parity(model_type):
    model = make_model(model_type)
    ids, hidden = inputs()
    positions = torch.arange(ids.shape[1])[None]
    docs = torch.zeros_like(ids)
    anchors = torch.tensor([2, 5])
    with torch.no_grad():
        features = model._block_features(ids, hidden, positions, docs, anchors)
        expected = model.lm_head(features[:1])
        reference = model.draft_block(ids[:, :4], hidden[:, :3])
        torch.testing.assert_close(expected, reference, atol=1e-6, rtol=1e-5)
        changed_ids, changed_hidden = ids.clone(), hidden.clone()
        changed_ids[:, 4:] = 23
        changed_hidden[:, 3:] = torch.randn_like(hidden[:, 3:]) * 10
        changed = model._block_features(
            changed_ids, changed_hidden, positions, docs, anchors
        )
        torch.testing.assert_close(features[:1], changed[:1], atol=1e-6, rtol=1e-5)


def test_packed_documents_and_padding_cannot_leak():
    model = make_model()
    ids, hidden = inputs(12)
    docs = torch.tensor([[0] * 5 + [1] * 5 + [-1] * 2])
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 0]])
    anchors = torch.tensor([6])
    with torch.no_grad():
        original = model._block_features(ids, hidden, positions, docs, anchors)
        ids[:, :5] = 25
        hidden[:, :5] *= 20
        ids[:, 10:] = 30
        hidden[:, 10:] *= 20
        changed = model._block_features(ids, hidden, positions, docs, anchors)
    torch.testing.assert_close(original, changed)


def test_targets_shift_two_tokens_and_do_not_cross_documents():
    model = make_model()
    ids, hidden = inputs(10)
    docs = torch.tensor([[0] * 5 + [1] * 5])
    loss_mask = torch.zeros_like(ids)
    loss_mask[:, [3, 4, 5]] = 1
    _, loss, metrics = model(ids, hidden, document_ids=docs, loss_mask=loss_mask)
    assert metrics["token_agreement_0_total"] == 2
    assert metrics["token_agreement_1_total"] == 1
    assert metrics["token_agreement_2_total"] == 0
    assert metrics["reference_prefix_length_total"] == 0
    features = model._block_features(
        ids, hidden, torch.arange(10)[None], docs, torch.tensor([1, 2])
    )
    logits = model.lm_head(features).float()
    step0 = nn.functional.cross_entropy(logits[:, 0], torch.tensor([3, 4]))
    step1 = nn.functional.cross_entropy(logits[:1, 1], torch.tensor([4]))
    torch.testing.assert_close(loss, (step0 + step1) / 4)


@pytest.mark.parametrize("steps", [1, 4])
def test_empty_packed_batch_has_finite_connected_zero_loss(steps):
    model = make_model(steps=steps).train()
    batch = CollateFn(8, 32, 1, torch.float32, shift_batch_mtp)([])
    _, loss, _ = model(**batch)
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(model.mask_hidden.grad).all()
    assert model.mtp_layers[0].input_proj.weight.grad is not None


def test_checkpoint_roundtrip_and_stitch_rejection(tmp_path, monkeypatch):
    model = make_model()
    model.save_pretrained(tmp_path)
    config = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert isinstance(config, PMTPSpeculatorConfig)
    assert config.num_speculative_steps == 4
    assert SpeculatorModel.registry is not None
    assert SpeculatorModel.registry["pmtp"] is PMTPDraftModel

    # The real loader requires the large verifier; restore only its frozen tensors here.
    def load_frozen(self):
        self.embed_tokens.weight.data.copy_(model.embed_tokens.weight)
        self.lm_head.weight.data.copy_(model.lm_head.weight)
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        del self.verifier_lm_head

    monkeypatch.setattr(PMTPDraftModel, "load_verifier_weights", load_frozen)
    restored = PMTPDraftModel.from_pretrained(tmp_path)
    assert isinstance(restored, PMTPDraftModel)
    torch.testing.assert_close(restored.mask_hidden, model.mask_hidden)
    torch.testing.assert_close(restored.mask_embedding, model.mask_embedding)
    ids, hidden = inputs(5)
    torch.testing.assert_close(
        restored.draft_block(ids, hidden[:, :-1]),
        model.draft_block(ids, hidden[:, :-1]),
    )
    with pytest.raises(ValueError, match="parallel proposer"):
        stitch(tmp_path, tmp_path / "absent-verifier", tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert (
        json.loads((tmp_path / "config.json").read_text())["speculators_model_type"]
        == "pmtp"
    )


def test_training_config_roundtrip_and_invalid_options(tmp_path):
    cfg = TrainConfig.from_sources(
        cli={
            "verifier_name_or_path": "dummy-verifier",
            "speculator_type": "pmtp",
            "num_speculative_steps": 4,
            "pmtp_max_anchors": 7,
        },
        argv=[],
    )
    assert cfg.loss.loss_fn == "ce"
    path = tmp_path / "run.yaml"
    path.write_text(cfg.dump_yaml(), encoding="utf-8")
    loaded = TrainConfig.from_sources(cli={}, config_path=str(path), argv=[])
    assert loaded.pmtp.pmtp_max_anchors == 7
    assert loaded.mtp.num_speculative_steps == 4
    with pytest.raises(ValueError, match="loss-fn=ce"):
        TrainConfig.from_flat({"speculator_type": "pmtp", "loss_fn": "kl_div"})
    with pytest.raises(ValueError, match="decoder config"):
        TrainConfig.from_sources(
            cli={"speculator_type": "pmtp", "num_layers": 2}, argv=[]
        )


def test_native_initialization_preserves_all_mtp_weights(monkeypatch):
    native = make_model("qwen3_5_moe_text")
    state = {
        name: value.clone()
        for name, value in native.state_dict().items()
        if name.startswith("mtp_layers.")
    }
    monkeypatch.setattr(MTPConverter, "convert_to_state_dict", lambda self, path: state)
    monkeypatch.setattr(
        VerifierConfig,
        "from_pretrained",
        lambda path: native.config.speculators_config.verifier,
    )
    monkeypatch.setattr(PMTPDraftModel, "load_verifier_weights", lambda self: None)
    restored = PMTPDraftModel.from_training_args(
        native.config.transformer_layer_config,
        verifier_name_or_path="dummy-verifier",
        num_speculative_steps=4,
    )
    for name, value in state.items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    state.pop(next(iter(state)))
    with pytest.raises(ValueError, match="initialization mismatch"):
        PMTPDraftModel.from_training_args(
            native.config.transformer_layer_config,
            verifier_name_or_path="dummy-verifier",
        )


def test_logit_chunks_bound_head_size_and_preserve_loss():
    model = make_model()
    ids, hidden = inputs(20)
    sizes = []
    hook = model.lm_head.register_forward_hook(
        lambda module, args, output: sizes.append(args[0].shape[0])
    )
    _, chunked, _ = model(ids, hidden)
    hook.remove()
    assert max(sizes) <= model.config.pmtp_logit_chunk_size
    assert (
        sum(sizes) == model.config.pmtp_max_anchors * model.config.num_speculative_steps
    )
    model.config.pmtp_logit_chunk_size = 100
    _, unchunked, _ = model(ids, hidden)
    torch.testing.assert_close(chunked, unchunked)
