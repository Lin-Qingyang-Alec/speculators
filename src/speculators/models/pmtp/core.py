"""One-pass, causal parallel MTP prototype (text-only, no serving KV cache).

Base row i combines h[i] with embedding(x[i+1]) and predicts x[i+2].
For a sampled anchor a, additional rows use learned placeholder inputs to
predict x[a+3:a+K+2]. They see base rows <= a and earlier rows in their own
group. In particular, they never see the teacher's future hidden states.
"""

from typing import Any, ClassVar

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.mtp.core import MTPDraftModel, _prepare_mtp_position_ids
from speculators.models.pmtp.config import PMTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


@SpeculatorModel.register("pmtp")
class PMTPDraftModel(MTPDraftModel):
    config_class: ClassVar[type[PMTPSpeculatorConfig]] = PMTPSpeculatorConfig  # type: ignore[misc]

    def __init__(self, config: PMTPSpeculatorConfig) -> None:
        # The dense custom mask below is implemented for SDPA only.
        config.transformer_layer_config._attn_implementation = "sdpa"  # noqa: SLF001
        super().__init__(config)
        self.mask_hidden = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.mask_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

    def _block_features(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        document_ids: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Return [anchors, K, H] features from exactly one decoder call."""
        n = input_ids.shape[1] - 1
        k = self.config.num_speculative_steps
        device = input_ids.device
        base = torch.arange(n, device=device)
        future_depth = torch.arange(1, k, device=device).repeat(anchors.numel())
        row_anchor = torch.cat((base, anchors.repeat_interleave(k - 1)))
        depth = torch.cat((torch.zeros_like(base), future_depth))
        docs = document_ids[0, :-1].clone()
        # A shifted token at a document boundary must not become a context key.
        docs[docs != document_ids[0, 1:]] = -1
        row_docs = docs[row_anchor]
        q_anchor, kv_anchor = row_anchor[:, None], row_anchor[None, :]
        q_depth, kv_depth = depth[:, None], depth[None, :]
        allowed = (
            (row_docs[:, None] == row_docs[None, :])
            & (row_docs[:, None] >= 0)
            & (
                ((kv_depth == 0) & (kv_anchor <= q_anchor))
                | ((q_anchor == kv_anchor) & (kv_depth <= q_depth))
            )
        )
        # Invalid queries have no loss and cannot be keys for valid queries.
        # Give them a finite self row instead of an all-masked softmax.
        allowed |= torch.diag(row_docs < 0)
        mask = torch.zeros_like(allowed, dtype=hidden_states.dtype)
        mask.masked_fill_(~allowed, float("-inf"))
        num_future = future_depth.numel()
        hidden = torch.cat(
            (hidden_states[:, :n], self.mask_hidden.expand(1, num_future, -1)), dim=1
        )
        embeddings = torch.cat(
            (
                self.embed_tokens(input_ids[:, 1:].long()),
                self.mask_embedding.expand(1, num_future, -1),
            ),
            dim=1,
        )
        positions = position_ids[:, row_anchor] + depth.unsqueeze(0)
        text_positions, rotary_positions = _prepare_mtp_position_ids(
            positions,
            positions.shape[1],
            self.config.transformer_layer_config.model_type,
        )
        output = self.mtp_layers[0](
            hidden_states=hidden,
            token_embeddings=embeddings,
            attention_mask=mask[None, None],
            position_ids=text_positions,
            position_embeddings=self.rotary_emb(hidden, rotary_positions),
        )
        return torch.cat(
            (
                output[0, anchors].unsqueeze(1),
                output[0, n:].reshape(anchors.numel(), k - 1, output.shape[-1]),
            ),
            dim=1,
        )

    def _token_loss(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.lm_head(features).float()
        losses = nn.functional.cross_entropy(logits, labels, reduction="none")
        return losses, logits.detach().argmax(-1)

    def forward(  # noqa: C901
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        step_weights: list[float] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple:
        """Sample prediction groups; return token IDs, weighted CE, and metrics.

        Logits are only materialized for sampled groups, in checkpointed chunks,
        to bound the large verifier vocabulary's activation memory. Agreement is
        measured against dataset tokens, not speculative serving acceptance.
        """
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 2:  # noqa: PLR2004
            raise ValueError("PMTP requires a packed [1, seq_len >= 2] input")
        if attention_mask is not None:
            raise ValueError(
                "PMTP uses document_ids (-1 for padding), not attention_mask"
            )
        device = input_ids.device
        s = input_ids.shape[1]
        k = self.config.num_speculative_steps
        if step_weights is not None and len(step_weights) != k:
            raise ValueError(f"step_weights must have {k} entries")
        if position_ids is None:
            position_ids = torch.arange(s, device=device)[None]
        if position_ids.ndim != input_ids.ndim:
            raise ValueError("PMTP currently supports text-only 2-D position_ids")
        if document_ids is None:
            # Infer contiguous documents from position resets for direct callers.
            document_ids = (position_ids == 0).long().cumsum(-1)
        if loss_mask is None:
            loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
        docs = document_ids[0]
        eligible = (
            loss_mask[0, 2:].bool()
            & (docs[:-2] >= 0)
            & (docs[:-2] == docs[1:-1])
            & (docs[:-2] == docs[2:])
        )
        anchors = eligible.nonzero().flatten()
        limit = self.config.pmtp_max_anchors
        if anchors.numel() > limit:
            indices = (
                torch.randperm(anchors.numel(), device=device)[:limit]
                if self.training
                else torch.linspace(0, anchors.numel() - 1, limit, device=device).long()
            )
            anchors = anchors[indices].sort().values
        has_targets = anchors.numel() > 0
        if not has_targets:
            anchors = torch.zeros(1, device=device, dtype=torch.long)
        features = self._block_features(
            input_ids, hidden_states, position_ids, document_ids, anchors
        )
        target_positions = anchors[:, None] + torch.arange(k, device=device)[None] + 2
        safe_positions = target_positions.clamp(max=s - 1)
        valid = (
            (target_positions < s)
            & (docs[safe_positions] == docs[anchors, None])
            & loss_mask[0, safe_positions].bool()
            & has_targets
        )
        labels = input_ids[0, safe_positions].long()
        losses, predictions = [], []
        chunk_size = self.config.pmtp_logit_chunk_size
        flat_features, flat_labels = features.flatten(0, 1), labels.flatten()
        for start in range(0, flat_labels.numel(), chunk_size):
            chunk = flat_features[start : start + chunk_size]
            targets = flat_labels[start : start + chunk_size]
            if self.training and torch.is_grad_enabled():
                loss_chunk, ids = checkpoint(
                    self._token_loss, chunk, targets, use_reentrant=False
                )
            else:
                loss_chunk, ids = self._token_loss(chunk, targets)
            losses.append(loss_chunk)
            predictions.append(ids)
        token_losses = torch.cat(losses).reshape_as(labels) * valid
        predicted = torch.cat(predictions).reshape_as(labels)
        weights = token_losses.new_tensor(step_weights or [1.0 / k] * k)
        per_step = token_losses.sum(0) / valid.sum(0).clamp(min=1)
        loss = (per_step * weights).sum()
        # Keep the placeholders connected even for K=1 / entirely empty batches.
        loss = loss + 0 * (self.mask_hidden.sum() + self.mask_embedding.sum())
        correct = (predicted == labels) & valid
        metrics = {"loss_sum": loss.detach(), "loss_total": loss.new_tensor(1)}
        for step in range(k):
            metrics[f"loss_step_{step}"] = per_step[step].detach()
            metrics[f"token_agreement_{step}_sum"] = correct[:, step].sum()
            metrics[f"token_agreement_{step}_total"] = valid[:, step].sum()
        # Only complete groups contribute to the prefix metric (no tail bias).
        full = valid.all(-1)
        prefix = correct.long().cumprod(-1).sum(-1)
        metrics["reference_prefix_length_sum"] = (prefix * full).sum()
        metrics["reference_prefix_length_total"] = full.sum()
        return predicted, loss, metrics

    @torch.no_grad()
    def draft_block(
        self, input_ids: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Uncached single-prefix reference; the last input token is already known.

        Verifier hidden states must cover input_ids[:-1]. Returns [1, K, V].
        This is a correctness reference, not a vLLM proposer or speed benchmark.
        """
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 2:  # noqa: PLR2004
            raise ValueError("draft_block requires one prefix with at least two tokens")
        n = input_ids.shape[1] - 1
        if hidden_states.shape[1] != n:
            raise ValueError("hidden_states must cover exactly input_ids[:-1]")
        positions = torch.arange(n + 1, device=input_ids.device)[None]
        features = self._block_features(
            input_ids,
            hidden_states,
            positions,
            torch.zeros_like(input_ids),
            torch.tensor([n - 1], device=input_ids.device),
        )
        return self.lm_head(features)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        *,
        num_speculative_steps: int = 3,
        verifier_name_or_path: str | None = None,
        pmtp_max_anchors: int = 32,
        pmtp_logit_chunk_size: int = 16,
        **kwargs: Any,  # noqa: ARG003
    ) -> "PMTPDraftModel":
        if verifier_name_or_path is None:
            raise ValueError("PMTP requires a verifier with native MTP weights")
        config = PMTPSpeculatorConfig(
            transformer_layer_config=verifier_config,
            pmtp_max_anchors=pmtp_max_anchors,
            pmtp_logit_chunk_size=pmtp_logit_chunk_size,
            speculators_config=SpeculatorsConfig(
                algorithm="pmtp",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=num_speculative_steps)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(verifier_name_or_path),
            ),
        )
        model = cls(config)
        from speculators.convert.mtp.converter import MTPConverter  # noqa: PLC0415

        state = MTPConverter().convert_to_state_dict(verifier_name_or_path)
        incompatible = model.load_state_dict(state, strict=False)
        missing = [k for k in incompatible.missing_keys if k.startswith("mtp_layers.")]
        if missing or incompatible.unexpected_keys:
            raise ValueError(f"Native MTP initialization mismatch: {incompatible}")
        model.load_verifier_weights()
        return model
