"""Experimental parallel prediction initialized from a native MTP head."""

from typing import Literal

from pydantic import Field

from speculators.config import SpeculatorModelConfig
from speculators.models.mtp.config import MTPSpeculatorConfig


@SpeculatorModelConfig.register("pmtp")
class PMTPSpeculatorConfig(MTPSpeculatorConfig):
    speculators_model_type: Literal["pmtp"] = "pmtp"  # type: ignore[assignment]
    architectures: list[str] = Field(default_factory=lambda: ["PMTPDraftModel"])
    # Checkpoints predating this field contain mask_hidden, not h_proj weights.
    # New training selects projection explicitly in PMTPArgs/from_training_args.
    pmtp_hidden_init: Literal["projection", "shared"] = "shared"
    pmtp_max_anchors: int = Field(default=32, ge=1)
    pmtp_logit_chunk_size: int = Field(default=16, ge=1)
