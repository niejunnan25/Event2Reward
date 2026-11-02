"""ProgressModelVLM: query tokens on a Qwen backbone, read out by two heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from event2reward.modeling.losses import ProgressModelLoss


@dataclass
class ProgressModelVLMOutput:
    loss: torch.Tensor | None
    progress_logits: torch.Tensor
    trend_logits: torch.Tensor
    progress: torch.Tensor
    metrics: dict[str, torch.Tensor]


class ProgressModelVLM(nn.Module):
    """A Qwen VLM read out at two query tokens by a progress and a trend head.

    The progress head predicts a distribution over 101 bins in [0, 1]; its
    expectation is the progress estimate, which event relations also constrain.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        progress_query_token_id: int,
        trend_query_token_id: int,
        num_bins: int = 101,
        head_dropout: float = 0.1,
        validate_query_tokens: bool = True,
        loss: ProgressModelLoss | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.progress_query_token_id = int(progress_query_token_id)
        self.trend_query_token_id = int(trend_query_token_id)
        self.num_bins = int(num_bins)
        self.validate_query_tokens = bool(validate_query_tokens)
        if loss is not None and loss.num_bins != self.num_bins:
            raise ValueError(
                "Progress head and loss must use the same number of bins, got "
                f"{self.num_bins} and {loss.num_bins}."
            )
        self.progress_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size // 2, self.num_bins),
        )
        self.trend_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size // 2, 3),
        )
        try:
            backbone_dtype = next(backbone.parameters()).dtype
            self.progress_head.to(dtype=backbone_dtype)
            self.trend_head.to(dtype=backbone_dtype)
        except StopIteration:
            pass
        self.loss_module = loss or ProgressModelLoss(num_bins=self.num_bins)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.backbone.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self) -> None:
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    @staticmethod
    def _last_hidden_state(outputs: Any) -> torch.Tensor:
        if getattr(outputs, "last_hidden_state", None) is not None:
            return outputs.last_hidden_state
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            return hidden_states[-1]
        if isinstance(outputs, tuple) and outputs:
            return outputs[0]
        raise RuntimeError("VLM backbone did not return hidden states")

    def _one_query_per_row(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        token_id: int,
        name: str,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = None
        if self.validate_query_tokens or positions is None:
            mask = input_ids.eq(token_id)
        if self.validate_query_tokens:
            assert mask is not None
            counts = mask.sum(dim=-1)
            if not bool(counts.eq(1).all()):
                raise ValueError(
                    f"each flattened clip must contain exactly one {name} query; "
                    f"got {counts.tolist()}"
                )
        if positions is None:
            assert mask is not None
            positions = mask.long().argmax(dim=-1)
        positions = positions.to(device=hidden.device, dtype=torch.long)
        return hidden.gather(
            dim=1,
            index=positions[:, None, None].expand(-1, 1, hidden.shape[-1]),
        ).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        target_progress: torch.Tensor | None = None,
        progress_valid: torch.Tensor | None = None,
        progress_target_kind: torch.Tensor | None = None,
        target_trend: torch.Tensor | None = None,
        trend_valid: torch.Tensor | None = None,
        progress_query_positions: torch.Tensor | None = None,
        trend_query_positions: torch.Tensor | None = None,
        relation: dict[str, torch.Tensor] | None = None,
        **vision_inputs: Any,
    ) -> ProgressModelVLMOutput:
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # Only the final hidden state feeds the two task heads. Returning
            # every transformer layer keeps a large, unused tuple of
            # activations alive until backward.
            output_hidden_states=False,
            # This is supervised representation learning rather than
            # autoregressive generation. KV caches only consume memory here,
            # especially once activation checkpointing is disabled.
            use_cache=False,
            return_dict=True,
            **vision_inputs,
        )
        hidden = self._last_hidden_state(backbone_outputs)
        progress_hidden = self._one_query_per_row(
            hidden,
            input_ids,
            self.progress_query_token_id,
            "progress",
            progress_query_positions,
        )
        trend_hidden = self._one_query_per_row(
            hidden,
            input_ids,
            self.trend_query_token_id,
            "trend",
            trend_query_positions,
        )
        progress_logits = self.progress_head(progress_hidden)
        trend_logits = self.trend_head(trend_hidden)

        if target_progress is None:
            from event2reward.modeling.losses import expected_progress

            progress = expected_progress(progress_logits)
            return ProgressModelVLMOutput(
                None, progress_logits, trend_logits, progress, {}
            )
        required = {
            "progress_valid": progress_valid,
            "target_trend": target_trend,
            "trend_valid": trend_valid,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing supervised inputs: {missing}")
        result = self.loss_module(
            progress_logits,
            trend_logits,
            target_progress=target_progress,
            progress_valid=progress_valid,
            progress_target_kind=progress_target_kind,
            target_trend=target_trend,
            trend_valid=trend_valid,
            relation=relation,
        )
        metrics = {
            "loss_progress": result.progress_loss.detach(),
            "loss_trend": result.trend_loss.detach(),
            "loss_relation": result.relation_loss.detach(),
            **{key: value.detach() for key, value in result.progress_metrics.items()},
            **{key: value.detach() for key, value in result.relation_metrics.items()},
        }
        return ProgressModelVLMOutput(
            result.loss,
            progress_logits,
            trend_logits,
            result.progress,
            metrics,
        )
