"""Progress distribution, trend and relation losses."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from event2reward.modeling.labels import ProgressTargetKind, RelationKind


def _validate_unit_interval(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")


def distributed_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return the true global masked mean under DDP gradient averaging."""

    if values.shape != mask.shape:
        raise ValueError(f"values/mask shape mismatch: {values.shape} vs {mask.shape}")
    weights = mask.to(device=values.device, dtype=values.dtype)
    local_sum = (values * weights).sum()
    # Every rank must enter this collective with the same dtype even when one
    # rank has no valid examples. Keep counts independent of the loss/autocast
    # dtype; this also avoids losing integer precision in BF16.
    global_count = mask.to(device=values.device, dtype=torch.float32).sum()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    return local_sum * float(world_size) / global_count.clamp_min(1.0)


def distributed_masked_means(
    pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[torch.Tensor, ...]:
    """Compute several exact DDP masked means with one count collective."""

    if not pairs:
        return ()
    for values, mask in pairs:
        if values.shape != mask.shape:
            raise ValueError(
                f"values/mask shape mismatch: {values.shape} vs {mask.shape}"
            )
    counts = torch.stack(
        [
            mask.to(device=values.device, dtype=torch.float32).sum()
            for values, mask in pairs
        ]
    )
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    return tuple(
        (values * mask.to(device=values.device, dtype=values.dtype)).sum()
        * float(world_size)
        / counts[index].clamp_min(1.0)
        for index, (values, mask) in enumerate(pairs)
    )


def mean_preserving_gaussian_targets(
    target: torch.Tensor,
    num_bins: int = 101,
    sigma_bins: float = 2.0,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Encode scalar progress without biasing the mean at 0/1 boundaries."""

    if num_bins < 2:
        raise ValueError("num_bins must be at least two")
    if sigma_bins <= 0:
        raise ValueError("sigma_bins must be positive")
    target = target.float()
    if validate:
        _validate_unit_interval(target, "progress target")
    bins = torch.arange(num_bins, device=target.device, dtype=torch.float32)
    scaled = target * float(num_bins - 1)
    lower = scaled.floor().long()
    upper = scaled.ceil().long().clamp(max=num_bins - 1)
    upper_weight = scaled - lower.float()

    def kernel(center: torch.Tensor) -> torch.Tensor:
        offset = bins - center.float().unsqueeze(-1)
        radius = torch.minimum(center, (num_bins - 1) - center).unsqueeze(-1)
        log_weights = -0.5 * (offset / float(sigma_bins)).square()
        log_weights = log_weights.masked_fill(offset.abs() > radius, -torch.inf)
        return torch.softmax(log_weights, dim=-1)

    return kernel(lower) * (1.0 - upper_weight).unsqueeze(-1) + kernel(
        upper
    ) * upper_weight.unsqueeze(-1)


def expected_progress(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(f"progress logits must be [N,K], got {tuple(logits.shape)}")
    support = torch.linspace(
        0.0, 1.0, logits.shape[-1], device=logits.device, dtype=torch.float32
    )
    return (logits.float().softmax(dim=-1) * support).sum(dim=-1)


@dataclass
class LossResult:
    loss: torch.Tensor
    progress_loss: torch.Tensor
    trend_loss: torch.Tensor
    relation_loss: torch.Tensor
    progress: torch.Tensor
    relation_metrics: dict[str, torch.Tensor]
    progress_metrics: dict[str, torch.Tensor]


@dataclass
class RelationLossResult:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class RelationLoss(nn.Module):
    """Event relations on the expected progress of each clip.

    Kind 3 (no further progress): members no higher than the anchor. Kind 4
    (unrecovered regression): members at least `margin` below it. Kind 5
    (progress stall): members within `margin` of it. Kind 6 (regression and
    recovery): members at least the margin below both the anchor and the
    recovery point. Anchors carry no
    gradient when `relation_detach_anchor` is set (always, for the training
    data), and regression anchors never do. Members are averaged within a
    group, then groups are averaged.
    """

    def forward_with_metrics(
        self,
        progress: torch.Tensor,
        anchor_index: torch.Tensor,
        secondary_anchor_index: torch.Tensor,
        member_indices: torch.Tensor,
        member_valid: torch.Tensor,
        relation_kind: torch.Tensor,
        margin: torch.Tensor,
        relation_valid: torch.Tensor,
        relation_loss_weight: torch.Tensor | None = None,
        relation_normalizer_weight: torch.Tensor | None = None,
        relation_detach_anchor: torch.Tensor | None = None,
    ) -> RelationLossResult:
        if member_indices.ndim != 2:
            raise ValueError("member_indices must be [G,M]")
        group_count, member_count = member_indices.shape
        expected = (group_count,)
        for name, value in {
            "anchor_index": anchor_index,
            "secondary_anchor_index": secondary_anchor_index,
            "relation_kind": relation_kind,
            "margin": margin,
            "relation_valid": relation_valid,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if member_valid.shape != (group_count, member_count):
            raise ValueError("member_valid must match member_indices")
        if group_count == 0:
            # Keep every rank in the identical collective sequence. A rank
            # without relation groups carries one masked dummy group through all reductions.
            if progress.numel() == 0:
                raise ValueError("relation loss needs at least one clip prediction")
            anchor_index = torch.zeros(1, dtype=torch.long, device=progress.device)
            secondary_anchor_index = torch.full(
                (1,), -1, dtype=torch.long, device=progress.device
            )
            member_indices = torch.zeros(
                (1, 1), dtype=torch.long, device=progress.device
            )
            member_valid = torch.zeros((1, 1), dtype=torch.bool, device=progress.device)
            relation_kind = torch.full(
                (1,),
                int(RelationKind.PROGRESS_STALL),
                dtype=torch.long,
                device=progress.device,
            )
            margin = torch.ones(1, dtype=torch.float32, device=progress.device)
            relation_valid = torch.zeros(1, dtype=torch.bool, device=progress.device)
            relation_loss_weight = torch.zeros(
                1, dtype=torch.float32, device=progress.device
            )
            relation_normalizer_weight = torch.zeros(
                1, dtype=torch.float32, device=progress.device
            )
            relation_detach_anchor = torch.zeros(
                1, dtype=torch.bool, device=progress.device
            )
            group_count, member_count = 1, 1
        if relation_loss_weight is None:
            relation_loss_weight = torch.ones(
                group_count, dtype=torch.float32, device=progress.device
            )
        if relation_normalizer_weight is None:
            relation_normalizer_weight = torch.ones(
                group_count, dtype=torch.float32, device=progress.device
            )
        if relation_detach_anchor is None:
            relation_detach_anchor = torch.zeros(
                group_count, dtype=torch.bool, device=progress.device
            )
        for name, value in {
            "relation_loss_weight": relation_loss_weight,
            "relation_normalizer_weight": relation_normalizer_weight,
        }.items():
            if value.shape != (group_count,):
                raise ValueError(f"{name} must have shape {(group_count,)}")
            if not torch.isfinite(value).all() or bool((value < 0).any()):
                raise ValueError(f"{name} must be finite and non-negative")
        if relation_detach_anchor.shape != (group_count,):
            raise ValueError(
                f"relation_detach_anchor must have shape {(group_count,)}"
            )
        relation_detach_anchor = relation_detach_anchor.bool()
        if bool((anchor_index < 0).any()) or bool(
            (anchor_index >= progress.numel()).any()
        ):
            raise ValueError("invalid relation anchor index")
        safe_members = member_indices.clamp_min(0)
        if bool((safe_members >= progress.numel()).any()):
            raise ValueError("invalid relation member index")

        valid = relation_valid.bool()
        members_valid = member_valid.bool() & valid.unsqueeze(-1)
        counts = members_valid.sum(dim=-1)
        if bool((valid & counts.eq(0)).any()):
            raise ValueError("every valid relation requires a member")
        anchor_raw = progress[anchor_index].float()
        anchor = anchor_raw.detach()
        members = progress[safe_members].float()
        kinds = relation_kind.long()
        regression = valid & kinds.eq(int(RelationKind.REGRESSION_AND_RECOVERY))
        rollback = regression
        stall = valid & kinds.eq(int(RelationKind.PROGRESS_STALL))
        recoverable = valid & kinds.eq(int(RelationKind.NO_FURTHER_PROGRESS))
        irrecoverable = valid & kinds.eq(int(RelationKind.UNRECOVERED_REGRESSION))
        unknown = valid & ~(rollback | stall | recoverable | irrecoverable)
        if bool(unknown.any()):
            raise ValueError("unknown relation kind")
        if not torch.isfinite(margin).all() or bool((margin < 0).any()):
            raise ValueError("relation margins must be finite and non-negative")
        strict = valid & ~recoverable
        if bool((margin[strict] <= 0).any()):
            raise ValueError("strict relations need a positive margin")
        if bool((margin[recoverable] != 0).any()):
            raise ValueError("recoverable not-higher relations require margin=0")

        secondary_valid = secondary_anchor_index.ge(0)
        if bool((rollback & ~secondary_valid).any()):
            raise ValueError("rollback requires a recovery anchor")
        if bool(((irrecoverable | recoverable | stall) & secondary_valid).any()):
            raise ValueError(
                "failure and progress stall relations must not have a secondary anchor"
            )
        safe_secondary = secondary_anchor_index.clamp_min(0)
        secondary = progress[safe_secondary].detach().float()

        # A regression margin larger than half the lower anchor cannot be met
        # in [0, 1]; clamp it there.
        effective_margin = torch.where(
            regression,
            torch.minimum(margin.float(), 0.5 * torch.minimum(anchor, secondary)),
            margin.float(),
        )
        rollback_floor = torch.minimum(anchor, secondary)
        rollback_member_gap = rollback_floor.unsqueeze(-1) - members
        rollback_dip_violation = (
            F.relu(effective_margin.unsqueeze(-1) - rollback_member_gap)
            .square()
            .masked_fill(~members_valid, 0.0)
            .sum(dim=-1)
            / counts.clamp_min(1).float()
        )
        rollback_violation = rollback_dip_violation

        failure_should_detach = relation_detach_anchor
        failure_anchor = torch.where(
            failure_should_detach.unsqueeze(-1),
            anchor.unsqueeze(-1),
            anchor_raw.unsqueeze(-1),
        )
        gaps = failure_anchor - members
        failure_violation = F.relu(margin.float().unsqueeze(-1) - gaps).square()
        failure_per_group = (
            failure_violation.masked_fill(~members_valid, 0.0).sum(dim=-1)
            / counts.clamp_min(1).float()
        )
        stall_deviation = (members - anchor.unsqueeze(-1)).abs()
        stall_per_group = (
            F.relu(stall_deviation - margin.float().unsqueeze(-1))
            .square()
            .masked_fill(~members_valid, 0.0)
            .sum(dim=-1)
            / counts.clamp_min(1).float()
        )

        loss_regression = distributed_masked_mean(rollback_violation, rollback)
        loss_stall = distributed_masked_mean(stall_per_group, stall)
        loss_recoverable = distributed_masked_mean(failure_per_group, recoverable)
        loss_irrecoverable = distributed_masked_mean(failure_per_group, irrecoverable)

        # The annotated event group is the statistical unit. Average members
        # inside each group above, then micro-average all valid groups globally.
        # Source sampling controls family exposure; do not reweight every active
        # family to an equal share a second time inside the loss.
        relation_per_group = torch.zeros_like(rollback_violation)
        relation_per_group = torch.where(
            rollback, rollback_violation, relation_per_group
        )
        relation_per_group = torch.where(
            stall, stall_per_group, relation_per_group
        )
        relation_per_group = torch.where(
            recoverable | irrecoverable, failure_per_group, relation_per_group
        )
        numerator_weight = relation_loss_weight.float() * valid.float()
        denominator_weight = relation_normalizer_weight.float() * valid.float()
        local_numerator = (relation_per_group * numerator_weight).sum()
        global_denominator = denominator_weight.sum().detach()
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_denominator, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        total = local_numerator * float(world_size) / global_denominator.clamp_min(1.0)

        family_masks = {
            "progress_stall": stall,
            "regression_and_recovery": rollback,
            "no_further_progress": recoverable,
            "unrecovered_regression": irrecoverable,
        }
        counts_global: dict[str, torch.Tensor] = {}
        for name, mask in family_masks.items():
            count = mask.float().sum().detach()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            counts_global[name] = count
        regression_satisfied = (
            ((rollback_member_gap >= effective_margin.unsqueeze(-1)) | ~members_valid)
            .all(dim=-1)
            .float()
        )
        stall_satisfied = (
            ((stall_deviation <= margin.float().unsqueeze(-1)) | ~members_valid)
            .all(dim=-1)
            .float()
        )
        failure_satisfied = (
            ((gaps >= margin.float().unsqueeze(-1)) | ~members_valid)
            .all(dim=-1)
            .float()
        )
        metrics = {
            "loss_relation_progress_stall": loss_stall,
            "loss_relation_regression_and_recovery": loss_regression,
            "loss_relation_no_further_progress": loss_recoverable,
            "loss_relation_unrecovered_regression": loss_irrecoverable,
            "relation_progress_stall_mean_deviation": distributed_masked_mean(
                stall_deviation, members_valid & stall.unsqueeze(-1)
            ),
            "relation_progress_stall_all_member_satisfaction": distributed_masked_mean(
                stall_satisfied, stall
            ),
            "relation_regression_and_recovery_mean_gap": distributed_masked_mean(
                rollback_member_gap, members_valid & rollback.unsqueeze(-1)
            ),
            "relation_regression_and_recovery_all_member_satisfaction": distributed_masked_mean(
                regression_satisfied, rollback
            ),
            "relation_no_further_progress_all_member_satisfaction": distributed_masked_mean(
                failure_satisfied, recoverable
            ),
            "relation_unrecovered_regression_all_member_satisfaction": distributed_masked_mean(
                failure_satisfied, irrecoverable
            ),
            "relation_loss_weight_mean": distributed_masked_mean(
                relation_loss_weight.float(), valid
            ),
        }
        metrics.update(
            {f"relation_count_{name}": value for name, value in counts_global.items()}
        )
        return RelationLossResult(total, metrics)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_with_metrics(*args, **kwargs).loss


class ProgressModelLoss(nn.Module):
    def __init__(
        self,
        *,
        num_bins: int = 101,
        gaussian_sigma_bins: float = 2.0,
        lambda_progress: float = 1.0,
        lambda_trend: float = 1.0,
        lambda_relation: float = 1.0,
        validate_targets: bool = True,
    ):
        super().__init__()
        self.num_bins = int(num_bins)
        self.gaussian_sigma_bins = float(gaussian_sigma_bins)
        self.lambda_progress = float(lambda_progress)
        self.lambda_trend = float(lambda_trend)
        self.lambda_relation = float(lambda_relation)
        self.validate_targets = bool(validate_targets)
        self.relation = RelationLoss()

    def forward(
        self,
        progress_logits: torch.Tensor,
        trend_logits: torch.Tensor,
        *,
        target_progress: torch.Tensor,
        progress_valid: torch.Tensor,
        progress_target_kind: torch.Tensor | None = None,
        target_trend: torch.Tensor,
        trend_valid: torch.Tensor,
        relation: dict[str, torch.Tensor] | None = None,
    ) -> LossResult:
        if progress_logits.ndim != 2 or progress_logits.shape[-1] != self.num_bins:
            raise ValueError(f"progress_logits must be [N,{self.num_bins}]")
        if trend_logits.shape != (progress_logits.shape[0], 3):
            raise ValueError("trend_logits must be [N,3]")
        target_progress = target_progress.to(progress_logits.device).float()
        progress_valid = progress_valid.to(progress_logits.device).bool()
        if progress_target_kind is None:
            progress_target_kind = torch.where(
                progress_valid,
                torch.full_like(
                    progress_valid,
                    int(ProgressTargetKind.EXPERT),
                    dtype=torch.long,
                ),
                torch.full_like(
                    progress_valid,
                    int(ProgressTargetKind.NONE),
                    dtype=torch.long,
                ),
            )
        progress_target_kind = progress_target_kind.to(
            progress_logits.device, dtype=torch.long
        )
        if progress_target_kind.shape != progress_valid.shape:
            raise ValueError("progress_target_kind must match progress_valid")
        valid_kinds = torch.tensor(
            [int(value) for value in ProgressTargetKind],
            device=progress_target_kind.device,
        )
        if not bool((progress_target_kind[..., None] == valid_kinds).any(-1).all()):
            raise ValueError("unknown progress_target_kind")
        target_trend = target_trend.to(trend_logits.device).long()
        trend_valid = trend_valid.to(trend_logits.device).bool()

        safe_progress = torch.where(
            progress_valid, target_progress, torch.zeros_like(target_progress)
        )
        soft_target = mean_preserving_gaussian_targets(
            safe_progress.clamp(0.0, 1.0),
            self.num_bins,
            self.gaussian_sigma_bins,
            # ClipRecord validates this invariant on CPU before collation.
            validate=self.validate_targets,
        )
        per_progress = -(soft_target * progress_logits.float().log_softmax(dim=-1)).sum(
            dim=-1
        )
        safe_trend = torch.where(
            trend_valid, target_trend, torch.zeros_like(target_trend)
        )
        per_trend = F.cross_entropy(trend_logits.float(), safe_trend, reduction="none")
        expert_progress = progress_valid & progress_target_kind.eq(
            int(ProgressTargetKind.EXPERT)
        )
        stall_success_progress = progress_valid & progress_target_kind.eq(
            int(ProgressTargetKind.STALL_SUCCESS)
        )
        completion_progress = progress_valid & progress_target_kind.eq(
            int(ProgressTargetKind.COMPLETION)
        )
        (
            progress_loss,
            trend_loss,
            expert_progress_loss,
            stall_success_progress_loss,
            completion_progress_loss,
        ) = distributed_masked_means(
            (
                (per_progress, progress_valid),
                (per_trend, trend_valid),
                (per_progress, expert_progress),
                (per_progress, stall_success_progress),
                (per_progress, completion_progress),
            )
        )
        progress = expected_progress(progress_logits)

        relation_loss = progress.sum() * 0.0
        relation_metrics: dict[str, torch.Tensor] = {}
        if relation is not None:
            relation_on_device = {
                key: value.to(progress.device) if torch.is_tensor(value) else value
                for key, value in relation.items()
            }
            relation_result = self.relation.forward_with_metrics(
                progress, **relation_on_device
            )
            relation_loss = relation_result.loss
            relation_metrics = relation_result.metrics
        total = (
            self.lambda_progress * progress_loss
            + self.lambda_trend * trend_loss
            + self.lambda_relation * relation_loss
        )
        progress_metrics = {
            "loss_progress_expert": expert_progress_loss,
            "loss_progress_stall_success": stall_success_progress_loss,
            "loss_progress_completion": completion_progress_loss,
        }
        return LossResult(
            total,
            progress_loss,
            trend_loss,
            relation_loss,
            progress,
            relation_metrics,
            progress_metrics,
        )
