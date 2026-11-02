"""Relation groups: index rows of clips plus a constraint between their predicted progress."""
from __future__ import annotations

import random
import re
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch

from event2reward.modeling.labels import ProgressTrend, RelationKind

from event2reward.data.collator import QwenProgressCollator
from event2reward.data.dataset import JsonlOffsetDataset
from event2reward.data.schema import ClipRecord

GOAL_CONTRACT = "language_matched_success_terminal_v1"
EVENT_KINDS = {
    RelationKind.NO_FURTHER_PROGRESS,
    RelationKind.UNRECOVERED_REGRESSION,
    RelationKind.PROGRESS_STALL,
    RelationKind.REGRESSION_AND_RECOVERY,
}


class JsonlRelationDataset(JsonlOffsetDataset):
    """Rows of `clips` plus an optional `relation`, as in the LIBERO indices.

    A row without a relation contributes absolute progress targets; expert
    clips marked `stage2_dynamic_sampling` draw a new query frame on every
    visit. In a row with a relation, the clips carry no absolute target; the
    relation constrains their predicted progress. Relations with
    `resample_each_epoch` draw new member frames on every visit.
    """

    def __init__(self, path: str | Path, *, require_goal_present: bool = False):
        super().__init__(path)
        self.require_goal_present = bool(require_goal_present)
        self.relation_kinds = array("b")
        self.clip_counts = array("H")

        def check_row(row: dict[str, Any]) -> bool:
            clips = row.get("clips")
            if not isinstance(clips, list) or not clips:
                raise ValueError("an index row needs a nonempty list of clips")
            relation = row.get("relation")
            kind = int(RelationKind.NONE) if relation is None else int(relation["kind"])
            if kind != int(RelationKind.NONE) and RelationKind(kind) not in EVENT_KINDS:
                raise ValueError(f"unsupported relation kind {kind}")
            if self.require_goal_present:
                for clip in clips:
                    metadata = clip.get("metadata", {})
                    if (
                        clip.get("goal_available") is not True
                        or not clip.get("goal_dataset_id")
                        or not clip.get("goal_video_path")
                        or metadata.get("goal_contract") != GOAL_CONTRACT
                        or re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("goal_sha256", ""))) is None
                    ):
                        raise ValueError("every clip needs an independent successful goal video")
            if len(clips) > 65_535:
                raise ValueError("relation group exceeds 65535 clips")
            self.relation_kinds.append(kind)
            self.clip_counts.append(len(clips))
            return True

        self._append_matching_offsets(check_row)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_with_epoch(index, 0)

    def clip_count(self, index: int) -> int:
        return int(self.clip_counts[index])

    def get_with_epoch(self, index: int, epoch: int) -> dict[str, Any]:
        row = self._row(index)
        try:
            clips = [ClipRecord.from_dict(value).to_dict() for value in row["clips"]]
            relation = row.get("relation")
            if relation is not None:
                clips, relation = _resample_relation_group(clips, relation, epoch)
                _validate_local_relation(relation, len(clips))
                # Event clips are supervised only through the relation.
                clips = [{**clip, "progress_valid": False, "trend_valid": False} for clip in clips]
                relation = {**relation, "loss_weight": 1.0, "normalizer_weight": 1.0,
                            "detach_anchor": True}
            else:
                clips = [_resample_expert_clip(clip, epoch) for clip in clips]
            clips = [{**clip, "metadata": {**clip.get("metadata", {}), "sampling_epoch": int(epoch)}}
                     for clip in clips]
            return {"clips": clips, "relation": relation}
        except Exception as exc:
            raise ValueError(f"invalid index row at byte offset {self.offsets[index]} in {self.path}") from exc


class OrdinaryRelationGroupDataset(torch.utils.data.Dataset):
    """Expose a clip dataset as one-clip groups without relations."""

    def __init__(self, dataset: torch.utils.data.Dataset):
        if len(dataset) <= 0:
            raise ValueError("ordinary relation-group source must not be empty")
        self.dataset = dataset
        self.relation_kinds = array("b", [int(RelationKind.NONE)]) * len(dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def clip_count(self, index: int) -> int:
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        return 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_with_epoch(index, 0)

    def get_with_epoch(self, index: int, epoch: int) -> dict[str, Any]:
        getter = getattr(self.dataset, "get_with_epoch", None)
        clip = getter(index, epoch) if getter is not None else self.dataset[index]
        return {"clips": [clip], "relation": None}


def _resample_expert_clip(clip: dict[str, Any], epoch: int) -> dict[str, Any]:
    """Draw this epoch's query frame of an expert clip; target = elapsed / total frames.

    The episode start and end (`stage2_required_frames`) are always among the
    `stage2_selection_count` frames drawn per epoch; the clip's slot picks one.
    """
    metadata = clip.get("metadata", {})
    if not metadata.get("stage2_dynamic_sampling"):
        return clip
    if metadata.get("category") != "clean_success":
        raise ValueError("per-epoch query sampling applies to clean expert clips")
    selected = _sample_frames(
        metadata["stage2_candidate_frames"],
        metadata["stage2_required_frames"],
        int(metadata["stage2_selection_count"]),
        seed=int(metadata["stage2_sampling_seed"]) + 1_000_003 * int(epoch),
    )
    frame = selected[int(metadata["stage2_selection_slot"])]
    stride = int(metadata["stage2_history_stride"])
    return {
        **clip,
        "history_frames": [max(0, frame - stride * offset) for offset in range(4, -1, -1)],
        "current_frame": frame,
        "target_progress": float(frame) / max(1, int(metadata["stage2_num_frames"]) - 1),
        "progress_valid": True,
        "target_trend": int(ProgressTrend.GROWTH),
        "trend_valid": frame >= 4 * stride,
        "history_transform": "natural",
    }


def _sample_frames(candidates: Iterable[int], required: Iterable[int], count: int, *, seed: int) -> list[int]:
    candidates = sorted({int(value) for value in candidates})
    required_set = {int(value) for value in required}
    if not required_set.issubset(candidates):
        raise ValueError(f"required frames are not candidates: {sorted(required_set - set(candidates))}")
    if len(required_set) > count:
        raise ValueError(f"{len(required_set)} required frames exceed the budget {count}")
    remaining = [value for value in candidates if value not in required_set]
    slots = min(count - len(required_set), len(remaining))
    rng = random.Random(seed)
    selected = list(required_set)
    if slots:
        selected.extend(rng.sample(remaining, k=slots))
    return sorted(selected)


def _resample_relation_group(
    clips: list[dict[str, Any]], relation: dict[str, Any], epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw this epoch's member frames: both ends of the candidates, then one per stratum."""

    if not relation.get("resample_each_epoch"):
        return clips, relation
    if relation.get("sampling_strategy") != "stratified_random":
        raise ValueError("resampled relation members require stratified_random")
    candidates = [int(value) for value in relation["candidate_frames"]]
    selected = _stratified_relation_members(
        candidates,
        int(relation["group_size"]),
        seed=int(relation["sampling_seed"]) + 2_000_033 * int(epoch),
    )
    old_members = [int(value) for value in relation["member_indices"]]
    if not old_members:
        raise ValueError("resampled relation has no member template")
    if old_members != list(range(old_members[0], len(clips))):
        raise ValueError("resampled relation members must form the tail of the group")
    prefix = clips[: old_members[0]]
    template = clips[old_members[0]]
    stride = int(relation["history_stride"])
    members = []
    for member_index, frame in enumerate(selected):
        members.append(
            {
                **template,
                "sample_id": (
                    f"{prefix[int(relation['anchor_index'])]['sample_id']}:"
                    f"relation-epoch-{int(epoch)}-member-{member_index}-frame-{frame}"
                ),
                "history_frames": [
                    frame - stride * offset for offset in range(4, -1, -1)
                ],
                "current_frame": frame,
                "target_progress": 0.0,
                "progress_valid": False,
                "target_trend": int(ProgressTrend.GROWTH),
                "trend_valid": False,
                "history_transform": "natural",
            }
        )
    updated_relation = {
        **relation,
        "member_indices": list(range(len(prefix), len(prefix) + len(members))),
        "sampling_epoch": int(epoch),
    }
    return [*prefix, *members], updated_relation


def _validate_local_relation(relation: dict[str, Any], clip_count: int) -> None:
    kind = RelationKind(int(relation["kind"]))
    anchor = int(relation["anchor_index"])
    secondary = int(relation.get("secondary_anchor_index", -1))
    members = [int(value) for value in relation["member_indices"]]
    used = [anchor, *members] + ([] if secondary < 0 else [secondary])
    if not members or min(used) < 0 or max(used) >= clip_count:
        raise ValueError("relation indices are outside the local clip group")
    regression_kinds = {RelationKind.REGRESSION_AND_RECOVERY}
    single_anchor_kinds = {
        RelationKind.NO_FURTHER_PROGRESS,
        RelationKind.UNRECOVERED_REGRESSION,
        RelationKind.PROGRESS_STALL,
    }
    if kind in regression_kinds and secondary < 0:
        raise ValueError("regression needs two surrounding anchors")
    if kind in single_anchor_kinds and secondary >= 0:
        raise ValueError("failure and progress stall relations have only one anchor")
    margin = float(relation["margin"])
    if kind == RelationKind.NO_FURTHER_PROGRESS:
        if margin != 0:
            raise ValueError("recoverable not-higher relation requires margin=0")
    elif margin <= 0:
        raise ValueError("strict relation margin must be positive")


def _validate_relation_conditioning(clips: list[dict[str, Any]]) -> None:
    fields = (
        "dataset_id",
        "physical_episode_id",
        "instruction",
        "video_path",
        "start_frame",
        "goal_frame",
        "goal_available",
        "goal_dataset_id",
        "goal_video_path",
    )
    reference = tuple(clips[0].get(field) for field in fields)
    if any(
        tuple(clip.get(field) for field in fields) != reference for clip in clips[1:]
    ):
        raise ValueError(
            "all clips in a relation must share start/goal/task conditioning"
        )


class QwenRelationCollator:
    """Flatten the clips of several groups into one batch and remap relation indices."""

    def __init__(self, clip_collator: QwenProgressCollator):
        self.clip_collator = clip_collator

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        clips: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        offset = 0
        for example in examples:
            local_clips = example["clips"]
            relation = example.get("relation")
            if relation is not None:
                _validate_local_relation(relation, len(local_clips))
                _validate_relation_conditioning(local_clips)
                anchor_clip = local_clips[int(relation["anchor_index"])]
                dropout_key = f"relation:{anchor_clip['sample_id']}"
                local_clips = [
                    {
                        **clip,
                        "metadata": {
                            **clip.get("metadata", {}),
                            "goal_dropout_group": dropout_key,
                        },
                    }
                    for clip in local_clips
                ]
                relations.append(
                    {
                        "anchor_index": offset + int(relation["anchor_index"]),
                        "secondary_anchor_index": (
                            -1
                            if int(relation.get("secondary_anchor_index", -1)) < 0
                            else offset + int(relation["secondary_anchor_index"])
                        ),
                        "member_indices": [
                            offset + int(value) for value in relation["member_indices"]
                        ],
                        "kind": int(relation["kind"]),
                        "margin": float(relation["margin"]),
                        "loss_weight": float(relation.get("loss_weight", 1.0)),
                        "normalizer_weight": float(
                            relation.get("normalizer_weight", 1.0)
                        ),
                        "detach_anchor": bool(relation.get("detach_anchor", False)),
                    }
                )
            clips.extend(local_clips)
            offset += len(local_clips)
        batch = self.clip_collator(clips)
        if not relations:
            batch["relation"] = {
                "anchor_index": torch.empty(0, dtype=torch.long),
                "secondary_anchor_index": torch.empty(0, dtype=torch.long),
                "member_indices": torch.empty((0, 0), dtype=torch.long),
                "member_valid": torch.empty((0, 0), dtype=torch.bool),
                "relation_kind": torch.empty(0, dtype=torch.long),
                "margin": torch.empty(0),
                "relation_valid": torch.empty(0, dtype=torch.bool),
                "relation_loss_weight": torch.empty(0),
                "relation_normalizer_weight": torch.empty(0),
                "relation_detach_anchor": torch.empty(0, dtype=torch.bool),
            }
            return batch
        width = max(len(value["member_indices"]) for value in relations)
        member_indices = torch.full((len(relations), width), -1, dtype=torch.long)
        member_valid = torch.zeros((len(relations), width), dtype=torch.bool)
        for row, relation in enumerate(relations):
            count = len(relation["member_indices"])
            member_indices[row, :count] = torch.tensor(relation["member_indices"])
            member_valid[row, :count] = True
        batch["relation"] = {
            "anchor_index": torch.tensor([x["anchor_index"] for x in relations]),
            "secondary_anchor_index": torch.tensor(
                [x["secondary_anchor_index"] for x in relations]
            ),
            "member_indices": member_indices,
            "member_valid": member_valid,
            "relation_kind": torch.tensor([x["kind"] for x in relations]),
            "margin": torch.tensor([x["margin"] for x in relations]),
            "relation_valid": torch.ones(len(relations), dtype=torch.bool),
            "relation_loss_weight": torch.tensor([x["loss_weight"] for x in relations]),
            "relation_normalizer_weight": torch.tensor(
                [x["normalizer_weight"] for x in relations]
            ),
            "relation_detach_anchor": torch.tensor(
                [x["detach_anchor"] for x in relations], dtype=torch.bool
            ),
        }
        return batch


def _stratified_relation_members(
    values: Iterable[int], count: int, *, seed: int
) -> list[int]:
    """Keep the first and last candidate, then sample one frame per interior stratum."""

    ordered = sorted({int(value) for value in values})
    if count <= 0 or not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    if count < 2:
        raise ValueError("relation group needs room for first and last members")
    rng = random.Random(seed)
    selected = [ordered[0], ordered[-1]]
    available = ordered[1:-1]
    slots = count - len(selected)
    strata = max(1, min(slots, len(available)))
    for index in range(strata):
        lo = (index * len(available)) // strata
        hi = ((index + 1) * len(available)) // strata
        if hi > lo:
            selected.append(rng.choice(available[lo:hi]))
    selected = list(dict.fromkeys(selected))
    if len(selected) < count:
        remaining = [frame for frame in available if frame not in selected]
        selected.extend(rng.sample(remaining, k=min(count - len(selected), len(remaining))))
    return sorted(selected[:count])
