"""Decode clips and build the Qwen conversation and targets for each one."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch
from PIL import Image, ImageOps

from event2reward.modeling.qwen import NULL_GOAL, PROGRESS_QUERY, TREND_QUERY

from event2reward.modeling.labels import ProgressTargetKind
from event2reward.data.schema import ClipRecord
from event2reward.data.video import DecordFrameReader



def _progress_target_kind(record: ClipRecord) -> int:
    return int(ProgressTargetKind.EXPERT if record.progress_valid else ProgressTargetKind.NONE)


class QwenProgressCollator:
    """Decode independent clips and build one Qwen conversation per clip."""

    def __init__(
        self,
        processor: Any,
        dataset_root: str | Mapping[str, str],
        *,
        goal_dropout: float = 0.30,
        allow_source_goal_dropout: bool = False,
        seed: int = 42,
        max_pixels: int | None = 256 * 28 * 28,
        image_size: int | None = None,
        allow_pyav_fallback: bool = True,
    ):
        if not 0.0 <= goal_dropout <= 1.0:
            raise ValueError("goal_dropout must be in [0,1]")
        self.processor = processor
        self.readers = (
            {
                str(key): DecordFrameReader(
                    value, allow_pyav_fallback=allow_pyav_fallback
                )
                for key, value in dataset_root.items()
            }
            if isinstance(dataset_root, Mapping)
            else {
                "*": DecordFrameReader(
                    dataset_root, allow_pyav_fallback=allow_pyav_fallback
                )
            }
        )
        self.allow_source_goal_dropout = allow_source_goal_dropout
        self.goal_dropout = float(goal_dropout)
        self.seed = int(seed)
        self.max_pixels = max_pixels
        self.image_size = None if image_size is None else int(image_size)
        if self.image_size is not None and self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def _canonicalize(
        self, image: Image.Image, spatial_transform: str = "center_square_v1"
    ) -> Image.Image:
        image = image.convert("RGB")
        if self.image_size is None:
            return image
        if spatial_transform == "lower80_square_v1":
            width, height = image.size
            box = (
                round(0.10 * width),
                round(0.15 * height),
                round(0.90 * width),
                round(0.95 * height),
            )
            cropped = image.crop(box)
            return cropped.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BICUBIC,
                reducing_gap=3.0,
            )
        if spatial_transform == "letterbox_square_v1":
            return ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
                centering=(0.5, 0.5),
            )
        if spatial_transform == "center_4x3_letterbox_v1":
            width, height = image.size
            crop_width = min(width, round(height * 4 / 3))
            left = (width - crop_width) // 2
            image = image.crop((left, 0, left + crop_width, height))
            return ImageOps.pad(
                image,
                (self.image_size, self.image_size),
                method=Image.Resampling.BILINEAR,
                color=(0, 0, 0),
                centering=(0.5, 0.5),
            )
        if spatial_transform != "center_square_v1":
            raise ValueError(f"unsupported spatial transform: {spatial_transform}")
        return ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )

    def _keep_goal(self, record: ClipRecord) -> bool:
        if not record.goal_available:
            return False
        # A stable per-clip draw avoids encoding dataset identity in goal
        # presence and remains exactly reproducible after a crash/resume,
        # independent of worker count, rank, and prefetch timing.
        dropout_key = record.metadata.get("goal_dropout_group", record.sample_id)
        sampling_epoch = int(record.metadata.get("sampling_epoch", 0))
        digest = hashlib.sha256(
            f"{self.seed}:{sampling_epoch}:{dropout_key}".encode()
        ).digest()
        draw = int.from_bytes(digest[:8], "big") / float(2**64)
        dropout = (float(record.metadata.get("source_goal_dropout", self.goal_dropout))
                   if self.allow_source_goal_dropout else self.goal_dropout)
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("source goal dropout must lie in [0,1]")
        return draw >= dropout

    def _read_observations(
        self,
        reader: DecordFrameReader,
        record: ClipRecord,
        indices: list[int],
        *,
        goal: bool = False,
    ) -> list[Image.Image]:
        spatial_transform = str(
            record.metadata.get("spatial_transform", "center_square_v1")
        )
        path = record.resolved_goal_video_path if goal else record.video_path
        query_mode = str(record.metadata.get("video_query_mode", "frame_index"))
        if query_mode == "timestamp":
            fps = float(record.metadata.get("fps", 0.0))
            tolerance_s = float(
                record.metadata.get("video_timestamp_tolerance_s", 0.0001)
            )
            if fps <= 0:
                raise ValueError(
                    "timestamp video queries require positive metadata fps"
                )
            start_frame = record.metadata.get("timeline_start_frame")
            end_frame = record.metadata.get("timeline_end_frame")
            start_time = record.metadata.get("timeline_start_time_seconds")
            end_time = record.metadata.get("timeline_end_time_seconds")
            if None not in (start_frame, end_frame, start_time, end_time):
                frame_span = int(end_frame) - int(start_frame)
                time_span = float(end_time) - float(start_time)
                if frame_span <= 0 or time_span <= 0:
                    raise ValueError("invalid affine timestamp anchors")
                timestamps = [
                    float(start_time)
                    + (int(index) - int(start_frame)) / frame_span * time_span
                    for index in indices
                ]
            else:
                timestamps = [index / fps for index in indices]
            images = reader.read_timestamps(
                path,
                timestamps,
                tolerance_s=tolerance_s,
            )
        elif query_mode == "frame_index":
            images = reader.read(path, indices)
        else:
            raise ValueError(f"unsupported video_query_mode: {query_mode}")
        return [self._canonicalize(image, spatial_transform) for image in images]

    def _conversation(
        self, record: ClipRecord, *, keep_goal: bool | None = None
    ) -> list[dict[str, Any]]:
        keep_goal = self._keep_goal(record) if keep_goal is None else bool(keep_goal)
        reader = self.readers.get(record.dataset_id, self.readers.get("*"))
        if reader is None:
            raise KeyError(f"no dataset root configured for {record.dataset_id}")
        episode_indices = [record.start_frame, *record.history_frames]
        goal_reader = (
            self.readers.get(record.resolved_goal_dataset_id, self.readers.get("*"))
            if keep_goal
            else None
        )
        if keep_goal and goal_reader is None:
            raise KeyError(
                f"no dataset root configured for {record.resolved_goal_dataset_id}"
            )
        coalesce_goal = bool(
            keep_goal
            and goal_reader is reader
            and record.resolved_goal_video_path == record.video_path
        )
        read_indices = [
            *episode_indices,
            *([record.goal_frame] if coalesce_goal else []),
        ]
        decoded_images = self._read_observations(reader, record, read_indices)
        goal_image = decoded_images.pop() if coalesce_goal else None
        episode_images = iter(decoded_images)
        start_image = next(episode_images)
        history_images = list(episode_images)
        if keep_goal and not coalesce_goal:
            goal_image = self._read_observations(goal_reader, record, [record.goal_frame], goal=True)[0]
        extras = {"max_pixels": self.max_pixels} if self.max_pixels is not None else {}
        content: list[dict[str, Any]] = [
            {"type": "text", "text": f"Robot task: {record.instruction}\nStart state:"},
            {"type": "image", "image": start_image, **extras},
            {"type": "text", "text": "\nGoal state:"},
        ]
        if keep_goal:
            content.append({"type": "image", "image": goal_image, **extras})
        else:
            content.append({"type": "text", "text": NULL_GOAL})
        content.append(
            {"type": "text", "text": "\nFive-frame history in displayed order:"}
        )
        for index, image in enumerate(history_images, 1):
            content.extend(
                [
                    {"type": "text", "text": f"\nHistory {index}:"},
                    {"type": "image", "image": image, **extras},
                ]
            )
        content.append(
            {
                "type": "text",
                "text": (
                    "\nPredict absolute task progress and the direction of progress in the history. "
                    f"Progress: {PROGRESS_QUERY} Trend: {TREND_QUERY}"
                ),
            }
        )
        return [{"role": "user", "content": content}]

    def _decode_examples(self, examples: list[dict[str, Any]]):
        records = [ClipRecord.from_dict(example) for example in examples]
        conversations = [self._conversation(record) for record in records]
        return records, conversations

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        records, conversations = self._decode_examples(examples)
        texts = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=False,
                enable_thinking=False,
            )
            for conversation in conversations
        ]
        try:
            from qwen_vl_utils import process_vision_info
        except ModuleNotFoundError:
            # Clips are already decoded PIL images; qwen-vl-utils is optional.
            image_inputs = [
                item["image"]
                for conversation in conversations
                for message in conversation
                for item in message["content"]
                if item.get("type") == "image"
            ]
            video_inputs, video_kwargs = None, {}
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                conversations,
                return_video_kwargs=True,
                return_video_metadata=True,
                image_patch_size=getattr(
                    self.processor.image_processor, "patch_size", 16
                ),
            )
        if video_inputs:
            raise RuntimeError(
                "clips must be passed as images, not as videos"
            )
        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
            **(video_kwargs or {}),
        )
        input_ids = batch.get("input_ids")
        tokenizer = getattr(self.processor, "tokenizer", None)
        if torch.is_tensor(input_ids) and tokenizer is not None:
            query_positions: dict[str, torch.Tensor] = {}
            for name, token in (
                ("progress", PROGRESS_QUERY),
                ("trend", TREND_QUERY),
            ):
                token_id = int(tokenizer.convert_tokens_to_ids(token))
                query_mask = input_ids.eq(token_id)
                counts = query_mask.sum(dim=-1)
                if not bool(counts.eq(1).all()):
                    raise ValueError(
                        f"each flattened clip must contain exactly one {name} query; "
                        f"got {counts.tolist()}"
                    )
                # Compute positions once on CPU. Boolean indexing in the model
                # otherwise has a data-dependent output shape and forces a CUDA
                # synchronization before both task heads on every step.
                query_positions[f"{name}_query_positions"] = query_mask.long().argmax(
                    dim=-1
                )
            batch.update(query_positions)
        batch.update(
            {
                "target_progress": torch.tensor([x.target_progress for x in records]),
                "progress_valid": torch.tensor([x.progress_valid for x in records]),
                "progress_target_kind": torch.tensor(
                    [_progress_target_kind(x) for x in records]
                ),
                "target_trend": torch.tensor([x.target_trend for x in records]),
                "trend_valid": torch.tensor([x.trend_valid for x in records]),
            }
        )
        return batch
