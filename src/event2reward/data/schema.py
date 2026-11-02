"""ClipRecord: one model query."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from event2reward.modeling.labels import ProgressTrend


@dataclass(frozen=True)
class ClipRecord:
    """One independently encoded five-frame VLM observation."""

    sample_id: str
    dataset_id: str
    physical_episode_id: str
    segment_id: str
    instruction: str
    video_path: str
    history_frames: tuple[int, int, int, int, int]
    start_frame: int
    goal_frame: int
    current_frame: int
    target_progress: float
    progress_valid: bool
    target_trend: int
    trend_valid: bool
    history_transform: str
    goal_available: bool = True
    episode_end_frame: int | None = None
    goal_dataset_id: str | None = None
    goal_video_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.history_frames) != 5:
            raise ValueError("history must contain exactly five frames")
        if self.target_trend not in {int(x) for x in ProgressTrend}:
            raise ValueError("target_trend must be growth=0, stable=1, or regression=2")
        if self.progress_valid and not 0.0 <= self.target_progress <= 1.0:
            raise ValueError("valid progress targets must lie in [0, 1]")
        external_goal = self.goal_video_path is not None or self.goal_dataset_id is not None
        if external_goal and self.episode_end_frame is None:
            raise ValueError("cross-episode goals require episode_end_frame")
        if self.goal_dataset_id is not None and self.goal_video_path is None:
            raise ValueError("goal_dataset_id requires goal_video_path")
        episode_end = self.goal_frame if self.episode_end_frame is None else self.episode_end_frame
        if episode_end < self.start_frame:
            raise ValueError("episode_end_frame must not precede start_frame")
        if any(frame < self.start_frame or frame > episode_end for frame in self.history_frames):
            raise ValueError("history crossed the logical segment boundary")
        if self.history_frames[-1] != self.current_frame:
            raise ValueError("the last observed frame must be the state receiving Progress")
        if self.goal_frame < 0:
            raise ValueError("goal_frame must be non-negative")
        if self.goal_video_path is None and self.goal_frame > episode_end:
            raise ValueError("an in-episode goal frame must lie inside the episode")

    @property
    def resolved_goal_dataset_id(self) -> str:
        return self.goal_dataset_id or self.dataset_id

    @property
    def resolved_goal_video_path(self) -> str:
        return self.goal_video_path or self.video_path

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["history_frames"] = list(self.history_frames)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClipRecord":
        copied = dict(value)
        copied["history_frames"] = tuple(copied["history_frames"])
        return cls(**copied)
