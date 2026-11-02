"""Label enumerations shared by data and losses."""
from __future__ import annotations

from enum import IntEnum


class ProgressTrend(IntEnum):
    """The direction of progress in the five-frame observation window."""

    GROWTH = 0
    STABLE = 1
    REGRESSION = 2


class ProgressTargetKind(IntEnum):
    """Where an absolute progress target comes from; used to split the logged loss."""

    NONE = 0
    EXPERT = 1
    STALL_SUCCESS = 2        # successful trajectory with progress stalls, stall time removed
    COMPLETION = 3           # confirmed completion frame


class RelationKind(IntEnum):
    """Event relations (see modeling.losses.RelationLoss)."""

    NONE = 0
    NO_FURTHER_PROGRESS = 3
    UNRECOVERED_REGRESSION = 4
    PROGRESS_STALL = 5
    REGRESSION_AND_RECOVERY = 6
