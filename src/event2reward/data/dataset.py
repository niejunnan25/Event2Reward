"""JSONL index datasets read by byte offset, and their concatenation across sources."""
from __future__ import annotations

import bisect
import json
from array import array
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset



class JsonlOffsetDataset(Dataset):
    """Memory-bounded JSONL dataset storing only 64-bit byte offsets."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = array("Q")
        self._handle = None

    def _append_matching_offsets(self, predicate) -> None:
        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip() and predicate(json.loads(line)):
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def _row(self, index: int) -> dict[str, Any]:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        return json.loads(self._handle.readline())

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __del__(self):
        if self._handle is not None:
            self._handle.close()


class EpochAwareConcatDataset(Dataset):
    """Concatenated sources; index `epoch * len(self) + i` reads item `i` at that epoch."""

    def __init__(self, datasets: Sequence[Dataset]):
        if not datasets:
            raise ValueError("datasets must not be empty")
        self.datasets = list(datasets)
        self.cumulative_sizes: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)
        if total <= 0:
            raise ValueError("datasets must contain at least one row")
        self.total_length = total

    def __len__(self) -> int:
        return self.total_length

    def _source_item(
        self, source_index: int, local_index: int, epoch: int
    ) -> dict[str, Any]:
        if source_index < 0 or source_index >= len(self.datasets):
            raise IndexError(f"source index {source_index} is out of range")
        source = self.datasets[source_index]
        if local_index < 0 or local_index >= len(source):
            raise IndexError(
                f"source-local index {local_index} is outside [0, {len(source)})"
            )
        getter = getattr(source, "get_with_epoch", None)
        value = getter(local_index, epoch) if getter is not None else source[local_index]
        if not isinstance(value, dict):
            raise TypeError("training datasets must return dictionaries")
        return value

    def __getitem__(self, encoded_index: int):
        encoded_index = int(encoded_index)
        if encoded_index < 0:
            raise IndexError("negative encoded indices are not supported")
        epoch, global_index = divmod(encoded_index, self.total_length)
        source_index = bisect.bisect_right(self.cumulative_sizes, global_index)
        source_start = 0 if source_index == 0 else self.cumulative_sizes[source_index - 1]
        local_index = global_index - source_start
        return self._source_item(source_index, local_index, epoch)
