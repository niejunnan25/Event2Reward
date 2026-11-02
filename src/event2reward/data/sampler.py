"""FixedClipBudgetBatchSampler: the source mixture of every index-data run."""
from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class FixedClipBudgetBatchSampler(Sampler[list[int]]):
    """Mix sources by clip fraction with exactly `clip_budget` clips per batch.

    A relation group holds several clips and is never split, so batches contain
    a varying number of groups. Each source receives groups while it is below
    its cumulative share of clips; a one-clip fill source tops up the remainder.
    Fractions count clips, not rows. Order is deterministic given the seed.
    """

    def __init__(
        self,
        datasets: Sequence[object],
        clip_fractions: Sequence[float],
        *,
        num_batches: int,
        clip_budget: int,
        fill_source_index: int,
        seed: int = 42,
    ):
        if len(datasets) != len(clip_fractions) or not datasets:
            raise ValueError(
                "datasets and clip_fractions must be non-empty and aligned"
            )
        if num_batches <= 0 or clip_budget <= 0:
            raise ValueError("num_batches and clip_budget must be positive")
        if not 0 <= fill_source_index < len(datasets):
            raise ValueError("fill_source_index is out of range")
        if any(float(value) < 0 for value in clip_fractions):
            raise ValueError("clip fractions must be non-negative")
        total_fraction = sum(float(value) for value in clip_fractions)
        if total_fraction <= 0:
            raise ValueError("at least one clip fraction must be positive")
        self.datasets = list(datasets)
        self.fractions = [float(value) / total_fraction for value in clip_fractions]
        self.num_batches = int(num_batches)
        self.clip_budget = int(clip_budget)
        self.fill_source_index = int(fill_source_index)
        self.seed = int(seed)
        self.lengths = [len(dataset) for dataset in datasets]  # type: ignore[arg-type]
        if any(length <= 0 for length in self.lengths):
            raise ValueError("every fixed-budget source must be non-empty")
        self.offsets = [0]
        for length in self.lengths[:-1]:
            self.offsets.append(self.offsets[-1] + int(length))
        self.total_length = sum(self.lengths)
        self._affine_parameters: dict[tuple[int, int], tuple[int, int]] = {}
        for source_index, dataset in enumerate(self.datasets):
            counter = getattr(dataset, "clip_count", None)
            if not callable(counter):
                raise TypeError(
                    f"source {source_index} does not expose clip_count(index)"
                )
            for local_index in range(len(dataset)):  # type: ignore[arg-type]
                cost = int(getattr(dataset, "max_clip_count", counter)(local_index))
                if cost <= 0 or cost > self.clip_budget:
                    raise ValueError(
                        f"source {source_index} row {local_index} has invalid clip cost {cost}"
                    )
                if source_index == self.fill_source_index and cost != 1:
                    raise ValueError("the fill source must contain only one-clip rows")

    def __len__(self) -> int:
        return self.num_batches

    def _rotating_local(self, source: int, position: int) -> tuple[int, int]:
        length = self.lengths[source]
        epoch, offset = divmod(int(position), length)
        key = (source, epoch)
        parameters = self._affine_parameters.get(key)
        if parameters is None:
            rng = random.Random(self.seed + 97_409 * (source + 1) + 1_000_003 * epoch)
            multiplier = rng.randrange(1, length + 1)
            while math.gcd(multiplier, length) != 1:
                multiplier = multiplier % length + 1
            parameters = (multiplier, rng.randrange(length))
            self._affine_parameters[key] = parameters
        multiplier, shift = parameters
        return (multiplier * offset + shift) % length, epoch

    def _encoded(self, source: int, position: int) -> tuple[int, int]:
        local, epoch = self._rotating_local(source, position)
        encoded = self.offsets[source] + local + epoch * self.total_length
        dataset = self.datasets[source]
        counter = getattr(dataset, "clip_count_for_epoch", None)
        cost = int(counter(local, epoch) if counter else dataset.clip_count(local))
        return encoded, cost

    def __iter__(self) -> Iterator[list[int]]:
        positions = [0] * len(self.datasets)
        emitted_clips = [0] * len(self.datasets)
        for batch_index in range(self.num_batches):
            remaining = self.clip_budget
            batch: list[int] = []
            target_total = (batch_index + 1) * self.clip_budget
            # Non-fill sources follow cumulative clip deficits. A relation group
            # is selected only when it improves the source's absolute deficit;
            # the unit-cost expert source then fills the exact batch remainder.
            source_order = sorted(
                (
                    index
                    for index in range(len(self.datasets))
                    if index != self.fill_source_index
                ),
                key=lambda index: (
                    -(target_total * self.fractions[index] - emitted_clips[index]),
                    index,
                ),
            )
            made_progress = True
            while made_progress:
                made_progress = False
                for source in source_order:
                    target = target_total * self.fractions[source]
                    deficit = target - emitted_clips[source]
                    if deficit <= 0:
                        continue
                    encoded, cost = self._encoded(source, positions[source])
                    if cost > remaining:
                        continue
                    before = abs(deficit)
                    after = abs(deficit - cost)
                    if after > before:
                        continue
                    batch.append(encoded)
                    positions[source] += 1
                    emitted_clips[source] += cost
                    remaining -= cost
                    made_progress = True
            fill = self.fill_source_index
            while remaining:
                encoded, cost = self._encoded(fill, positions[fill])
                if cost != 1:
                    raise RuntimeError("a fill source returned a multi-clip row")
                batch.append(encoded)
                positions[fill] += 1
                emitted_clips[fill] += 1
                remaining -= 1
            random.Random(self.seed + 2_000_033 * batch_index).shuffle(batch)
            yield batch
