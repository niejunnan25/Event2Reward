"""Turn a dataset manifest into training microbatches.

`index` data (LIBERO) reads JSONL indices through a
fixed-clip-budget sampler: every microbatch holds exactly `batch_size` clips,
mixed across sources by their configured fractions. `trajectories` data (small
real-robot sets) plans every query ahead from annotated trajectories.
"""
import json
import os
from pathlib import Path

from event2reward.data.trajectories import Media, read_trajectories
from .sampling import build_plan, event_inventory, stall_success, supervision

def read_index_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "event2reward.dataset.v1":
        raise ValueError(f"{path}: expected schema event2reward.dataset.v1")
    resolve = lambda value: os.path.abspath(path.parent / value)
    roots = {name: resolve(value) for name, value in manifest["media_roots"].items()}
    indices = {name: resolve(value) for name, value in manifest["indices"].items()}
    return roots, indices


def build_index_dataset(source, path, config):
    from event2reward.data.relation import JsonlRelationDataset
    return JsonlRelationDataset(path, require_goal_present=config["data"]["index"]["require_goal_present"])


class IndexPipeline:
    def __init__(self, config, processor=None, num_processes=1):
        from event2reward.data.dataset import EpochAwareConcatDataset
        from event2reward.data.sampler import FixedClipBudgetBatchSampler
        self.config = config
        t, sources = config["training"], config["data"]["index"]["sources"]
        self.roots, indices = read_index_manifest(config["data"]["manifest"])
        missing = [s["family"] for s in sources if s["family"] not in indices]
        if missing:
            raise ValueError(f"dataset manifest has no index for: {missing}")
        self.datasets = []
        for source in sources:
            dataset = build_index_dataset(source, indices[source["family"]], config)
            if len(dataset) == 0:
                raise ValueError(f"empty index: {source['family']}")
            self.datasets.append(dataset)
        self.dataset = EpochAwareConcatDataset(self.datasets)
        # The sampler plans the whole learning-rate schedule, so stopping early
        # (steps < schedule_steps) reproduces the prefix of the longer run.
        self.sampler = FixedClipBudgetBatchSampler(
            self.datasets, [float(s["fraction"]) for s in sources],
            num_batches=t["schedule_steps"] * t["accumulation"] * num_processes,
            clip_budget=t["batch_size"],
            fill_source_index=next(i for i, s in enumerate(sources) if s["role"] == "fill"),
            seed=t["seed"])
        self.processor = processor

    def summary(self):
        sources = self.config["data"]["index"]["sources"]
        return {"format": "index", "sources": {s["family"]: {"rows": len(d), "fraction": s["fraction"]}
                                               for s, d in zip(sources, self.datasets)},
                "microbatches": len(self.sampler)}

    def loader(self):
        import torch
        from torch.utils.data import DataLoader
        from event2reward.data.collator import QwenProgressCollator
        from event2reward.data.relation import QwenRelationCollator
        c, t = self.config, self.config["training"]
        collator = QwenRelationCollator(QwenProgressCollator(
            self.processor, self.roots, allow_source_goal_dropout=True,
            goal_dropout=c["data"]["index"]["goal_dropout"], seed=t["seed"],
            max_pixels=c["data"]["max_pixels"], image_size=c["data"]["image_size"]))
        kwargs = {"batch_sampler": self.sampler, "collate_fn": collator, "num_workers": t["num_workers"],
                  "pin_memory": torch.cuda.is_available(), "persistent_workers": t["num_workers"] > 0}
        if t["num_workers"]:
            kwargs.update(prefetch_factor=t["prefetch_factor"], in_order=True)
        return DataLoader(self.dataset, **kwargs)


class TrajectoryPipeline:
    """Planned queries over annotated trajectories; runs on a single process."""

    def __init__(self, config, processor=None):
        self.config = config
        self.records, _ = read_trajectories(config["data"]["manifest"], config["data"]["trajectories"]["task"])
        self.lookup = {r["uid"]: r for r in self.records}
        self.plan = build_plan(self.records, config)
        self.media = Media()
        self.processor = processor

    def summary(self):
        from collections import Counter
        usable, skipped = event_inventory(self.records, self.config)
        s = self.config["data"]["trajectories"]["sampling"]
        notes = []
        for task in sorted({r["task"] for r in self.records}):
            if s["relations"]["groups"] and not usable[task]:
                notes.append(f"{task}: no usable events; its event groups use expert frames")
            if s["progress"]["stall_success"] and not any(r["task"] == task and stall_success(r)
                                                               for r in self.records):
                notes.append(f"{task}: no successful trajectories with progress stalls; "
                             "its stall_success queries use expert frames")
        return {"format": "trajectories", "trajectories": len(self.records),
                "per_task": dict(Counter(r["task"] for r in self.records)),
                "events": {task: dict(Counter(e["type"] for _, e in pairs)) for task, pairs in sorted(usable.items())},
                "skipped_events": skipped, "notes": notes,
                "microbatches": len(self.plan),
                "queries": sum(len(b["clips"]) for b in self.plan),
                "supervision": dict(Counter(q["target_source"] for b in self.plan for q in b["clips"]))}

    def batches(self, start=0):
        from event2reward.inference import Observation, prepare_batch
        inputs = self.config["data"]
        for row in self.plan[start:]:
            observations = []
            for q in row["clips"]:
                r, g = self.lookup[q["uid"]], self.lookup[q["goal_uid"]]
                observations.append(Observation(r["instruction"], self.media.frame(r, 0),
                                                self.media.frame(g, q["goal_frame"]),
                                                [self.media.frame(r, f) for f in q["history"]]))
            batch = prepare_batch(self.processor, observations, image_size=inputs["image_size"],
                                  max_pixels=inputs["max_pixels"])
            batch.update(supervision(row, "cpu", self.config["loss"]["terminal_weight"]))
            batch["completion_mask"] = [q["target_source"] == "completion" for q in row["clips"]]
            yield batch
