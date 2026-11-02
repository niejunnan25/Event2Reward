"""LIBERO: write `dataset.json` for annotated relation indices of one suite."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def write_dataset_manifest(indices_dir: str | Path, output: str | Path,
                           media_roots: Mapping[str, str | Path]) -> Path:
    """Write `dataset.json` for annotated LIBERO relation indices of one suite.

    Families follow the index file names: `train_expert_taskNN.jsonl` ->
    `expert_taskNN`, `train_kindK.jsonl` -> `event_kindK`, and
    `train_expert_all.jsonl` -> `expert_fill`. Labels are not modified.
    """
    indices_dir, output = Path(indices_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    indices, files = {}, []
    for path in sorted(indices_dir.glob("train_*.jsonl")):
        name = path.stem.removeprefix("train_")
        family = ("expert_fill" if name == "expert_all" else name if name.startswith("expert_task")
                  else f"event_{name}" if name.startswith("kind") else None)
        if family is not None:
            indices[family] = os.path.relpath(path, output.parent)
            files.append(path)
    if "expert_fill" not in indices:
        raise FileNotFoundError(indices_dir / "train_expert_all.jsonl")
    needed = set()
    for path in files:
        with path.open() as rows:
            for line in rows:
                for clip in json.loads(line)["clips"]:
                    needed.update(v for v in (clip["dataset_id"], clip.get("goal_dataset_id")) if v)
    missing = sorted(needed - set(media_roots))
    if missing:
        raise ValueError(f"the indices read media from {sorted(needed)}; add --media-root for {missing}")
    roots = {}
    for name, root in media_roots.items():
        root = Path(root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        roots[name] = os.path.relpath(root, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "event2reward.dataset.v1", "media_roots": roots, "indices": indices}
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output
