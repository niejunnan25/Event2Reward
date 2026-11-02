"""Annotation exports: write `event2reward.trajectories.v1` for training.

`event2reward annotate export` writes one `event2reward.annotation.v1` row per
annotated video (src/event2reward/annotation/schema.py). Each row becomes one
trajectory:

- success                                          -> role expert, C = final_frame
- progress stall, regression and recovery, multiple -> role event,  C = final_frame
- no further progress, unrecovered regression       -> role event,  C = null

`final_frame` becomes S, the last frame used, and in a successful video also
C, its completion frame. Rows labeled uncertain are left out and reported.

The paper's four events keep their names as the event types of the trajectory
format (docs/data.md).
"""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path

from event2reward.annotation.schema import SCHEMA, check


def trajectory_event(event: dict, final: int) -> dict:
    out = {"type": event["type"], "anchor_frame": event["a_frame"],
           "end_frame": event.get("r_frame", final)}
    if "loss_frame" in event:
        out["loss_frame"] = event["loss_frame"]
    return out


def trajectory(row: dict, project_dir: Path, output_dir: Path) -> dict | None:
    a = check(row, row)
    if a["label"] == "uncertain":
        return None
    video = (project_dir / row["video"]).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"{row['uid']}: {video}")
    return {"uid": row["uid"], "task": row["task"], "instruction": row["instruction"], "split": row["split"],
            "role": "event" if a["events"] else "expert", "video_path": os.path.relpath(video, output_dir),
            "times": row["times"], "S": a["final_frame"],
            "C": a["final_frame"] if row["mode"] == "success" else None,
            "events": [trajectory_event(e, a["final_frame"]) for e in a["events"]]}


def convert_annotations(export: str | Path, project_dir: str | Path, output: str | Path) -> dict:
    """Write `dataset.json` from an annotation export; return what was used and left out."""
    from event2reward.data.trajectories import read_trajectories
    project_dir, output = Path(project_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in Path(export).read_text().splitlines() if line.strip()]
    if any(row.get("schema") != SCHEMA for row in rows):
        raise ValueError(f"expected rows of {SCHEMA}, from event2reward annotate export")
    trajectories, used, left_out = [], [], []
    for row in rows:
        t = trajectory(row, project_dir, output.parent)
        if t is None:
            left_out.append({"uid": row["uid"], "reason": "uncertain"})
        else:
            trajectories.append(t)
            used.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"schema": "event2reward.trajectories.v1", "trajectories": trajectories},
                                 indent=1, ensure_ascii=False) + "\n")
    try:
        read_trajectories(output)
    except ValueError:
        output.unlink()
        raise
    tasks = {}
    for t, row in zip(trajectories, used):
        entry = tasks.setdefault(t["task"], {"splits": Counter(), "expert": 0, "event": 0, "events": Counter()})
        entry["splits"][t["split"]] += 1
        entry[t["role"]] += 1
        entry["events"].update(e["type"] for e in row["events"])
    return {"output": str(output), "trajectories": len(trajectories),
            "tasks": {name: {k: dict(v) if isinstance(v, Counter) else v for k, v in e.items()}
                      for name, e in tasks.items()},
            "left_out": left_out}
