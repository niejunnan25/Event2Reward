"""Annotated trajectories (`event2reward.trajectories.v1`): reading, checks and frame access."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

# The paper's four events: the first two end in recovery, the last two last until the end.
EVENT_TYPES = {"progress_stall", "regression_and_recovery", "no_further_progress", "unrecovered_regression"}
TERMINAL = {"no_further_progress", "unrecovered_regression"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_trajectories(manifest_path, task=None):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "event2reward.trajectories.v1":
        raise ValueError("expected event2reward.trajectories.v1; use the environment importer first")
    records = deepcopy(manifest["trajectories"])
    seen = set()
    for r in records:
        if not isinstance(r.get("uid"), str) or not r["uid"] or r["uid"] in seen:
            raise ValueError("trajectory uid must be nonempty and unique")
        seen.add(r["uid"])
        if r.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"explicit train/val/test split required: {r['uid']}")
        if r.get("role") not in {"expert", "event"}:
            raise ValueError("role must be expert or event; it is not inferred from success")
        if not r.get("task") or not r.get("instruction"):
            raise ValueError("task and instruction required")
        times = r["times"]
        if not times or times[0] != 0 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in times) or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("times must start at zero and be finite, strictly increasing model seconds")
        if type(r["S"]) is not int or not 0 < r["S"] < len(times):
            raise ValueError("S must be a valid inclusive last frame")
        if r["C"] is not None and (type(r["C"]) is not int or not 0 < r["C"] <= r["S"]):
            raise ValueError("C must be a confirmed completion frame or null")
        if r["role"] == "expert" and (r["C"] is None or r.get("events")):
            raise ValueError("clean experts require completion and no events")
        if bool(r.get("frame_directory")) == bool(r.get("video_path")):
            raise ValueError("provide exactly one of frame_directory or video_path")
        for key in ("frame_directory", "video_path"):
            if r.get(key):
                r[key] = str((path.parent / r[key]).resolve())
        r.setdefault("frame_pattern", "{frame:06d}.png")
        pattern = r["frame_pattern"]
        if r.get("frame_directory"):
            names = [pattern.format(frame=f) for f in range(r["S"] + 1)]
            if any(not name or Path(name).name != name or name in {".", ".."} for name in names):
                raise ValueError("frame_pattern must produce a filename")
            if len(set(names)) != len(names):
                raise ValueError("frame_pattern must map each frame to a distinct filename")
        r.setdefault("events", [])
        last = -1
        for e in sorted(r["events"], key=lambda x: x["anchor_frame"]):
            if e.get("type") not in EVENT_TYPES:
                raise ValueError(f"{r['uid']}: event type must be one of {sorted(EVENT_TYPES)}")
            a, b, loss = e["anchor_frame"], e["end_frame"], e.get("loss_frame")
            if type(a) is not int or type(b) is not int or not 0 <= a < b <= r["S"] or a < last:
                raise ValueError(f"{r['uid']}: event intervals must be ordered, nonoverlapping and inside the trajectory")
            last = b
            if loss is not None and (e["type"] not in {"regression_and_recovery", "unrecovered_regression"} or type(loss) is not int
                                     or not a < loss <= b or e["type"] == "regression_and_recovery" and loss == b):
                raise ValueError(f"{r['uid']}: loss_frame belongs to regression_and_recovery or unrecovered_regression, after the anchor "
                                 "and inside the event (before the recovery of a regression)")
            if e["type"] in TERMINAL:
                if r["C"] is not None:
                    raise ValueError(f"{r['uid']}: no further progress and unrecovered regression mean failure; set C to null")
            elif r["C"] is not None and b > r["C"]:
                raise ValueError(f"{r['uid']}: {e['type']} must end by the completion frame C")
        if r["role"] == "event" and not r["events"]:
            raise ValueError("event trajectories require event annotations")
    selected = [r for r in records if r["split"] == "train" and (task is None or r["task"] == task)]
    if not selected:
        raise ValueError("no training trajectories match data.trajectories.task")
    for name in {r["task"] for r in selected}:
        experts = [r for r in selected if r["task"] == name and r["role"] == "expert"]
        if len(experts) < 2:
            raise ValueError(f"{name}: at least two training experts are needed for independent same-task goals")
    # A mislabeled duplicate must not silently cross the evaluation split.
    media_splits = {}
    for r in records:
        media = r.get("frame_directory") or r["video_path"]
        if media in media_splits and media_splits[media] != r["split"]:
            raise ValueError("the same physical media appears across train/evaluation splits")
        if media in media_splits:
            raise ValueError("duplicate physical media: provide one trajectory record per video")
        media_splits[media] = r["split"]
    return selected, manifest


class Media:
    def __init__(self):
        self.reader = None

    def frame(self, record, frame):
        if record.get("frame_directory"):
            path = Path(record["frame_directory"]) / record["frame_pattern"].format(frame=frame)
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        if self.reader is None:
            from event2reward.data.video import DecordFrameReader
            self.reader = DecordFrameReader("/", allow_pyav_fallback=True)
        return self.reader.read(record["video_path"], [frame])[0]
