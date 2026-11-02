"""Annotation projects: a folder of videos, their frame times and the saved annotations.

    my_project/
      project.json        the videos: uid, task, instruction, mode, split, media path, frame times
      annotations.json    the saved annotation of each video, by uid
      media/              the videos (copied, or re-encoded with --transcode)

`add_videos` adds a folder of successful or failed videos of one task;
`export` writes one `event2reward.annotation.v1` row per annotated video.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil

from .schema import PROJECT_SCHEMA, SCHEMA, check

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
SPLITS = {"train", "val", "test"}


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_project(root: str | Path) -> dict:
    root = Path(root)
    project = json.loads((root / "project.json").read_text())
    if project.get("schema") != PROJECT_SCHEMA:
        raise ValueError(f"{root} is not an annotation project ({PROJECT_SCHEMA})")
    return project


def read_annotations(root: str | Path) -> dict:
    path = Path(root) / "annotations.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def frame_times(path: str | Path) -> list[float]:
    """Presentation time of every frame in seconds, the first frame at 0."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = stream.time_base   # read while the file is open; PyAV clears it on close
        pts = sorted(p.pts for p in container.demux(stream) if p.pts is not None and p.size)
    if len(pts) < 2:
        raise ValueError(f"{path}: fewer than two video frames")
    times = [round(float((p - pts[0]) * base), 6) for p in pts]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError(f"{path}: frame timestamps must increase")
    return times


def transcode(source: Path, target: Path) -> None:
    """Re-encode as H.264 with a keyframe every second, which browsers seek quickly."""
    import av
    times = frame_times(source)
    rate = max(1, round((len(times) - 1) / times[-1]))
    with av.open(str(source)) as reader, av.open(str(target), "w", options={"movflags": "+faststart"}) as writer:
        src = reader.streams.video[0]
        out = writer.add_stream("libx264", rate=rate, options={"g": str(rate), "crf": "18"})
        out.width, out.height, out.pix_fmt = src.codec_context.width, src.codec_context.height, "yuv420p"
        for i, frame in enumerate(reader.decode(src)):
            image = frame.reformat(format="yuv420p")
            image.pts, image.time_base = i, Fraction(1, rate)
            for packet in out.encode(image):
                writer.mux(packet)
        for packet in out.encode():
            writer.mux(packet)


def add_videos(root: str | Path, *, task: str, instruction: str, mode: str, videos: str | Path,
               split: str = "train", reencode: bool = False) -> list[str]:
    """Add every video in a folder as one task's successful or failed videos; return their uids."""
    if mode not in {"success", "failure"}:
        raise ValueError("mode must be success or failure")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}")
    if not task.strip() or not instruction.strip():
        raise ValueError("task and instruction are required")
    root, folder = Path(root), Path(videos)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"no videos in {folder}")
    (root / "media").mkdir(parents=True, exist_ok=True)
    path = root / "project.json"
    project = read_project(root) if path.is_file() else {"schema": PROJECT_SCHEMA, "videos": []}
    known = {v["uid"] for v in project["videos"]}
    added = []
    for file in files:
        uid = f"{task}/{mode}/{file.stem}"
        if uid in known:
            continue
        target = root / "media" / task / mode / (file.stem + (".mp4" if reencode else file.suffix.lower()))
        target.parent.mkdir(parents=True, exist_ok=True)
        if reencode:
            transcode(file, target)
        else:
            shutil.copy2(file, target)
        project["videos"].append({"uid": uid, "task": task, "instruction": instruction.strip(), "mode": mode,
                                  "split": split, "video": target.relative_to(root).as_posix(),
                                  "source": file.name, "times": frame_times(target)})
        added.append(uid)
    _write_json(path, project)
    return added


def save_annotation(root: str | Path, uid: str, annotation: dict) -> dict:
    root = Path(root)
    video = next((v for v in read_project(root)["videos"] if v["uid"] == uid), None)
    if video is None:
        raise ValueError(f"unknown video {uid}")
    stored = check(annotation, video) | {"saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    annotations = read_annotations(root)
    annotations[uid] = stored
    _write_json(root / "annotations.json", annotations)
    return stored


def export(root: str | Path) -> list[dict]:
    """One row per annotated video, in project order."""
    root = Path(root)
    annotations = read_annotations(root)
    rows = []
    for v in read_project(root)["videos"]:
        if v["uid"] in annotations:
            a = check(annotations[v["uid"]], v) | {"saved_at": annotations[v["uid"]].get("saved_at")}
            rows.append({"schema": SCHEMA, **{k: v[k] for k in ("uid", "task", "instruction", "mode", "split",
                                                                  "video", "times")}, **a})
    return rows


def write_export(root: str | Path, output: str | Path) -> int:
    rows = export(root)
    Path(output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return len(rows)
