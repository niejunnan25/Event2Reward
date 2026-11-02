"""What an annotation is: labels, event points and their checks.

The labels and events are the paper's. Successful and failed videos are
annotated separately.

- Successful video: `success` (no event), `progress_stall`,
  `regression_and_recovery`, `multiple_events` (two or more of these two), or
  `uncertain`. A progress stall runs from a (the stall begins) to r (progress
  resumes); a regression and recovery from a (the last state before the loss)
  to r (the first state where it is restored).
- Failed video: `no_further_progress` (a: the last state with real progress),
  `unrecovered_regression` (a: the last state before the loss; optionally
  `loss_frame`, where the loss is visible), or `uncertain`. These events last
  until the final frame.

`final_frame` (S, the last frame by default) is the last frame used for
training; in a successful video it is also the completion frame. The page, the
server and the converter share these checks.
"""
from __future__ import annotations

SCHEMA = "event2reward.annotation.v1"
PROJECT_SCHEMA = "event2reward.annotation-project.v1"

LABELS = {
    "success": ("success", "progress_stall", "regression_and_recovery", "multiple_events", "uncertain"),
    "failure": ("no_further_progress", "unrecovered_regression", "uncertain"),
}
RECOVERED = {"progress_stall", "regression_and_recovery"}    # events with a and r
TERMINAL = {"no_further_progress", "unrecovered_regression"}  # events with a, lasting until S
SINGLE = {label: label for label in RECOVERED | TERMINAL}     # labels with exactly one event of their own name


def check(annotation: dict, video: dict) -> dict:
    """Return the annotation in its stored form, or raise ValueError with the reason."""
    uid, mode, last = video["uid"], video["mode"], len(video["times"]) - 1
    label = annotation.get("label")
    if label not in LABELS[mode]:
        raise ValueError(f"{uid}: a {mode} video takes one of {list(LABELS[mode])}")
    frame = lambda value, name: _frame(value, name, last, uid)
    final = frame(annotation.get("final_frame", last), "final_frame")
    if final == 0:
        raise ValueError(f"{uid}: final_frame must lie after frame 0")
    events = []
    for e in annotation.get("events") or []:
        kind = e.get("type")
        if kind not in RECOVERED | TERMINAL:
            raise ValueError(f"{uid}: event type must be one of {sorted(RECOVERED | TERMINAL)}")
        event = {"type": kind, "a_frame": frame(e.get("a_frame"), "a_frame")}
        if kind in RECOVERED:
            event["r_frame"] = frame(e.get("r_frame"), "r_frame")
        if e.get("loss_frame") is not None:
            if kind != "unrecovered_regression":
                raise ValueError(f"{uid}: only an unrecovered regression has a loss_frame")
            event["loss_frame"] = frame(e["loss_frame"], "loss_frame")
        events.append(event)
    events.sort(key=lambda e: e["a_frame"])
    if label == "multiple_events":
        if len(events) < 2 or any(e["type"] not in RECOVERED for e in events):
            raise ValueError(f"{uid}: multiple_events needs two or more progress stalls or regressions and recoveries")
    elif label in SINGLE:
        if len(events) != 1 or events[0]["type"] != label:
            raise ValueError(f"{uid}: {label} needs exactly one {label} event")
    elif events:
        raise ValueError(f"{uid}: {label} has no events")
    previous_r = -1
    for e in events:
        a = e["a_frame"]
        if e["type"] in RECOVERED:
            if not previous_r < a < e["r_frame"] <= final:
                raise ValueError(f"{uid}: events must be ordered, must not overlap, and must end by final_frame")
            previous_r = e["r_frame"]
        else:
            if not 0 < a < final:
                raise ValueError(f"{uid}: a must lie after frame 0 and before final_frame")
            if "loss_frame" in e and not a < e["loss_frame"] <= final:
                raise ValueError(f"{uid}: loss_frame must come after a and by final_frame")
    return {"label": label, "events": events, "final_frame": final}


def _frame(value, name, last, uid):
    if type(value) is not int or not 0 <= value <= last:
        raise ValueError(f"{uid}: {name} must be a frame between 0 and {last}")
    return value
