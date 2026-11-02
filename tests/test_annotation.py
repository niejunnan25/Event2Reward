"""The annotation tool: checks, projects, the server, and conversion to training data."""
import json
import threading
import urllib.request

import pytest

from event2reward.annotation import project as store
from event2reward.annotation.schema import check
from event2reward.data.converters.annotations import convert_annotations
from event2reward.data.trajectories import read_trajectories

INSTRUCTION = "push the block"


def video(mode, frames=41):
    return {"uid": f"push/{mode}/v", "mode": mode, "times": [i / 10 for i in range(frames)]}


STALL = lambda a, r: {"type": "progress_stall", "a_frame": a, "r_frame": r}
REGRESSION = lambda a, r: {"type": "regression_and_recovery", "a_frame": a, "r_frame": r}


@pytest.mark.parametrize("annotation, stored", [
    ({"label": "success"}, {"events": [], "final_frame": 40}),
    ({"label": "success", "final_frame": 35}, {"final_frame": 35}),
    ({"label": "regression_and_recovery", "events": [REGRESSION(5, 20)]}, {"events": [REGRESSION(5, 20)]}),
    ({"label": "multiple_events", "events": [REGRESSION(25, 30), STALL(5, 10)]}, {"events": [STALL(5, 10), REGRESSION(25, 30)]}),
    ({"label": "uncertain"}, {"events": []}),
])
def test_valid_success_annotations(annotation, stored):
    result = check(annotation, video("success"))
    assert {k: result[k] for k in stored} == stored


def test_valid_failure_annotation():
    result = check({"label": "unrecovered_regression", "final_frame": 35,
                    "events": [{"type": "unrecovered_regression", "a_frame": 20, "loss_frame": 22}]}, video("failure"))
    assert result["events"] == [{"type": "unrecovered_regression", "a_frame": 20, "loss_frame": 22}]


@pytest.mark.parametrize("mode, annotation, message", [
    ("success", {"label": "unrecovered_regression"}, "takes one of"),
    ("success", {"label": "progress_stall"}, "exactly one progress_stall"),
    ("success", {"label": "progress_stall", "events": [STALL(5, None)]}, "r_frame"),
    ("success", {"label": "regression_and_recovery", "final_frame": 15, "events": [REGRESSION(5, 20)]}, "end by final_frame"),
    ("success", {"label": "multiple_events", "events": [STALL(5, 10)]}, "two or more"),
    ("success", {"label": "multiple_events", "events": [STALL(5, 15), STALL(10, 20)]}, "not overlap"),
    ("success", {"label": "success", "events": [STALL(5, 10)]}, "has no events"),
    ("success", {"label": "progress_stall", "events": [{"type": "hesitation", "a_frame": 5, "r_frame": 10}]}, "event type"),
    ("failure", {"label": "no_further_progress", "final_frame": 5,
                 "events": [{"type": "no_further_progress", "a_frame": 5}]}, "before final_frame"),
    ("failure", {"label": "no_further_progress", "events": [{"type": "no_further_progress", "a_frame": 5, "loss_frame": 8}]},
     "only an unrecovered regression"),
    ("failure", {"label": "unrecovered_regression", "events": [{"type": "unrecovered_regression", "a_frame": 50}]},
     "between 0 and 40"),
])
def test_invalid_annotations(mode, annotation, message):
    with pytest.raises(ValueError, match=message):
        check(annotation, video(mode))


def write_video(path, frames=41, fps=10):
    import av
    import numpy as np
    from fractions import Fraction
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 4 * i, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


@pytest.fixture
def annotated_project(tmp_path):
    pytest.importorskip("av")
    for name in ("clean1", "clean2", "setback"):
        write_video(tmp_path / "videos/success" / f"{name}.mp4")
    write_video(tmp_path / "videos/failure/drop.mp4", frames=31, fps=30)
    write_video(tmp_path / "videos/held_out/clean3.mp4")
    root = tmp_path / "project"
    store.add_videos(root, task="push", instruction=INSTRUCTION, mode="success", videos=tmp_path / "videos/success")
    store.add_videos(root, task="push", instruction=INSTRUCTION, mode="failure", videos=tmp_path / "videos/failure")
    store.add_videos(root, task="push", instruction=INSTRUCTION, mode="success", videos=tmp_path / "videos/held_out",
                     split="test")
    for uid, annotation in {
        "push/success/clean1": {"label": "success"},
        "push/success/clean2": {"label": "success", "final_frame": 38},
        "push/success/setback": {"label": "regression_and_recovery", "events": [REGRESSION(10, 20)]},
        "push/failure/drop": {"label": "unrecovered_regression", "events": [{"type": "unrecovered_regression", "a_frame": 12}]},
        "push/success/clean3": {"label": "uncertain"},
    }.items():
        store.save_annotation(root, uid, annotation)
    return root


def test_projects_add_save_and_export(annotated_project, tmp_path):
    videos = {v["uid"]: v for v in store.read_project(annotated_project)["videos"]}
    assert videos["push/failure/drop"]["times"][1] == pytest.approx(1 / 30, abs=1e-4)   # frame times, any rate
    assert videos["push/success/clean3"]["split"] == "test"
    # Adding the same folder again skips the videos already there.
    assert store.add_videos(annotated_project, task="push", instruction=INSTRUCTION, mode="success",
                            videos=tmp_path / "videos/success") == []
    rows = store.export(annotated_project)
    assert [r["uid"] for r in rows] == list(videos) and all(r["schema"] == "event2reward.annotation.v1" for r in rows)
    with pytest.raises(ValueError, match="exactly one regression_and_recovery"):
        store.save_annotation(annotated_project, "push/success/setback", {"label": "regression_and_recovery"})


def test_reencoded_videos_keep_every_frame(tmp_path):
    pytest.importorskip("av")
    write_video(tmp_path / "videos/v.mp4", frames=31, fps=30)
    store.add_videos(tmp_path / "p", task="push", instruction=INSTRUCTION, mode="success", videos=tmp_path / "videos",
                     reencode=True)
    (v,) = store.read_project(tmp_path / "p")["videos"]
    assert v["video"] == "media/push/success/v.mp4" and len(v["times"]) == 31
    assert v["times"][-1] == pytest.approx(1.0, abs=1e-3)


def test_exports_become_trajectories(annotated_project, tmp_path):
    export = tmp_path / "annotations.jsonl"
    assert store.write_export(annotated_project, export) == 5
    report = convert_annotations(export, annotated_project, tmp_path / "data/dataset.json")
    assert report["left_out"] == [{"uid": "push/success/clean3", "reason": "uncertain"}]
    task = report["tasks"]["push"]
    assert task["expert"] == 2 and task["event"] == 2
    assert task["events"] == {"regression_and_recovery": 1, "unrecovered_regression": 1}
    records, _ = read_trajectories(tmp_path / "data/dataset.json")
    by_uid = {r["uid"]: r for r in records}
    assert by_uid["push/success/clean2"]["C"] == by_uid["push/success/clean2"]["S"] == 38 and by_uid["push/success/clean2"]["role"] == "expert"
    assert by_uid["push/success/setback"]["events"] == [{"type": "regression_and_recovery", "anchor_frame": 10, "end_frame": 20}]
    # The paper's events become the trajectory-format types.
    assert by_uid["push/failure/drop"]["C"] is None and by_uid["push/failure/drop"]["S"] == 30
    assert by_uid["push/failure/drop"]["events"] == [{"type": "unrecovered_regression", "anchor_frame": 12, "end_frame": 30}]


def test_the_server_serves_saves_and_exports(annotated_project):
    from http.server import ThreadingHTTPServer
    from event2reward.annotation.server import Handler
    server = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"root": annotated_project}))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urllib.request.urlopen(url + "/").read()
        assert b"Event2Reward" in page and b"ProgressMark" not in page
        project = json.load(urllib.request.urlopen(url + "/api/project"))
        assert len(project["videos"]) == 5 and project["labels"]["failure"][0] == "no_further_progress"
        media = urllib.request.Request(url + "/" + project["videos"][0]["video"], headers={"Range": "bytes=0-99"})
        with urllib.request.urlopen(media) as response:
            assert response.status == 206 and len(response.read()) == 100
        post = lambda body: urllib.request.Request(url + "/api/annotation", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"})
        saved = json.load(urllib.request.urlopen(post({"uid": "push/success/clean3",
                                                       "annotation": {"label": "success"}})))
        assert saved["final_frame"] == 40 and "completion_frame" not in saved
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(post({"uid": "push/success/clean3", "annotation": {"label": "unrecovered_regression"}}))
        assert error.value.code == 400
        rows = urllib.request.urlopen(url + "/api/export.jsonl").read().decode().splitlines()
        assert json.loads(rows[-1])["label"] == "success"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(url + "/../project.json")
    finally:
        server.shutdown()
