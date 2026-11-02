"""Training data: manifests, index datasets, the sampler and trajectory planning."""
import copy
import json

import pytest

from event2reward.data.trajectories import read_trajectories
from event2reward.train.config import DEFAULTS, _merge, validate_config
from event2reward.train.sampling import build_plan, corrected_time_progress
from conftest import trajectory_records

INDEX_SOURCES = [{"family": "expert", "type": "relation", "role": "expert", "fraction": 0.5},
                 *[{"family": f"event_kind{k}", "type": "relation", "role": "event", "fraction": 0.125}
                   for k in range(3, 7)],
                 {"family": "expert_fill", "type": "relation", "role": "fill", "fraction": 0.0}]


def config(**sections):
    base = {"model_path": "m", "output_dir": "o",
            "training": {"steps": 3, "schedule_steps": 9, "warmup_steps": 1, "save_every": 1}}
    for key, value in sections.items():
        base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    c = _merge(DEFAULTS, base)
    validate_config(c)
    return c


def trajectory_config(manifest="dataset.json", trajectories=None, **sections):
    sections = {"loss": {"trend": 0.0}, "tuning": {"heads": {"progress": True, "trend": False, "fp32": False}},
                **sections}
    return config(data={"format": "trajectories", "manifest": str(manifest), "trajectories": trajectories or {}},
                  **sections)


def test_index_pipeline_fills_every_microbatch_exactly(synthetic_index):
    from event2reward.train.pipelines import IndexPipeline
    c = config(data={"format": "index", "manifest": str(synthetic_index), "index": {"sources": INDEX_SOURCES}})
    pipeline = IndexPipeline(c)
    assert len(pipeline.sampler) == 9
    for indices in pipeline.sampler:
        assert sum(len(pipeline.dataset[i]["clips"]) for i in indices) == 10


def test_index_manifest_must_name_every_configured_family(synthetic_index, tmp_path):
    from event2reward.train.pipelines import IndexPipeline
    manifest = json.loads(synthetic_index.read_text())
    del manifest["indices"]["event_kind5"]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({**manifest, "media_roots": {"example_media": str(synthetic_index.parent)},
                                "indices": {k: str(synthetic_index.parent / v) for k, v in manifest["indices"].items()}}))
    with pytest.raises(ValueError, match="event_kind5"):
        IndexPipeline(config(data={"format": "index", "manifest": str(path), "index": {"sources": INDEX_SOURCES}}))


def test_libero_manifest_from_index_files(tmp_path):
    from event2reward.data.converters.libero import write_dataset_manifest
    indices, media = tmp_path / "indices", tmp_path / "media"
    indices.mkdir(); media.mkdir()
    for name in ("expert_all", "expert_task00", "expert_task01", "kind3", "kind5", "notes"):
        (indices / f"train_{name}.jsonl").write_text("")
    path = write_dataset_manifest(indices, tmp_path / "out/dataset.json", {"libero": media})
    manifest = json.loads(path.read_text())
    assert set(manifest["indices"]) == {"expert_fill", "expert_task00", "expert_task01", "event_kind3", "event_kind5"}
    assert manifest["media_roots"] == {"libero": "../media"}
    with pytest.raises(FileExistsError):
        write_dataset_manifest(indices, path, {"libero": media})
    (indices / "train_kind3.jsonl").write_text(json.dumps({"clips": [{"dataset_id": "libero", "goal_dataset_id": "goals"}]}) + "\n")
    with pytest.raises(ValueError, match="goals"):
        write_dataset_manifest(indices, tmp_path / "out2/dataset.json", {"libero": media})


def test_trajectory_records_are_validated(tmp_path):
    rows = trajectory_records()
    for mutate, message in [(lambda r: r[0].update(C=None), "clean experts"),
                            (lambda r: r[0].update(role="success"), "role"),
                            (lambda r: r[3]["events"][0].update(end_frame=99), "inside"),
                            (lambda r: r[4].update(C=40), "mean failure"),
                            (lambda r: r[1].update(frame_directory="e0"), "duplicate")]:
        broken = copy.deepcopy(rows)
        mutate(broken)
        path = tmp_path / "dataset.json"
        path.write_text(json.dumps({"schema": "event2reward.trajectories.v1", "trajectories": broken}))
        with pytest.raises(ValueError, match=message):
            read_trajectories(path)


def test_plan_supervision_and_goals(trajectory_data):
    c = trajectory_config(trajectory_data)
    records, _ = read_trajectories(trajectory_data)
    plan = build_plan(records, c)
    assert len(plan) == 3 * c["training"]["accumulation"]
    lookup = {r["uid"]: r for r in records}
    for batch in plan:
        sources = [q["target_source"] for q in batch["clips"]]
        assert len(sources) == 10
        assert sources.count("completion") == 1
        assert sources.count("stall_success") == 1
        assert sources.count("relation") == 5
        for q in batch["clips"]:
            goal = lookup[q["goal_uid"]]
            assert goal["role"] == "expert" and goal["uid"] != q["uid"]
            assert max(q["history"]) <= q["frame"]
        relation = batch["relations"][0]
        assert batch["clips"][relation["anchor_index"]]["uid"] in {"stall", "failure"}


def test_stalls_do_not_count_as_progress():
    record = dict(times=[i / 10 for i in range(41)], C=40,
                  events=[dict(type="progress_stall", anchor_frame=10, end_frame=20)])
    assert corrected_time_progress(record, 10) == corrected_time_progress(record, 20) == pytest.approx(1 / 3)
    assert corrected_time_progress(record, 40) == 1.0
    with pytest.raises(ValueError):
        corrected_time_progress(dict(record, C=None), 5)


def test_sampling_budget_must_match_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        trajectory_config(trajectories={"sampling": {"progress": {"expert": 4}}})


def test_expert_clips_draw_a_new_query_frame_each_epoch(tmp_path):
    from event2reward.data.converters.synthetic import make_example
    from event2reward.data.relation import JsonlRelationDataset
    root = make_example(tmp_path / "data").parent
    row = json.loads((root / "expert.jsonl").read_text().splitlines()[0])
    rows = []
    for slot in range(8):
        clip = copy.deepcopy(row["clips"][0])
        clip["metadata"].update(category="clean_success", stage2_dynamic_sampling=True,
                                stage2_candidate_frames=list(range(64)), stage2_required_frames=[0, 63],
                                stage2_selection_count=8, stage2_sampling_seed=7, stage2_selection_slot=slot,
                                stage2_num_frames=64, stage2_history_stride=1)
        rows.append({"clips": [clip], "relation": None})
    (root / "dynamic.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = JsonlRelationDataset(root / "dynamic.jsonl")
    for epoch in (0, 1):
        frames = [dataset.get_with_epoch(i, epoch)["clips"][0] for i in range(8)]
        assert {0, 63} <= {c["current_frame"] for c in frames}
        for c in frames:
            assert c["progress_valid"] and c["target_progress"] == c["current_frame"] / 63
            assert c["history_frames"][-1] == c["current_frame"] and min(c["history_frames"]) >= 0
    first = [dataset.get_with_epoch(i, 0)["clips"][0]["current_frame"] for i in range(8)]
    again = [dataset.get_with_epoch(i, 0)["clips"][0]["current_frame"] for i in range(8)]
    later = [dataset.get_with_epoch(i, 1)["clips"][0]["current_frame"] for i in range(8)]
    assert first == again and first != later


def test_media_roots_reached_through_a_symlink(tmp_path, synthetic_index):
    """Clip paths written through a symlinked mount still lie inside their root."""
    from event2reward.data.video import DecordFrameReader
    real = synthetic_index.parent
    (tmp_path / "mount").symlink_to(real)
    reader = DecordFrameReader(str(real))
    assert reader._resolve(str(tmp_path / "mount/trajectory.mp4")).name == "trajectory.mp4"
    with pytest.raises(ValueError, match="escapes"):
        reader._resolve(str(tmp_path / "elsewhere.mp4"))


def test_expert_frames_get_growth_trend_labels_when_trend_is_trained(trajectory_data):
    records, _ = read_trajectories(trajectory_data)
    libero_history = {"history_offsets": [0.4, 0.3, 0.2, 0.1, 0]}
    off = build_plan(records, trajectory_config(trajectory_data, trajectories=libero_history))
    assert not any(q["trend_valid"] for b in off for q in b["clips"])
    c = trajectory_config(trajectory_data, trajectories=libero_history, loss={"trend": 1.0},
                          tuning={"heads": {"progress": True, "trend": True, "fp32": False}})
    on = build_plan(records, c)
    labeled = [q for b in on for q in b["clips"] if q["trend_valid"]]
    assert labeled and all(q["target_source"] == "expert" for q in labeled)
    times = {r["uid"]: r["times"] for r in records}
    for q in labeled:
        assert times[q["uid"]][q["frame"]] >= 0.4 and len(set(q["history"])) == 5
    # The labels do not change which frames are sampled.
    assert [[q["frame"] for q in b["clips"]] for b in off] == [[q["frame"] for q in b["clips"]] for b in on]


def four_event_records():
    """Three experts and one trajectory of each event type, 41 frames at 10 Hz."""
    rows = trajectory_records()
    end = 40
    common = {k: rows[0][k] for k in ("task", "instruction", "split", "times", "S")}
    rows.append(dict(common, uid="regression", role="event", C=end, frame_directory="r",
                     events=[dict(type="regression_and_recovery", anchor_frame=10, loss_frame=14, end_frame=30)]))
    rows.append(dict(common, uid="drop", role="event", C=None, frame_directory="d",
                     events=[dict(type="unrecovered_regression", anchor_frame=20, end_frame=end)]))
    return rows


def test_all_four_event_types_become_libero_relations():
    records = four_event_records()
    plan = build_plan(records, trajectory_config(trajectories={"sampling": {"relations": {"groups": 1}}},
                                                 training={"steps": 8}))
    seen = {}
    for batch in plan:
        relation = batch["relations"][0]
        clips = batch["clips"]
        frames = [clips[i]["frame"] for i in relation["member_indices"]]
        uid = clips[relation["anchor_index"]]["uid"]
        seen[uid] = relation["kind"]
        assert len(clips) == 10
        if relation["kind"] == 6:       # reference, recovery frame, three members after the loss
            assert relation["margin"] == 0.1 and clips[relation["secondary_anchor_index"]]["frame"] == 30
            assert len(frames) == 3 and all(14 <= f < 30 for f in frames)
        else:
            assert relation["secondary_anchor_index"] == -1 and len(frames) == 4
        if relation["kind"] == 4:       # no loss frame: members from 5 frames after the anchor
            assert relation["margin"] == 0.1 and all(25 <= f <= 40 for f in frames)
        if relation["kind"] == 5:
            assert relation["margin"] == 0.02 and all(10 < f < 20 for f in frames)
    assert seen == {"stall": 5, "failure": 3, "regression": 6, "drop": 4}


def test_event_types_take_equal_turns():
    rows = trajectory_records()
    # Four progress stall trajectories, one no further progress.
    rows += [dict(rows[3], uid=f"stall{i}", frame_directory=f"h{i}") for i in range(3)]
    for balance, plateau_share in [("type", 0.5), ("trajectory", 0.2)]:
        plan = build_plan(rows, trajectory_config(trajectories={"sampling": {"relations": {"balance": balance}}},
                                                  training={"steps": 20, "schedule_steps": 20}))
        kinds = [b["relations"][0]["kind"] for b in plan]
        assert kinds.count(3) / len(kinds) == plateau_share


def test_missing_events_fall_back_to_expert_frames(tmp_path):
    experts = [r for r in trajectory_records() if r["role"] == "expert"]
    plan = build_plan(experts, trajectory_config())
    for batch in plan:
        assert not batch["relations"] and len(batch["clips"]) == 10
        assert {q["target_source"] for q in batch["clips"]} == {"expert", "completion"}
    # Only a failure: stall_success queries use expert frames too.
    failure_only = experts + [r for r in trajectory_records() if r["uid"] == "failure"]
    for batch in build_plan(failure_only, trajectory_config()):
        sources = [q["target_source"] for q in batch["clips"]]
        assert "stall_success" not in sources and sources.count("relation") == 5


def test_short_events_are_left_out_and_reported(trajectory_data):
    from event2reward.train.pipelines import TrajectoryPipeline
    rows = trajectory_records()
    rows[3]["events"] = [dict(type="progress_stall", anchor_frame=10, end_frame=13)]   # two frames inside
    for r in rows:
        r["frame_directory"] = str(trajectory_data.parent / r["frame_directory"])
    manifest = trajectory_data.parent / "short.json"
    manifest.write_text(json.dumps({"schema": "event2reward.trajectories.v1", "trajectories": rows}))
    summary = TrajectoryPipeline(trajectory_config(manifest)).summary()
    assert summary["events"] == {"push": {"no_further_progress": 1}}
    assert summary["skipped_events"][0]["uid"] == "stall" and summary["notes"] == []


def test_only_stall_successes_get_extra_absolute_targets():
    from event2reward.train.sampling import stall_success
    rows = four_event_records()
    assert [r["uid"] for r in rows if stall_success(r)] == ["stall"]
    plan = build_plan(rows, trajectory_config(training={"steps": 30, "schedule_steps": 30}))
    drawn = {q["uid"] for b in plan for q in b["clips"] if q["target_source"] == "stall_success"}
    assert drawn == {"stall"}   # the successful regression only forms event groups


def test_mixed_groups_pad_member_indices():
    import torch
    from event2reward.train.sampling import supervision
    rows = four_event_records()
    c = trajectory_config(trajectories={"sampling": {"progress": {"expert": 0, "stall_success": 0, "completion": 0},
                                                     "relations": {"groups": 2, "members": 4}}},
                          training={"steps": 4})
    for batch in build_plan(rows, c):
        result = supervision(batch, "cpu")["relation"]
        widths = result["member_valid"].sum(dim=1).tolist()
        assert result["member_indices"].shape == (2, 4)
        kinds = result["relation_kind"].tolist()
        assert widths == [3 if k == 6 else 4 for k in kinds]
        assert torch.equal(result["secondary_anchor_index"].ge(0), result["relation_kind"].eq(6))

