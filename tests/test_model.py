"""Bundles, loading and inference."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from event2reward import Event2Reward, bundle
from event2reward.inference import (Observation, canonical_image, causal_history, history_indices,
                                    prepare_batch)

SAVE = SimpleNamespace(save_pretrained=lambda _: None)


def test_manifest_checks_every_model_file(tiny_bundle):
    for asset in ("state.json", "backbone/tokenizer.json", "model/model.safetensors"):
        root, _ = tiny_bundle(asset.replace("/", "_"))
        bundle.read_manifest(root)
        (root / asset).write_bytes(b"changed")
        with pytest.raises(ValueError):
            bundle.read_manifest(root)


def test_only_the_current_bundle_format_loads(tiny_bundle):
    root, _ = tiny_bundle()
    manifest = json.loads((root / "event2reward.json").read_text())
    manifest["format"] = "event2reward.bundle.v1"
    (root / "event2reward.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsupported bundle format"):
        bundle.read_manifest(root)


@pytest.mark.parametrize("escape", ["../outside.json", "/outside.json", "backbone/../outside.json"])
def test_manifest_paths_stay_inside_the_bundle(tiny_bundle, escape):
    root, _ = tiny_bundle()
    manifest = json.loads((root / "event2reward.json").read_text())
    manifest["files"][escape] = {"bytes": 1, "sha256": "0"}
    (root / "event2reward.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        bundle.read_manifest(root)


def test_model_card_edits_keep_a_bundle_valid(tiny_bundle):
    root, model = tiny_bundle()
    for name in ("README.md", "NOTICE"):
        (root / name).write_text(name)
    manifest = json.loads((root / "event2reward.json").read_text())
    manifest["files"] = bundle.file_records(root)
    (root / "event2reward.json").write_text(json.dumps(manifest))
    (root / "README.md").write_text("an updated model card\n")
    (root / "NOTICE").unlink()
    bundle.read_manifest(root)
    saved = bundle.save_instance(model, SAVE, root, root.parent / "saved")
    assert (saved / "README.md").read_text() == "an updated model card\n"
    assert "NOTICE" not in bundle.read_manifest(saved)["files"]


def test_save_writes_current_parameters_and_manifest_updates(tiny_bundle):
    root, model = tiny_bundle()
    original = bundle.sha256(root / "model/model.safetensors")
    with torch.no_grad():
        model.weight.fill_(7)
    saved = bundle.save_instance(model, SAVE, root, root.parent / "saved", updates={"saved_step": 3})
    from safetensors.torch import load_file
    assert torch.equal(load_file(str(saved / "model/model.safetensors"))["weight"], model.weight)
    assert bundle.sha256(root / "model/model.safetensors") == original
    assert bundle.read_manifest(saved)["saved_step"] == 3
    with pytest.raises(FileExistsError):
        bundle.save_instance(model, SAVE, root, saved)
    with pytest.raises(ValueError, match="inside"):
        bundle.save_instance(model, SAVE, root, root / "nested")


def test_hub_snapshot_symlinks_load_and_save(tmp_path, tiny_bundle, monkeypatch):
    source, model = tiny_bundle()
    snapshot = tmp_path / "models--owner--model/snapshots/revision"
    blobs = tmp_path / "models--owner--model/blobs"
    blobs.mkdir(parents=True)
    for asset in source.rglob("*"):
        if asset.is_file():
            blob = blobs / bundle.sha256(asset)
            blob.write_bytes(asset.read_bytes())
            link = snapshot / asset.relative_to(source)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(blob)
    monkeypatch.setattr(bundle, "load_model", lambda path, **kw: (SimpleNamespace(model=model, processor=SAVE),
                                                                  bundle.read_manifest(path)))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: str(snapshot))
    loaded = Event2Reward.from_pretrained("owner/model", device="cpu")
    loaded.save_pretrained(tmp_path / "saved")
    assert not any(p.is_symlink() for p in (tmp_path / "saved").rglob("*"))


def test_shared_hub_blob_cache_is_accepted(tmp_path, tiny_bundle):
    """huggingface_hub >= 1.x may link repository blobs into a cache-wide hub/blobs/."""
    source, _ = tiny_bundle()
    repo = tmp_path / "hub/models--owner--model"
    snapshot, blobs, shared = repo / "snapshots/revision", repo / "blobs", tmp_path / "hub/blobs/ab"
    blobs.mkdir(parents=True)
    shared.mkdir(parents=True)
    for asset in source.rglob("*"):
        if asset.is_file():
            digest = bundle.sha256(asset)
            if not (blobs / digest).exists():   # identical files share one blob
                (shared / digest).write_bytes(asset.read_bytes())
                (blobs / digest).symlink_to(shared / digest)
            link = snapshot / asset.relative_to(source)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(blobs / digest)
    bundle.read_manifest(snapshot)
    (snapshot / "state.json").unlink()
    (snapshot / "state.json").symlink_to(tmp_path / "elsewhere.json")
    (tmp_path / "elsewhere.json").write_text("{}")
    with pytest.raises(ValueError, match="escapes"):
        bundle.read_manifest(snapshot)


def test_local_directories_never_download(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda **kw: pytest.fail("network used"))
    monkeypatch.setattr(bundle, "load_model", lambda path, **kw: (
        SimpleNamespace(model=torch.nn.Linear(1, 1), processor=None), {}))
    assert Event2Reward.from_pretrained(tmp_path, device="cpu").bundle_path == tmp_path


def test_download_failure_explains_both_possibilities(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise PermissionError("access denied")
    monkeypatch.setattr("huggingface_hub.snapshot_download", fail)
    with pytest.raises(OSError, match="does not exist.*permission"):
        Event2Reward.from_pretrained(tmp_path / "missing", device="cpu")


def test_histories_are_causal():
    assert history_indices(0) == [0] * 5
    assert history_indices(3) == [0, 0, 1, 2, 3]
    assert history_indices(12, start=10, stride=2) == [10, 10, 10, 10, 12]
    times = [0, 0.1, 0.25, 0.4, 1.0, 1.1]
    assert causal_history(times, 5, [1, 0.5, 0.2, 0.1, 0]) == [1, 3, 3, 4, 5]
    for args in [(-1, 0, 1), (2, 3, 1), (4, 0, 0)]:
        with pytest.raises(ValueError):
            history_indices(args[0], start=args[1], stride=args[2])


@pytest.mark.parametrize("shape", [(31, 57, 3), (3, 31, 57), (9, 12, 1), (61, 29, 4)])
def test_images_of_any_layout_become_square_rgb(shape):
    image = canonical_image(np.random.default_rng(7).integers(0, 256, size=shape, dtype=np.uint8))
    assert image.mode == "RGB" and image.size == (224, 224)
    floating = canonical_image(np.random.default_rng(7).integers(0, 256, size=shape).astype(np.float32) / 255)
    assert floating.size == (224, 224)


def test_invalid_observations_are_rejected():
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        Observation("task", image, None, [image] * 5)
    with pytest.raises(ValueError):
        Observation("task", image, image, [image] * 4)
    with pytest.raises(ValueError):
        canonical_image(np.full((8, 8, 3), np.nan))
    with pytest.raises(ValueError):
        prepare_batch(None, [])


def fake_engine(config=None):
    engine = object.__new__(Event2Reward)
    engine.config = config or {}
    engine.predict = lambda rows: [{"progress": float(r.history[-1][0, 0, 0])} for r in rows]
    return engine


def test_trajectory_scoring_is_independent_of_batching():
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(9)]
    engine = fake_engine()
    a = engine.predict_trajectory(frames=frames, task="task", goal=frames[-1], batch_size=1)
    b = engine.predict_trajectory(frames=frames, task="task", goal=frames[-1], batch_size=4)
    assert a == b
    with pytest.raises(ValueError):
        engine.predict_trajectory(frames=frames, task="task", goal=frames[-1], timestamps=[0] * 9)


def test_time_based_models_use_their_offsets():
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(9)]
    engine = fake_engine({"history_offsets_seconds": [0.4, 0.2, 0.1, 0.05, 0]})
    with pytest.raises(ValueError, match="timestamps"):
        engine.predict_trajectory(frames=frames, task="task", goal=frames[-1])
    rows = engine.predict_trajectory(frames=frames, task="task", goal=frames[-1],
                                     timestamps=[i / 10 for i in range(9)])
    assert rows[8]["history_indices"] == [4, 6, 7, 7, 8]


def test_real_processor_batch_and_forward(tiny_qwen):
    model, processor = tiny_qwen()
    engine = Event2Reward(SimpleNamespace(model=model, processor=processor), device="cpu")
    image = np.zeros((32, 32, 3), np.uint8)
    rows = engine.predict([Observation("push the block", image, image, [image] * 5)] * 2)
    assert len(rows) == 2 and all(0 <= r["progress"] <= 1 for r in rows)
    assert rows[0]["trend"] in {"growth", "stable", "regression"}
