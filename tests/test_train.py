"""The trainer: parameter selection, optimization, checkpoints, resume and export.

Each training run uses a fresh process because accelerate keeps global state.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import torch
from torch import nn

from event2reward.train.config import DEFAULTS, _merge, validate_config
from event2reward.train.optimizer import MasterAdamW, lr_multiplier


def run_python(code, tmp_path, cwd=None):
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(code))
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=cwd,
                            env={"ACCELERATE_USE_CPU": "true", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
                                 "PYTHONPATH": ":".join(sys.path)})
    assert result.returncode == 0, result.stderr[-3000:]
    return result.stdout


def trajectory_yaml(tmp_path, bundle, manifest, mode, output, **training):
    import yaml
    config = {
        "model_path": str(bundle), "output_dir": str(output),
        "data": {"format": "trajectories", "manifest": str(manifest)},
        "tuning": {"mode": mode, "lora": {"layers": [2, 3], "rank": 2, "alpha": 4},
                   "full": {"vision_layers": 1}, "heads": {"progress": True, "trend": False, "fp32": True}},
        "optimizer": {"lr": {"lora": 1e-3, "language": 1e-4, "vision": 1e-4, "heads": 1e-3}},
        "loss": {"trend": 0.0, "completion_weight": 2.0, "terminal_weight": 2.0},
        "training": {"steps": 3, "schedule_steps": 6, "warmup_steps": 1, "accumulation": 1,
                     "precision": "fp32", "gradient_checkpointing": False, "save_every": 1,
                     "keep_optimizer_states": 2, "num_workers": 0, "log_every": 1, **training},
    }
    path = tmp_path / f"{output.name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_train_resume_load_and_export(tmp_path, tiny_qwen_bundle, trajectory_data, mode):
    full = trajectory_yaml(tmp_path, tiny_qwen_bundle, trajectory_data, mode, tmp_path / "full")
    resumed = trajectory_yaml(tmp_path, tiny_qwen_bundle, trajectory_data, mode, tmp_path / "resumed",
                              resume_from=str(tmp_path / "full/checkpoint-000002"))
    for path in (full, resumed):
        run_python(f"""
            from event2reward.train import load_config, train
            train(load_config({str(path)!r}))
        """, tmp_path)
    from safetensors.torch import load_file
    a = load_file(str(tmp_path / "full/checkpoint-000003/weights.safetensors"))
    b = load_file(str(tmp_path / "resumed/checkpoint-000003/weights.safetensors"))
    assert a.keys() == b.keys()
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    if mode == "lora":
        assert all(n.startswith("progress_head.") or n.endswith(("lora_A", "lora_B")) for n in a)
    else:
        assert any(".visual.blocks.1." in n for n in a) and not any(".visual.blocks.0." in n for n in a)
    kept = sorted(p.parent.name for p in (tmp_path / "full").glob("checkpoint-*/optimizer.pt"))
    assert kept == ["checkpoint-000002", "checkpoint-000003"]
    metrics = [json.loads(l) for l in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [1, 2, 3] and all(m["loss"] > 0 for m in metrics)

    out = run_python(f"""
        import numpy as np, json
        from event2reward import Event2Reward
        from event2reward.train.checkpoint import export_bundle, load_checkpoint
        checkpoint = {str(tmp_path / "full/checkpoint-000003")!r}
        frames = [np.zeros((32, 32, 3), np.uint8)] * 3
        times = [0.0, 0.5, 1.0]
        direct = load_checkpoint(checkpoint, device="cpu").predict_trajectory(
            frames=frames, task="push", goal=frames[0], timestamps=times)
        path = export_bundle(checkpoint, {str(tmp_path / "exported")!r})
        loaded = Event2Reward.from_pretrained(path, device="cpu")
        loaded.config["inference_precision"] = "fp32"
        exported = loaded.predict_trajectory(frames=frames, task="push", goal=frames[0], timestamps=times)
        print(json.dumps([[r["progress"] for r in direct], [r["progress"] for r in exported],
                          loaded.config["history_offsets_seconds"]]))
    """, tmp_path)
    direct, exported, offsets = json.loads(out.strip().splitlines()[-1])
    assert str(tmp_path) not in (tmp_path / "exported/state.json").read_text()
    assert offsets == [4, 2, 1, 0.25, 0]
    assert direct == pytest.approx(exported, abs=2e-3)   # export stores BF16 weights


def test_run_records_hold_no_machine_paths(tmp_path, tiny_qwen_bundle, trajectory_data):
    import yaml
    (tmp_path / "models").mkdir()
    (tmp_path / "models/Stage2").symlink_to(tiny_qwen_bundle)
    config = yaml.safe_load(trajectory_yaml(tmp_path, tiny_qwen_bundle, trajectory_data, "lora",
                                            tmp_path / "rel", steps=1).read_text())
    config.update(model_path="models/Stage2", output_dir="outputs/run")
    config["data"]["manifest"] = "dataset.json"
    (tmp_path / "rel.yaml").write_text(yaml.safe_dump(config))
    run_python("""
        from event2reward.train import load_config, train
        from event2reward.train.checkpoint import load_checkpoint
        train(load_config("rel.yaml"))
        load_checkpoint("outputs/run/checkpoint-000001", device="cpu")
    """, tmp_path, cwd=tmp_path)
    for name in ("outputs/run/config.yaml", "outputs/run/checkpoint-000001/state.json"):
        assert str(tmp_path) not in (tmp_path / name).read_text()


def test_resume_rejects_a_changed_run(tmp_path, tiny_qwen_bundle, trajectory_data):
    first = trajectory_yaml(tmp_path, tiny_qwen_bundle, trajectory_data, "lora", tmp_path / "first", steps=1)
    run_python(f"""
        from event2reward.train import load_config, train
        train(load_config({str(first)!r}))
    """, tmp_path)
    changed = trajectory_yaml(tmp_path, tiny_qwen_bundle, trajectory_data, "lora", tmp_path / "changed",
                              resume_from=str(tmp_path / "first/checkpoint-000001"), seed=7)
    from event2reward.train import load_config
    from event2reward.train.checkpoint import check_resume
    with pytest.raises(ValueError, match="training.seed"):
        check_resume(tmp_path / "first/checkpoint-000001", load_config(changed))


def test_index_training_runs_on_synthetic_data(tmp_path, tiny_qwen_bundle, synthetic_index):
    import yaml
    from test_data import INDEX_SOURCES
    config = {"model_path": str(tiny_qwen_bundle), "output_dir": str(tmp_path / "run"),
              "data": {"format": "index", "manifest": str(synthetic_index),
                       "index": {"sources": INDEX_SOURCES, "require_goal_present": True}},
              "tuning": {"full": {"vision_layers": 1}},
              "training": {"steps": 2, "schedule_steps": 2, "accumulation": 2, "precision": "fp32",
                           "gradient_checkpointing": False, "save_every": 2, "num_workers": 0, "log_every": 1}}
    path = tmp_path / "index.yaml"
    path.write_text(yaml.safe_dump(config))
    run_python(f"""
        from event2reward.train import load_config, train
        train(load_config({str(path)!r}))
    """, tmp_path)
    assert (tmp_path / "run/checkpoint-000002/weights.safetensors").is_file()


@pytest.mark.skipif(sys.platform == "darwin", reason="the macOS MPS backend has no distributed barrier")
@pytest.mark.skipif(torch.cuda.device_count() == 1, reason="NCCL needs one GPU per process")
def test_two_process_data_parallel(tmp_path, tiny_qwen_bundle, synthetic_index):
    import yaml
    from test_data import INDEX_SOURCES
    config = {"model_path": str(tiny_qwen_bundle), "output_dir": str(tmp_path / "run"),
              "data": {"format": "index", "manifest": str(synthetic_index), "index": {"sources": INDEX_SOURCES}},
              "tuning": {"full": {"vision_layers": 1}},
              "training": {"steps": 2, "schedule_steps": 2, "precision": "fp32", "gradient_checkpointing": False,
                           "save_every": 1, "num_workers": 0, "log_every": 1}}
    path = tmp_path / "ddp.yaml"
    path.write_text(yaml.safe_dump(config))
    # torchrun with ACCELERATE_USE_CPU gives real DistributedDataParallel on CPU.
    import os
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "CUDA_HOME", "CUDA_VISIBLE_DEVICES")}
    if not torch.cuda.is_available():
        env["ACCELERATE_USE_CPU"] = "true"
    import socket
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = str(probe.getsockname()[1])
    result = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", port,
                             "-m", "event2reward", "train", str(path)], capture_output=True, text=True,
                            env={**env, "HOME": str(tmp_path), "PYTHONPATH": ":".join(sys.path)}, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert (tmp_path / "run/checkpoint-000002/weights.safetensors").is_file()


def test_parameter_selection():
    from event2reward.train.parameters import configure_parameters

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.visual = nn.Module()
            self.backbone.visual.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.backbone.visual.merger = nn.Linear(4, 4)
            self.backbone.language_model = nn.Module()
            layer = nn.Module(); layer.self_attn = nn.Module(); layer.self_attn.q_proj = nn.Linear(4, 4)
            self.backbone.language_model.layers = nn.ModuleList([layer])
            self.progress_head, self.trend_head = nn.Linear(4, 101), nn.Linear(4, 3)

    base = _merge(DEFAULTS, {"model_path": "m", "output_dir": "o",
                             "data": {"format": "index", "manifest": "d", "index": {"sources": [
                                 {"family": "a", "type": "relation", "role": "fill", "fraction": 1.0}]}},
                             "training": {"steps": 1, "schedule_steps": 1, "save_every": 1}})
    base["tuning"]["full"]["vision_layers"] = 1
    model = Tiny()
    groups, summary = configure_parameters(model, base)
    assert not model.backbone.visual.blocks[0].weight.requires_grad
    assert model.backbone.visual.blocks[1].weight.requires_grad and model.backbone.visual.merger.weight.requires_grad
    assert set(summary) == {"language", "vision", "heads"}
    base["tuning"].update(mode="lora", lora={"layers": [0], "rank": 2, "alpha": 4, "dropout": 0.0})
    base["tuning"]["heads"]["trend"] = False
    base["loss"]["trend"] = 0.0
    validate_config(base)
    model = Tiny()
    groups, summary = configure_parameters(model, base)
    assert set(summary) == {"lora", "heads"}
    assert not model.backbone.language_model.layers[0].self_attn.q_proj.base.weight.requires_grad


def test_learning_rate_schedule():
    assert lr_multiplier(0, 100, 10) == pytest.approx(0.1)
    assert lr_multiplier(10, 100, 10) == pytest.approx(1.0)
    assert lr_multiplier(100, 100, 10) == pytest.approx(0.0)
    # Stopping at step 300 of a 900-step schedule keeps a high learning rate.
    assert lr_multiplier(299, 900, 15) > 0.7


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_master_weights_keep_small_updates(dtype):
    p, reference = nn.Parameter(torch.ones(3, dtype=dtype)), nn.Parameter(torch.ones(3))
    optimizer = MasterAdamW([{"params": [p], "lr": 2e-6}], weight_decay=0.01)
    reference_optimizer = torch.optim.AdamW([{"params": [reference], "lr": 2e-6}], weight_decay=0.01)
    for _ in range(1000):
        for parameter, opt in ((p, optimizer), (reference, reference_optimizer)):
            parameter.grad = torch.ones_like(parameter)
            opt.step()
            opt.zero_grad(set_to_none=True)
    torch.testing.assert_close(p, reference.to(dtype), rtol=0, atol=0)
    assert not torch.equal(p, torch.ones_like(p))
    if dtype == torch.bfloat16:
        restored = MasterAdamW([{"params": [nn.Parameter(p.detach().clone())], "lr": 2e-6}], weight_decay=0.01)
        restored.load_state_dict(optimizer.state_dict())
        torch.testing.assert_close(restored.master_pairs[0][1], optimizer.master_pairs[0][1], rtol=0, atol=0)
