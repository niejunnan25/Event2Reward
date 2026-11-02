"""The shipped configurations parse, and the paper configurations match the runs."""
import json
from pathlib import Path

import pytest

from event2reward.train import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").rglob("*.yaml"))
CHECKED = [*CONFIGS, ROOT / "examples/annotation/finetune.yaml"]
PAPER = ROOT / "tests/fixtures/paper"   # configurations recorded by the original training runs


def test_configs_exist():
    names = {p.relative_to(ROOT / "configs").as_posix() for p in CONFIGS}
    assert names == {*(f"finetune/libero/{s}.yaml" for s in ("spatial", "object", "goal", "libero_10")),
                     *(f"finetune/real_world_franka/{t}_{m}.yaml"
                       for t in ("plug", "ram", "ethernet", "bowls", "buttons", "vegetables") for m in ("full", "lora"))}


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_config_has_the_same_complete_skeleton(path):
    """Every config writes every field, in the order of DEFAULTS, so any two configs diff by values only."""
    import yaml
    from event2reward.train.config import DEFAULTS

    def paths(node, prefix=""):
        for key, value in node.items():
            yield prefix + key
            if isinstance(value, dict):
                yield from paths(value, prefix + key + ".")

    assert list(paths(yaml.safe_load(path.read_text()))) == list(paths(DEFAULTS))


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_config_is_valid(path):
    config = load_config(path, base_dir=ROOT)
    assert config["model_path"].startswith(str(ROOT))


def common(config):
    t, opt, full = config["training"], config["optimizer"], config["tuning"]["full"]
    return {"steps": t["steps"], "warmup_steps": t["warmup_steps"], "seed": t["seed"],
            "per_device_batch_size": t["batch_size"], "gradient_accumulation_steps": t["accumulation"],
            "language_learning_rate": opt["lr"]["language"], "vision_learning_rate": opt["lr"]["vision"],
            "head_learning_rate": opt["lr"]["heads"], "vision_train_last_n_layers": full["vision_layers"],
            "train_vision_merger": full["vision_merger"], "lambda_progress": config["loss"]["progress"],
            "lambda_trend": config["loss"]["trend"], "lambda_relation": config["loss"]["relation"],
            "goal_dropout": config["data"]["index"]["goal_dropout"], "event_progress_objective": "relation_only",
            "image_size": config["data"]["image_size"], "max_pixels": config["data"]["max_pixels"]}


def recorded(run):
    r = dict(run)
    r["steps"] = r.get("stop_after_step") or r["max_steps"]
    keys = {"steps", "warmup_steps", "seed", "per_device_batch_size", "gradient_accumulation_steps",
            "language_learning_rate", "vision_learning_rate", "head_learning_rate", "vision_train_last_n_layers",
            "train_vision_merger", "lambda_progress", "lambda_trend", "lambda_relation", "goal_dropout",
            "event_progress_objective", "image_size", "max_pixels"}
    return {k: r[k] for k in keys}


@pytest.mark.parametrize("suite", ["spatial", "object", "goal", "10"])
def test_libero_configs_match_the_paper_runs(suite):
    config = load_config(ROOT / f"configs/finetune/libero/{'libero_10' if suite == '10' else suite}.yaml", base_dir=ROOT)
    run = json.loads((PAPER / f"libero_{suite}.json").read_text())
    assert common(config) == recorded(run)
    assert config["training"]["schedule_steps"] == run["max_steps"]
    assert config["data"]["index"]["require_goal_present"] == run["require_goal_present"]
    assert {s["family"]: s["fraction"] for s in config["data"]["index"]["sources"]} == \
           {s["family"]: s["clip_fraction"] for s in run["train_sources"]}
    fill = [s["family"] for s in config["data"]["index"]["sources"] if s["role"] == "fill"]
    assert fill == [s["family"] for s in run["train_sources"] if s["role"] == "expert_fill"]


def test_overrides_and_unknown_keys(tmp_path):
    path = ROOT / "configs/finetune/real_world_franka/buttons_lora.yaml"
    config = load_config(path, ["training.steps=10", "model_path=/data/Stage2"], base_dir=ROOT)
    assert config["training"]["steps"] == 10 and config["model_path"] == "/data/Stage2"
    with pytest.raises(ValueError, match="unknown configuration key: training.step"):
        load_config(path, ["training.step=10"], base_dir=ROOT)


def test_franka_full_configs_follow_the_libero_adaptation():
    libero = load_config(ROOT / "configs/finetune/libero/goal.yaml", base_dir=ROOT)
    for task in ("plug", "ram", "ethernet", "bowls", "buttons", "vegetables"):
        full = load_config(ROOT / f"configs/finetune/real_world_franka/{task}_full.yaml", base_dir=ROOT)
        for section in ("tuning", "optimizer", "loss"):
            assert full[section] == libero[section], section
        for key in ("steps", "schedule_steps", "warmup_steps", "batch_size", "seed", "precision"):
            assert full["training"][key] == libero["training"][key]
        # LIBERO: 2 GPUs x 4 microbatches; Franka trajectories: 1 GPU x 8.
        assert full["training"]["accumulation"] == 2 * libero["training"]["accumulation"]
