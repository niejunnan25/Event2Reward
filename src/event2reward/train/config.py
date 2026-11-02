"""One YAML schema for every fine-tuning recipe.

Relative paths are resolved against the directory you run the command from
(normally the repository root). Unknown keys are errors, so a typo cannot
silently fall back to a default.
"""
from copy import deepcopy
import math
import os
from pathlib import Path

import yaml

DEFAULTS = {
    "model_path": None,
    "output_dir": None,
    "data": {
        "format": None,               # index | trajectories; selects the block below
        "manifest": None,
        "image_size": 224,
        "max_pixels": 50176,
        "index": {
            "goal_dropout": 0.0,
            "require_goal_present": False,
            "sources": [],            # sampling families, see SOURCE_KEYS
        },
        "trajectories": {
            "task": None,             # restrict to one task name
            "history_offsets": [4, 2, 1, 0.25, 0],
            "training_history_scales": [1.0],
            "sampling": {
                "progress": {"expert": 3, "stall_success": 1, "completion": 1},   # absolute targets
                "relations": {"groups": 1, "members": 4, "balance": "type"},           # event groups
            },
        },
    },
    "tuning": {
        "mode": "full",               # full | lora; selects the block below
        "full": {"language": True, "vision_layers": 3, "vision_merger": True},
        "lora": {"layers": list(range(24, 32)), "rank": 8, "alpha": 16, "dropout": 0.05},
        "heads": {"progress": True, "trend": True, "fp32": False},
    },
    "optimizer": {
        "lr": {"language": 2e-6, "vision": 5e-7, "lora": 1e-4, "heads": 1e-5},
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "fp32_master_weights": False,
        "fused": False,
    },
    "loss": {"progress": 1.0, "trend": 1.0, "relation": 10.0,
             "completion_weight": 1.0, "terminal_weight": 1.0},
    "training": {
        "steps": None,
        "schedule_steps": None,
        "warmup_steps": 0,
        "batch_size": 10,
        "accumulation": 1,
        "seed": 42,
        "precision": "bf16",
        "attention": "sdpa",
        "gradient_checkpointing": True,
        "save_every": None,
        "save_steps": None,
        "keep_optimizer_states": 1,
        "log_every": 10,
        "num_workers": 2,
        "prefetch_factor": 2,
        "resume_from": None,
    },
}

SOURCE_TYPES = {"relation"}
SOURCE_KEYS = {"family", "type", "role", "fraction"}
PATH_KEYS = (("model_path",), ("output_dir",), ("data", "manifest"), ("training", "resume_from"))


def _merge(base, override, prefix=""):
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"unknown configuration key: {prefix}{key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            result[key] = _merge(result[key], value, f"{prefix}{key}.")
        else:
            result[key] = deepcopy(value)
    return result


def apply_overrides(document, overrides):
    """`--set training.steps=10` style overrides; values are parsed as YAML."""
    for item in overrides or ():
        key, separator, raw = item.partition("=")
        if not separator or not key:
            raise ValueError(f"override must look like section.key=value: {item}")
        *parents, leaf = key.split(".")
        node = document
        for name in parents:
            node = node.setdefault(name, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override inside a non-mapping: {key}")
        node[leaf] = yaml.safe_load(raw)
    return document


def _map_paths(config, convert):
    config = deepcopy(config)
    for keys in PATH_KEYS:
        node = config
        for name in keys[:-1]:
            node = node[name]
        if node[keys[-1]] is not None:
            node[keys[-1]] = convert(node[keys[-1]])
    return config


def portable_paths(config, base_dir=None):
    """Paths under the working directory become relative, so run records hold no machine paths."""
    base = os.path.abspath(base_dir or os.getcwd())
    def convert(path):
        relative = os.path.relpath(path, base)
        return path if relative == ".." or relative.startswith(".." + os.sep) else relative
    return _map_paths(config, convert)


def absolute_paths(config, base_dir=None):
    """Inverse of portable_paths: relative paths are taken from the working directory."""
    base = base_dir or os.getcwd()
    return _map_paths(config, lambda path: os.path.abspath(os.path.join(base, path)))


def load_config(path, overrides=(), *, base_dir=None):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = _merge(DEFAULTS, apply_overrides(raw, overrides))
    base = Path.cwd() if base_dir is None else Path(base_dir)
    for keys in PATH_KEYS:
        node = config
        for name in keys[:-1]:
            node = node[name]
        if node[keys[-1]] is not None:
            # Absolute, but symlinks kept as written (e.g. models/ linked to a data disk).
            node[keys[-1]] = os.path.abspath(base / Path(node[keys[-1]]).expanduser())
    validate_config(config)
    return config


def _positive_int(value, name, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
            or value < 0 or positive and value == 0:
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")


def validate_config(c):
    for key in ("model_path", "output_dir"):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError(f"{key} is required")
    d, tu, t = c["data"], c["tuning"], c["training"]
    if d["format"] not in {"index", "trajectories"}:
        raise ValueError("data.format must be index or trajectories")
    if not isinstance(d["manifest"], str):
        raise ValueError("data.manifest is required")
    _positive_int(d["image_size"], "data.image_size")
    _positive_int(d["max_pixels"], "data.max_pixels")
    # Only the block that data.format selects is checked and used.
    if d["format"] == "index":
        index = d["index"]
        _validate_sources(index["sources"])
        _number(index["goal_dropout"], "data.index.goal_dropout")
        if index["goal_dropout"] > 1:
            raise ValueError("data.index.goal_dropout must lie in [0, 1]")
        if type(index["require_goal_present"]) is not bool:
            raise ValueError("data.index.require_goal_present must be true or false")
        if index["require_goal_present"] and index["goal_dropout"]:
            raise ValueError("data.index.require_goal_present needs goal_dropout 0")
        for name in ("completion_weight", "terminal_weight"):
            if c["loss"][name] != 1:
                raise ValueError(f"loss.{name} applies to trajectories data; set it to 1.0")
    else:
        _validate_trajectory_inputs(c)
    if tu["mode"] not in {"full", "lora"}:
        raise ValueError("tuning.mode must be full or lora")
    full = tu["full"]
    if full["vision_layers"] != "all":
        _positive_int(full["vision_layers"], "tuning.full.vision_layers", zero=True)
    if full["vision_layers"] == 0 and full["vision_merger"]:
        raise ValueError("a frozen vision encoder cannot train its merger")
    lora = tu["lora"]
    _positive_int(lora["rank"], "tuning.lora.rank")
    _number(lora["alpha"], "tuning.lora.alpha", positive=True)
    _number(lora["dropout"], "tuning.lora.dropout")
    if lora["dropout"] >= 1:
        raise ValueError("tuning.lora.dropout must be below 1")
    if not lora["layers"] or len(set(lora["layers"])) != len(lora["layers"]):
        raise ValueError("tuning.lora.layers must be distinct layer indices")
    for i in lora["layers"]:
        _positive_int(i, "tuning.lora.layers", zero=True)
    for section, names in ((full, ("language", "vision_merger")), (tu["heads"], ("progress", "trend", "fp32")),
                           (c["optimizer"], ("fp32_master_weights", "fused")),
                           (t, ("gradient_checkpointing",))):
        for name in names:
            if type(section[name]) is not bool:
                raise ValueError(f"{name} must be true or false")
    if tu["heads"]["trend"] != (c["loss"]["trend"] > 0):
        raise ValueError("tuning.heads.trend must be true exactly when loss.trend > 0")
    opt = c["optimizer"]
    if set(opt["lr"]) - {"language", "vision", "heads", "lora"}:
        raise ValueError("optimizer.lr keys are language, vision, heads and lora")
    for name, value in opt["lr"].items():
        _number(value, f"optimizer.lr.{name}", positive=True)
    _number(opt["weight_decay"], "optimizer.weight_decay")
    _number(opt["grad_clip"], "optimizer.grad_clip", positive=True)
    for name, value in c["loss"].items():
        _number(value, f"loss.{name}")
    if not any(c["loss"][k] > 0 for k in ("progress", "trend", "relation")):
        raise ValueError("enable at least one of loss.progress, loss.trend, loss.relation")
    for name in ("steps", "schedule_steps", "batch_size", "accumulation", "log_every", "prefetch_factor"):
        _positive_int(t[name], f"training.{name}")
    for name in ("warmup_steps", "seed", "keep_optimizer_states", "num_workers"):
        _positive_int(t[name], f"training.{name}", zero=True)
    if t["steps"] > t["schedule_steps"] or t["warmup_steps"] >= t["schedule_steps"]:
        raise ValueError("require steps <= schedule_steps and warmup_steps < schedule_steps")
    if t["precision"] not in {"bf16", "fp32"}:
        raise ValueError("training.precision must be bf16 or fp32")
    if t["attention"] not in {"sdpa", "eager"}:
        raise ValueError("training.attention must be sdpa or eager")
    if (t["save_every"] is None) == (t["save_steps"] is None):
        raise ValueError("set exactly one of training.save_every and training.save_steps")
    if t["save_every"] is not None:
        _positive_int(t["save_every"], "training.save_every")
    else:
        steps = t["save_steps"]
        if not isinstance(steps, list) or steps != sorted(set(steps)) or not steps \
                or any(type(s) is not int or not 0 < s <= t["schedule_steps"] for s in steps):
            raise ValueError("training.save_steps must be increasing steps within the schedule")
        if t["steps"] not in steps:
            raise ValueError("training.save_steps must include training.steps")


def _validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise ValueError("index data needs data.index.sources")
    families = [s.get("family") for s in sources if isinstance(s, dict)]
    if len(families) != len(sources) or len(set(families)) != len(families) \
            or any(not isinstance(f, str) or not f for f in families):
        raise ValueError("every data.index.sources entry needs a unique family name")
    for s in sources:
        if set(s) - SOURCE_KEYS:
            raise ValueError(f"unknown data.index.sources keys for {s['family']}: {sorted(set(s) - SOURCE_KEYS)}")
        if s.get("type") not in SOURCE_TYPES:
            raise ValueError(f"{s['family']}: type must be one of {sorted(SOURCE_TYPES)}")
        _number(s.get("fraction"), f"{s['family']}.fraction")
    if sum(s["fraction"] for s in sources) <= 0:
        raise ValueError("data.index.sources fractions must have a positive total")
    if sum(s.get("role") == "fill" for s in sources) != 1:
        raise ValueError("exactly one data.index.sources entry needs role: fill")


def _validate_trajectory_inputs(c):
    inp, t = c["data"]["trajectories"], c["training"]
    s = inp["sampling"]
    offsets = inp["history_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 5 or offsets[-1] != 0 \
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in offsets) \
            or any(a < b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("data.trajectories.history_offsets: five descending seconds before the query, ending at 0")
    scales = inp["training_history_scales"]
    if not isinstance(scales, list) or not scales:
        raise ValueError("data.trajectories.training_history_scales must be a nonempty list")
    for v in scales:
        _number(v, "data.trajectories.training_history_scales", positive=True)
    p, r = s["progress"], s["relations"]
    for group, values in (("progress", p), ("relations", {k: r[k] for k in ("groups", "members")})):
        for name, value in values.items():
            _positive_int(value, f"data.trajectories.sampling.{group}.{name}", zero=True)
    if r["balance"] not in {"type", "trajectory"}:
        raise ValueError("data.trajectories.sampling.relations.balance must be type or trajectory")
    budget = sum(p.values()) + r["groups"] * (1 + r["members"])
    if budget != t["batch_size"]:
        raise ValueError(f"data.trajectories.sampling uses {budget} queries per microbatch "
                         f"but training.batch_size is {t['batch_size']}")
    if r["groups"] and not r["members"]:
        raise ValueError("data.trajectories.sampling.relations.groups need members")
