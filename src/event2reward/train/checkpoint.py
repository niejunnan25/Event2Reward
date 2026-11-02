"""Training checkpoints, resuming, and conversion to inference bundles.

A checkpoint directory `checkpoint-000300/` holds:

- `weights.safetensors`: the trainable tensors only (LoRA adapters and heads, or
  the updated full-model parameters). The frozen rest comes from `model_path`.
- `state.json`: step, the complete configuration, code version. Paths under
  the directory training ran from are stored relative to it.
- `optimizer.pt`, `rng/`: optimizer, scheduler and random states for resuming.
  Only the latest `training.keep_optimizer_states` checkpoints keep them.
"""
import json
import random
import shutil
from pathlib import Path

from .config import absolute_paths, portable_paths

WEIGHTS, STATE, OPTIMIZER = "weights.safetensors", "state.json", "optimizer.pt"


def trainable_state(model):
    return {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path):
    import torch
    from safetensors.torch import load_file
    state = load_file(str(Path(path) / WEIGHTS))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        missing, extra = sorted(parameters.keys() - state.keys()), sorted(state.keys() - parameters.keys())
        raise ValueError(f"checkpoint tensors do not match the configuration: missing={missing[:3]} extra={extra[:3]}")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f"checkpoint shape differs: {name}")
            parameter.copy_(state[name])


def code_version():
    import subprocess
    from event2reward import __version__
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"package": __version__, "git_commit": commit}


def save_checkpoint(accelerator, model, optimizer, scheduler, config, step):
    import torch
    from safetensors.torch import save_file
    output = Path(config["output_dir"])
    final, temporary = output / f"checkpoint-{step:06d}", output / f".checkpoint-{step:06d}.partial"
    if final.exists():
        raise FileExistsError(final)
    if accelerator.is_main_process:
        shutil.rmtree(temporary, ignore_errors=True)
        (temporary / "rng").mkdir(parents=True)
    accelerator.wait_for_everyone()
    rng = {"python": random.getstate(), "torch": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(rng, temporary / "rng" / f"rank{accelerator.process_index}.pt")
    if accelerator.is_main_process:
        save_file(trainable_state(accelerator.unwrap_model(model)), str(temporary / WEIGHTS))
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                   temporary / OPTIMIZER)
        (temporary / STATE).write_text(json.dumps(
            {"step": step, "config": portable_paths(config), "code": code_version()}, indent=2) + "\n")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        temporary.rename(final)
        resumable = sorted(p for p in output.glob("checkpoint-*") if (p / OPTIMIZER).is_file())
        keep = config["training"]["keep_optimizer_states"]
        for old in resumable[:len(resumable) - keep] if keep else resumable:
            (old / OPTIMIZER).unlink()
            shutil.rmtree(old / "rng")
    accelerator.wait_for_everyone()
    return final


def read_state(path):
    """The checkpoint's state; its relative paths are resolved from the working directory."""
    state = json.loads((Path(path) / STATE).read_text())
    state["config"] = absolute_paths(state["config"])
    if not (Path(path) / WEIGHTS).is_file():
        raise FileNotFoundError(Path(path) / WEIGHTS)
    return state


def check_resume(path, config):
    """Only execution settings may change when resuming; the run itself may not."""
    state = read_state(path)
    if not (Path(path) / OPTIMIZER).is_file():
        raise ValueError(f"{path} has no optimizer state; it can be loaded but not resumed")
    allowed = {"output_dir", "training"}
    runtime = {"steps", "save_every", "save_steps", "keep_optimizer_states", "log_every",
               "num_workers", "prefetch_factor", "resume_from", "gradient_checkpointing"}
    before, after = state["config"], config
    changed = sorted(k for k in after if k not in allowed and before.get(k) != after[k])
    changed += sorted(f"training.{k}" for k in after["training"]
                      if k not in runtime and before["training"].get(k) != after["training"][k])
    if changed:
        raise ValueError(f"resuming with a different configuration: {changed}")
    if not 0 < state["step"] < config["training"]["steps"]:
        raise ValueError(f"checkpoint step {state['step']} is not before training.steps")
    return state["step"]


def restore_training(accelerator, optimizer, scheduler, path):
    import torch
    saved = torch.load(Path(path) / OPTIMIZER, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    rng = torch.load(Path(path) / "rng" / f"rank{accelerator.process_index}.pt", weights_only=False)
    random.setstate(rng["python"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def materialize(path, *, model_path=None, dtype=None):
    """Base model + checkpoint tensors, LoRA merged: a plain Event2Reward model."""
    import torch
    from event2reward.bundle import load_model
    from .lora import merge
    from .parameters import configure_parameters
    state = read_state(path)
    config = state["config"]
    base = model_path or config["model_path"]
    if not Path(base).is_dir():
        raise FileNotFoundError(f"base model {base} not found; run from the training directory or pass model_path")
    loaded, manifest = load_model(base, torch_dtype=dtype or torch.bfloat16)
    configure_parameters(loaded.model, config)
    load_trainable(loaded.model, path)
    merge(loaded.model)
    loaded.model.requires_grad_(False)
    for parameter in loaded.model.parameters():
        if parameter.dtype != (dtype or torch.bfloat16):
            parameter.data = parameter.data.to(dtype or torch.bfloat16)
    manifest = dict(manifest)
    if config["data"]["format"] == "trajectories":
        manifest["history_offsets_seconds"] = config["data"]["trajectories"]["history_offsets"]
    manifest["image_size"], manifest["max_pixels"] = config["data"]["image_size"], config["data"]["max_pixels"]
    return loaded, manifest, base


def load_checkpoint(path, *, model_path=None, device="cuda:0"):
    """An inference model straight from a training checkpoint."""
    import torch
    from event2reward.inference import Event2Reward
    dtype = torch.float32 if str(device) == "cpu" else torch.bfloat16
    loaded, manifest, base = materialize(path, model_path=model_path, dtype=dtype)
    if str(device) == "cpu":
        manifest["inference_precision"] = "fp32"
    return Event2Reward(loaded, device=device, bundle_path=base, config=manifest)


def export_bundle(path, output, *, model_path=None):
    """Write a self-contained bundle that `Event2Reward.from_pretrained` loads."""
    from event2reward.bundle import save_instance
    loaded, manifest, base = materialize(path, model_path=model_path)
    updates = {k: manifest[k] for k in ("history_offsets_seconds", "image_size", "max_pixels") if k in manifest}
    saved = read_state(path)
    updates["saved_step"] = saved["step"]
    # The training configuration without machine-specific paths.
    config = {k: v for k, v in saved["config"].items() if k not in ("model_path", "output_dir")}
    config["data"] = {k: v for k, v in config["data"].items() if k != "manifest"}
    config["training"] = {k: v for k, v in config["training"].items() if k != "resume_from"}
    state = {"step": saved["step"], "base_model": Path(base).name, "config": config, "code": saved["code"]}
    return save_instance(loaded.model, loaded.processor, base, output, updates=updates, state=state)
