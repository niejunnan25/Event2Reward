"""The training loop shared by every fine-tuning recipe."""
import json
import time
from pathlib import Path

import yaml


def load_base_model(config):
    import torch
    from event2reward.bundle import load_model
    loaded, _ = load_model(config["model_path"], attention=config["training"]["attention"],
                           torch_dtype=torch.bfloat16 if config["training"]["precision"] == "bf16" else torch.float32)
    return loaded.model, loaded.processor


def check(config):
    """Validate data and sampling on CPU without loading the model."""
    from .pipelines import IndexPipeline, TrajectoryPipeline
    from event2reward.bundle import read_manifest
    read_manifest(config["model_path"], verify=False)
    pipeline = IndexPipeline(config) if config["data"]["format"] == "index" else TrajectoryPipeline(config)
    return pipeline.summary()


def _completion_loss(result, batch, weight):
    """Extra weight on confirmed-completion queries of trajectories data."""
    from event2reward.modeling.losses import mean_preserving_gaussian_targets
    mask = batch["completion_mask"] & batch["progress_valid"]
    targets = mean_preserving_gaussian_targets(batch["target_progress"], 101, 2.0)
    ce = -(targets * result.progress_logits.float().log_softmax(-1)).sum(-1)
    return (weight - 1) * (ce * mask).sum() / batch["progress_valid"].sum().clamp_min(1)


def train(config, *, model_factory=None):
    """Run `config`; `model_factory(config) -> (model, processor)` replaces loading in tests."""
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    from event2reward.modeling.losses import ProgressModelLoss
    from .config import portable_paths
    from .checkpoint import check_resume, load_trainable, restore_training, save_checkpoint
    from .optimizer import build_optimizer, lr_multiplier
    from .parameters import configure_parameters, training_mode
    from .pipelines import IndexPipeline, TrajectoryPipeline

    t, loss_cfg = config["training"], config["loss"]
    index = config["data"]["format"] == "index"
    accelerator = Accelerator(
        gradient_accumulation_steps=t["accumulation"],
        mixed_precision="bf16" if t["precision"] == "bf16" else "no",
        step_scheduler_with_optimizer=False,
        # A relation group is atomic, so microbatches hold a varying number of rows.
        dataloader_config=DataLoaderConfiguration(even_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(
            bucket_cap_mb=200, gradient_as_bucket_view=True, broadcast_buffers=False)])
    if not index and accelerator.num_processes > 1:
        raise ValueError("trajectories data trains on one GPU; launch a single process")
    if t["precision"] == "bf16" and accelerator.device.type == "cuda":
        from event2reward.env import check_hopper_training
        check_hopper_training()
    output = Path(config["output_dir"])
    resume = t["resume_from"]
    start = check_resume(resume, config) if resume else 0
    if output.exists() and not resume and any(output.iterdir()):
        raise FileExistsError(f"output_dir already contains files: {output}")
    set_seed(t["seed"], device_specific=True)

    model, processor = (model_factory or load_base_model)(config)
    if accelerator.device.type == "cpu":
        from event2reward.modeling.qwen import use_torch_kernels
        use_torch_kernels(model)
    model.loss_module = ProgressModelLoss(lambda_progress=loss_cfg["progress"], lambda_trend=loss_cfg["trend"],
                                          lambda_relation=loss_cfg["relation"])
    if t["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    groups, trainable_summary = configure_parameters(model, config)
    if resume:
        load_trainable(model, resume)
    optimizer = build_optimizer(groups, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, t["schedule_steps"], t["warmup_steps"]))
    if index:
        pipeline = IndexPipeline(config, processor, accelerator.num_processes)
        model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, pipeline.loader(), scheduler)
        skipped = start * t["accumulation"]
        batches = iter(accelerator.skip_first_batches(loader, skipped) if skipped else loader)
    else:
        pipeline = TrajectoryPipeline(config, processor)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        batches = pipeline.batches(start * t["accumulation"])
    if resume:
        restore_training(accelerator, optimizer, scheduler, resume)
    # Clip the model parameters' gradients; FP32 master copies receive them at step().
    trainable = [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad]

    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.yaml").write_text(yaml.safe_dump(portable_paths(config), sort_keys=False))
        print(json.dumps({"data": pipeline.summary(), "trainable_parameters": trainable_summary}), flush=True)

    save_steps = set(t["save_steps"] or range(t["save_every"], t["steps"] + 1, t["save_every"])) | {t["steps"]}
    window, window_steps, begin = {}, 0, time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start, t["steps"]):
        training_mode(accelerator.unwrap_model(model))
        for _ in range(t["accumulation"]):
            batch = next(batches)
            completion = batch.pop("completion_mask", None)
            batch = {k: (v.to(accelerator.device) if torch.is_tensor(v) else
                         {kk: vv.to(accelerator.device) for kk, vv in v.items()} if isinstance(v, dict) else v)
                     for k, v in batch.items()}
            with accelerator.accumulate(model):
                result = model(**batch)
                loss = result.loss
                if completion is not None and loss_cfg["completion_weight"] != 1 and any(completion):
                    batch["completion_mask"] = torch.tensor(completion, device=accelerator.device)
                    loss = loss + loss_cfg["progress"] * _completion_loss(result, batch, loss_cfg["completion_weight"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}")
                accelerator.backward(loss)
                for key, value in {"loss": loss.detach(), **result.metrics}.items():
                    window[key] = window.get(key, 0.0) + float(value) / t["accumulation"]
                if accelerator.sync_gradients:
                    norm = accelerator.clip_grad_norm_(trainable, config["optimizer"]["grad_clip"])
                    if not torch.isfinite(norm):
                        raise FloatingPointError(f"non-finite gradient norm at step {step + 1}")
                    window["grad_norm"] = window.get("grad_norm", 0.0) + float(norm)
                    learning_rates = {g["name"]: g["lr"] for g in optimizer.param_groups}
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        window_steps += 1
        if (step + 1) % t["log_every"] == 0 or step + 1 == t["steps"]:
            row = {k: v / window_steps for k, v in window.items()}
            row = {"step": step + 1, **row, "lr": learning_rates,
                   "seconds_per_step": (time.monotonic() - begin) / window_steps}
            if accelerator.is_main_process:
                with (output / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            window, window_steps, begin = {}, 0, time.monotonic()
        if step + 1 in save_steps:
            save_checkpoint(accelerator, model, optimizer, scheduler, config, step + 1)
    accelerator.wait_for_everyone()
    accelerator.end_training()
    return output / f"checkpoint-{t['steps']:06d}"
