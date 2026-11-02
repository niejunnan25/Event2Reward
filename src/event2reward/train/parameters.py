"""Choose trainable parameters and group them by learning rate."""
from .lora import inject


def configure_parameters(model, config):
    """Freeze everything, then unfreeze what `tuning` selects.

    Returns optimizer groups named language / vision / lora / heads.
    """
    from event2reward.modeling.qwen import inspect_qwen_backbone
    tuning, lr = config["tuning"], config["optimizer"]["lr"]
    components = inspect_qwen_backbone(model.backbone)
    model.requires_grad_(False)
    if tuning["mode"] == "lora":
        depth = len(getattr(components.language_model, "layers", []))
        if any(i >= depth for i in tuning["lora"]["layers"]):
            raise ValueError(f"tuning.lora.layers exceed the model's {depth} language layers")
        inject(model, tuning["lora"])
    else:
        full = tuning["full"]
        components.language_model.requires_grad_(full["language"])
        layers = full["vision_layers"]
        if layers == "all":
            components.visual.requires_grad_(True)
            for merger in components.vision_mergers:
                merger.requires_grad_(full["vision_merger"])
        elif layers:
            if layers > len(components.vision_blocks):
                raise ValueError(f"tuning.full.vision_layers exceeds {len(components.vision_blocks)} vision blocks")
            for block in components.vision_blocks[-layers:]:
                block.requires_grad_(True)
            for merger in components.vision_mergers:
                merger.requires_grad_(full["vision_merger"])
    heads = tuning["heads"]
    if heads["fp32"]:
        model.progress_head.float()
        model.trend_head.float()
    model.progress_head.requires_grad_(heads["progress"])
    model.trend_head.requires_grad_(heads["trend"])

    visual = {id(p) for p in components.visual.parameters()}
    head = {id(p) for h in (model.progress_head, model.trend_head) for p in h.parameters()}
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = ("heads" if id(parameter) in head else "vision" if id(parameter) in visual
                     else "lora" if name.endswith((".lora_A", ".lora_B")) else "language")
            groups.setdefault(group, []).append(parameter)
    if not groups:
        raise ValueError("the configuration leaves no trainable parameters")
    summary = {name: sum(p.numel() for p in ps) for name, ps in groups.items()}
    return [{"params": ps, "lr": lr[name], "name": name} for name, ps in groups.items()], summary


def training_mode(model):
    """Training mode, except frozen modules keep dropout off."""
    model.train()
    for module in (model.backbone.visual, model.trend_head, model.progress_head):
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()
