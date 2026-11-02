"""AdamW with FP32 master weights for low-precision trainable parameters.

Forward weights remain BF16 for the model's kernels. Small updates accumulate in
FP32 rather than rounding to zero at every optimizer step. The master weights and
moments both participate in checkpoint/resume.
"""
import math

import torch


class MasterAdamW(torch.optim.AdamW):
    def __init__(self, groups, **kwargs):
        self.master_pairs = []
        optimizer_groups = []
        for group in groups:
            parameters = []
            for parameter in group["params"]:
                if parameter.dtype in (torch.bfloat16, torch.float16):
                    master = torch.nn.Parameter(parameter.detach().float().clone())
                    self.master_pairs.append((parameter, master))
                    parameters.append(master)
                elif parameter.dtype == torch.float32:
                    parameters.append(parameter)
                else:
                    raise ValueError("fine-tuning supports only FP32/BF16/FP16 parameter storage")
            optimizer_groups.append({**group, "params": parameters})
        super().__init__(optimizer_groups, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MasterAdamW does not support closures")
        for parameter, master in self.master_pairs:
            master.grad = None if parameter.grad is None else parameter.grad.detach().float()
        result = super().step()
        for parameter, master in self.master_pairs:
            parameter.copy_(master)
        return result

    def zero_grad(self, set_to_none=True):
        for parameter, _ in self.master_pairs:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.detach_()
                    parameter.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        state = super().state_dict()
        state["fp32_master_weights"] = [p.detach().cpu().clone() for _, p in self.master_pairs]
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        masters = state.pop("fp32_master_weights", None)
        if masters is None or len(masters) != len(self.master_pairs):
            raise ValueError("optimizer master-weight layout differs from checkpoint")
        for value, (_, parameter) in zip(masters, self.master_pairs):
            if value.dtype != torch.float32 or value.shape != parameter.shape:
                raise ValueError("optimizer master-weight dtype/shape mismatch")
        super().load_state_dict(state)
        with torch.no_grad():
            for value, (parameter, master) in zip(masters, self.master_pairs):
                master.copy_(value)
                parameter.copy_(master)


def build_optimizer(groups, config):
    """Plain AdamW, or AdamW over FP32 master copies of BF16 parameters."""
    opt = config["optimizer"]
    if opt["fp32_master_weights"]:
        if opt["fused"]:
            raise ValueError("optimizer.fused is not available with fp32_master_weights")
        return MasterAdamW(groups, weight_decay=opt["weight_decay"])
    return torch.optim.AdamW(groups, weight_decay=opt["weight_decay"], fused=True if opt["fused"] else None)


def lr_multiplier(step, schedule_steps, warmup_steps):
    """Linear warmup, then cosine decay to zero at schedule_steps."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, schedule_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
