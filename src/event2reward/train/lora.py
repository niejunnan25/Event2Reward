"""LoRA adapters on the attention projections of selected language layers."""
import math
import re
import torch
from torch import nn

TARGET = re.compile(
    r'backbone\.language_model\.layers\.(\d+)\.'
    r'(linear_attn\.(in_proj_qkv|out_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj))$'
)


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError('LoRA requires Linear and positive rank')
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        original = self.base(x)
        # Keep master adapter weights in FP32; autocast handles CUDA matmuls.
        delta = nn.functional.linear(nn.functional.linear(self.dropout(x).float(), self.lora_A), self.lora_B)
        return original + (delta * self.scale).to(original.dtype)


def inject(model, config):
    """Wrap the attention projections of the selected language layers."""
    selected = []
    layers = set(config['layers'])
    for name, module in list(model.named_modules()):
        match = TARGET.fullmatch(name)
        if match and int(match.group(1)) in layers:
            if not isinstance(module, nn.Linear):
                raise TypeError(f'expected Linear at {name}, got {type(module)}')
            parent_name, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), leaf,
                    LoRALinear(module, config['rank'], config['alpha'], config['dropout']))
            selected.append(name)
    if not selected:
        raise ValueError('no LoRA target projections in the selected layers')
    return selected


def merge(model):
    """Merge for inference only, replacing each adapter with its base Linear."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = module.lora_B.float() @ module.lora_A.float() * module.scale
                module.base.weight.copy_((module.base.weight.float() + delta).to(module.base.weight.dtype))
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, leaf, module.base)
    return model
