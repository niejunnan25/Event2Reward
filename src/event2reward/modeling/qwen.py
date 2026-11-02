"""Query tokens, the loaded-model pair, and the Qwen components that training selects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from event2reward.modeling.model import ProgressModelVLM

PROGRESS_QUERY = "<|progress_query|>"
TREND_QUERY = "<|trend_query|>"
NULL_GOAL = "<|null_goal|>"


@dataclass(frozen=True)
class LoadedModel:
    model: ProgressModelVLM
    processor: Any


@dataclass(frozen=True)
class QwenBackboneComponents:
    visual: torch.nn.Module
    language_model: torch.nn.Module
    vision_blocks: Any
    vision_mergers: tuple[torch.nn.Module, ...]


def inspect_qwen_backbone(backbone: torch.nn.Module) -> QwenBackboneComponents:
    """Locate the vision encoder, its blocks and mergers, and the language model.

    Fails loudly if the backbone layout differs, so no parameter group can be
    silently left out of training.
    """
    visual = getattr(backbone, "visual", None)
    language_model = getattr(backbone, "language_model", None)
    if not isinstance(visual, torch.nn.Module):
        raise RuntimeError("cannot locate Qwen visual encoder at backbone.visual")
    if not isinstance(language_model, torch.nn.Module):
        raise RuntimeError("cannot locate Qwen language model at backbone.language_model")
    vision_blocks = getattr(visual, "blocks", None)
    if vision_blocks is None or not hasattr(vision_blocks, "__len__"):
        raise RuntimeError("cannot locate Qwen visual blocks at backbone.visual.blocks")
    merger = getattr(visual, "merger", None)
    if not isinstance(merger, torch.nn.Module):
        raise RuntimeError("cannot locate Qwen visual merger at backbone.visual.merger")
    mergers = [merger]
    deepstack_mergers = getattr(visual, "deepstack_merger_list", None)
    if deepstack_mergers is not None:
        mergers.append(deepstack_mergers)
    return QwenBackboneComponents(visual=visual, language_model=language_model,
                                  vision_blocks=vision_blocks, vision_mergers=tuple(mergers))


def use_torch_kernels(model: torch.nn.Module) -> torch.nn.Module:
    """Switch Qwen3.5 linear-attention layers to their PyTorch implementation.

    With flash-linear-attention and causal-conv1d installed, Transformers uses
    their CUDA-only kernels in every layer. A model placed on CPU needs the
    PyTorch versions instead.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    except ImportError:
        return model
    for module in model.modules():
        if isinstance(module, qwen.Qwen3_5GatedDeltaNet):
            module.causal_conv1d_fn = None
            module.causal_conv1d_update = qwen.torch_causal_conv1d_update
            module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
            if not isinstance(module.norm, qwen.Qwen3_5RMSNormGated):
                norm = qwen.Qwen3_5RMSNormGated(module.head_v_dim, eps=module.layer_norm_epsilon)
                norm.weight = module.norm.weight
                module.norm = norm
    return model
