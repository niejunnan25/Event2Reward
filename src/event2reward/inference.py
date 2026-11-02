"""Inference: score observations with a task, a start image, a goal image and five causal frames."""
from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch

PROGRESS_QUERY = "<|progress_query|>"
TREND_QUERY = "<|trend_query|>"


def canonical_image(value: Any, image_size: int = 224) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            value = np.asarray(image.convert("RGB"))
    array = np.asarray(value)
    if array.ndim != 3 or not array.size:
        raise ValueError(f"expected a nonempty RGB-compatible image, got {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"expected RGB-compatible channels, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("image contains NaN or infinity")
    if array.dtype != np.uint8:
        if float(np.max(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.ascontiguousarray(array)).convert("RGB")
    return ImageOps.fit(image, (image_size, image_size),
                        method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))


def history_indices(query: int, *, start: int = 0, stride: int = 1) -> list[int]:
    if any(type(v) is not int for v in (query, start, stride)):
        raise TypeError("query, start and stride must be integers")
    if start < 0 or query < start or stride <= 0:
        raise ValueError("require 0 <= start <= query and positive stride")
    return [max(start, query - offset * stride) for offset in range(4, -1, -1)]


def causal_history(times, query, offsets_seconds):
    """Latest frame at or before `times[query] - offset` for each offset (seconds)."""
    return [max(0, min(query, bisect_right(times, times[query] - o + 1e-10) - 1)) for o in offsets_seconds]


@dataclass(frozen=True)
class Observation:
    task: str
    start: Any
    goal: Any
    history: Sequence[Any]

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task language must be nonempty")
        if self.start is None or self.goal is None:
            raise ValueError("start and independent successful reference goal are required")
        if len(self.history) != 5:
            raise ValueError("history must contain exactly five causal observations")


def build_conversation(*, task, start, goal, history, max_pixels):
    if len(history) != 5:
        raise ValueError("history must contain exactly five frames")
    extras = {"max_pixels": int(max_pixels)}
    content = [
        {"type": "text", "text": f"Robot task: {task.strip()}\nStart state:"},
        {"type": "image", "image": start, **extras},
        {"type": "text", "text": "\nGoal state:"},
        {"type": "image", "image": goal, **extras},
        {"type": "text", "text": "\nFive-frame history in displayed order:"},
    ]
    for index, image in enumerate(history, 1):
        content.extend([{"type": "text", "text": f"\nHistory {index}:"},
                        {"type": "image", "image": image, **extras}])
    content.append({"type": "text", "text":
        "\nPredict absolute task progress and the direction of progress in the history. "
        f"Progress: {PROGRESS_QUERY} Trend: {TREND_QUERY}"})
    return [{"role": "user", "content": content}]


def prepare_batch(processor, observations, *, image_size=224, max_pixels=50176):
    if not observations:
        raise ValueError("batch must not be empty")
    conversations = [build_conversation(task=o.task,
        start=canonical_image(o.start, image_size), goal=canonical_image(o.goal, image_size),
        history=[canonical_image(x, image_size) for x in o.history], max_pixels=max_pixels)
        for o in observations]
    texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=False,
              add_vision_id=False, enable_thinking=False) for c in conversations]
    images = [item["image"] for c in conversations for message in c
              for item in message["content"] if item.get("type") == "image"]
    batch = processor(text=texts, images=images, padding=True, return_tensors="pt")
    for name, token in (("progress", PROGRESS_QUERY), ("trend", TREND_QUERY)):
        mask = batch["input_ids"].eq(int(processor.tokenizer.convert_tokens_to_ids(token)))
        if not bool(mask.sum(-1).eq(1).all()):
            raise ValueError(f"each example must contain exactly one {name} token")
        batch[f"{name}_query_positions"] = mask.long().argmax(-1)
    return batch


def distribution_statistics(logits):
    if logits.ndim != 2 or logits.shape[-1] != 101:
        raise ValueError("expected progress logits [N,101]")
    probabilities = logits.float().softmax(-1)
    support = torch.linspace(0., 1., 101, device=logits.device)
    mean = (probabilities * support).sum(-1)
    variance = (probabilities * (support - mean.unsqueeze(-1)).square()).sum(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    return mean, variance, entropy


class Event2Reward:
    def __init__(self, loaded, *, device="cuda:0", bundle_path=None, config=None):
        from .modeling.qwen import use_torch_kernels
        self.device = torch.device(device)
        if self.device.type == "cpu":
            use_torch_kernels(loaded.model)
        self.model = loaded.model.to(self.device).eval()
        self.processor = loaded.processor
        self.bundle_path = Path(bundle_path).resolve() if bundle_path else None
        self.config = config or {"image_size": 224, "max_pixels": 50176}

    @classmethod
    def from_pretrained(cls, name_or_path, *, revision=None, cache_dir=None,
                        device="cuda:0", attention="sdpa", verify=True):
        from .bundle import load_model
        candidate = Path(name_or_path).expanduser()
        if candidate.is_dir():
            path = candidate.resolve()
        else:
            from huggingface_hub import snapshot_download
            try:
                path = snapshot_download(repo_id=str(name_or_path), revision=revision, cache_dir=cache_dir)
            except Exception as exc:
                raise OSError(
                    f"Could not load {name_or_path!r}: the local directory {str(candidate)!r} "
                    "does not exist, and the Hugging Face Hub download failed. "
                    "The Hub repository may not exist or you may lack permission; "
                    "also check the revision, network connection and offline cache. "
                    f"Original error: {exc}"
                ) from exc
        loaded, config = load_model(path, attention=attention, verify=verify)
        return cls(loaded, device=device, bundle_path=path, config=config)

    def prepare(self, observations):
        return prepare_batch(self.processor, observations, image_size=self.config["image_size"],
                             max_pixels=self.config["max_pixels"])

    @torch.inference_mode()
    def predict(self, observations, *, include_distribution=False):
        batch = self.prepare(observations)
        inputs = {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in batch.items()}
        context = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" and self.config.get("inference_precision", "bf16") == "bf16" else nullcontext()
        with context:
            output = self.model(**inputs)
        mean, variance, entropy = distribution_statistics(output.progress_logits)
        trends = output.trend_logits.float().softmax(-1)
        labels = ("growth", "stable", "regression")
        result=[]
        for i in range(len(observations)):
            row={"progress":float(mean[i]), "trend":labels[int(trends[i].argmax())],
                 "trend_probabilities":trends[i].cpu().tolist(),
                 "progress_variance":float(variance[i]), "progress_entropy":float(entropy[i])}
            if include_distribution:
                row["progress_distribution"]=output.progress_logits[i].float().softmax(-1).cpu().tolist()
            result.append(row)
        return result

    def predict_trajectory(self, *, frames, task, goal, query_indices=None, stride=1,
                           batch_size=1, timestamps=None):
        """Score every query frame of one trajectory with causal histories.

        Models fine-tuned with time-based histories (`history_offsets_seconds` in
        their manifest) need `timestamps`; otherwise histories step back `stride`
        frames at a time.
        """
        if not len(frames) or type(batch_size) is not int or batch_size <= 0:
            raise ValueError("nonempty frames and positive integer batch_size required")
        queries=list(range(len(frames))) if query_indices is None else list(query_indices)
        if any(q >= len(frames) for q in queries):
            raise ValueError("query outside trajectory")
        if timestamps is not None:
            if len(timestamps)!=len(frames) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<=0):
                raise ValueError("timestamps must be finite, strictly increasing and match frames")
        offsets=self.config.get("history_offsets_seconds")
        if offsets is not None:
            if timestamps is None:
                raise ValueError("this model uses time-based histories; pass timestamps")
            indices=[causal_history(list(timestamps), q, offsets) for q in queries]
        else:
            indices=[history_indices(q, stride=stride) for q in queries]
        result=[]
        for offset in range(0,len(queries),batch_size):
            group=indices[offset:offset+batch_size]
            outputs=self.predict([Observation(task,frames[0],goal,[frames[i] for i in h]) for h in group])
            for j,row in enumerate(outputs,offset):
                row.update(query_index=queries[j],history_indices=indices[j])
                if timestamps is not None:row['timestamp']=float(timestamps[queries[j]])
                result.append(row)
        return result

    def save_pretrained(self, destination):
        from .bundle import save_instance
        if self.bundle_path is None:
            raise ValueError("save_pretrained needs a model loaded from a bundle")
        save_instance(self.model, self.processor, self.bundle_path, destination)
