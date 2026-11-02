import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def write_bundle(root, model):
    """A minimal bundle around `model`; enough for manifest, save and load paths."""
    from safetensors.torch import save_model
    from event2reward import bundle
    (root / "model").mkdir(parents=True)
    (root / "backbone").mkdir()
    save_model(model, str(root / "model/model.safetensors"))
    for name in ("state.json", *(f"backbone/{x}" for x in bundle.SIDECARS)):
        (root / name).write_text("{}")
    manifest = dict(format=bundle.FORMAT, num_bins=101, history_frames=5, files=bundle.file_records(root))
    (root / "event2reward.json").write_text(json.dumps(manifest))
    return model


@pytest.fixture
def tiny_bundle(tmp_path):
    return lambda name="bundle": (tmp_path / name, write_bundle(tmp_path / name, torch.nn.Linear(3, 2)))


def processor_files():
    """Tokenizer and image-processor files of the released backbone; nothing is downloaded."""
    path = Path(os.environ.get("EVENT2REWARD_TEST_PROCESSOR", ROOT / "models/Stage2/backbone"))
    if not (path / "tokenizer.json").is_file():
        pytest.skip("set EVENT2REWARD_TEST_PROCESSOR to a Stage2 backbone/ directory")
    return path


@pytest.fixture(scope="session")
def tiny_qwen():
    """A 4-layer random Qwen3.5 with the real processor: real code paths at CPU speed."""
    def build(seed=42, dtype=torch.float32):
        from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5Model
        from event2reward.modeling.model import ProgressModelVLM
        sidecars = processor_files()
        processor = AutoProcessor.from_pretrained(sidecars, local_files_only=True)
        raw = json.loads((sidecars / "config.json").read_text())
        token = processor.tokenizer.convert_tokens_to_ids
        config = Qwen3_5Config(
            text_config={"vocab_size": len(processor.tokenizer), "hidden_size": 32, "intermediate_size": 64,
                         "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
                         "head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                         "linear_key_head_dim": 8, "linear_value_head_dim": 8},
            vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                           "out_hidden_size": 32, "patch_size": 16},
            **{k: raw[k] for k in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")})
        config._attn_implementation = "sdpa"
        torch.manual_seed(seed)
        model = ProgressModelVLM(Qwen3_5Model(config), hidden_size=32,
                                 progress_query_token_id=token("<|progress_query|>"),
                                 trend_query_token_id=token("<|trend_query|>"))
        from event2reward.modeling.qwen import use_torch_kernels
        return use_torch_kernels(model).to(dtype), processor
    return build


@pytest.fixture(scope="session")
def synthetic_index(tmp_path_factory):
    from event2reward.data.converters.synthetic import make_example
    return make_example(tmp_path_factory.mktemp("synthetic") / "data")


def trajectory_records(experts=3, frames=41):
    times = [i / 10 for i in range(frames)]
    end = frames - 1
    common = dict(task="push", instruction="push the block", split="train", times=times, S=end)
    rows = [dict(common, uid=f"expert{i}", role="expert", C=end, events=[], frame_directory=f"e{i}")
            for i in range(experts)]
    rows.append(dict(common, uid="stall", role="event", C=end, frame_directory="h",
                     events=[dict(type="progress_stall", anchor_frame=10, end_frame=20)]))
    rows.append(dict(common, uid="failure", role="event", C=None, frame_directory="f",
                     events=[dict(type="no_further_progress", anchor_frame=25, end_frame=end)]))
    return rows


@pytest.fixture
def trajectory_data(tmp_path):
    """dataset.json in event2reward.trajectories.v1 with PNG frame directories."""
    from PIL import Image, ImageDraw
    rows = trajectory_records()
    for k, r in enumerate(rows):
        directory = tmp_path / r["frame_directory"]
        directory.mkdir()
        for f in range(len(r["times"])):
            image = Image.new("RGB", (64, 48), (30, 30, 35))
            x = 4 + int(40 * f / (len(r["times"]) - 1)) + k
            ImageDraw.Draw(image).rectangle((x, 20, x + 8, 28), fill=(220, 45, 35))
            image.save(directory / f"{f:06d}.png")
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"schema": "event2reward.trajectories.v1", "trajectories": rows}))
    return path


@pytest.fixture(scope="session")
def tiny_qwen_bundle(tiny_qwen, tmp_path_factory):
    """The tiny Qwen saved as a real bundle, so training loads it like Stage2."""
    import shutil
    from safetensors.torch import save_model
    from event2reward import bundle
    model, processor = tiny_qwen()
    root = tmp_path_factory.mktemp("tiny") / "Stage2"
    (root / "model").mkdir(parents=True)
    (root / "backbone").mkdir()
    save_model(model, str(root / "model/model.safetensors"))
    for name in bundle.SIDECARS:
        shutil.copy(processor_files() / name, root / "backbone" / name)
    model.backbone.config.save_pretrained(root / "backbone")
    (root / "state.json").write_text("{}")
    token = processor.tokenizer.convert_tokens_to_ids
    manifest = dict(format=bundle.FORMAT, num_bins=101, history_frames=5, hidden_size=32, head_dropout=0.1,
                    vocab_size=len(processor.tokenizer), image_size=224, max_pixels=50176,
                    query_token_ids={"progress": token("<|progress_query|>"), "trend": token("<|trend_query|>"),
                                     "null_goal": token("<|null_goal|>")},
                    files=bundle.file_records(root))
    (root / "event2reward.json").write_text(json.dumps(manifest))
    return root
