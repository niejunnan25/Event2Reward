# Installation

- [Requirements](#requirements)
- [Inference](#inference)
- [Download a model](#download-a-model)
- [Training](#training)
- [Check your environment](#check-your-environment)
- [Reference environment](#reference-environment)
- [Without the fast kernels](#without-the-fast-kernels)
- [Troubleshooting](#troubleshooting)

## Requirements

- Linux x86_64 and Conda; the commands below use Python 3.10.
- An NVIDIA GPU whose driver reports **CUDA Version ≥ 12.8** in `nvidia-smi`. This is the driver's compatibility level, not an installed compiler.
- **Inference needs no CUDA Toolkit.** The default Linux `torch==2.8.0` wheel from PyPI brings the CUDA 12.8 runtime.

These requirements are for inference and training. The [annotation tool](annotation.md) runs on any computer, with or without a GPU, on Linux, macOS or Windows: `pip install -e ".[video]"` in any Python 3.10 to 3.13 environment. Usually you annotate on your own laptop, export, copy the project folder (it holds the videos and the export) to the GPU server, and convert and train there.

## Inference

```bash
git clone https://github.com/niejunnan25/Event2Reward.git
cd Event2Reward
conda create -n event2reward python=3.10 -y
conda activate event2reward
pip install -e .

# Recommended: the fast kernels used for the paper (prebuilt wheels, no compiler needed)
pip install flash-linear-attention==0.5.2 fla-core==0.5.2
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

The last two commands are optional. Without them Transformers uses its PyTorch implementation, which is slower and gives slightly different outputs ([measurements](#without-the-fast-kernels)). Install them to reproduce the quick-start output exactly. Models placed on CPU always use the PyTorch implementation.

## Download a model

Stage1 and Stage2 are public on the Hugging Face Hub. `from_pretrained` downloads and caches a model by name:

```python
from event2reward import Event2Reward
model = Event2Reward.from_pretrained("niejunnan25/Event2Reward-Qwen3.5-4B-Stage2")
```

To keep a copy in the repository, which the training configurations expect under `models/`:

```bash
bash scripts/download_models.sh Stage2          # or: Stage1 Stage2
```

`from_pretrained` also accepts `revision` and `cache_dir`, and a local directory in place of a name. `HF_HUB_OFFLINE=1` uses only the local cache.

## Training

```bash
pip install -e ".[train]"
```

On Hopper GPUs (H100, H200, H20), the backward pass of flash-linear-attention compiles TileLang kernels. Install TileLang and provide a CUDA 12.8 compiler:

```bash
pip install tilelang==0.1.9 apache-tvm-ffi==0.1.10
export CUDA_HOME=/usr/local/cuda-12.8          # your CUDA 12.8 toolkit
```

If no toolkit is installed, follow NVIDIA's [CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/). The compiler must be at least 12.8 and at most the CUDA Version that `nvidia-smi` reports. `apache-tvm-ffi` is pinned because newer releases can fail when TileLang first loads. Other GPUs should not need this step, but training was only tested on Hopper.

Continue with [training](training.md).

## Check your environment

```bash
event2reward env           # or: event2reward env --json
```

The command reports the versions, GPU and kernels it finds and ends with a conclusion such as `ready for inference`. It always exits with status 0. A missing fast kernel does not block inference, and the Hopper compiler setup only concerns training. Training runs the same Hopper check before it starts.

## Reference environment

The paper's reward models and the stored quick-start output were produced with:

| Component | Version |
|---|---|
| OS / NVIDIA driver | Ubuntu 22.04 / 570.195.03 |
| GPU | H200 |
| Python | 3.10.20 |
| PyTorch / Transformers / Accelerate | 2.8.0+cu128 / 5.2.0 / 1.12.0 |
| Triton / Pillow / Tokenizers | 3.4.0 / 11.3.0 / 0.22.1 |
| flash-linear-attention, fla-core / causal-conv1d | 0.5.2 / 1.7.0 |
| TileLang / apache-tvm-ffi / nvcc (Hopper training) | 0.1.9 / 0.1.10 / 12.8.93 |

To pin every Python package to these versions:

```bash
pip install -e ".[train]" -c docs/requirements-tested.txt
```

then install the fast kernels as above. With this environment, Stage2 reproduces `examples/quickstart/expected_output.json` exactly:

```bash
event2reward infer --bundle niejunnan25/Event2Reward-Qwen3.5-4B-Stage2 \
  --observations examples/quickstart/observations.json --output outputs/prediction.json
```

Speed and memory (BF16, SDPA attention, batch 1, preprocessing included): a Stage2 query takes 0.090 s on H200 (median of 30 after 5 warm-up queries). Peak allocated memory is 8.7 GiB; loading the model with integrity checks takes about 13 s.

## Without the fast kernels

Over eight fixed inputs, the PyTorch implementation differs from the fast kernels by at most 0.169 in the Stage2 progress logits and 0.017 in progress (observed differences, not bounds). It takes 0.320 s per query instead of 0.090 s on H200.

## Troubleshooting

- **Slow or blocked downloads.** Select a PyPI mirror with `-i https://mirrors.aliyun.com/pypi/simple/` or `-i https://pypi.tuna.tsinghua.edu.cn/simple`. For Hugging Face, set `HF_ENDPOINT=https://hf-mirror.com` before starting Python. If GitHub is unreachable, download the causal-conv1d wheel elsewhere and run `pip install <file>.whl`.
- **No pip in an offline Conda environment.** Run `python -m ensurepip --upgrade`, then `python -m pip install --upgrade pip`.
- **Version conflicts.** Use a separate Conda environment; RL frameworks often pin other torch or Transformers versions. Set `PYTHONNOUSERSITE=1` if user-site packages interfere with imports.
- **Triton compilation errors.** Triton needs a C compiler such as `gcc`.
- **Model not found.** Check the model name, e.g. `niejunnan25/Event2Reward-Qwen3.5-4B-Stage2`. A complete local model directory works with the same API.
