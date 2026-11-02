# Training

Every fine-tuning recipe runs through the same command and the same YAML layout:

```bash
event2reward train configs/finetune/real_world_franka/buttons_full.yaml   # one GPU
event2reward train configs/finetune/libero/goal.yaml --gpus 2         # data-parallel
event2reward train configs/finetune/libero/goal.yaml --check         # validate config and data, no training
```

Run commands from the repository root. Relative paths in a config resolve against the directory you run from. Use `--set` to change any value without editing the file:

```bash
event2reward train configs/finetune/libero/goal.yaml \
  --set model_path=/data/models/Stage2 --set data.manifest=/data/libero/goal/dataset.json
```

Install the [training environment](installation.md#training) first, and download the starting model with `bash scripts/download_models.sh Stage2` (or `Stage1`).

## Configurations

| Config | What it trains | Data |
|---|---|---|
| `configs/finetune/libero/{spatial,object,goal,libero_10}.yaml` | Stage2 → the four LIBERO suite models | Index format in [data](data.md#libero) |
| `configs/finetune/real_world_franka/*.yaml` | Stage2 → the reward models of the paper's six real-world tasks (`_full`: full fine-tuning; `_lora`: LoRA) | Format in [data](data.md#trajectory-data) |

Each configuration holds the settings of its original run. For LIBERO, `tests/test_configs.py` checks them against the configurations those runs recorded.

## Config reference

Every configuration has the same fields in the same order, so comparing two of them shows only the settings that differ (`diff configs/finetune/libero/goal.yaml configs/finetune/real_world_franka/buttons_full.yaml`). Two switches select which block is used; the other block stays in the file with its defaults:

- `data.format` selects `data.trajectories` (annotated trajectories: Franka, your own data) or `data.index` (prepared JSONL indices: Stage2, LIBERO).
- `tuning.mode` selects `tuning.full` or `tuning.lora`; `optimizer.lr.language` and `vision` apply to `full`, `optimizer.lr.lora` to `lora`.

```yaml
model_path: models/Stage2          # bundle to start from
output_dir: outputs/libero/goal

data:
  format: index                    # index | trajectories
  manifest: data/libero/goal/dataset.json
  image_size: 224
  max_pixels: 50176
  index:
    goal_dropout: 0.0              # share of clips shown without their goal image
    require_goal_present: true
    sources: [...]                 # index families to mix, and their shares
  trajectories:
    task: null                     # one task of the manifest; null: every task
    history_offsets: [4, 2, 1, 0.25, 0]
    training_history_scales: [1.0]
    sampling:
      progress: {expert: 3, stall_success: 1, completion: 1}
      relations: {groups: 1, members: 4, balance: type}

tuning:
  mode: full                       # full | lora
  full: {language: true, vision_layers: 3, vision_merger: true}
  lora: {layers: [24, ..., 31], rank: 8, alpha: 16, dropout: 0.05}
  heads: {progress: true, trend: true, fp32: false}

optimizer:
  lr: {language: 2.0e-6, vision: 5.0e-7, lora: 1.0e-4, heads: 1.0e-5}
  weight_decay: 0.01
  grad_clip: 1.0
  fp32_master_weights: false
  fused: false

loss: {progress: 1.0, trend: 1.0, relation: 10.0, completion_weight: 1.0, terminal_weight: 1.0}

training:
  steps: 500                       # optimizer steps to run
  schedule_steps: 500              # length of the warmup + cosine learning-rate schedule
  warmup_steps: 25
  batch_size: 10                   # clips per microbatch per GPU
  accumulation: 4                  # microbatches per GPU per optimizer step
  seed: 42
  precision: bf16
  attention: sdpa
  gradient_checkpointing: true
  save_every: 50                   # or save_steps: [1000, 2000, ...]
  save_steps: null
  keep_optimizer_states: 1         # latest checkpoints that can be resumed
  log_every: 10
  num_workers: 2
  prefetch_factor: 2
  resume_from: null                # a checkpoint directory to continue from
```

The shipped files carry a comment on every field. Unknown keys are errors, so a misspelled key fails at load time instead of being ignored. `tests/test_configs.py` checks that every shipped configuration follows this layout exactly; the defaults are in [`src/event2reward/train/config.py`](../src/event2reward/train/config.py).

**`tuning`.** `full` updates the original weights of the language model, the last `vision_layers` vision blocks (`all`, or `0` to freeze vision), the vision merger and the heads. `lora` freezes the backbone and trains rank-`rank` adapters on the attention projections of the listed language layers, plus the heads. Each parameter group uses its own learning rate from `optimizer.lr`.

**`training.steps` and `schedule_steps`.** The learning rate warms up linearly for `warmup_steps`, then follows a cosine down to zero at `schedule_steps`. Training stops after `steps`.

**Index data (`data.format: index`).** Each `data.index.sources` entry names an index family of the manifest and its `fraction` of the clips in a microbatch. A relation group is never split, so microbatches hold a varying number of groups but always exactly `batch_size` clips. The one source with `role: fill` absorbs rounding. Source `type` is `relation`: rows with clips and an optional relation, as in LIBERO.

### What a microbatch holds

With trajectory data (`data.format: trajectories`), every microbatch is planned ahead and alternates between tasks. `data.trajectories.sampling` sets how many queries of each kind it holds, in two groups whose counts add up to `training.batch_size`. `progress` queries get an absolute progress target; `relations` queries get only the relation of their event:

| Key | Frames from | Progress target |
|---|---|---|
| `progress.expert` | clean successful demonstrations, stratified over time | elapsed time / completion time |
| `progress.stall_success` | successful trajectories with progress stalls (and no other events) | the same, without the stall time |
| `progress.completion` | the completion frame `C` of any successful trajectory | 1 |
| `relations.groups` × (1 + `relations.members`) | an event's anchor frame and member frames inside the event | none; the event's relation ([event types](data.md#trajectory-data)) |

**The provided configurations** give half of every microbatch absolute progress on clean demonstrations and the other half to event relations: `expert: 5` and one event group of 1 + 4 frames, with `stall_success` and `completion` at 0. The Franka `*_full.yaml` configurations and [`examples/annotation/finetune.yaml`](../examples/annotation/finetune.yaml) use it.

Every event trajectory takes part through its event groups, whatever the `progress` counts: a successful trajectory with a progress stall learns that progress stands still during the stall. `stall_success` supervises such trajectories once more, with absolute progress that leaves out the stall time; it helps when clean demonstrations are few. `completion` adds a completion frame to every microbatch, and `loss.completion_weight` weights it up, when the model should be sure that a task is finished. Both take places from the other kinds:

```yaml
sampling:
  progress: {expert: 4, stall_success: 1, completion: 0}
  relations: {groups: 1, members: 4, balance: type}       # 4 + 1 + 0 + 1 × (1 + 4) = 10
```

The Franka `*_lora.yaml` runs use `3 + 1 + 1 + 1 × (1 + 4)`.

A regression and recovery group holds its recovery frame as a second reference and one member fewer, so every group has the same size. With `relations.balance: type`, the event types present in a task take equal turns, and the events of each type take equal turns: any subset of the four types works, and with all four each takes a quarter of the event groups, as in the LIBERO runs. With `balance: trajectory`, event trajectories take equal turns, as in the Franka runs. A task without events fills its event queries with expert frames, and a task without successful trajectories with progress stalls fills its `stall_success` queries with them; events too short for `members` frames are left out. `--check` reports the events used per task, the events left out and these fallbacks. Goal images always come from another expert of the same task. `loss.completion_weight` and `loss.terminal_weight` reweight completion queries and terminal-failure relations. With `loss.trend > 0`, expert frames whose whole history lies inside the trajectory are labeled trend "growth", as LIBERO expert clips are. History frames are chosen by time, `data.trajectories.history_offsets` seconds before the query; `training_history_scales` randomly stretches them during training. This data format trains on one GPU.

## Outputs and checkpoints

```text
outputs/libero/goal/
  config.yaml                 the complete configuration of the run
  metrics.jsonl               losses, gradient norm and learning rates every log_every steps
  checkpoint-000050/
    weights.safetensors       the trainable tensors (LoRA adapters and heads, or the updated weights)
    state.json                step, configuration and code version
    optimizer.pt, rng/        optimizer, schedule and random states, for resuming
```

The rest of the model comes from `model_path`. To use a checkpoint:

```python
from event2reward.train.checkpoint import load_checkpoint
model = load_checkpoint("outputs/libero/goal/checkpoint-000500", device="cuda:0")
model.predict([...])                           # same interface as Event2Reward
```

To share it, export a self-contained bundle (LoRA merged into the weights):

```bash
event2reward export --checkpoint outputs/libero/goal/checkpoint-000500 --output models/libero-goal
```

Models trained with time-based histories store their offsets in the bundle. `predict_trajectory` then needs `timestamps` and selects history frames by time.

**Resuming.** Set `training.resume_from` to a checkpoint that still has `optimizer.pt` and use the same configuration. Only execution settings may differ: `steps`, saving, logging, workers and gradient checkpointing. The data order, learning rate and random state continue exactly where the run stopped.

## LIBERO

The four LIBERO suite models start from Stage2. Each run has 500 steps of 80 clips: 2 GPUs × 4 microbatches × 10 clips. On one GPU, `--set training.accumulation=8` gives the same 80 clips per step. Half the clips are successful expert demonstrations (5% per task) and half are event groups (12.5% per event kind). All clips are conditioned on an independent successful goal.

```bash
bash scripts/download_models.sh Stage2
event2reward convert libero --indices /path/to/annotations/goal \
  --media-root libero=/path/to/lerobot/libero --output data/libero/goal/dataset.json
event2reward train configs/finetune/libero/goal.yaml --check
event2reward train configs/finetune/libero/goal.yaml --gpus 2
```

[Data](data.md#libero) describes the files. The synthetic dataset runs the full pipeline, data, model and training, without any data download:

```bash
event2reward convert synthetic --output data/synthetic
event2reward train configs/finetune/libero/goal.yaml --set data.manifest=data/synthetic/dataset.json \
  --set 'data.index.sources=[{family: expert, type: relation, role: expert, fraction: 0.5}, {family: event_kind3, type: relation, role: event, fraction: 0.5}, {family: expert_fill, type: relation, role: fill, fraction: 0.0}]' \
  --set training.steps=2 --set output_dir=outputs/synthetic
```

## Real-world Franka

The reward models of the paper's six real-world tasks on Franka robots: three precise insertion tasks, Plug Insert, RAM Insert and Ethernet Insert, and three manipulation tasks, Stack Bowls, Press Buttons and Plate Vegetables. Each task has a full fine-tuning and a LoRA configuration:

| Config | Training |
|---|---|
| `<task>_full.yaml` | The training parameters of the LIBERO models, on less data: the language model, the last 3 vision blocks, the vision merger and both heads, 500 steps of 80 clips (about 1 hour on one H200). Each microbatch holds 5 frames of successful trajectories and one event group. |
| `<task>_lora.yaml` | LoRA adapters on the last eight language layers with the backbone frozen, on one GPU. Each microbatch holds 3 expert frames, 1 frame of a successful trajectory with events, 1 completion frame and one event group. History offsets are set per task (`[8, 4, 2, 0.5, 0]` seconds for buttons), and the vegetables run also rescales them during training. |

Both configurations keep the event order of the runs, `relations.balance: trajectory`.

```bash
event2reward train configs/finetune/real_world_franka/buttons_full.yaml
event2reward train configs/finetune/real_world_franka/buttons_lora.yaml
```

The configurations serve as starting points for your own robot.

## A new environment

To add your own robot, simulator or task, you annotate videos and convert them; the trainer and the settings stay the same. [`examples/annotation/`](../examples/annotation) walks through the steps on five LIBERO videos.

1. **Annotate** with the [annotation tool](annotation.md): successful videos and failed videos separately, marking where a success stalls or loses progress and where a failure stops progressing. Each task needs at least two clean successful videos, which also provide the goal images. Any subset of the four event types works.

   ```bash
   event2reward annotate add my_project --task stack_bowls --instruction "stack the bowls" \
     --success videos/stack_bowls/success --failure videos/stack_bowls/failure
   event2reward annotate serve my_project
   event2reward annotate export my_project --output annotations.jsonl
   ```

2. **Convert** the export to trajectories ([format](data.md#from-the-annotation-tool)):

   ```bash
   event2reward convert annotations --input annotations.jsonl --project my_project --output data/stack_bowls/dataset.json
   ```

   Without the tool, write `event2reward.trajectories.v1` directly ([format](data.md#trajectory-data)).
3. **Copy the configuration** [`examples/annotation/finetune.yaml`](../examples/annotation/finetune.yaml) and set `output_dir`, `data.manifest` and, for one task of several, `data.trajectories.task`. Its `sampling` block is explained in [what a microbatch holds](#what-a-microbatch-holds). `data.trajectories.history_offsets` spans 0.4 s, the history of the LIBERO models; choose longer offsets if a visible step of your task takes longer. For a lighter run on one GPU with a frozen backbone, start from `configs/finetune/real_world_franka/buttons_lora.yaml` instead and set `relations.balance: type`. With 10 clips per microbatch, BF16 and gradient checkpointing, full fine-tuning peaks at about 41 GiB of GPU memory and LoRA at about 11 GiB (measured on one H200).
4. **Check, then train:**

   ```bash
   event2reward train configs/finetune/stack_bowls.yaml --check
   event2reward train configs/finetune/stack_bowls.yaml
   ```

   `--check` reports the trajectories per task, the events used per type, the events left out and the planned queries per supervision type, without loading the model.

5. **Export and use the model.** `event2reward export --checkpoint outputs/stack_bowls/checkpoint-000500 --output models/stack_bowls` writes a model directory that `Event2Reward.from_pretrained("models/stack_bowls")` loads ([inference](inference.md)). [`examples/annotation/compare.py`](../examples/annotation/compare.py) scores held-out videos before and after fine-tuning.
