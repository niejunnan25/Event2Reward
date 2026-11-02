# Data

Training reads a `dataset.json` manifest. There are two formats. **Trajectory** data lists annotated trajectories, and the trainer samples the training queries from them; use it for your own data, for example from the annotation tool. **Index** data lists the training clips themselves, prepared in advance; the LIBERO models were trained on it.

## LIBERO

Each LIBERO suite consists of these index files:

| File | Content |
|---|---|
| `train_expert_task00.jsonl` … `task09.jsonl` | Successful expert demonstrations, one clip per row, per task |
| `train_kind3.jsonl` … `train_kind6.jsonl` | Event groups: no further progress, unrecovered regression, progress stall, regression and recovery |
| `train_expert_all.jsonl` | All expert clips, used to fill microbatches |

The clips reference videos of the [LIBERO datasets](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets) and policy rollouts recorded in the LIBERO simulator. To build the manifest, point `event2reward convert libero` at the index directory and at each media root that the clips' `dataset_id` names (it lists any that are missing):

```bash
event2reward convert libero --indices /path/to/annotations/goal \
  --media-root libero=/path/to/lerobot/libero --output data/libero/goal/dataset.json
```

## Index data

```json
{
  "schema": "event2reward.dataset.v1",
  "media_roots": {"libero": "../../media/libero"},
  "indices": {"expert_task00": "indices/train_expert_task00.jsonl", "...": "..."}
}
```

`media_roots` maps each `dataset_id` used by the clips to a directory. `indices` maps family names, which the configuration's `data.index.sources` refers to, to JSONL files. Paths are relative to the manifest.

**`relation` rows** hold `clips` and an optional `relation`. Each clip names its video (`dataset_id`, `video_path`), the task `instruction`, and five causal `history_frames` that end at `current_frame`. It also names an independent successful goal (`goal_dataset_id`, `goal_video_path`, `goal_frame`) and its targets: `target_progress` in [0, 1] with a `progress_valid` mask, and `target_trend` (0 growth, 1 stable, 2 regression) with a `trend_valid` mask. The complete field list and checks are in [`data/schema.py`](../src/event2reward/data/schema.py) (`ClipRecord`).

A relation constrains the predicted progress of its member clips relative to an anchor clip:

| `kind` | Event | Constraint |
|---|---|---|
| 3 | no further progress | members no higher than the anchor |
| 4 | unrecovered regression | members strictly below the anchor |
| 5 | progress stall | members within `margin` of the anchor |
| 6 | regression and recovery | members below both the anchor and the recovery point `secondary_anchor_index` |

The loss definitions in [`modeling/losses.py`](../src/event2reward/modeling/losses.py) are the reference for each inequality.

In LIBERO indices, expert clips marked `stage2_dynamic_sampling` draw a new query frame from their episode on each visit, always including its first and last frame, and relation rows with `resample_each_epoch` draw new member frames.

`event2reward convert synthetic --output data/synthetic` writes a complete small index dataset with one source of every kind.

## Trajectory data

```json
{
  "schema": "event2reward.trajectories.v1",
  "trajectories": [
    {"uid": "bowls_expert_03", "task": "bowls", "instruction": "stack the bowls", "split": "train",
     "role": "expert", "frame_directory": "bowls/expert_03", "times": [0.0, 0.1, 0.2, "..."],
     "S": 212, "C": 205, "events": []},
    {"uid": "bowls_event_01", "task": "bowls", "instruction": "stack the bowls", "split": "train",
     "role": "event", "video_path": "bowls/event_01.mp4", "times": ["..."], "S": 260, "C": 251,
     "events": [{"type": "progress_stall", "anchor_frame": 80, "end_frame": 121}]}
  ]
}
```

| Field | Meaning |
|---|---|
| `role` | `expert`: a clean successful demonstration. `event`: a trajectory with annotated events. |
| `times` | Timestamp of every frame in seconds, starting at 0. History offsets are measured on this clock. |
| `S` | Last frame used. |
| `C` | Confirmed completion frame, at most `S`, or `null` for a failure. The annotation tool sets it to `S` for successes. |
| `events` | Annotated events, in time order, each with `type`, `anchor_frame` and `end_frame` (and optionally `loss_frame`); see below. |
| `frame_directory` or `video_path` | Frames as `000000.png`, … (`frame_pattern` changes the names), or one video file. |
| `split` | `train`, `val` or `test`; only `train` is used for training. |

Each task needs at least two training experts, because a query's goal image always comes from a different expert. Annotate events only where they happened; the trainer does not infer success or events.

The four event types are the paper's four events, as in the LIBERO annotations, and each becomes a relation of the same kind and margin as in the LIBERO indices. `anchor_frame` is the paper's a and `end_frame` its r, or the last frame for the two events that last until the end:

| `type` | Paper event | Annotation | Trajectory | Relation of the member frames to the anchor |
|---|---|---|---|---|
| `progress_stall` | Progress stall | `anchor_frame`: the stall begins; `end_frame`: progress resumes | any | within 0.02 (kind 5) |
| `regression_and_recovery` | Regression and recovery | `anchor_frame`: the last state before the loss; `loss_frame` (optional): the loss is visible; `end_frame`: the first state where it is restored | any | below both the anchor and the `end_frame` by a margin of min(0.1, half the lower of the two) (kind 6) |
| `no_further_progress` | No further progress | `anchor_frame`: the last state with real progress; `end_frame`: the last frame | `C: null` | no higher (kind 3) |
| `unrecovered_regression` | Unrecovered regression | `anchor_frame`: the last state before the loss; `loss_frame` (optional): the loss is visible; `end_frame`: the last frame | `C: null` | at least 0.1 below (kind 4) |

Member frames lie inside the event. For `regression_and_recovery` and `unrecovered_regression` they start at `loss_frame` or, without one, 0.5 s after the anchor (5 frames at 10 Hz, as in the LIBERO indices).

### From the annotation tool

The [annotation tool](annotation.md) exports one JSON line per annotated video. Convert an export into a trajectory dataset:

```bash
event2reward convert annotations --input annotations.jsonl --project my_project --output data/stack_bowls/dataset.json
```

`--project` is the annotation project folder, which holds the videos. Successes become experts, the other annotated videos event trajectories; each event takes the type of its paper event in the table above. The tool's last frame `S` of a successful video is also its completion frame `C`, so end a success where the task is done; failures get `C: null`. Uncertain videos are left out, and the command lists them. Each video keeps its task, instruction and split from the project.

## Licenses

Third-party videos and annotations keep their original licenses. The project's Apache-2.0 license covers the code and synthetic samples. The quick-start frames and the demonstrations in `examples/annotation/project/` come from the LIBERO datasets, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
