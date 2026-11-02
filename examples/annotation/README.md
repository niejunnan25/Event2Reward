# From annotated videos to a fine-tuned reward model

Five LIBERO-10 videos of one task, *put both the alphabet soup and the tomato sauce in the basket*, annotated with the [annotation tool](../../docs/annotation.md):

| Video | Split | Annotation |
|---|---|---|
| `demo_199`, `demo_008` | train | success |
| `demo_340` | train | regression and recovery: a = 169, the last state before the loss; r = 210, where it is restored |
| `rollout_484` | train | unrecovered regression: a = 261, the last state before the loss; the sauce visibly falls back by frame 270 (l) |
| `rollout_502` | test | success, held out |

`project/` is the annotation project as the tool leaves it, and `annotations.jsonl` its export. Run the steps from the repository root.

### 1. Annotate

The project was created from folders of videos, then annotated in the browser:

```bash
event2reward annotate add my_project --task soup_and_sauce \
  --instruction "put both the alphabet soup and the tomato sauce in the basket" \
  --success videos/success --failure videos/failure
event2reward annotate add my_project --task soup_and_sauce \
  --instruction "put both the alphabet soup and the tomato sauce in the basket" \
  --success videos/test --split test
event2reward annotate serve my_project
```

To see the annotations, open the example project:

```bash
event2reward annotate serve examples/annotation/project
```

`demo_340` among the successful videos shows the points a and r; `rollout_484` among the failed videos shows a and l. Saving changes writes to `project/annotations.json`; work on a copy to keep the example as it is.

### 2. Export and convert

```bash
event2reward annotate export examples/annotation/project --output annotations.jsonl   # = examples/annotation/annotations.jsonl
event2reward convert annotations --input examples/annotation/annotations.jsonl \
  --project examples/annotation/project --output data/example/dataset.json
```

The conversion reports one task with three successes (one of them in the test split) and two event trajectories, a regression and recovery and an unrecovered regression. `data/example/dataset.json` lists the four training trajectories ([format](../../docs/data.md#trajectory-data)).

### 3. Fine-tune

[`finetune.yaml`](finetune.yaml) is an example configuration: 500 steps of 80 clips on one GPU, half frames of successes with absolute progress, half event groups, the two events taking turns.

```bash
bash scripts/download_models.sh Stage2
event2reward train examples/annotation/finetune.yaml --check
event2reward train examples/annotation/finetune.yaml          # about 1 hour on one H200
event2reward export --checkpoint outputs/example/checkpoint-000500 --output models/example
```

`--check` prints the plan without loading the model: 4,000 microbatches, 20,000 expert queries and 20,000 relation queries. Training writes `outputs/example/metrics.jsonl`, with the progress and relation losses every 10 steps.

### 4. Compare

```bash
pip install matplotlib
python examples/annotation/compare.py --before models/Stage2 --after models/example \
  --project examples/annotation/project --output outputs/example/curves.png
```

Both models score every frame of each held-out video, with the last frame (S) of a successful training video as the goal image. The script writes the predictions as JSON and plots them.

Four training videos are enough to run the pipeline, not to train a reward model you would use: the paper's Franka models saw 5 successful and 5 suboptimal trajectories per task. Load the exported model like any other ([inference](../../docs/inference.md)):

```python
from event2reward import Event2Reward
model = Event2Reward.from_pretrained("models/example")
rows = model.predict_trajectory(frames=frames, task="put both the alphabet soup and the tomato sauce in the basket",
                                goal=goal_image, timestamps=timestamps)
```

Fine-tuned models choose their history frames by time, so `predict_trajectory` needs the frame `timestamps`.
