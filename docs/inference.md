# Inference

## Input and output

Each `Observation` contains task text, the episode start image, an **independent successful goal image for the same task**, and exactly five causal history frames in chronological order. Supply PIL images, RGB arrays, or image paths. Images are center-cropped and resized to 224 × 224. Use an external camera consistently; the examples do not use the wrist camera.

For Stage2 and models fine-tuned from it, sample observations at 10 Hz. For query index q, use `[max(0, q-4), ..., q]`: early histories repeat the first frame. Never use future observations or substitute the evaluated video's final frame for the independent goal.

`progress` is the expectation over 101 progress bins in [0, 1]. It is not a success probability, can decrease, and may underestimate a completed task or overestimate a failed one. Outputs also contain progress variance/entropy and auxiliary trend probabilities in `growth`, `stable`, `regression` order. No universal success threshold is supplied.

Stage1 is primarily the initialization of Stage2 and uses a 0.25-second training history interval. The released loader always center-crops. Stage1 training instead squared frames per source dataset (center crop, letterbox, or a fixed crop), so to match its training inputs, square non-square camera frames before passing them in.

## Single and batch prediction

```python
from event2reward import Event2Reward, Observation
model = Event2Reward.from_pretrained("niejunnan25/Event2Reward-Qwen3.5-4B-Stage2")
observation = Observation("place the bowl on the plate", "start.png", "goal.png",
                          [f"h{i}.png" for i in range(5)])
result = model.predict([observation], include_distribution=True)
batch_result = model.predict([observation, observation])
```

Batching requires additional memory. `include_distribution=True` adds the full 101-bin probability vector. `model.prepare(observations)` exposes the model inputs for numerical comparisons.

## Trajectories and MP4

```python
results = model.predict_trajectory(
    frames=frames, task="place the bowl on the plate", goal="goal.png",
    batch_size=1, stride=1, timestamps=timestamps,
)
```

`frames` must already be sampled at the observation interval; `timestamps` are strictly increasing seconds, with one timestamp per frame. Models fine-tuned with time-based histories (the Franka recipes) choose their five history frames by `timestamps`, which are then required. Optional `query_indices` selects queries without changing the history. Each result includes its query/history indices and timestamp. To sample an MP4 at 10 Hz, take the latest frame at or before each 0.1 s tick (needs `pip install av`):

```python
import av

def frames_at_10hz(path):
    with av.open(str(path)) as video:
        decoded = [(float(f.pts * f.time_base), f.to_image()) for f in video.decode(video=0)]
    start, end = decoded[0][0], decoded[-1][0]
    frames, timestamps, j = [], [], 0
    for i in range(int((end - start) / 0.1) + 1):
        while j + 1 < len(decoded) and decoded[j + 1][0] - start <= i * 0.1 + 1e-9:
            j += 1
        frames.append(decoded[j][1])
        timestamps.append(i * 0.1)
    return frames, timestamps

frames, timestamps = frames_at_10hz("episode.mp4")
results = model.predict_trajectory(frames=frames, task="place the bowl on the plate", goal="goal.png",
                                   timestamps=timestamps)
```

## CLI

```bash
event2reward infer --bundle niejunnan25/Event2Reward-Qwen3.5-4B-Stage2 --observations examples/quickstart/observations.json --output outputs/prediction.json
```

The observations JSON is a list of objects with `task`, `start`, `goal`, and `history`. Image paths resolve relative to that JSON file. `--bundle` also accepts a local model directory; `--device` defaults to `cuda:0`.

## Save and export

```python
model.save_pretrained("models/saved-copy")
reloaded = Event2Reward.from_pretrained("models/saved-copy")
```

The destination must not exist. This saves the current parameters and loading assets. To turn a training checkpoint into such a bundle, use `event2reward export` ([checkpoints](training.md#outputs-and-checkpoints)).
