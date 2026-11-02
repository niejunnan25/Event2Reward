# Examples

`quickstart/` holds one query for the [quick start](../README.md#quick-start): the start frame, five history frames ending 6.0 s into a successful LIBERO-Spatial episode (10 Hz, oldest first), and a goal frame from another successful episode of the same task. All images are lossless PNGs of the decoded frames.

`quickstart/expected_output.json` is the Stage2 output for this query with the [reference environment](../docs/installation.md#reference-environment):

```bash
event2reward infer --bundle niejunnan25/Event2Reward-Qwen3.5-4B-Stage2 \
  --observations examples/quickstart/observations.json --output outputs/prediction.json
```

`annotation/` fine-tunes Stage2 on annotated videos and compares the result with Stage2 on a held-out video; [its README](annotation/README.md) walks through every step.

Interactive examples of whole episodes, in simulation and on real robots, are on the [project page](https://niejunnan25.github.io/Event2Reward/).

The quick-start frames and the three demonstrations in `annotation/project/` come from the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) datasets (Liu et al., 2023), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The two other videos are policy rollouts we recorded in the LIBERO simulator.
