"""Score the test videos of an annotation project before and after fine-tuning.

    python examples/annotation/compare.py --before models/Stage2 --after models/example \
      --project examples/annotation/project --output examples/annotation/curves.png

For every video of one split (test by default), both models score each frame
with its causal history; the goal image is the last frame (S) of a clean
successful training video of the same task. Writes the predictions as JSON and,
with matplotlib installed, a plot next to them.
"""
import argparse
import json
from pathlib import Path

import av

from event2reward import Event2Reward
from event2reward.annotation.project import read_annotations, read_project


def frames(path):
    with av.open(str(path)) as video:
        return [frame.to_image() for frame in video.decode(video=0)]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--before", required=True, help="model before fine-tuning, e.g. models/Stage2")
    p.add_argument("--after", required=True, help="the exported fine-tuned model")
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True, help="the plot (.png); predictions go to the same name with .json")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    root = Path(args.project)
    videos, annotations = read_project(root)["videos"], read_annotations(root)
    goals = {}
    for v in videos:
        a = annotations.get(v["uid"])
        if v["split"] == "train" and a and a["label"] == "success":
            goals.setdefault(v["task"], frames(root / v["video"])[a["final_frame"]])
    tests = [v for v in videos if v["split"] == args.split]
    if not tests:
        raise SystemExit(f"the project has no {args.split} videos")
    results = {}
    for name in ("before", "after"):
        model = Event2Reward.from_pretrained(getattr(args, name), device=args.device)
        for v in tests:
            rows = model.predict_trajectory(frames=frames(root / v["video"]), task=v["instruction"],
                                            goal=goals[v["task"]], timestamps=v["times"])
            results.setdefault(v["uid"], {"times": v["times"], "label": annotations.get(v["uid"], {}).get("label")})
            results[v["uid"]][name] = [round(r["progress"], 4) for r in rows]
        del model
    output = Path(args.output)
    output.with_suffix(".json").write_text(json.dumps(results, indent=1) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"wrote {output.with_suffix('.json')}; pip install matplotlib for the plot")
        return
    fig, axes = plt.subplots(len(results), 1, figsize=(7, 2.6 * len(results)), squeeze=False)
    for ax, (uid, r) in zip(axes[:, 0], results.items()):
        ax.plot(r["times"], r["before"], color="#8c8c8c", label="Stage2 (before)")
        ax.plot(r["times"], r["after"], color="#d9822b", label="fine-tuned (after)")
        ax.set(title=f"{uid} ({r['label']})", xlabel="time (s)", ylabel="progress", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"wrote {output} and {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
