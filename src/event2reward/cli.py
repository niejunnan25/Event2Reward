"""`event2reward` command line: infer, annotate, convert, train, export, env."""
import argparse
import json
from pathlib import Path


def _parse_roots(values):
    roots = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in roots:
            raise SystemExit(f"--media-root expects DATASET_ID=PATH once per dataset: {value}")
        roots[name] = path
    return roots


def _train(args):
    from .train import check, load_config, train
    config = load_config(args.config, args.set)
    if args.gpus > 1:
        import subprocess
        import sys
        command = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu",
                   "--num_processes", str(args.gpus), "--num_machines", "1",
                   "--mixed_precision", "bf16" if config["training"]["precision"] == "bf16" else "no",
                   "--dynamo_backend", "no",
                   "-m", "event2reward", "train", args.config, *sum((["--set", s] for s in args.set), [])]
        raise SystemExit(subprocess.run(command).returncode)
    if args.check:
        print(json.dumps(check(config), indent=2))
    else:
        print(train(config))


def _infer(args):
    from .inference import Event2Reward, Observation
    path = Path(args.observations).resolve()
    rows = json.loads(path.read_text())
    image = lambda p: p if Path(p).is_absolute() else str(path.parent / p)
    observations = [Observation(r["task"], image(r["start"]), image(r["goal"]), [image(x) for x in r["history"]])
                    for r in rows]
    model = Event2Reward.from_pretrained(args.bundle, device=args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.predict(observations), indent=2) + "\n")


def _annotate(args):
    from .annotation import project
    if args.action == "add":
        if not (args.success or args.failure):
            raise SystemExit("give --success and/or --failure folders")
        for mode in ("success", "failure"):
            folder = getattr(args, mode)
            if folder:
                added = project.add_videos(args.project, task=args.task, instruction=args.instruction, mode=mode,
                                           videos=folder, split=args.split, reencode=args.reencode)
                print(f"{args.task}: added {len(added)} {mode} videos")
    elif args.action == "serve":
        from .annotation.server import serve
        serve(args.project, port=args.port, open_browser=not args.no_browser)
    else:
        print(f"{project.write_export(args.project, args.output)} annotated videos -> {args.output}")


def _convert(args):
    if args.dataset == "annotations":
        from .data.converters.annotations import convert_annotations
        print(json.dumps(convert_annotations(args.input, args.project, args.output), indent=2, ensure_ascii=False))
    elif args.dataset == "libero":
        from .data.converters.libero import write_dataset_manifest
        print(write_dataset_manifest(args.indices, args.output, _parse_roots(args.media_root)))
    else:
        from .data.converters.synthetic import make_example
        print(make_example(args.output))


def main(argv=None):
    p = argparse.ArgumentParser(prog="event2reward")
    sub = p.add_subparsers(dest="command", required=True)

    infer = sub.add_parser("infer", help="score observations listed in a JSON file")
    infer.add_argument("--bundle", required=True, help="local bundle directory or Hugging Face repo id")
    infer.add_argument("--observations", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cuda:0")

    train = sub.add_parser("train", help="fine-tune from a YAML config")
    train.add_argument("config")
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                       help="override a config value, e.g. --set training.steps=10")
    train.add_argument("--check", action="store_true", help="validate config and data without training")
    train.add_argument("--gpus", type=int, default=1, help="data-parallel processes (index data only)")

    export = sub.add_parser("export", help="turn a training checkpoint into a model bundle")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--model-path", help="base bundle, if it moved since training")

    annotate = sub.add_parser("annotate", help="annotate videos for fine-tuning")
    actions = annotate.add_subparsers(dest="action", required=True)
    add = actions.add_parser("add", help="add one task's successful and/or failed videos to a project")
    add.add_argument("project", help="project folder, created if needed")
    add.add_argument("--task", required=True, help="short task name, e.g. stack_bowls")
    add.add_argument("--instruction", required=True, help="the task instruction given to the model")
    add.add_argument("--success", help="folder of successful videos")
    add.add_argument("--failure", help="folder of failed videos")
    add.add_argument("--split", default="train", choices=["train", "val", "test"])
    add.add_argument("--reencode", action="store_true",
                     help="re-encode as H.264 with a keyframe every second, for fast seeking in the browser")
    serve = actions.add_parser("serve", help="open the annotation page")
    serve.add_argument("project")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    export_annotations = actions.add_parser("export", help="write the saved annotations as JSONL")
    export_annotations.add_argument("project")
    export_annotations.add_argument("--output", required=True)

    convert = sub.add_parser("convert", help="prepare training data")
    kinds = convert.add_subparsers(dest="dataset", required=True)
    annotations = kinds.add_parser("annotations", help="trajectories dataset.json from an annotation export")
    annotations.add_argument("--input", required=True, help="the exported JSONL")
    annotations.add_argument("--project", required=True, help="the annotation project folder (its videos)")
    annotations.add_argument("--output", required=True)
    libero = kinds.add_parser("libero", help="dataset.json for annotated LIBERO relation indices")
    libero.add_argument("--indices", required=True, help="directory with train_*.jsonl")
    libero.add_argument("--media-root", action="append", required=True, metavar="DATASET_ID=PATH")
    libero.add_argument("--output", required=True)
    synthetic = kinds.add_parser("synthetic", help="a tiny synthetic index dataset for pipeline checks")
    synthetic.add_argument("--output", required=True)

    env = sub.add_parser("env", help="report the installed environment")
    env.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    if args.command == "train":
        _train(args)
    elif args.command == "infer":
        _infer(args)
    elif args.command == "export":
        from .train.checkpoint import export_bundle
        print(export_bundle(args.checkpoint, args.output, model_path=args.model_path))
    elif args.command == "annotate":
        _annotate(args)
    elif args.command == "convert":
        _convert(args)
    else:
        from .env import main as env_main
        env_main(["--json"] if args.json else [])


if __name__ == "__main__":
    main()
