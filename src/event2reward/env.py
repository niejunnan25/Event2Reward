"""`event2reward env`: report versions, kernels and GPU readiness for inference and training."""
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_import(name):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(name)
        return {"available": True, "error": None}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def nvcc_info():
    home = os.environ.get("CUDA_HOME")
    path = str(Path(home) / "bin/nvcc") if home else shutil.which("nvcc")
    output = command_output([path, "--version"]) if path else None
    match = re.search(r"release\s+(\d+)\.(\d+)", output or "")
    release = tuple(map(int, match.groups())) if match else None
    driver_output = command_output(["nvidia-smi"])
    driver_match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", driver_output or "")
    driver_release = tuple(map(int, driver_match.groups())) if driver_match else None
    return {"path": path, "version": ".".join(map(str, release)) if release else None,
            "at_least_12_8": release is not None and release >= (12, 8),
            "driver_cuda_max": ".".join(map(str, driver_release)) if driver_release else None,
            "within_driver_limit": (release <= driver_release
                                    if release and driver_release else None)}


def compiler_problem(info):
    if not info["at_least_12_8"]:
        return "nvcc >= 12.8 is required"
    if info.get("within_driver_limit") is None:
        return "cannot verify the driver's CUDA limit; check nvidia-smi"
    if not info["within_driver_limit"]:
        return (f"nvcc {info['version']} exceeds the driver's CUDA {info['driver_cuda_max']} limit; "
                "select a compatible CUDA Toolkit with CUDA_HOME")
    return None


def hopper_workaround_required(triton):
    from packaging.version import InvalidVersion, Version
    try:
        return triton is None or Version(triton) < Version("3.7.1")
    except InvalidVersion:
        return True


def tilelang_problem(import_status):
    if not import_status["available"]:
        return "tilelang==0.1.9 is required"
    # This pair imports successfully but fails when TVM first loads after model
    # execution. Do not reject other versions without compatibility evidence.
    if version("tilelang") == "0.1.9" and version("apache-tvm-ffi") == "0.1.15.post0":
        return ("tilelang 0.1.9 with apache-tvm-ffi 0.1.15.post0 can fail during backward; "
                "install tilelang==0.1.9 apache-tvm-ffi==0.1.10")
    return None


def check_hopper_training():
    """Fail before reading datasets for the known Hopper backward-kernel issue."""
    import torch
    if not torch.cuda.is_available():
        return
    hopper = any(torch.cuda.get_device_capability(i)[0] == 9
                 for i in range(torch.cuda.device_count()))
    if not hopper:
        return
    if not hopper_workaround_required(version("triton")):
        return
    problems = []
    problem = tilelang_problem(probe_import("tilelang"))
    if problem:
        problems.append(problem)
    problem = compiler_problem(nvcc_info())
    if problem:
        problems.append(problem)
    if problems:
        raise RuntimeError("Hopper training: " + "; ".join(problems)
                           + "; see docs/installation.md#training")


def collect_environment():
    result = {"python": platform.python_version(), "os": platform.platform(),
              "torch": version("torch"), "transformers": version("transformers"),
              "triton": version("triton"), "cuda_runtime": None, "cuda_available": False,
              "gpus": [], "driver": None, "errors": []}
    try:
        import torch
        result.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                result["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                       "capability": list(torch.cuda.get_device_capability(index))})
    except Exception as exc:
        result["errors"].append(str(exc))
    if shutil.which("nvidia-smi"):
        result["driver"] = command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    for name in ("fla", "causal_conv1d"):
        result[name] = probe_import(name)
    result["fast_path_available"] = False
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
        result["fast_path_available"] = bool(module.is_fast_path_available)
    except Exception as exc:
        result["errors"].append(str(exc))
    missing = []
    if not result["cuda_available"]:
        missing.append("CUDA GPU / compatible NVIDIA driver (docs/installation.md#requirements)")
    if not result["torch"] or not result["transformers"] or result["errors"]:
        missing.append("working PyTorch and Transformers installation (docs/installation.md#inference)")
    result["inference_ready"] = not missing
    adaptation_missing = []
    result["hopper"] = any(g["capability"][0] == 9 for g in result["gpus"])
    if result["hopper"]:
        result["tilelang"] = probe_import("tilelang")
        result["tilelang_version"] = version("tilelang")
        result["apache_tvm_ffi"] = version("apache-tvm-ffi")
        result["nvcc"] = nvcc_info()
        if hopper_workaround_required(result["triton"]):
            problem = tilelang_problem(result["tilelang"])
            if problem:
                adaptation_missing.append(problem + " (docs/installation.md#training)")
            problem = compiler_problem(result["nvcc"])
            if problem:
                adaptation_missing.append(problem + " (docs/installation.md#training)")
    result["adaptation_missing"] = adaptation_missing
    result["adaptation_ready"] = not missing and not adaptation_missing
    result["missing"] = missing + adaptation_missing
    if missing:
        result["conclusion"] = "missing: " + "; ".join(result["missing"])
    elif adaptation_missing:
        result["conclusion"] = "ready for inference; missing for adaptation: " + "; ".join(adaptation_missing)
    else:
        result["conclusion"] = "ready"
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="event2reward env")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = collect_environment()
    except Exception as exc:
        result = {"conclusion": f"missing: environment diagnostics failed: {exc}; see docs/installation.md"}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            if key != "conclusion":
                print(f"{key}: {value}")
        print(result["conclusion"])
    return 0


if __name__ == "__main__":
    main()
