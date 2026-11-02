import json
from types import SimpleNamespace
import pytest
from event2reward import env


def test_json_diagnostic_returns_zero_for_incomplete_environment(monkeypatch, capsys):
    monkeypatch.setattr(env, "collect_environment", lambda: {"conclusion": "missing: CUDA"})
    assert env.main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out)["conclusion"] == "missing: CUDA"


def test_diagnostic_failure_is_reported_not_raised(monkeypatch, capsys):
    def fail():
        raise RuntimeError("driver unavailable")
    monkeypatch.setattr(env, "collect_environment", fail)
    assert env.main(["--json"]) == 0
    assert "driver unavailable" in json.loads(capsys.readouterr().out)["conclusion"]


def test_nvcc_home_is_authoritative(monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/compiler")
    monkeypatch.setattr(env, "command_output", lambda args: "Cuda compilation tools, release 12.8, V12.8.93")
    result = env.nvcc_info()
    assert result["path"] == "/compiler/bin/nvcc"
    assert result["at_least_12_8"]
    monkeypatch.setattr(env, "command_output", lambda args: "release 12.4, V12.4.131")
    assert not env.nvcc_info()["at_least_12_8"]


def test_optional_import_failure_is_diagnostic(monkeypatch):
    def fail(name):
        raise ImportError("optional kernel missing")
    monkeypatch.setattr(env.importlib, "import_module", fail)
    assert env.probe_import("fla")["available"] is False


@pytest.mark.parametrize("triton,adaptation_ready", [("3.4.0", False), ("3.7.1", True)])
def test_hopper_inference_does_not_need_compiler(monkeypatch, triton, adaptation_ready):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "H200")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (9, 0))
    monkeypatch.setattr(env, "version", lambda name: triton if name == "triton" else "installed")
    monkeypatch.setattr(env, "probe_import", lambda name: {"available": False, "error": "not installed"})
    monkeypatch.setattr(env, "nvcc_info", lambda: {"at_least_12_8": False})
    monkeypatch.setattr(env, "command_output", lambda args: "570.195.03")
    monkeypatch.setattr(env.importlib, "import_module", lambda name: SimpleNamespace(is_fast_path_available=False))
    result = env.collect_environment()
    assert result["inference_ready"]
    assert result["adaptation_ready"] == adaptation_ready
    assert result["conclusion"].startswith("ready")
    if not adaptation_ready:
        assert "missing for adaptation" in result["conclusion"]


@pytest.mark.parametrize('compiler,driver,problem', [
    ('12.8', '12.8', None), ('12.8', '13.0', None),
    ('13.0', '12.8', 'exceeds'), ('12.4', '12.8', '>= 12.8'),
    ('12.8', None, 'cannot verify'),
])
def test_compiler_must_fit_driver(monkeypatch, compiler, driver, problem):
    monkeypatch.setenv('CUDA_HOME', '/compiler')
    def output(args):
        if args[0] == 'nvidia-smi':
            return f'CUDA Version: {driver}' if driver else None
        return f'Cuda compilation tools, release {compiler}, V{compiler}.0'
    monkeypatch.setattr(env, 'command_output', output)
    result = env.compiler_problem(env.nvcc_info())
    assert result is None if problem is None else problem in result


def test_training_rejects_nvcc_newer_than_driver(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (9, 0))
    monkeypatch.setattr(env, 'version', lambda _: '3.4.0')
    monkeypatch.setattr(env, 'probe_import', lambda _: {'available': True})
    monkeypatch.setattr(env, 'nvcc_info', lambda: {
        'at_least_12_8': True, 'within_driver_limit': False,
        'version': '13.0', 'driver_cuda_max': '12.8'})
    with pytest.raises(RuntimeError, match="nvcc 13.0 exceeds the driver's CUDA 12.8"):
        env.check_hopper_training()


def test_env_entrypoint_survives_broken_torch_import():
    import subprocess
    import sys
    from pathlib import Path
    script = '''
import importlib.abc
import runpy
import sys
class BrokenTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch intentionally unavailable for diagnostic test")
sys.meta_path.insert(0, BrokenTorch())
sys.argv = ["event2reward", "env", "--json"]
runpy.run_module("event2reward", run_name="__main__")
'''
    result = subprocess.run([sys.executable, '-c', script],
                            cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    diagnostic = json.loads(result.stdout)
    assert not diagnostic['inference_ready']
    assert any('torch intentionally unavailable' in error for error in diagnostic['errors'])


@pytest.mark.parametrize('ffi,ready', [('0.1.10', True), ('0.1.15.post0', False)])
def test_known_tilelang_ffi_conflict_is_reported_and_rejected(monkeypatch, ffi, ready):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda _: 'H200')
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda _: (9, 0))
    versions = {'triton': '3.4.0', 'tilelang': '0.1.9', 'apache-tvm-ffi': ffi}
    monkeypatch.setattr(env, 'version', lambda name: versions.get(name, 'installed'))
    # This conflict appears only after model work, so a successful import is insufficient.
    monkeypatch.setattr(env, 'probe_import', lambda _: {'available': True, 'error': None})
    monkeypatch.setattr(env, 'nvcc_info', lambda: {
        'version': '12.8', 'at_least_12_8': True,
        'driver_cuda_max': '12.8', 'within_driver_limit': True})
    monkeypatch.setattr(env, 'command_output', lambda _: '570.195.03')
    monkeypatch.setattr(env.importlib, 'import_module',
                        lambda _: SimpleNamespace(is_fast_path_available=True))
    diagnostic = env.collect_environment()
    assert diagnostic['inference_ready']
    assert diagnostic['adaptation_ready'] is ready
    if ready:
        env.check_hopper_training()
    else:
        assert 'apache-tvm-ffi==0.1.10' in diagnostic['conclusion']
        with pytest.raises(RuntimeError, match='apache-tvm-ffi==0.1.10'):
            env.check_hopper_training()
