"""Read-only completion validation for the explicitly authenticated GC-off loader.

The original controller stays byte-for-byte frozen. Raw log bytes remain the
artifact: only its text view hides the single authenticated runtime success line.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_LEGACY_VERIFY = None
PREFIX = "RUNTIME_LOADER_SUCCESS "


def bind_legacy_verifier(verifier):
    """Capture the frozen callable before a bridge adapts that module in memory."""
    global _LEGACY_VERIFY
    if _LEGACY_VERIFY is None:
        _LEGACY_VERIFY = verifier


class LogView:
    def __init__(self, path, text):
        self.path = Path(path)
        self.text = text

    def read_text(self, *args, **kwargs):
        return self.text

    def read_bytes(self):
        return self.path.read_bytes()

    def __str__(self):
        return str(self.path)

    def __fspath__(self):
        return str(self.path)


def authenticated_log_view(folder, log_path):
    folder, log_path = Path(folder), Path(log_path)
    raw = log_path.read_text()
    lines = raw.splitlines()
    if not lines:
        raise RuntimeError("Empty completion log")
    if lines[-1].strip() == "DONE " + str(folder):
        execution_path = folder / "execution.json"
        if execution_path.exists():
            execution = json.loads(execution_path.read_text())
            sources = execution.get("runtime_recovery", {}).get("runtime_loader_source_hashes", {})
            if sources:
                raise RuntimeError("Cached-loader stage lacks its authenticated terminal runtime policy")
        return log_path, None
    if len(lines) < 2 or not lines[-1].startswith(PREFIX) or lines[-2].strip() != "DONE " + str(folder):
        raise RuntimeError("Completion log must end in exact DONE plus one runtime success suffix")
    if sum(line.startswith(PREFIX) for line in lines) != 1:
        raise RuntimeError("Duplicate runtime success suffix")
    try:
        suffix = json.loads(lines[-1][len(PREFIX):])
        policy_path = folder / "runtime_loader_policy.json"
        policy = json.loads(policy_path.read_text())
        execution = json.loads((folder / "execution.json").read_text())
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError("Runtime suffix is not backed by readable actual artifacts") from error
    if policy != suffix:
        raise RuntimeError("Runtime policy artifact and terminal JSON differ")
    if execution.get("status") not in ("completed", "success") or execution.get("returncode") != 0:
        raise RuntimeError("Runtime stage did not actually finish with return code zero")
    expected = execution.get("runtime_recovery", {}).get("runtime_loader_source_hashes", {})
    if not expected or policy.get("source_hashes") != expected:
        raise RuntimeError("Runtime loader source hashes differ from execution provenance")
    for path, digest in expected.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Actual runtime loader source changed: " + path)
    requirements = {
        "policy": "public_pickle_module_class_cache_and_full_cyclic_gc_deferral",
        "automatic_gc_enabled": False,
        "manual_gc_collect": "disabled",
        "young_gc_interval": 0,
        "tensor_and_data_conversion": False,
        "objective_or_model_change": False,
        "only_explicit_weights_only_false_loads": True,
        "custom_pickle_module_preserved": True,
        "success_exit": "flush_then_os._exit(0)",
    }
    for key, value in requirements.items():
        if policy.get(key) != value or type(policy.get(key)) is not type(value):
            raise RuntimeError("Runtime policy guard differs: " + key)
    stats = policy.get("class_cache", {})
    created, reuses = stats.get("classes_created"), stats.get("class_reuses")
    if (type(created) is not int or not 1 <= created <= 2
            or type(reuses) is not int or reuses < 0):
        raise RuntimeError("Actual class cache statistics are invalid")
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol.get("young_gc_interval") != 0 or protocol.get("defer_cyclic_gc") is not True:
        raise RuntimeError("Runner did not use the certified GC-off runtime")
    return LogView(log_path, "\n".join(lines[:-1]) + "\n"), policy_path


def verify_completion(folder, log_path, spec):
    """Validate actual rows using the frozen validator and preserve raw hashes."""
    view, policy_path = authenticated_log_view(folder, log_path)
    if _LEGACY_VERIFY is None:
        from tools import continue_overnight_init_diffusion as controller
        if controller.verify_completion is verify_completion:
            raise RuntimeError("Bridge must bind the original verifier before installing its adapter")
        bind_legacy_verifier(controller.verify_completion)
    rows, artifacts = _LEGACY_VERIFY(Path(folder), view, spec)
    if policy_path is not None:
        artifacts["runtime_loader_policy"] = {"path": str(policy_path),
            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest()}
    return rows, artifacts
