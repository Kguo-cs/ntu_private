# recursive_spectral_norm.py
import re
import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import spectral_norm as spectral_norm_param
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    _PARAM_API = True
except Exception:
    from torch.nn.utils import spectral_norm as spectral_norm_wrap, remove_spectral_norm as remove_sn_legacy
    _PARAM_API = False


# ---- Defaults: what to skip/apply ----------------------------------------------------
_DEFAULT_EXCLUDED_TYPES = (
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.Embedding, nn.EmbeddingBag,
)
_DEFAULT_INCLUDED_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)

def _has_weight_param(m: nn.Module, name: str = "weight") -> bool:
    p = getattr(m, name, None)
    return isinstance(p, torch.nn.Parameter) and p.ndim >= 2

def _already_spectral_normed(m: nn.Module, name: str = "weight") -> bool:
    if _PARAM_API:
        return is_parametrized(m) and hasattr(getattr(m, "parametrizations", object()), name)
    else:
        # best-effort check on legacy API
        return hasattr(m, name + "_orig") and hasattr(m, name + "_u") and hasattr(m, name + "_v")

# ---- Core: add/remove recursively ----------------------------------------------------
def add_spectral_norm_recursively(
    model: nn.Module,
    *,
    include_types=_DEFAULT_INCLUDED_TYPES,
    exclude_types=_DEFAULT_EXCLUDED_TYPES,
    include_name_regex: str | None = None,   # e.g., r"(q_proj|k_proj|v_proj|out_proj)"
    exclude_name_regex: str | None = None,   # e.g., r"(embedding|ln|norm)"
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dry_run: bool = False,
) -> list[str]:
    """
    Walks the entire module tree and adds spectral norm to eligible submodules.
    Returns fully-qualified parameter names that were (or would be) wrapped.
    """
    wrapped: list[str] = []
    inc_pat = re.compile(include_name_regex) if include_name_regex else None
    exc_pat = re.compile(exclude_name_regex) if exclude_name_regex else None

    for module_name, module in model.named_modules():
        if module_name == "":  # skip the root's own param unless it's directly one of the include_types
            pass

        # Type gating
        type_ok = isinstance(module, include_types)
        type_blocked = isinstance(module, exclude_types)

        # Name gating
        name_ok = (inc_pat.search(module_name) is not None) if inc_pat else True
        name_blocked = (exc_pat.search(module_name) is not None) if exc_pat else False

        if not type_ok or type_blocked or not name_ok or name_blocked:
            continue
        if not _has_weight_param(module, name=name):
            continue
        if _already_spectral_normed(module, name=name):
            continue

        fqname = f"{module_name}.{name}" if module_name else name
        if dry_run:
            wrapped.append(fqname)
            continue

        try:
            if _PARAM_API:
                spectral_norm_param(module, name=name, n_power_iterations=n_power_iterations, eps=eps)
            else:
                spectral_norm_wrap(module, name=name, n_power_iterations=n_power_iterations, eps=eps)
            wrapped.append(fqname)
        except Exception as e:
            print(f"[SN] Skip {fqname}: {e}")

    return wrapped


def remove_spectral_norm_recursively(
    model: nn.Module,
    *,
    name: str = "weight",
) -> list[str]:
    """Removes spectral norm wherever present. Returns fully-qualified names removed."""
    removed: list[str] = []
    for module_name, module in model.named_modules():
        fqname = f"{module_name}.{name}" if module_name else name
        try:
            if _PARAM_API:
                if _already_spectral_normed(module, name=name):
                    remove_parametrizations(module, name, leave_parametrized=False)
                    removed.append(fqname)
            else:
                # Legacy remove tries; if not present it raises -> ignore
                remove_sn_legacy(module, name=name)
                removed.append(fqname)
        except Exception:
            pass
    return removed
