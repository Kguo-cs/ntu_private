"""Load only sibling numerical files. No AST exec, external repo or network.

`repo` is accepted for old SMART callers, but is never read or imported.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from types import SimpleNamespace
if __package__:
    from . import sd_metric_helpers as metrics
    from . import sd_lane_helpers as lanes
else:
    import sd_metric_helpers as metrics
    import sd_lane_helpers as lanes


def load_official_backend(repo=None):
    del repo
    # Optional exact one-time freeze of the user's own official revision.
    # This is a sibling Python import, not an external-repository import.
    frozen_path = Path(__file__).with_name("sd_frozen_official.py")
    if frozen_path.is_file():
        if __package__:
            from . import sd_frozen_official as numerical
        else:
            import sd_frozen_official as numerical
        paths = (frozen_path, Path(__file__))
    else:
        numerical = metrics
        paths = (Path(metrics.__file__), Path(lanes.__file__), Path(__file__))
    values = {k: v for k, v in vars(numerical).items() if not k.startswith('_')}
    values.update(source_root=str(Path(__file__).resolve().parent),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return SimpleNamespace(**values)
