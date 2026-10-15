"""Audit raw angular-velocity support, optionally applying a documented bound."""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--heading-bound-mode', choices=['audit', 'clip'], required=True)
    p.add_argument('--heading-bound-report', type=Path, required=True)
    a, remaining = p.parse_known_args()
    import torch
    from src.smart.model.smart_gail import SMART_GAIL
    from tools.bounded_heading_velocity_20261015 import install_inference_bound
    from tools import run_audited_capped_init_eval_20261015 as native
    original = SMART_GAIL.__init__

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        before = torch.get_rng_state().clone()
        install_inference_bound(self.encoder.init_decoder.G1, apply=a.heading_bound_mode == 'clip')
        assert torch.equal(before, torch.get_rng_state())

    SMART_GAIL.__init__ = init
    try:
        sys.argv = [sys.argv[0], *remaining]
        native.main()
        flow = native.MODEL.encoder.init_decoder.G1
        rows = flow._heading_bound_audit
        assert rows
        clipped = sum(int(r[0].cpu()) for r in rows)
        total = sum(int(r[1].cpu()) for r in rows)
        maximum = max(float(r[2].cpu()) for r in rows)
        report = dict(status='NATIVE_GPU_SUPPORT_PROBE_COMPLETE', mode=a.heading_bound_mode,
            bound_radians=__import__('math').pi, scalar_predictions=total,
            predictions_outside_support=clipped, outside_fraction=clipped / max(total, 1),
            maximum_absolute_raw_omega=maximum, native_prediction_calls=len(rows),
            excludes_conditioned_heading_rows=True, parameters_changed=False,
            training_objective_changed=False, quality_improvement_unproven=True,
            actual_EMA_updates=native.MODEL.encoder.init_decoder.ema.num_updates,
            source_hashes={str(x.relative_to(ROOT)): hashlib.sha256(x.read_bytes()).hexdigest()
                           for x in [Path(__file__), ROOT / 'tools/bounded_heading_velocity_20261015.py']})
        a.heading_bound_report.parent.mkdir(parents=True, exist_ok=True)
        a.heading_bound_report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        SMART_GAIL.__init__ = original


if __name__ == '__main__':
    main()
