"""Update audited overnight comparisons without touching active training code."""
from __future__ import annotations
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_init_overnight_20261014 import ART, METRICS, FOCUS, audit_rows
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now
from tools.run_init_diffusion_evening_20261013 import score

END = datetime.fromisoformat('2026-10-14T10:00:00+08:00').timestamp()
CONTROL_FILES = [
    ROOT / 'output/reports/init_diffusion_evening_20261013/final_long/rebased/sampler_merge/observations.json',
    ROOT / 'logs/control657_paired_20261012/train4817/summary.json',
    ROOT / 'logs/control657_paired_20261012/train5817/summary.json',
]


def refresh():
    source = ART / 'observations.json'
    rows = read(source)
    audit = audit_rows(rows)
    audit.update(observations_sha256=sha(source), auditor_sha256=sha(ROOT / 'tools/audit_init_overnight_20261014.py'))
    save(ART / 'native_metric_audit.json', audit)
    controls = {}
    for path in CONTROL_FILES:
        for row in read(path):
            if row['variant'] not in ('fresh657', 'control657') or row['num_samples'] != 4096:
                continue
            # Prefer the random4096 control over sorted4096 cohort probes.
            key = (row['seed'], row['budget'], row['eval_seed'], row['eval_manifest_sha256'])
            controls.setdefault(key, row)
    result = []
    for arm, seed, budget, scenes in sorted({(r['variant'], r['seed'], r['budget'], r['num_samples']) for r in rows}):
        selected = [r for r in rows if (r['variant'], r['seed'], r['budget'], r['num_samples']) == (arm, seed, budget, scenes)]
        assert len(selected) == 2
        means = {k: statistics.fmean(r[k] for r in selected) for k in METRICS + ('collision_rate',)}
        item = dict(arm=arm, TRAIN=seed, steps=budget, scenes_per_GEN=scenes, GEN=[2817, 3817], metrics=means,
                    checkpoints=sorted(set(r['checkpoint'] for r in selected)))
        ref = []
        for row in selected:
            key = (seed, budget, row['eval_seed'], row['eval_manifest_sha256'])
            if key in controls:
                ref.append(controls[key])
        if len(ref) == 2:
            item['against657'] = score(selected, ref)
            item['657_caveat'] = 'Historical completed control; identical initial tensors/map/calibration/TRAIN order/exposures/GEN/cohort required by score. Inactive source versions differ.'
        result.append(item)
    save(ART / 'live_results.json', dict(updated_utc=now(), observations_sha256=sha(source),
        independent_metric_audit=audit['status'], cells=len(rows), results=result,
        control_source_hashes={str(p.relative_to(ROOT)): sha(p) for p in CONTROL_FILES},
        caveat='Means across two GEN seeds. No full128epoch improvement claim. Collision is descriptive, not a selection objective.'))
    return len(rows)


def main():
    seen = None
    while time.time() < END:
        path = ART / 'observations.json'
        signature = sha(path)
        if signature != seen:
            try:
                count = refresh()
                seen = signature
                save(ART / 'result_watcher_status.json', dict(status='AUDITED', updated_utc=now(), pid=os.getpid(), cells=count))
                print('AUDITED_CELLS ' + str(count), flush=True)
            except Exception as exc:
                save(ART / 'result_watcher_status.json', dict(status='AUDIT_FAILED', updated_utc=now(), pid=os.getpid(),
                    error_type=type(exc).__name__, error=str(exc)))
                print('AUDIT_FAILED ' + type(exc).__name__, flush=True)
                # Preserve the failed observation signature and retry when files change.
                seen = signature
        time.sleep(min(30, max(0, END - time.time())))
    count = refresh()
    save(ART / 'result_watcher_status.json', dict(status='DEADLINE_AUDITED', updated_utc=now(), pid=os.getpid(), cells=count))


if __name__ == '__main__':
    main()
