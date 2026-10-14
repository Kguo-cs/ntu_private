"""Read cached W&B snapshots and track common-epoch comparisons to roadK8."""
from datetime import datetime
from pathlib import Path
import math
import os
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import read, save, sha, now

ART = ROOT / 'output/reports/init_diffusion_overnight_20261014'
END = datetime.fromisoformat('2026-10-14T10:00:00+08:00').timestamp()
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd', 'width_jsd', 'speed_jsd', 'collision_rate')
FOCUS = ('speed_jsd', 'ang_dev_jsd', 'length_jsd')
REFERENCE = '0vftb3cc'
CANDIDATES = ('doke6bbr', 'nnsjk5vl', 'zww9tc6r', 'iu3fa4gx')


def refresh():
    path = ART / 'wandb/latest_analysis.json'
    analysis = read(path)
    manifest = {r['id']: r for r in analysis['manifest']}
    epochs = {}; provenance = {}
    for rid in (REFERENCE,) + CANDIDATES:
        if rid not in manifest:
            continue
        item = manifest[rid]; history = read(item['history_file']); groups = {}
        for row in history['rows']:
            if row.get('epoch') is None or not all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in METRICS):
                continue
            groups.setdefault(int(row['epoch']), []).append(row)
        epochs[rid] = {e: {k: statistics.fmean(v[k] for v in rows) for k in METRICS} for e, rows in groups.items()}
        provenance[rid] = dict(name=item['name'], path=item['history_file'], sha256=sha(item['history_file']),
                              fetched_utc=history['fetched_at_utc'], conditions=item['conditions'])
    output = []
    for rid in CANDIDATES:
        for e in sorted(set(epochs.get(rid, {})) & set(epochs.get(REFERENCE, {}))):
            a, b = epochs[rid][e], epochs[REFERENCE][e]
            ratios = {k: a[k]/b[k] for k in METRICS if b[k] > 0}
            if not all(k in ratios for k in FOCUS):
                continue
            output.append(dict(candidate=rid, baseline=REFERENCE, epoch=e,
                metrics=a, reference_metrics=b, ratios=ratios,
                R_focus=statistics.fmean(ratios[k] for k in FOCUS),
                R_jsd=statistics.fmean(ratios[k] for k in METRICS[:6]),
                same_configured_TRAIN=manifest[rid]['conditions']['seed'] == manifest[REFERENCE]['conditions']['seed']))
    result = dict(reviewed_utc=now(), snapshot_utc=analysis['inventory_fetched_at_utc'], comparisons=output,
        provenance=provenance, guidance={'doke6bbr':'final-only .05', 'nnsjk5vl':'multi-step .3', REFERENCE:'off'},
        caveat='Observed W&B comparisons at common epochs. Actual cohort/GEN hashes not fetched. Guidance and GPU hardware differ; not a controlled causal estimate. Final-only guidance is expected to primarily alter positions, not a proof of identical attribute metrics.',
        hardware={'doke6bbr':'A6000',REFERENCE:'A6000','nnsjk5vl':'RTX5090','iu3fa4gx':'RTX5090','zww9tc6r':'A6000'})
    save(ART / 'wandb/roadK8_same_epoch_latest.json', result)
    filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    save(ART / 'wandb/paired_reviews' / filename, result)
    return result


if __name__ == '__main__':
    seen = None
    while time.time() < END:
        signature = sha(ART / 'wandb/latest_analysis.json')
        if signature != seen:
            try:
                result = refresh(); seen = signature
                save(ART / 'wandb/pair_watcher_status.json', dict(status='REVIEWED', updated_utc=now(), pid=os.getpid(),
                    snapshot_utc=result['snapshot_utc'], rows=len(result['comparisons'])))
                print('PAIRED_REVIEW ' + result['snapshot_utc'], flush=True)
            except Exception as exc:
                save(ART / 'wandb/pair_watcher_status.json', dict(status='REVIEW_FAILED', updated_utc=now(),
                    pid=os.getpid(), error_type=type(exc).__name__))
        time.sleep(min(30, max(0, END-time.time())))
    refresh()
