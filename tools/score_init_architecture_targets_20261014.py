"""Keep all-seven target checks separate from average-R architecture ranking."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import now, read, save, sha
from tools import score_init_sd_vw_targets_20261014 as target


def main():
    art = ROOT / 'output/reports/init_diffusion_expanded_20261014/map_architecture'
    sources, cells, keys = {}, [], set()
    for name in ('observations.json', 'mask_followup/observations.json', 'mask_midpoint/observations.json'):
        path = art / name
        if not path.exists():
            continue
        sources[name] = sha(path)
        for row in read(path):
            key = row['metrics_path']
            if key in keys:
                continue
            keys.add(key)
            report = read(key)
            cells.append(dict(method='raw', arm=row['variant'], TRAIN=row['seed'],
                              updates=row['budget'], GEN=row['eval_seed'], scenes=row['num_samples'],
                              checkpoint=row['checkpoint'],
                              sampling_steps=report['decoder']['sampling_steps'],
                              metrics_path=key, report_sha256=sha(key),
                              scope=target.report_scope(report), **target.score(row)))
    for name in ('size_calibration/quick_comparison.json', 'size_calibration/full50000_comparison.json'):
        path = art / name
        if not path.exists():
            continue
        sources[name] = sha(path)
        for row in read(path)['rows']:
            key = row['metrics_path']
            assert key not in keys
            keys.add(key)
            report=read(key)
            cells.append(dict(method='TRAIN-only size postprocessing', arm=row['variant'], TRAIN=row['seed'],
                              updates=row['budget'], GEN=row['eval_seed'], scenes=row['num_samples'],
                              checkpoint=row['checkpoint'],
                              sampling_steps=report['decoder']['sampling_steps'],
                              calibration_sha256=report['postprocessing']['calibration_sha256'],
                              metrics_path=key, report_sha256=sha(key),
                              scope=target.report_scope(report), **target.score(row)))
    full_groups = {}
    for cell in cells:
        if cell['scenes'] == 50000:
            full_groups.setdefault((cell['method'], cell['arm'], cell['TRAIN'], cell['updates'],
                                    cell['checkpoint'], cell['sampling_steps'], cell.get('calibration_sha256')), []).append(cell)
    successes = [dict(method=k[0], arm=k[1], TRAIN=k[2], updates=k[3]) for k, rows in full_groups.items()
                 if {r['GEN'] for r in rows} == {2817, 3817}
                 and all(r['passed_metrics'] == 7 and r['scope']['numerical_comparison_verifiable'] for r in rows)]
    result = dict(at_utc=now(), targets_sha256=sha(target.TARGET_FILE), source_files=sources,
                  auditor_sha256=sha(__file__), unique_actual_cells=len(cells), cells=cells,
                  full50000_all_GEN_successes=successes,
                  full128epoch_verified=False,
                  caveat='Raw and TRAIN-only postprocessing are separate methods. Reused control rows '
                         'deduplicated by actual metric report. BothGEN and complete50K required; '
                         'averageR cannot establish all-seven success. Init fixes ego conditions.')
    save(art / 'all_seven_target_progress.json', result)
    print(dict(cells=len(cells), full50k_all_GEN_successes=len(successes)))


if __name__ == '__main__':
    main()
