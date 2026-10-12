"""Pre-register a separate position-only inference experiment against fresh N40 anchors.

--freeze-protocol is CPU/std-library metadata work before the first sampler cell.
--select never evaluates or trains: it accepts only a complete certified8-cell
grid, then writes one immutable decision which full50k must not revise.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ART = ROOT / 'docs/experiments/init_diffusion_overnight_search_20261011'
GRID = ROOT / 'logs/init_diffusion_hybrid_sampler_cohort_20261012/grid/summary.json'
OPTIONS = {'n40_eps005': (40, .05), 'hybrid_pos0001': (40, .05)}
TRAIN = (4817, 5817)
GEN = (2817, 3817)
BASELINE = 'n40_eps005'
METRICS = ('nearest_dist_jsd', 'lat_dev_jsd', 'ang_dev_jsd', 'length_jsd',
           'width_jsd', 'speed_jsd', 'collision_rate')
VALUES = METRICS + ('generated_onroad_fraction', 'num_generated_vehicles', 'num_gt_vehicles')
CRITERIA = {
    'grid': 'exact two options x two reused TRAIN checkpoints x two GEN =8 completed cells',
    'options': {k: {'steps': v[0], 'eps': v[1]} for k, v in OPTIONS.items()},
    'TRAIN_seeds': list(TRAIN), 'GEN_seeds': list(GEN), 'scenes': 4096,
    'default': BASELINE, 'baseline_origin': 'four fresh native N40/eps.05 anchors; hybrid changes only low-t position denominator floor',
    'original_strict_sampler': 'FAILED_DEFAULT20_PARITY permanently excluded; tolerance unchanged',
    'numerical_margin': 'R <=0.98 required, descriptive guard not statistical CI',
    'averaging': 'arithmetic GEN mean inside each TRAIN, then equal arithmetic mean across TRAIN',
    'R': 'arithmetic mean of seven candidate/base ratios; each ratio uses pooled metric means',
    'gates_pooled_AND_each_TRAIN': {'speed_jsd_ratio_max': 1.15, 'length_jsd_ratio_max': 1.15,
        'width_jsd_ratio_max': 1.15, 'onroad_fraction_min': 'same-baseline-.02',
        'generated_vehicle_count_over_GT': [.9, 1.1]},
    'selection': 'eligible minimum pooled R <=0.98; otherwise fresh native N40',
    'exact_R_tie': ['fewer_steps', 'smaller_abs_eps_minus_training_eps_0.05', 'option_name'],
    'partial_or_native_execution_failure': 'reject; no ranking and no frozen native sampler JSON',
    'new_training_runs': 0, 'main_model_selection_unchanged': True,
    'native16_selection_unchanged': True, 'hybrid_position': 'only position denominator1e-4 for0<t<.05; other components retainN40eps.05',
    'full50000': 'evaluate only the frozen option; no reselection from full results',
    'cost': 'both options40 model evaluations; only position low-time damping changes; no training updates',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save_new(path, value):
    """An existing protocol/decision is immutable, never silently replaced."""
    path = Path(path)
    require(not path.exists(), 'Refusing to overwrite existing frozen artifact: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        temporary = Path(handle.name)
    try:
        require(not path.exists(), 'Frozen artifact appeared while writing')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_hashes():
    paths = (Path(__file__).resolve(), ROOT / 'tools/evaluate_init_diffusion_hybrid_sampler.py', ROOT / 'tools/hybrid_position_sampler_hook.py',
             ROOT / 'tools/evaluate_init_diffusion_native_sampler.py', ROOT / 'tools/evaluate_init_diffusion_sampler.py',
             ROOT / 'tools/runtime_cached_torch_loader_sweep.py', ROOT / 'tools/torch_pickle_class_cache.py',
             ROOT / 'tools/runtime_completion.py', ROOT / 'tools/continue_overnight_init_diffusion.py')
    return {str(p): sha(p) for p in paths}


def freeze_protocol(path=ART / 'hybrid_sampler_selection_protocol.json'):
    path = Path(path)
    if path.exists():
        return validate_protocol(path)
    value = {'status': 'FROZEN_BEFORE_SAMPLER_EXECUTION', 'created_utc': datetime.now(timezone.utc).isoformat(),
             'criteria': CRITERIA, 'criteria_sha256': digest(CRITERIA), 'source_hashes': source_hashes(),
             'CPU_only': True, 'GPU_started': False, 'selection_made': False,
             'frozen_sampler_json_created': False, 'full50000_cannot_reselect': True,
             'native16_frozen_decision': {'path': str(ART / 'frozen_native_sampler.json'), 'sha256': sha(ART / 'frozen_native_sampler.json')}}
    # Freeze must precede actual cells, not retrospectively label a result.
    runs = GRID.parent
    existing = list(runs.glob('*.execution.json')) if runs.is_dir() else []
    require(not existing and not GRID.is_file(), 'Sampler execution already exists; cannot preregister posthoc')
    save_new(path, value)
    return value


def validate_protocol(path):
    value = read(path)
    require(value.get('status') == 'FROZEN_BEFORE_SAMPLER_EXECUTION' and value.get('selection_made') is False
            and value.get('full50000_cannot_reselect') is True, 'Invalid sampler selection protocol')
    require(value['criteria'] == CRITERIA and value['criteria_sha256'] == digest(CRITERIA), 'Frozen criterion changed')
    require(value['source_hashes'] == source_hashes(), 'Frozen sampler/selector/source SHA changed')
    require(value['native16_frozen_decision']['sha256'] == sha(value['native16_frozen_decision']['path']), 'Prior native16 frozen selection changed')
    datetime.fromisoformat(value['created_utc'])
    return value


def compare(values, baseline):
    require(all(math.isfinite(values[k]) and values[k] >= 0 and math.isfinite(baseline[k])
                and baseline[k] > 0 for k in METRICS), 'Ratios need finite nonnegative values and positive baseline')
    ratios = {k: values[k] / baseline[k] for k in METRICS}
    checks = {k: ratios[k] <= 1.15 for k in ('length_jsd', 'width_jsd', 'speed_jsd')}
    checks.update(onroad=0 <= values['generated_onroad_fraction'] <= 1 and
                  values['generated_onroad_fraction'] >= baseline['generated_onroad_fraction'] - .02,
                  vehicle_ratio=values['num_gt_vehicles'] > 0 and
                  .9 <= values['num_generated_vehicles'] / values['num_gt_vehicles'] <= 1.1)
    return {'metrics': values, 'ratios': ratios, 'R': sum(ratios.values()) / len(METRICS),
            'gate_checks': checks, 'eligible': all(checks.values())}


def rank_actual_rows(rows, *, complete):
    """Pure arithmetic helper, with no fabricated or padded independent runs."""
    require(complete is True, 'Incomplete sampler grid cannot be selected')
    expected = {(s, g, o) for s in TRAIN for g in GEN for o in OPTIONS}
    identities = [(r['TRAIN_seed'], r['GEN_seed'], r['option']) for r in rows]
    require(len(rows) == 8 and len(set(identities)) == 8 and set(identities) == expected,
            'Sampler grid must contain exactly the8 paired cells')
    require(all((r['steps'], r['eps']) == OPTIONS[r['option']] and r['scenes'] == 4096
                and r['actual_training_updates_in_this_job'] == 0 and r['evaluation_only'] is True for r in rows),
            'Sampler mode/scope/representation or inference-only guard differs')
    require(all(row.get('hybrid_position') is (row['option'] == 'hybrid_pos0001') for row in rows), 'Hybrid position mode differs from option')
    require(all(row.get('main_endpoint_parity_claimed') is False for row in rows), 'Old replay parity cannot be claimed in the new native cohort')
    means = {}
    for option in OPTIONS:
        means[option] = {s: {k: sum(r[k] for r in rows if r['option'] == option and r['TRAIN_seed'] == s) / len(GEN)
                             for k in VALUES} for s in TRAIN}
    pooled = {o: {k: sum(means[o][s][k] for s in TRAIN) / len(TRAIN) for k in VALUES} for o in OPTIONS}
    assessed = {}
    for option in OPTIONS:
        value = compare(pooled[option], pooled[BASELINE])
        per_train = {str(s): compare(means[option][s], means[BASELINE][s]) for s in TRAIN}
        value.update(per_TRAIN=per_train, pooled_gates_pass=value['eligible'],
                     each_TRAIN_gates_pass=all(v['eligible'] for v in per_train.values()),
                     eligible=value['eligible'] and all(v['eligible'] for v in per_train.values()),
                     independent_reused_TRAIN_N=2, new_training_runs=0)
        assessed[option] = value
    eligible = [o for o in OPTIONS if assessed[o]['eligible'] and assessed[o]['R'] <= .98]
    selected = min(eligible, key=lambda o: (assessed[o]['R'], OPTIONS[o][0], abs(OPTIONS[o][1] - .05), o)) if eligible else BASELINE
    return {'selected_option': selected, 'comparisons': assessed,
            'selection_reason': 'eligible minimum pooled R <=0.98 using fresh native anchors' if eligible else 'no eligible 2-percent-margin improvement: retain fresh native N40',
            'steps': OPTIONS[selected][0], 'eps': OPTIONS[selected][1],
            'model_evaluations_ratio_to_baseline40': OPTIONS[selected][0] / OPTIONS[BASELINE][0],
            'actual_new_training_runs': 0, 'full50000_reselection_permitted': False}


def verify_actual_grid(summary_path, protocol_path, *, main_art=ART):
    """No torch/checkpoint payload import: strict existing artifact certification."""
    protocol = validate_protocol(protocol_path)
    summary_path = Path(summary_path).resolve()
    summary = read(summary_path)
    require(summary.get('complete') is True and summary.get('main_selection_unchanged') is True
            and summary.get('new_selection_made') is False, 'Actual complete no-selection grid required')
    from types import SimpleNamespace
    from tools import evaluate_init_diffusion_hybrid_sampler as driver
    plan = read(summary_path.parent / 'plan.json')
    require(plan == summary['plan'] and plan['stage'] == 'grid', 'Actual grid plan differs')
    rebuilt = driver.make_plan(SimpleNamespace(main_art=Path(main_art), stage='grid', manifest=driver.MANIFEST,
                                               frozen_sampler=None, full_train_seeds=list(TRAIN)))
    require(rebuilt == plan, 'Actual grid/main/checkpoint/config/data/source cohort changed')
    require(plan['driver_sha256'] == protocol['source_hashes'][str(ROOT / 'tools/evaluate_init_diffusion_hybrid_sampler.py')],
            'Actual grid differs from preregistered driver')
    created = datetime.fromisoformat(protocol['created_utc']).timestamp()
    require(len(summary['rows']) == len(plan['cells']) == 8, 'Missing actual cells')
    for row, cell in zip(summary['rows'], plan['cells']):
        folder = driver.cell_folder(summary_path.parent, cell)
        driver.validate_completed_cell(row, cell, folder, plan, require_execution=True)
        execution = read(folder.parent / (folder.name + '.execution.json'))
        require(execution['started_unix'] >= created, 'Sampler execution predates frozen selection protocol')
    return summary, rank_actual_rows(summary['rows'], complete=True)


def select(summary_path=GRID, protocol_path=ART / 'hybrid_sampler_selection_protocol.json',
           output=ART / 'frozen_hybrid_sampler.json', *, main_art=ART):
    summary, proof = verify_actual_grid(summary_path, protocol_path, main_art=main_art)
    value = {'selection_is_frozen': True, 'selected_option': proof['selected_option'],
             'grid_summary': str(Path(summary_path).resolve()), 'grid_summary_sha256': sha(summary_path),
             'protocol': str(Path(protocol_path).resolve()), 'protocol_sha256': sha(protocol_path),
             'created_utc': datetime.now(timezone.utc).isoformat(), 'criterion_proof': proof,
             'selection_is_inference_only': True, 'main_selection_unchanged': True,
             'fresh_native_baselines': True, 'baseline_option': BASELINE, 'native16_selection_unchanged': True, 'original_sampler_status': 'FAILED_DEFAULT20_PARITY',
             'original_main_replay_parity_claimed': False,
             'main_decision_sha256': summary['plan']['main_decision_sha256'],
             'actual_training_updates_in_this_job': 0, 'full50000_cannot_reselect': True,
             'selector_sha256': sha(__file__)}
    if Path(output).exists():
        existing = read(output)
        require({k:v for k,v in existing.items() if k != 'created_utc'} ==
                {k:v for k,v in value.items() if k != 'created_utc'}, 'Refusing to revise frozen sampler decision')
        return existing
    save_new(output, value)
    return value


class SelectorTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [{'TRAIN_seed': s, 'GEN_seed': g, 'option': o, 'steps': steps, 'eps': eps,
                 'scenes': 4096, 'evaluation_only': True, 'actual_training_updates_in_this_job': 0,
                 **{k: 1. for k in METRICS}, 'generated_onroad_fraction': .85,
                 'num_generated_vehicles': 100., 'num_gt_vehicles': 100.,
                 'main_endpoint_parity_claimed': False, 'hybrid_position': o == 'hybrid_pos0001'} for s in TRAIN for g in GEN for o,(steps,eps) in OPTIONS.items()]

    def test_incomplete_and_duplicate_cannot_select(self):
        with self.assertRaises(ValueError): rank_actual_rows(self.rows()[:-1], complete=False)
        rows=self.rows(); rows[-1]=rows[0]
        with self.assertRaises(ValueError): rank_actual_rows(rows, complete=True)

    def test_per_train_gate_failure_cannot_hide_in_pooled_mean(self):
        rows=self.rows()
        for r in rows:
            if r['option']=='hybrid_pos0001':
                r['speed_jsd']=1.4 if r['TRAIN_seed']==5817 else .6
                r['length_jsd']=.8
        result=rank_actual_rows(rows,complete=True)
        self.assertTrue(result['comparisons']['hybrid_pos0001']['pooled_gates_pass'])
        self.assertFalse(result['comparisons']['hybrid_pos0001']['eligible'])
        self.assertEqual(result['selected_option'],BASELINE)

    def test_baseline_retained_when_no_eligible_strict_improvement(self):
        self.assertEqual(rank_actual_rows(self.rows(),complete=True)['selected_option'],BASELINE)

    def test_two_percent_margin_and_no_old_parity_claim(self):
        rows=self.rows()
        for r in rows:
            if r['option']!='n40_eps005':
                for k in METRICS: r[k]=.985
        self.assertEqual(rank_actual_rows(rows,complete=True)['selected_option'],BASELINE)
        rows[0]['main_endpoint_parity_claimed']=True
        with self.assertRaises(ValueError): rank_actual_rows(rows,complete=True)

    def test_ties_prefer_fewer_steps_then_training_epsilon(self):
        rows=self.rows()
        for r in rows:
            if r['option']!='n40_eps005':
                for k in METRICS: r[k]=.9
        self.assertEqual(rank_actual_rows(rows,complete=True)['selected_option'],'hybrid_pos0001')
        for r in rows:
            if r['option']=='hybrid_pos0001':
                for k in METRICS: r[k]=1.
        self.assertEqual(rank_actual_rows(rows,complete=True)['selected_option'],BASELINE)

    def test_gen_then_train_ratio_of_means_and_training_n(self):
        rows=self.rows()
        for r in rows:
            r['speed_jsd']=1. if r['TRAIN_seed']==4817 else 3.
            if r['option']=='hybrid_pos0001': r['speed_jsd']*=.8
        value=rank_actual_rows(rows,complete=True)['comparisons']['hybrid_pos0001']
        self.assertAlmostEqual(value['ratios']['speed_jsd'],.8)
        self.assertEqual(value['independent_reused_TRAIN_N'],2)
        self.assertEqual(value['new_training_runs'],0)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--select',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ART/'hybrid_sampler_selection_protocol.json')
    parser.add_argument('--grid-summary',type=Path,default=GRID)
    parser.add_argument('--output',type=Path,default=ART/'frozen_hybrid_sampler.json')
    parser.add_argument('--main-art',type=Path,default=ART)
    args=parser.parse_args(argv)
    require(sum((args.freeze_protocol,args.select,args.self_test))==1,'Choose exactly one action')
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelectorTests)).wasSuccessful() else 1
    value=freeze_protocol(args.protocol) if args.freeze_protocol else select(args.grid_summary,args.protocol,args.output,main_art=args.main_art)
    print(json.dumps({'action':'freeze_protocol' if args.freeze_protocol else 'select_actual_complete_grid',
                      'path':str(args.protocol if args.freeze_protocol else args.output),'created_utc':value['created_utc'],
                      'selected_option':value.get('selected_option'),'GPU_started':False,'torch_imported':'torch' in sys.modules}))
    return 0


if __name__=='__main__':
    import gc
    gc.disable()
    result=main();sys.stdout.flush();sys.stderr.flush();os._exit(result)
