"""Serialize production-cohort confirmation and independent-seed evaluation.

This is the frozen final stage of the 2026-10-11 scratch loss search. It does
not change or add loss recipes. Run only after the two initial production
arms have started; it waits for their completed summary before using the GPU.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
ARTIFACTS = Path('docs/experiments/init_diffusion_scratch_loss_search_20261011')
RUNS = Path('logs/component_loss_scratch')
CONTROL = 'warm_winner'
NEW = ['geo_mse32_col20_heading_l1_shape2', 'geometry_strong_heading_l1_width_p2']
SCALES = dict(nearest_dist_jsd=1., lat_dev_jsd=1., ang_dev_jsd=10.,
              length_jsd=20., width_jsd=15., speed_jsd=20., collision_rate=5.)


def run(stage, command):
    print('START ' + stage, flush=True)
    with open('logs/component_loss_scratch_' + stage + '.log', 'w') as handle:
        status = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    if status:
        raise SystemExit(f'{stage} failed with returncode {status}; confirmation queue stopped.')


def base(scenes, manifest=None):
    result = [PYTHON, 'tools/scratch_component_loss_sweep.py', '--scenes', str(scenes),
              '--batch-size', '512', '--workers', '0', '--lr', '5e-4', '--evaluate-ema',
              '--recipe', 'logs/component_loss_scratch/design/anchor_recipe.json',
              '--variant-file', 'logs/component_loss_scratch/design/adaptive_variants.json',
              '--normalizer-json', 'logs/component_loss_scratch/calibration/normalizer.json']
    if manifest:
        result += ['--eval-manifest', str(manifest), '--extra-eval-seed', '3817']
    return result


def endpoint(folder, variant):
    rows = json.loads((folder / 'summary.json').read_text())
    matches = [row for row in rows if row['variant'] == variant
               and row['budget'] == 2048 and row['evaluation_weights'] == 'ema'
               and row['eval_seed'] == 2817]
    if len(matches) != 1:
        raise ValueError(f'Expected one frozen EMA endpoint for {variant}, got {len(matches)}')
    row = matches[0]
    row['selection_score'] = sum(row[key] / scale for key, scale in SCALES.items()) / 7
    row['vehicle_count_ratio'] = row['num_generated_vehicles'] / row['num_gt_vehicles']
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-independent-seeds', action='store_true',
                        help='Use the saved production selection and completed seed817 heldout; start fresh independent seeds only')
    args = parser.parse_args()
    if args.start_independent_seeds:
        selection = json.loads((ARTIFACTS / 'production_candidate_confirmation_selection.json').read_text())
        names = selection['confirmation_finalists']
        if (names != [CONTROL, selection['selected']] or selection['selected'] not in NEW
                or len(json.loads((RUNS / 'heldout_seed817/summary.json').read_text())) != 8):
            raise ValueError('Missing frozen production selection or completed seed817 heldout')
        heldout_base = base(4096, ARTIFACTS / 'heldout4096_files.json')
        for seed in (1817, 2817):
            run(f'batch512_seed{seed}', heldout_base + [
                '--steps', '2048', '--checkpoint-steps', '1536', '--allow-multiple-epochs',
                '--variants', ','.join(names), '--seed', str(seed),
                '--output', str(RUNS / f'batch512_seed{seed}')])
        print('SEED CONFIRMATIONS DONE', flush=True)
        return
    frozen = {
        'candidate_variants': NEW, 'control': CONTROL, 'training_seed': 817,
        'selection_budget': 2048, 'selection_weights': 'ema',
        'selection_generation_seed': 2817, 'score_scales': SCALES,
        'eligibility': {'minimum_onroad_fraction': 'control minus 0.02',
                        'vehicle_count_ratio_range': [0.9, 1.1]},
        'tie_break': 'lexicographic variant name after ascending score',
        'confirmation_training_seeds': [817, 1817, 2817],
        'confirmation_generation_seeds': [2817, 3817],
        'confirmation_manifest': str(ARTIFACTS / 'heldout4096_files.json'),
        'note': 'Frozen before production endpoints complete; no new loss recipe.'}
    (ARTIFACTS / 'production_confirmation_protocol.json').write_text(
        json.dumps(frozen, indent=2) + '\n')
    log = Path('logs/component_loss_scratch_batch512_seed817.log')
    while not (log.exists() and 'DONE ' in log.read_text()):
        if log.exists() and ('Traceback (most recent call last)' in log.read_text()
                             or 'ValueError:' in log.read_text()):
            raise SystemExit('Initial production stage failed; queue stopped.')
        time.sleep(5)
    alternative = RUNS / 'batch512_alternative_seed817'
    run('batch512_alternative_seed817', base(1024) + [
        '--steps', '2048', '--checkpoint-steps', '768,1536', '--allow-multiple-epochs',
        '--variants', NEW[1], '--seed', '817', '--output', str(alternative)])
    original = RUNS / 'batch512_seed817'
    control = endpoint(original, CONTROL)
    candidates = [endpoint(original, NEW[0]), endpoint(alternative, NEW[1])]
    for row in candidates:
        row['eligible'] = (row['generated_onroad_fraction'] >=
                           control['generated_onroad_fraction'] - .02
                           and .9 <= row['vehicle_count_ratio'] <= 1.1)
    eligible = sorted((row for row in candidates if row['eligible']),
                      key=lambda row: (row['selection_score'], row['variant']))
    if not eligible:
        (ARTIFACTS / 'production_candidate_confirmation_selection.json').write_text(
            json.dumps({'protocol': frozen, 'control': control, 'candidates': candidates,
                        'selected': None}, indent=2) + '\n')
        raise SystemExit('Both new recipes fail the coverage/count guard; inspect results.')
    winner = eligible[0]
    names = [CONTROL, winner['variant']]
    selection = {'protocol': frozen, 'control': control, 'candidates': candidates,
                 'selected': winner['variant'], 'confirmation_finalists': names}
    (ARTIFACTS / 'production_candidate_confirmation_selection.json').write_text(
        json.dumps(selection, indent=2) + '\n')
    checkpoint_map = {CONTROL: control['checkpoint'], winner['variant']: winner['checkpoint']}
    checkpoint_file = RUNS / 'design/heldout_seed817_checkpoints.json'
    checkpoint_file.write_text(json.dumps(checkpoint_map, indent=2) + '\n')
    heldout_base = base(4096, ARTIFACTS / 'heldout4096_files.json')
    run('heldout_seed817', heldout_base + [
        '--steps', '1', '--checkpoint-steps', '', '--variants', '',
        '--evaluation-checkpoints', str(checkpoint_file), '--seed', '817',
        '--output', str(RUNS / 'heldout_seed817')])
    for seed in (1817, 2817):
        run(f'batch512_seed{seed}', heldout_base + [
            '--steps', '2048', '--checkpoint-steps', '1536', '--allow-multiple-epochs',
            '--variants', ','.join(names), '--seed', str(seed),
            '--output', str(RUNS / f'batch512_seed{seed}')])
    print('SEED CONFIRMATIONS DONE', flush=True)


if __name__ == '__main__':
    main()
