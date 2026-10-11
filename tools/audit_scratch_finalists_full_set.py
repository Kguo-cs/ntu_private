"""Select from the frozen heldout matrix, then audit both finalists on 50k.

The full-set evaluation never changes the heldout deployment decision. Only
scratch checkpoints already trained by this search are loaded; no training
or outside repository is involved here. --wait serializes this after the
independent-seed confirmation queue.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/ke/miniconda3/envs/sim/bin/python'
ARTIFACTS = ROOT / 'docs/experiments/init_diffusion_scratch_loss_search_20261011'
RUNS = ROOT / 'logs/component_loss_scratch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    log = ROOT / 'logs/component_loss_scratch_batch512_seed2817.log'
    while args.wait and not (log.exists() and 'DONE ' in log.read_text()):
        if log.exists() and 'Traceback (most recent call last)' in log.read_text():
            raise SystemExit('Final confirmation failed; no full-set audit started.')
        time.sleep(5)
    selection = json.loads((ARTIFACTS / 'production_candidate_confirmation_selection.json').read_text())
    candidate = selection['selected']
    checkpoint_file = RUNS / 'design/heldout_seed817_checkpoints.json'
    final_checkpoints = json.loads(checkpoint_file.read_text())
    bridge_checkpoints = {name: str(Path(path).parent / 'step_1536.ckpt')
                          for name, path in final_checkpoints.items()}
    if any(not Path(path).is_file() for path in bridge_checkpoints.values()):
        raise FileNotFoundError('Missing existing seed817 milestone for paired late-budget curves')
    bridge_map = RUNS / 'design/heldout_bridge_seed817_checkpoints.json'
    bridge_map.write_text(json.dumps(bridge_checkpoints, indent=2) + '\n')
    # Both milestone curves must contain the same independent training seeds.
    # This adds evaluation only; the frozen EMA2048 selection remains unchanged.
    with (ROOT / 'logs/component_loss_scratch_heldout_bridge_seed817.log').open('w') as handle:
        subprocess.run([PYTHON, 'tools/scratch_component_loss_sweep.py', '--steps', '1',
                        '--checkpoint-steps', '', '--variants', '', '--evaluation-checkpoints',
                        str(bridge_map), '--scenes', '4096', '--eval-manifest',
                        str(ARTIFACTS / 'heldout4096_files.json'), '--evaluation-weights', 'both',
                        '--batch-size', '512', '--workers', '0', '--lr', '5e-4', '--seed', '817',
                        '--eval-seed', '2817', '--extra-eval-seed', '3817',
                        '--recipe', 'logs/component_loss_scratch/design/anchor_recipe.json',
                        '--variant-file', 'logs/component_loss_scratch/design/adaptive_variants.json',
                        '--normalizer-json', 'logs/component_loss_scratch/calibration/normalizer.json',
                        '--output', str(RUNS / 'heldout_bridge_seed817')],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    roots = [RUNS / name for name in ('heldout_seed817', 'heldout_bridge_seed817',
                                      'batch512_seed1817', 'batch512_seed2817')]
    heldout = ARTIFACTS / 'heldout_final_stage'
    subprocess.run([PYTHON, 'tools/analyze_scratch_loss_search.py', '--runs',
                    *map(str, roots), '--output', str(heldout), '--anchor', 'warm_winner',
                    '--plot-variants', 'warm_winner,' + candidate], check=True)
    paired = ARTIFACTS / 'paired_finalists'
    subprocess.run([PYTHON, 'tools/summarize_paired_scratch_finalists.py',
                    '--observations', str(heldout / 'observations.json'),
                    '--protocol', str(ARTIFACTS / 'final_deployment_selection_protocol.json'),
                    '--candidate', candidate, '--output', str(paired)], check=True)
    decision = json.loads((paired / 'paired_summary.json').read_text())
    if not decision['selected_variant']:
        raise SystemExit('No eligible complete heldout decision; full-set audit stopped.')
    print('FROZEN DEPLOYMENT DECISION ' + decision['selected_variant'], flush=True)
    with (ROOT / 'logs/component_loss_scratch_full50000_seed817.log').open('w') as handle:
        subprocess.run([PYTHON, 'tools/scratch_component_loss_sweep.py', '--steps', '1',
                        '--checkpoint-steps', '', '--variants', '', '--evaluation-checkpoints',
                        str(checkpoint_file), '--scenes', '50000', '--require-full-set',
                        '--evaluation-weights', 'ema', '--batch-size', '512', '--workers', '0',
                        '--lr', '5e-4', '--seed', '817', '--eval-seed', '2817',
                        '--recipe', 'logs/component_loss_scratch/design/anchor_recipe.json',
                        '--variant-file', 'logs/component_loss_scratch/design/adaptive_variants.json',
                        '--normalizer-json', 'logs/component_loss_scratch/calibration/normalizer.json',
                        '--output', str(RUNS / 'full50000_seed817')],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print('FULL50000 AUDIT DONE', flush=True)


if __name__ == '__main__':
    main()
