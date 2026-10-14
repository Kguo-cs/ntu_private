"""Refresh the conservative recommendation from observed, same-epoch evidence."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import now, read, save, sha

ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'


def main():
    analysis_path = ART / 'wandb/latest_analysis.json'
    comparisons_path = ART / 'wandb/current_parent_comparisons.json'
    analysis, comparisons = read(analysis_path), read(comparisons_path)
    assert comparisons['source_analysis_sha256'] == sha(analysis_path)
    manifest = {r['id']: r for r in analysis['manifest']}
    latest = {}
    for rid in ('kw1ayqd5', 'efzhwmu3', 'cchgfuhr', 'bq7z12zy'):
        run = manifest[rid]
        paired = next(r for r in comparisons['comparisons'] if
                      r['candidate'] == rid and r['parent'] == '0vftb3cc'
                      and r['last_shared_epoch'])
        latest[rid] = dict(epoch=run['last_validation_epoch'],
                           metrics=run['last_validation_metrics'],
                           shared_epoch=paired['epoch'],
                           ratios_vs_same_epoch_roadK8=paired['ratios'],
                           R_focus=paired['R_focus'], R_jsd=paired['R_jsd'])
    record = ART / 'current_recommendation.json'
    previous = read(record)
    stamp = now()
    archive = ART / 'registrations/interim_recommendations' / (
        stamp.replace(':', '').replace('+', '_') + '.json')
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record, archive)
    config = 'init_diffusion_657_n20_road_k8_full'
    config_path = ROOT / 'configs/experiment' / (config + '.yaml')
    # Preserve the old selected arm in an explicit historical field, so the
    # active config cannot accidentally be mistaken for the affine candidate.
    historical = previous.get('historical_selected_candidate') or {
        k: previous[k] for k in ('config', 'evaluation_config', 'config_sha256',
                                'TRAIN', 'updates', 'scenes', 'GEN', 'means',
                                'ratios_vs_roadK8') if k in previous}
    native = ART / 'map_architecture'
    result = dict(
        at_utc=stamp, status='KEEP_ROADK8_NO_VERIFIED_NEW_FULL128_SUPERIORITY',
        config=config, recommended_full_config=config,
        evaluation_config='init_diffusion_657_n20_road_k8_full_eval',
        config_sha256=sha(config_path), historical_selected_candidate=historical,
        remote_latest_validations=latest,
        inventory_fetched_at_utc=analysis['inventory_fetched_at_utc'],
        analysis_sha256=sha(analysis_path), comparison_sha256=sha(comparisons_path),
        native_architecture_status=read(native / 'status.json'),
        native_architecture_observations_sha256=sha(native / 'observations.json'),
        completed_native_architecture_cells=len(read(native / 'observations.json')),
        remote_observation_priority=['cchgfuhr', 'bq7z12zy', 'efzhwmu3', 'kw1ayqd5'],
        recommend_duplicate_full_run=False,
        reason='No completed new full128-epoch result jointly improves the target attributes. '
               'Native14336 full50K affine improves speed/angle but worsens sizes. '
               'Remote affine also worsens sizes;24-step matching4 remains unfinished '
               'and has worse length at its shared epoch. Newly permitted architecture '
               'changes require their own paired confirmation.',
        limitation='Remote comparisons are observational. Same epoch does not establish '
                   'identical members, source, precision, hardware or GEN. New architecture '
                   'phase uses its own fresh control. Short-run selection is not proof '
                   'of full-training superiority.',
        previous_record=str(archive.relative_to(ROOT)), final_goal_complete=False)
    save(record, result)
    print(dict(recommended=config, snapshot=result['inventory_fetched_at_utc'],
               last_epochs={k: v['epoch'] for k, v in latest.items()}))


if __name__ == '__main__':
    main()
