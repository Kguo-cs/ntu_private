"""Registered continuation after the opt-in e998 sampler integration.

Uniform defaults preserve the training objective and same-arm K8 lineage.
Reserve the remaining evening budget for paired N20 sampler/guidance tests.
"""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import resume_init_diffusion_final_long_20261013 as prior
from tools import run_init_diffusion_long_audit_20261013 as core

read, save, sha, now = core.read, core.save, core.sha, core.now
OLD, OLD_RUNS = prior.ART, prior.RUNS
ART, RUNS = OLD / 'sampler_merge', OLD_RUNS / 'sampler_merge'
original_sources = prior.sources


def sources():
    return {**original_sources(),
            'src/smart/diffusion/sampling_time_grid.py': sha(ROOT / 'src/smart/diffusion/sampling_time_grid.py'),
            str(Path(__file__).relative_to(ROOT)): sha(__file__)}


def prepare():
    assert not (ART / 'protocol.json').exists()
    plan = read(OLD / 'protocol.json')
    rows = read(OLD / 'observations.json')
    completed = [r for r in rows if r['variant'] == 'freshK8' and r['budget'] == 8192
                 and not r['evaluation_only']]
    assert len(completed) == 2 and {r['eval_seed'] for r in completed} == {2817, 3817}
    cp = completed[0]['checkpoint']
    folder = Path(cp).parent.parent
    proof = read(folder / 'verified_completion.json')
    assert proof['status'] == 'VERIFIED_COMPLETE'
    assert proof['summary_sha256'] == sha(folder / 'summary.json')
    changed = [p for p in plan['source_hashes']
               if sha(ROOT / p) != plan['source_hashes'][p]]
    allowed = {'src/smart/diffusion/initial_diffusion.py', 'src/smart/diffusion/scale_flow.py',
               'src/smart/model/smart.py', 'tools/overnight_init_diffusion_sweep.py'}
    assert set(changed) <= allowed, changed
    for n in ('recipe.json', 'variants.json', 'sorted128.json', 'sorted256.json', 'sorted4096.json'):
        save(ART / n, read(OLD / n))
    plan = copy.deepcopy(plan)
    plan['stages'] = [dict(arm='freshK8', seed=4817, steps=target,
                          initial_resume={'freshK8': cp}, initial_resume_step=8192,
                          scenes=4096,
                          manifest=str(ROOT / 'tmp/control657_paired_20261012/eval4096_files.json'),
                          checkpoints=[], label=f'long_freshK8_{target}')
                      for target in (12288, 16384)]
    plan.update(registered_utc=now(), source_hashes=sources(),
                previous_protocol_sha256=sha(OLD / 'protocol.json'),
                local_input_hashes={n: sha(ART / n) for n in
                                   ('recipe.json', 'variants.json', 'sorted128.json', 'sorted256.json', 'sorted4096.json')},
                source_change_audit=dict(changed=changed, reason='Opt-in inference time grid; uniform default retained.',
                                         unit_tests={'heading': 22, 'guidance_coordinates': 11, 'time_grid': 6},
                                         objective_changed=False, parameters_changed=False),
                deferred_lane_continuation='User reports poor long-run lane methods; preserve completed6826, reserve time for guidance/grid pairing and50K.',
                sampler_followup='Same saved EMA weights, same cohorts/GEN, observed NFE20; off/light/cc and power/cosine compared without fit.')
    plan['imported_sources'][str((OLD / 'observations.json').relative_to(ROOT))] = sha(OLD / 'observations.json')
    for path in (Path(cp), folder / 'summary.json', folder / 'verified_completion.json'):
        plan['imported_sources'][str(path.relative_to(ROOT))] = sha(path)
    save(ART / 'protocol.json', plan)
    save(ART / 'observations.json', rows)
    core.comparisons(rows)
    core.status('REGISTERED')


prior.ART, prior.RUNS = ART, RUNS
prior.recovery.ART, prior.recovery.RUNS = ART, RUNS
core.ART, core.RUNS, core.sources = ART, RUNS, sources
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    prior.execute() if args.execute else prepare()
