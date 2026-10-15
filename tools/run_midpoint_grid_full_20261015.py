"""Explicit full50K check of roadK8's small paired8K uniform-grid gain."""
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_midpoint_grid_20261015 as grid


def main():
    parent = grid.ART
    assert grid.read(parent / 'status.json')['status'] == 'GRID_FINITE_PLAN_COMPLETE'
    evidence = next(r for r in grid.read(parent / 'comparisons.json')['comparisons']
                    if r['model'] == 'roadK8' and r['power'] == 1.)
    assert len(evidence['paired']) == 2
    assert all(p['ratios']['ang_dev_jsd'] < 1 for p in evidence['paired'])
    grid.ART = parent.parent / 'midpoint_grid_full_v1'
    grid.RUNS = ROOT / 'logs/init_diffusion_continuous_20261014/midpoint_grid_full_v1'
    grid.ART.mkdir(parents=True, exist_ok=True)
    grid.RUNS.mkdir(parents=True, exist_ok=True)
    if (grid.ART / 'protocol.json').exists():
        raise RuntimeError('Full grid check already registered')
    for power in (1., 1.2):
        grid.save(grid.ART / f'power{power:g}.json', grid.read(parent / f'power{power:g}.json'))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [grid.RUNNER, *sorted((ROOT / 'src/smart').rglob('*.py'))]}
    grid.save(grid.ART / 'protocol.json', dict(registered_utc=grid.now(), paired8K=evidence,
                                             decision='Explicit agent follow-up:2.4% angular gain is smaller than '
                                             'automatic3% gate, but roadK8 is close to the user threshold. '
                                             'Measure full50K rather than infer success.',
                                             powers=[1., 1.2], GEN=[2817, 3817], scenes=50000,
                                             source_hashes=hashes, NFE=24, objective_unchanged=True))
    rows = []
    for seed in (2817, 3817):
        for power in (1., 1.2):
            for name, expected in hashes.items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
            row = grid.evaluate('roadK8', power, 50000, seed)
            if row is None:
                return
            rows.append(row)
            grid.save(grid.ART / 'observations.json', rows)
    grid.save(grid.ART / 'status.json', dict(status='FULL_GRID_CHECK_COMPLETE', completed_utc=grid.now(),
                                           next='Review both full50K GENs; no default or deployment change.'))


if __name__ == '__main__':
    main()
