"""TRAIN-only paired source assignment diagnostic; no model training or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.runtime_cached_torch_loader_sweep import install_cached_loader
install_cached_loader(torch)
import hydra
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from src.smart.datasets import MultiDataset
from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
from src.smart.diffusion import scale_flow

ART = ROOT / 'output/reports/init_diffusion_expanded_20261014'
OUT = ART / 'matching_assignment_diagnostic'
FIELDS = {'position': slice(0, 2), 'heading_chord': slice(2, 4),
          'length': slice(4, 5), 'width': slice(5, 6), 'speed': slice(6, 7)}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(4817)
    config = ART / 'recommended_shape_affine_full_resolved.yaml'
    cfg = OmegaConf.load(config)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu().train()
    decoder = model.encoder.init_decoder
    flow = decoder.G1
    assert not flow.joint_type and flow.velocity_representation == 'speed'
    assert flow.model.attribute_source is None
    with (ROOT / 'logs/component_loss_scratch/calibration/calibration_files.pkl').open('rb') as f:
        files = pickle.load(f)['files'][:256]
    manifest = OUT / 'train256.pkl'
    with manifest.open('wb') as f:
        pickle.dump({'files': files}, f)
    dataset = MultiDataset(cfg.data.train_raw_dir, WaymoTargetBuilderTrain(32),
                           sample_list=str(manifest), sample_list_check_exists=False)
    loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)
    original_match = scale_flow.get_closest_sum_idx_fast
    captured = []
    def capture(*args, **kwargs):
        idx = original_match(*args, **kwargs)
        captured.append(idx.clone())
        return idx
    scale_flow.get_closest_sum_idx_fast = capture
    values = {arm: {k: [] for k in (*FIELDS, 'heading_arc')} for arm in ('roadK8', 'match2', 'match4')}
    changed = {'match2': 0, 'match4': 0}
    total = 0
    source_seeds = (6817, 7817, 8817, 9817)
    per_source = []
    try:
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                _, agents = model.token_processor(batch)
                decoder._prepare_ego_context(agents)
                x, _ = flow.model.get_input(agents)
                movable = ~flow.model._ego_role_mask(agents)
                for source_seed in source_seeds:
                    sources, indices = {}, {}
                    for arm in values:
                        weight = {'roadK8': 1., 'match2': 2., 'match4': 4.}[arm]
                        flow.matching_weights = (None if arm == 'roadK8' else
                            dict(position=1., heading=weight, length=1., width=1., velocity=weight))
                        torch.manual_seed(source_seed + batch_i * 100000)
                        captured.clear()
                        sources[arm] = flow._sample_noise(x, agents)
                        assert len(captured) == 1
                        indices[arm] = captured[0]
                        residual = (sources[arm][movable] - x[movable]) / flow.model.normal_scale
                        for name, fields in FIELDS.items():
                            values[arm][name].append(residual[:, fields].square().mean(-1))
                        delta = torch.atan2(sources[arm][movable, 3], sources[arm][movable, 2]) - torch.atan2(x[movable, 3], x[movable, 2])
                        values[arm]['heading_arc'].append(torch.atan2(delta.sin(), delta.cos()).square())
                    # Undo both permutations: the underlying complete source rows must match exactly.
                    a = sources['roadK8'][movable][torch.argsort(indices['roadK8'])]
                    for arm in ('match2', 'match4'):
                        b = sources[arm][movable][torch.argsort(indices[arm])]
                        assert torch.equal(a, b)
                        assert torch.equal(sources['roadK8'][~movable], sources[arm][~movable])
                    for arm in indices:
                        idx = indices[arm]
                        for label in ('batch', 'type'):
                            labels = agents[label][movable]
                            assert torch.equal(labels, labels[idx])
                    n = int(movable.sum())
                    c = {arm: int((indices['roadK8'] != indices[arm]).sum())
                         for arm in ('match2', 'match4')}
                    total += n
                    for arm in changed:
                        changed[arm] += c[arm]
                    per_source.append(dict(batch=batch_i, source_seed=source_seed, agents=n, changed=c['match4'], changed_match2=c['match2']))
                print(f'batch {batch_i + 1}/{len(loader)} complete', flush=True)
    finally:
        scale_flow.get_closest_sum_idx_fast = original_match
    means = {a: {k: float(torch.cat(v).mean()) for k, v in groups.items()} for a, groups in values.items()}
    result = dict(status='PASS', at_utc=datetime.now(timezone.utc).isoformat(), scenes=len(files),
                  source_seeds=source_seeds, agent_source_pairs=total, changed_assignment_fraction=changed['match4'] / total,
                  changed_assignment_fractions={a: n / total for a, n in changed.items()},
                  mean_squared_endpoint_distances=means,
                  match4_over_roadK8={k: means['match4'][k] / means['roadK8'][k] for k in FIELDS | {'heading_arc': None}},
                  match2_over_roadK8={k: means['match2'][k] / means['roadK8'][k] for k in FIELDS | {'heading_arc': None}},
                  source_rows_identical_after_inverse_permutation=True, scene_and_type_preserved=True,
                  ego_sources_identical=True, per_source=per_source,
                  source_sha256=sha(__file__), manifest_sha256=sha(manifest), config_sha256=sha(config),
                  scope='256 TRAIN calibration scenes, four paired CPU source seeds; actual native _sample_noise and Hungarian code. No network prediction, checkpoint evaluation, validation labels or training updates.',
                  caveat='Endpoint distances describe coupling and may influence learning difficulty; they do not prove the cause of eventual JSD differences. Heading chord uses saved normalization; heading arc is radians squared. All non-ego agent types are pooled; type labels stay fixed during matching.')
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:result[k] for k in ('changed_assignment_fraction', 'mean_squared_endpoint_distances', 'match4_over_roadK8')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
