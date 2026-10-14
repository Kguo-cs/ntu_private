"""Describe actual frozen map contexts on TRAIN; no generation-quality claims."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import copy
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_wandb_cc_scratch_20261013 import now, save, sha


def main():
    import hydra
    import torch
    import torch.nn.functional as F
    from torch_geometric.loader import DataLoader
    from omegaconf import OmegaConf
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    from src.smart.datasets import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderTrain
    install_cached_loader(torch)
    torch.set_num_threads(2)
    art = ROOT / 'output/reports/init_diffusion_expanded_20261014/map_architecture'
    cfg_path = ROOT / 'logs/init_diffusion_expanded_20261014/map_architecture/roadK8_T4817_2048_N4096/roadK8/config.yaml'
    cfg = OmegaConf.load(cfg_path)
    torch.manual_seed(4817)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False).cpu().eval()
    source = ROOT / 'logs/component_loss_scratch/calibration/calibration_files.pkl'
    with source.open('rb') as f:
        files = pickle.load(f)['files'][:16]
    manifest = art / 'context_train16.pkl'
    with manifest.open('wb') as f:
        pickle.dump({'files': files}, f)
    dataset = MultiDataset(cfg.data.train_raw_dir, WaymoTargetBuilderTrain(32),
                           sample_list=str(manifest), sample_list_check_exists=False)
    batch = next(iter(DataLoader(dataset, batch_size=16, num_workers=0, shuffle=False)))
    maps, agents = model.token_processor(copy.deepcopy(batch))
    encoder = model.encoder.map_encoder
    original = encoder.edge_encoder.build_map2map_edge
    captured = {}

    def capture(*args, **kwargs):
        edge, relation = original(*args, **kwargs)
        pos_src, pos_dst = args[0], args[2]
        degree = torch.bincount(edge[1], minlength=len(pos_dst))
        distances = (pos_src[edge[0]] - pos_dst[edge[1]]).norm(dim=-1)
        captured.update(source_nodes=len(pos_src), query_nodes=len(pos_dst),
                        edges=edge.shape[1], mean_neighbors=degree.float().mean().item(),
                        queries_without_neighbors=int((degree == 0).sum()),
                        mean_neighbor_distance=distances.mean().item(),
                        p90_neighbor_distance=torch.quantile(distances, .9).item())
        return edge, relation

    encoder.edge_encoder.build_map2map_edge = capture
    records = []
    with torch.inference_mode():
        for mode in ('road_edges', 'lanes_only', 'lanes_and_edges', 'all'):
            model.token_processor.init_map_output_types = mode
            reference = None
            for radius, neighbors in ((20., 30), (40., 64)):
                encoder.pl2pl_radius, encoder.pt2pt_neighbor = radius, neighbors
                output = encoder(maps, tokenized_agent=agents)
                stats = dict(captured)
                stats.update(mask=mode, radius=radius, neighbor_cap=neighbors,
                             returned_nodes=len(output['pt_token']),
                             scenes_with_queries=int(output['batch'].unique().numel()))
                if reference is None:
                    reference = {k: v.clone() for k, v in output.items()}
                else:
                    assert all(torch.equal(output[k], reference[k]) for k in ('batch', 'position', 'orientation'))
                    cosine = F.cosine_similarity(output['pt_token'], reference['pt_token'])
                    stats['mean_feature_cosine_vs_r20_k30'] = cosine.mean().item()
                records.append(stats)
    save(art / 'map_context_train_diagnostic.json', dict(
        status='PASS', at_utc=now(), TRAIN_scenes=16,
        manifest_sha256=sha(manifest), original_TRAIN_manifest_sha256=sha(source),
        config_sha256=sha(cfg_path), script_sha256=sha(__file__), device='cpu',
        map_layers=encoder.num_layers,
        half_extent=float(model.token_processor.init_map_half_extent), rows=records,
        scope='Actual frozen AIRL map encoder forward on16 TRAIN scenes. Graph and '
              'feature descriptions only; not generated-agent metrics or an ablation '
              'of learned quality. CPU graph ties may differ from GPU.'))
    print('TRAIN_MAP_CONTEXT_DIAGNOSTIC_SUCCESS', records, flush=True)


if __name__ == '__main__':
    main()
