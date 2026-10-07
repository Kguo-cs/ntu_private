"""Compute posterior-sample latent normalization for a trained VectorWorld VAE."""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import sys

import torch
from torch_geometric.data import HeteroData, Batch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.smart.vectorworld.decoder import VectorWorldInitDecoder
from src.smart.vectorworld.checkpoints import plain, DATASET_KEYS, validate_stats, resolve_path
from src.smart.scenario_dreamer.preprocessed import adapt_preprocessed_scene
from src.smart.tokens.token_processor import TokenProcessor

@torch.no_grad()
def compute_stats(ae_checkpoint, data_dir, *, max_scenes=2048, batch_size=32, sample_list=None,
                  seed=0, device="cpu", base_config=None):
    if (not isinstance(max_scenes, int) or isinstance(max_scenes, bool) or max_scenes < 1
            or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("max_scenes and batch_size must be positive integers")
    root = Path(data_dir).expanduser().resolve()
    manifest = Path(sample_list).expanduser().resolve() if sample_list else None
    if manifest is not None:
        with manifest.open("rb") as handle:
            record = pickle.load(handle)
        names = record.get("files") if isinstance(record, dict) else None
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or Path(name).name != name
                       or name in (".", "..") or Path(name).suffix != ".pkl" for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("sample_list must contain unique native .pkl basenames in a files list")
        paths = [root / name for name in names[:max_scenes]]
    else:
        paths = sorted(root.glob("*.pkl"))[:max_scenes]
    if not paths:
        raise ValueError(f"No native motion .pkl scenes found: {root}")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected native scenes: {missing[:3]}")
    # Resolve the lightweight scene selection before constructing/loading the AE.
    torch.manual_seed(seed)
    processor = TokenProcessor(map_token_file="map_traj_token5.pkl", agent_token_file="agent_vocab_555_s2.pkl",
        map_token_sampling=OmegaConf.create(dict(num_k=1, temp=1.)),
        agent_token_sampling=OmegaConf.create(dict(num_k=1, temp=1.)),
        pred_init=True, learn_init=True, scenario_dreamer_init=True).to(device).eval()
    decoder = VectorWorldInitDecoder(processor, training_stage="autoencoder",
                                      ae_checkpoint=ae_checkpoint).to(device).eval()
    sums, sumsqs, counts = {}, {}, {}
    for start in range(0, len(paths), batch_size):
        graphs = []
        for path in paths[start:start + batch_size]:
            with path.open("rb") as handle:
                scene = pickle.load(handle)
            graphs.append(HeteroData(adapt_preprocessed_scene(scene, path.name)))
        batch = Batch.from_data_list(graphs).to(device)
        tokens, agent = processor(batch)
        agent["tokenized_map"] = tokens
        graph, _, _, _ = decoder._build_graph(agent)
        am, lm, av, lv = decoder.autoencoder.forward_encoder(graph, return_stats=True)
        # Native LDM training draws from the AE posterior, rather than training
        # solely on mu. Estimate its moments analytically, avoiding sample noise.
        for kind, mean, log_var in (("agent", am, av), ("lane", lm, lv)):
            dim = int(decoder.ae_config[f"{kind}_latent_dim"])
            if (mean.ndim != 2 or mean.shape[1] != dim or len(mean) == 0
                    or log_var.shape != mean.shape):
                raise ValueError(f"Invalid {kind} posterior shape: expected nonempty [nodes, {dim}] mean/logvar")
            mean, log_var = mean.double().cpu(), log_var.double().cpu()
            second_moment = mean.square() + log_var.exp()
            if not torch.isfinite(mean).all() or not torch.isfinite(log_var).all() or not torch.isfinite(second_moment).all():
                raise ValueError(f"Non-finite {kind} posterior moments")
            sums[kind] = sums.get(kind, 0) + mean.sum(0)
            sumsqs[kind] = sumsqs.get(kind, 0) + second_moment.sum(0)
            counts[kind] = counts.get(kind, 0) + len(mean)
    cfg = OmegaConf.load(base_config or Path(__file__).parent / "smart/vectorworld/waymo_flow.yaml")
    for key in DATASET_KEYS:
        if key in decoder.ae_cfg.dataset and "latents_" not in key:
            cfg.dataset[key] = plain(decoder.ae_cfg.dataset[key])
    for kind in ("agent", "lane"):
        dim = int(decoder.ae_config[f"{kind}_latent_dim"])
        cfg.model[f"{kind}_latent_dim"] = dim
        mean = sums[kind] / counts[kind]
        variance = (sumsqs[kind] / counts[kind] - mean.square()).clamp_min(1e-12)
        cfg.dataset[f"{kind}_latents_mean"] = mean.tolist()
        cfg.dataset[f"{kind}_latents_std"] = variance.sqrt().tolist()
    validate_stats(cfg)
    return cfg, dict(ae_checkpoint=str(resolve_path(ae_checkpoint)),
                    data_dir=str(root), num_scenes=len(paths),
                    requested_max_scenes=max_scenes, batch_size=batch_size,
                    sample_list=str(manifest) if manifest is not None else None,
                    sample_list_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest is not None else None,
                    agent_nodes=counts["agent"], lane_nodes=counts["lane"], seed=seed,
                    agent_latent_dim=int(cfg.model.agent_latent_dim), lane_latent_dim=int(cfg.model.lane_latent_dim),
                    estimator="per-dimension E[mu], E[mu^2+exp(logvar)]",
                    ordered_scene_names_sha256=hashlib.sha256("\n".join(p.name for p in paths).encode()).hexdigest())

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Full generator YAML usable as ldm_config")
    parser.add_argument("--sample-list", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--max-scenes", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg, report = compute_stats(args.ae_checkpoint, args.data_dir, max_scenes=args.max_scenes,
        batch_size=args.batch_size, sample_list=args.sample_list, seed=args.seed,
        device=args.device, base_config=args.base_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Latent stats from {report['num_scenes']} scenes -> {args.output}")

if __name__ == "__main__":
    main()
