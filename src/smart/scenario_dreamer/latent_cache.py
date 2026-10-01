"""Versioned AE posterior sidecars for the existing Scenario Dreamer pipeline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf

FORMAT_VERSION = 1
PREPROCESS_KEYS = (
    "max_num_agents", "max_num_lanes", "num_points_per_lane", "fov",
    "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
    "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y",
)


def encoder_fingerprint(autoencoder, ae_config, dataset_config):
    """Identify actual AE tensors and encoding conventions, including after resume."""
    config = {
        "version": FORMAT_VERSION,
        "ae": OmegaConf.to_container(ae_config, resolve=True),
        "preprocess": {key: dataset_config[key] for key in PREPROCESS_KEYS},
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, value in sorted(autoencoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def cache_path(root, filename):
    if Path(filename).name != filename:
        raise ValueError("Latent cache keys must be scene basenames")
    # Avoid adding another million files to one flat directory.
    shard = hashlib.sha256(filename.encode()).hexdigest()[:2]
    return Path(root) / shard / (filename + ".pt")


def atomic_save(path, value, *, json_format=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        if json_format:
            temporary.write_text(json.dumps(value, indent=2) + "\n")
        else:
            torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_manifest(root):
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing latent cache manifest: {path}; run action=cache_latents first")
    manifest = json.loads(path.read_text())
    if manifest.get("version") != FORMAT_VERSION or not manifest.get("encoder_fingerprint"):
        raise ValueError(f"Unsupported latent cache manifest: {path}")
    return manifest


def load_record(root, filename, source_sha256, manifest, num_agents, num_lanes):
    path = cache_path(root, filename)
    if not path.is_file():
        raise FileNotFoundError(f"Missing latent cache for {filename}: {path}; complete action=cache_latents first")
    record = torch.load(path, map_location="cpu", weights_only=True)
    if (record.get("version") != FORMAT_VERSION or record.get("filename") != filename
            or record.get("encoder_fingerprint") != manifest["encoder_fingerprint"]):
        raise ValueError(f"Incompatible latent cache record: {path}")
    if record.get("source_sha256") != source_sha256:
        raise ValueError(f"Stale latent cache: source scene changed for {filename}; regenerate its cache")
    for kind, count in (("agent", num_agents), ("lane", num_lanes)):
        for stat in ("mu", "log_var"):
            key = f"{kind}_{stat}"
            value = record.get(key)
            expected = (count, manifest[f"{kind}_latent_dim"])
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or tuple(value.shape) != expected or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid {key} in latent cache {path}; expected finite float32 {expected}")
    return record


@torch.no_grad()
def precache(decoder, dataset, output_dir, *, batch_size=16, num_workers=0,
             device="cuda", max_scenes=None, overwrite=False):
    """Cache unnormalized posterior statistics in the original pickle node order."""
    from torch_geometric.data import Batch
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from tqdm import tqdm

    if not dataset.scenario_dreamer_preprocessed or not dataset.record_latent_source:
        raise ValueError("Pre-caching requires official AE scenes with record_latent_source=True")
    if dataset.latent_cache_dir is not None:
        raise ValueError("Pre-caching must read raw scenes, not another latent cache")
    if decoder.map_source != "exact":
        raise ValueError("Pre-caching requires map_source=exact")
    if batch_size < 1 or num_workers < 0 or (max_scenes is not None and max_scenes < 1):
        raise ValueError("batch_size/max_scenes must be positive and num_workers nonnegative")
    selected = min(len(dataset), max_scenes) if max_scenes is not None else len(dataset)
    if not selected:
        raise ValueError("No scenes selected for latent pre-caching")
    identity = encoder_fingerprint(decoder.autoencoder, decoder.ae_config, decoder.cfg.dataset)
    manifest = {
        "version": FORMAT_VERSION, "encoder_fingerprint": identity,
        "agent_latent_dim": int(decoder.ae_config.agent_latent_dim),
        "lane_latent_dim": int(decoder.ae_config.lane_latent_dim),
        "ae_config": OmegaConf.to_container(decoder.ae_config, resolve=True),
        "preprocess": {key: decoder.cfg.dataset[key] for key in PREPROCESS_KEYS},
        "source_root": str(Path(dataset.raw_paths[0]).parent.resolve()),
        "node_order": "original AE pickle; ego first; original lane order",
        "statistics": "unnormalized posterior mean and log variance",
    }
    output_dir = Path(output_dir).expanduser().resolve()
    if (output_dir / "manifest.json").exists():
        previous = read_manifest(output_dir)
        if previous["encoder_fingerprint"] != identity:
            raise ValueError("Output cache belongs to a different AE/config; choose a new output_dir")
    else:
        atomic_save(output_dir / "manifest.json", manifest, json_format=True)
    decoder = decoder.to(device).eval()
    loader = DataLoader(Subset(dataset, range(selected)), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, persistent_workers=num_workers > 0)
    moments = {kind: [0, 0.0, 0.0] for kind in ("agent", "lane")}

    def accumulate(record):
        for kind in moments:
            mean = record[f"{kind}_mu"].double()
            variance = record[f"{kind}_log_var"].double().exp()
            moments[kind][0] += mean.numel()
            moments[kind][1] += mean.sum().item()
            moments[kind][2] += (mean.square() + variance).sum().item()

    written = skipped = 0
    for batch in tqdm(loader, desc="Caching AE posteriors"):
        scenes = batch.to_data_list()
        pending = []
        for scene in scenes:
            filename = scene.scenario_dreamer_cache_file
            if not overwrite and cache_path(output_dir, filename).is_file():
                record = load_record(output_dir, filename, scene.sd_source_sha256, manifest,
                                     scene["sd_agent"].num_nodes, scene["sd_lane"].num_nodes)
                accumulate(record)
                skipped += 1
            else:
                pending.append(scene)
        if not pending:
            continue
        batch = Batch.from_data_list(pending).to(device)
        tokenized_map, agent = decoder.token_processor(batch)
        agent["tokenized_map"] = tokenized_map
        graph, rows, _, _ = decoder._build_graph(agent)
        am, lm, av, lv = decoder.autoencoder.forward_encoder(graph, return_stats=True)
        # Reverse canonical recursive sorting before splitting into sidecars.
        agent_order = rows.argsort()
        lane_order = graph["lane"].source_row.argsort()
        am, av = am[agent_order].cpu(), av[agent_order].cpu()
        lm, lv = lm[lane_order].cpu(), lv[lane_order].cpu()
        agent_batch = batch["sd_agent"].batch.cpu()
        lane_batch = batch["sd_lane"].batch.cpu()
        for index, scene in enumerate(pending):
            n = scene["sd_agent"].num_nodes
            # SMART stores ego last; cache stores original ego-first order.
            ego_first = torch.cat((torch.tensor([n - 1]), torch.arange(n - 1)))
            record = {
                "version": FORMAT_VERSION, "encoder_fingerprint": identity,
                "filename": scene.scenario_dreamer_cache_file,
                "source_sha256": scene.sd_source_sha256,
                "agent_mu": am[agent_batch == index][ego_first].contiguous(),
                "agent_log_var": av[agent_batch == index][ego_first].contiguous(),
                "lane_mu": lm[lane_batch == index].contiguous(),
                "lane_log_var": lv[lane_batch == index].contiguous(),
            }
            if not all(torch.isfinite(record[key]).all() for key in ("agent_mu", "agent_log_var", "lane_mu", "lane_log_var")):
                raise ValueError(f"Non-finite AE posterior for {record['filename']}")
            atomic_save(cache_path(output_dir, record["filename"]), record)
            accumulate(record)
            written += 1
    statistics = {}
    for kind, (count, total, second) in moments.items():
        mean = total / count
        statistics[f"{kind}_latents_mean"] = mean
        statistics[f"{kind}_latents_std"] = max(0, second / count - mean * mean) ** 0.5
    summary = {
        "encoder_fingerprint": identity, "selected_scenes": selected, "source_scenes": len(dataset),
        "partial": selected != len(dataset), "written": written, "reused": skipped,
        "latent_statistics": statistics,
        "statistics_method": "posterior population moments: E[z]=mu, E[z^2]=mu^2+exp(log_var)",
    }
    atomic_save(output_dir / "summary.json", summary, json_format=True)
    return summary


def run_precache(cfg):
    """src.run action=cache_latents; reuse the native dataset, tokens and AE."""
    from src.smart.datasets.scalable_dataset import MultiDataset
    from src.smart.datamodules.target_builder import WaymoTargetBuilderVal
    from src.smart.tokens.token_processor import TokenProcessor
    from .decoder import ScenarioDreamerInitDecoder

    options = cfg.latent_cache
    split = options.split
    if split not in ("train", "val", "test"):
        raise ValueError("latent_cache.split must be train, val or test")
    if cfg.ckpt_path is not None:
        raise ValueError("For pre-caching, set decoder.scenario_dreamer.ae_checkpoint instead of ckpt_path")
    model_options = dict(cfg.model.model_config.decoder.scenario_dreamer)
    if model_options.get("ae_checkpoint") is None:
        raise ValueError("Latent pre-caching requires a trained ae_checkpoint")
    if model_options.get("ldm_checkpoint") is not None or model_options.get("ldm_config") is not None:
        raise ValueError("Pre-cache the AE with ldm_checkpoint=null and ldm_config=null")
    model_options.update(training_stage="autoencoder", map_source="exact")
    processor = TokenProcessor(**cfg.model.model_config.token_processor)
    decoder = ScenarioDreamerInitDecoder(processor, **model_options)
    dataset = MultiDataset(
        cfg.data[f"{split}_raw_dir"], WaymoTargetBuilderVal(),
        scenario_dreamer_preprocessed=True, record_latent_source=True,
        sample_list=cfg.data.scenario_dreamer_eval_set if split != "train" else None,
    )
    summary = precache(decoder, dataset, options.output_dir, batch_size=options.batch_size,
                       num_workers=cfg.data.num_workers, device=options.device,
                       max_scenes=options.max_scenes, overwrite=options.overwrite)
    print(json.dumps({"output_dir": str(options.output_dir), **summary}, indent=2))
    return summary
