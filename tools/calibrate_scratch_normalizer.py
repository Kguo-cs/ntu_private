"""Fit the InitDiffusion noise statistics on training data, without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def state_dimensions(size_representation, velocity_representation):
    """Names describe the actual coordinates returned by InitDenoiser.get_input."""
    if size_representation not in ("linear", "log"):
        raise ValueError("size_representation must be linear or log")
    if velocity_representation not in ("speed", "vector"):
        raise ValueError("velocity_representation must be speed or vector")
    sizes = ["length", "width"] if size_representation == "linear" else ["log_length", "log_width"]
    motion = ["speed"] if velocity_representation == "speed" else ["velocity_x", "velocity_y"]
    return ["x", "y", "cos_heading", "sin_heading", *sizes, *motion]


def state_observations(state, size_valid_mask=None):
    """Exclude missing log-size annotations, including their finite input fills."""
    if state.ndim != 2 or state.shape[1] not in (7, 8):
        raise ValueError("Calibration state must be [N,7] or [N,8]")
    if state.device.type != "cpu":
        raise ValueError("Normalizer calibration must run on CPU")
    if not torch.isfinite(state).all():
        raise ValueError("Nonfinite training calibration state")
    observed = torch.ones_like(state, dtype=torch.bool)
    if size_valid_mask is not None:
        if size_valid_mask.shape != (len(state), 2):
            raise ValueError("Size annotation mask must align with [N,2] size fields")
        observed[:, 4:6] = size_valid_mask.bool()
    values = torch.where(observed, state.double(), torch.zeros_like(state, dtype=torch.float64))
    return values.sum(0), values.square().sum(0), observed.sum(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=4717)
    parser.add_argument("--size-representation", choices=("linear", "log"), default="linear")
    parser.add_argument("--velocity-representation", choices=("speed", "vector"), default="speed")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.scenes < 1 or args.workers < 0:
        parser.error("--scenes must be positive and --workers nonnegative")
    started = time.monotonic()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "src/waymo_data/full"
    with (cache / "training_map2_sd_files.pkl").open("rb") as f:
        all_files = pickle.load(f)["files"]
    files = random.Random(args.seed).sample(all_files, args.scenes)
    manifest = output / "calibration_files.pkl"
    with manifest.open("wb") as f:
        pickle.dump({"files": files}, f)
    (output / "calibration_files.txt").write_text("\n".join(files) + "\n")
    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="run", overrides=["experiment=init_diffusion_lane_conditioned_mixed_loss"])
    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT / "src")
        cfg.paths.output_dir = str(output)
        cfg.paths.log_dir = str(output)
        cfg.ckpt_path = None
        cfg.action = "fit"
        cfg.model.model_config.finetune = False
        cfg.model.model_config.decoder.init_diffusion.normalizer_path = None
        cfg.model.model_config.decoder.init_diffusion.size_representation = args.size_representation
        cfg.model.model_config.decoder.init_diffusion.velocity_representation = args.velocity_representation
        cfg.model.model_config.decoder.init_diffusion.invalid_size_policy = "mask"
        cfg.data.train_batch_size = 64
        cfg.data.num_workers = args.workers
        cfg.data.pin_memory = False
        cfg.data.shuffle = False
        cfg.data.persistent_workers = False
        cfg.data.scenario_dreamer_train_sample_list = str(manifest)
        # setup also constructs validation: select a tiny known filename list.
        cfg.data.scenario_dreamer_eval_set = str(manifest)
    # Avoid using the training manifest against the different validation root.
    tiny_val = output / "unused_val_files.pkl"
    with tiny_val.open("wb") as f:
        pickle.dump({"files": [next((cache / "scenario_dreamer_val").glob("*.pt")).name]}, f)
    cfg.data.scenario_dreamer_eval_set = str(tiny_val)
    OmegaConf.resolve(cfg)
    torch.set_num_threads(4)
    L.seed_everything(args.seed, workers=True)
    dm = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.train()
    dm.setup("fit")
    dimensions = state_dimensions(args.size_representation, args.velocity_representation)
    count = 0
    scene_count = 0
    invalid_size_fields = 0
    invalid_size_agents = 0
    total = torch.zeros(len(dimensions), dtype=torch.float64)
    squares = torch.zeros_like(total)
    observations = torch.zeros(len(dimensions), dtype=torch.long)
    with torch.no_grad():
        for batch_i, batch in enumerate(dm.train_dataloader()):
            _, agents = model.token_processor(batch)
            model.encoder.init_decoder._prepare_ego_context(agents)
            _, state = model.encoder.init_decoder.G1.model.get_input(agents)
            if state.shape[-1] != len(dimensions):
                raise ValueError("Configured representation differs from actual get_input state")
            size_valid_mask = (agents.get("_init_diffusion_size_valid_mask")
                               if args.size_representation == "log" else None)
            batch_total, batch_squares, batch_observations = state_observations(state, size_valid_mask)
            count += len(state)
            scene_count += int(agents["num_graphs"])
            total += batch_total
            squares += batch_squares
            observations += batch_observations
            if size_valid_mask is not None:
                invalid_size_fields += int((~size_valid_mask).sum())
                invalid_size_agents += int((~size_valid_mask).any(-1).sum())
            elif args.size_representation == "linear":
                invalid_size = ~torch.isfinite(state[:, 4:6]) | (state[:, 4:6] <= 0)
                invalid_size_fields += int(invalid_size.sum())
                invalid_size_agents += int(invalid_size.any(-1).sum())
            if (batch_i + 1) % 16 == 0:
                print(f"CALIBRATION scenes={scene_count} agents={count}", flush=True)
    if scene_count != args.scenes:
        raise ValueError(f"Calibration loaded {scene_count} scenes, requested {args.scenes}")
    if (observations == 0).any():
        raise ValueError("Calibration has a coordinate with no valid TRAIN observations")
    mean = total / observations
    scale = (squares / observations - mean.square()).clamp_min(1e-12).sqrt()
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU calibration unexpectedly initialized CUDA")
    report = dict(normal_mean=[mean.tolist()], normal_scale=[scale.tolist()],
                  scenes=args.scenes, agents=count, seed=args.seed,
                  train_manifest=str(manifest), source="training_map2_sd; no checkpoint loaded",
                  train_manifest_sha256=hashlib.sha256((output / "calibration_files.txt").read_bytes()).hexdigest(),
                  dimensions=dimensions, size_representation=args.size_representation,
                  velocity_representation=args.velocity_representation, workers=args.workers,
                  observation_counts=observations.tolist(), invalid_size_fields=invalid_size_fields,
                  invalid_size_agents=invalid_size_agents,
                  invalid_size_statistics=("Log-size nonfinite/nonpositive annotations are excluded per coordinate; "
                                           "finite model input fills never enter statistics. Linear mode preserves "
                                           "the original all-finite-state observation algorithm."),
                  log_input_fill=("Current InitDenoiser mask policy: same-type log mean, then batch log mean, "
                                  "then nominal token dimensions. generate_type=true uses one common group, "
                                  "preventing type leakage through the fill."),
                  cpu_only=True, cuda_initialized=False, checkpoint_loaded=False,
                  calibration_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256((cache / "training_map2_sd_files.pkl").read_bytes()).hexdigest(),
                  sample_algorithm="random.Random(seed).sample(original training_map2_sd_files.pkl files, scenes)",
                  elapsed_seconds=time.monotonic() - started)
    report["metadata_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    (output / "normalizer.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "calibration_audit.json").write_text(json.dumps({
        "normalizer_sha256": hashlib.sha256((output / "normalizer.json").read_bytes()).hexdigest(),
        "metadata_sha256": report["metadata_sha256"],
        "train_manifest_sha256": report["train_manifest_sha256"],
        "cpu_only": True, "cuda_initialized": False, "checkpoint_loaded": False,
    }, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
