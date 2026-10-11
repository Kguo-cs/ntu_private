"""Fit the InitDiffusion noise statistics on training data, without loading weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=4717)
    args = parser.parse_args()
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
        cfg.data.train_batch_size = 64
        cfg.data.num_workers = 4
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
    count = 0
    total = torch.zeros(7, dtype=torch.float64)
    squares = torch.zeros_like(total)
    with torch.no_grad():
        for batch_i, batch in enumerate(dm.train_dataloader()):
            _, agents = model.token_processor(batch)
            model.encoder.init_decoder._prepare_ego_context(agents)
            _, state = model.encoder.init_decoder.G1.model.get_input(agents)
            state = state.double()
            if not torch.isfinite(state).all():
                raise ValueError("Nonfinite training calibration state")
            count += len(state)
            total += state.sum(0)
            squares += state.square().sum(0)
            if (batch_i + 1) % 16 == 0:
                print(f"CALIBRATION scenes={(batch_i+1)*64} agents={count}", flush=True)
    mean = total / count
    scale = (squares / count - mean.square()).clamp_min(1e-12).sqrt()
    report = dict(normal_mean=[mean.tolist()], normal_scale=[scale.tolist()],
                  scenes=args.scenes, agents=count, seed=args.seed,
                  train_manifest=str(manifest), source="training_map2_sd; no checkpoint loaded",
                  train_manifest_sha256=hashlib.sha256((output / "calibration_files.txt").read_bytes()).hexdigest(),
                  dimensions=["x", "y", "cos_heading", "sin_heading", "length", "width", "speed"])
    (output / "normalizer.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
