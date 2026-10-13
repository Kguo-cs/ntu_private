"""Evaluate one checkpoint under several inference variants in one process.

Example:
  python tools/cc_sweep_eval.py --ckpt path/last.ckpt --out results.jsonl \
      --variants '[{"name":"base"},{"name":"n40","steps":40}]' -- experiment=...
Each variant may set: ema (bool), steps (int),
guidance (collision_guidance mapping), exact (bool), seed (int),
time_grid (sampling_time_grid mapping),
a2a_radius / a2a_max_num_neighbors / map2agent_radius /
map2agent_max_num_neighbors (inference-time graph override). Metrics come from the
native ScenarioDreamer evaluator via the model's own test loop.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--variants", required=True, help="JSON list or @file")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--limit", type=float, default=1.0, help="limit_test_batches")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    args.ckpt, args.out = str(Path(args.ckpt).resolve()), str(Path(args.out).resolve())

    gc.disable()
    import torch
    from tools.runtime_cached_torch_loader_sweep import install_cached_loader
    install_cached_loader(torch)
    import hydra
    import lightning as L
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("sim_root", lambda: str(ROOT), replace=True)
    variants = json.loads(Path(args.variants[1:]).read_text() if args.variants.startswith("@") else args.variants)
    workdir = Path(args.workdir or Path(args.out).with_suffix("")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    with initialize_config_dir(str(ROOT / "configs"), version_base=None):
        cfg = compose("run", overrides=["action=test"] + list(args.overrides))
    torch.set_float32_matmul_precision(cfg.get("float32_matmul_precision", "highest"))
    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=True)
    decoder = model.encoder.init_decoder
    flow = decoder.G1
    base_guidance = dict(flow.collision_guidance)
    from src.smart.diffusion.sampling_time_grid import parse_sampling_time_grid
    base_time_grid = dict(flow.sampling_time_grid)
    graph_names = ("a2a_radius", "a2a_max_num_neighbors", "map2agent_radius", "map2agent_max_num_neighbors")
    base_graph = {name: getattr(flow.model, name) for name in graph_names}

    for variant in variants:
        decoder.use_ema = bool(variant.get("ema", True))
        decoder.sampling_steps = int(variant.get("steps", 20))
        flow.sampling_exact_x0 = bool(variant.get("exact", False))
        flow.sampling_time_grid = (parse_sampling_time_grid(variant["time_grid"])
                                   if "time_grid" in variant else dict(base_time_grid))
        flow.collision_guidance = dict(base_guidance, **variant.get("guidance", {}))
        for name, value in base_graph.items():
            setattr(flow.model, name, type(value)(variant.get(name, value)))
        L.seed_everything(int(variant.get("seed", cfg.seed)), workers=True)
        trainer = L.Trainer(accelerator="gpu", devices=1, logger=False, precision=cfg.trainer.precision,
                            limit_test_batches=args.limit, enable_checkpointing=False,
                            enable_progress_bar=False, enable_model_summary=False)
        started = time.time()
        trainer.test(model=model, datamodule=datamodule, verbose=False)
        report = json.loads((workdir / "sd_agent_metrics.json").read_text())
        record = dict(variant=variant, ckpt=args.ckpt, seconds=round(time.time() - started, 1),
                      metrics=report["agent_metrics"],
                      num_generated_vehicles=report["num_generated_vehicles"],
                      num_gt_vehicles=report["num_gt_vehicles"],
                      num_samples=report["num_samples"])
        evaluator = model.sd_evaluator
        record["histograms"] = dict(generated=[h.tolist() for h in evaluator.generated.histograms],
                                    real=[h.tolist() for h in evaluator.real.histograms])
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        print("RESULT", json.dumps(record), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
