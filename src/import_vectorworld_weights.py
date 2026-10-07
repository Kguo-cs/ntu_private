"""Import inference weights from a local VectorWorld checkout into sim.

Drops optimizer states and storage credentials; keeps native parameter names,
embedded VAE, EMA, and public configs so runtime needs no external checkout.
"""
from pathlib import Path
import argparse
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.smart.vectorworld.checkpoints import public_config

def import_weights(source_root, output_root, variants=("autoencoder", "flow", "meanflow", "diffusion")):
    source_root, output_root = Path(source_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = {"autoencoder": "sd_ae_waymo_motion_large/last.ckpt", "flow": "flow.ckpt",
               "meanflow": "mean_flow.ckpt", "diffusion": "diffusion.ckpt"}
    paths = []
    for variant in variants:
        path = source_root / sources[variant]
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        hp = checkpoint["hyper_parameters"]
        cfg = public_config(hp["cfg"])
        prefixes = ("model.",) if variant == "autoencoder" else ("gen_model.", "autoencoder.model.")
        state = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefixes)}
        if not state:
            raise ValueError(f"No VectorWorld weights found: {path}")
        output = {"state_dict": state, "hyper_parameters": {"cfg": cfg},
                  "global_step": int(checkpoint.get("global_step", 0)),
                  "epoch": int(checkpoint.get("epoch", 0))}
        if variant != "autoencoder":
            output["hyper_parameters"]["cfg_ae"] = public_config(hp["cfg_ae"])
            output["ema_state_dict"] = checkpoint["ema_state_dict"]
        target = output_root / f"{variant}.ckpt"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an imported checkpoint: {target}")
        temp = target.with_suffix(".ckpt.tmp")
        torch.save(output, temp)
        temp.replace(target)
        print(f"{variant}: {len(state)} tensors -> {target}", flush=True)
        paths.append(target)
        del checkpoint, state, output
    return paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/ke/code/VectorWorld/tools/outputs/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "waymo_data/vectorworld/checkpoints")
    parser.add_argument("--variants", nargs="+", choices=("autoencoder", "flow", "meanflow", "diffusion"),
                        default=("autoencoder", "flow", "meanflow", "diffusion"))
    args = parser.parse_args()
    import_weights(args.source, args.output, args.variants)

if __name__ == "__main__":
    main()
