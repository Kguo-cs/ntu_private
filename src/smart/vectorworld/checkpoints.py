"""Read native VectorWorld and SMART checkpoints without retaining storage credentials."""
from pathlib import Path
import math
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
DATASET_KEYS = (
    "max_num_agents", "max_num_lanes", "num_map_ids", "num_agent_types", "num_lane_types",
    "num_lane_connection_types", "num_points_per_lane", "fov", "motion",
    "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
    "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y",
    "agent_latents_mean", "agent_latents_std", "lane_latents_mean", "lane_latents_std",
)
TRAIN_KEYS = ("loss_type", "lane_weight", "guidance_scale", "ema_decay")
PREFIX = "encoder.init_decoder."

def plain(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value

def public_config(cfg):
    """Only model hyperparameters and public normalization constants are saved."""
    model = plain(cfg["model"])
    for key in ("autoencoder_path", "autoencoder_run_name", "flow_grpo",
                "meanflow_distill_teacher_ckpt"):
        model.pop(key, None)
    dataset = {key: plain(cfg["dataset"][key]) for key in DATASET_KEYS if key in cfg["dataset"]}
    train = {key: plain(cfg["train"][key]) for key in TRAIN_KEYS if key in cfg.get("train", {})}
    return OmegaConf.create(dict(model=model, dataset=dataset, train=train, dataset_name="waymo"))

def resolve_path(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()

def load_checkpoint(path):
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing VectorWorld checkpoint: {path}")
    # Released files include optimizer states; mmap avoids loading these into RAM.
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Expected a Lightning/SMART checkpoint with state_dict: {path}")
    return ckpt

def strip_weights(state, prefix):
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not weights:
        raise ValueError(f"Checkpoint contains no weights with prefix {prefix}")
    return weights

def read_autoencoder(ckpt):
    hp = ckpt.get("hyper_parameters", {})
    if "cfg" in hp:
        cfg = public_config(hp["cfg"])
        prefix = "model."
    else:
        extra = ckpt["state_dict"].get(PREFIX + "_extra_state", {})
        if extra.get("loss_kind") != "vectorworld" or "ae_config" not in extra:
            raise ValueError("Expected a VectorWorld AE checkpoint, not a Scenario Dreamer AE")
        cfg = public_config(extra["ae_config"])
        prefix = PREFIX + "autoencoder."
    return cfg, strip_weights(ckpt["state_dict"], prefix)

def read_generator(ckpt):
    hp = ckpt.get("hyper_parameters", {})
    if "cfg_ae" in hp and "cfg" in hp:
        cfg, ae_cfg = public_config(hp["cfg"]), public_config(hp["cfg_ae"])
        generator_prefix, ae_prefix = "gen_model.", "autoencoder.model."
        ema = ckpt.get("ema_state_dict")
    else:
        extra = ckpt["state_dict"].get(PREFIX + "_extra_state", {})
        if extra.get("loss_kind") != "vectorworld" or extra.get("training_stage") != "ldm":
            raise ValueError("Expected a VectorWorld generator checkpoint")
        cfg, ae_cfg = public_config(extra["ldm_config"]), public_config(extra["ae_config"])
        generator_prefix, ae_prefix = PREFIX + "diff_model.", PREFIX + "autoencoder."
        ema = extra.get("ema")
    if ema is None:
        raise ValueError("VectorWorld generator checkpoint must contain its EMA state")
    return (cfg, ae_cfg, strip_weights(ckpt["state_dict"], generator_prefix),
            strip_weights(ckpt["state_dict"], ae_prefix), ema)

def validate_stats(cfg):
    for kind in ("agent", "lane"):
        dim = int(cfg.model[f"{kind}_latent_dim"])
        for stat in ("mean", "std"):
            key = f"{kind}_latents_{stat}"
            if key not in cfg.dataset or cfg.dataset[key] is None:
                raise ValueError(f"Missing {key}; compute latent statistics with the trained VectorWorld AE")
            x = np.asarray(plain(cfg.dataset[key]), dtype=np.float64)
            if x.ndim > 1 or (x.ndim == 1 and x.size != dim) or not np.isfinite(x).all():
                raise ValueError(f"Invalid {key}: expected a finite scalar or {dim}-element vector")
            if stat == "std" and (x <= 0).any():
                raise ValueError(f"{key} must be strictly positive")
