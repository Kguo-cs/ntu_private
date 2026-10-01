"""Scenario Dreamer as an ordinary SMART init_decoder (no external checkout)."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from .core.autoencoder import AutoEncoder
from .core.ldm import LDM
from .core.data_helpers import unnormalize_scene
from .data import build_graph, rotate

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "src/waymo_data/scenario_dreamer/checkpoints"


def _checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing Scenario Dreamer checkpoint: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" not in checkpoint or "cfg" not in checkpoint.get("hyper_parameters", {}):
        raise ValueError(f"Expected an official Scenario Dreamer Lightning checkpoint: {path}")
    return checkpoint


class ScenarioDreamerInitDecoder(nn.Module):
    """Train with the existing initial_logit loss; infer the existing 5-tuple.

    Inference supports joint lane/agent generation or lane conditioning, with fixed scene counts.
    Official checkpoint parameter names and EMA order are retained in the core.
    AE parameters remain frozen; SMART's optimizer trains the LDM.
    """
    use_gan = False
    learn_autoencoder = False
    use_rl = False
    loss_kind = "scenario_dreamer"

    def __init__(self, token_processor, *, ae_checkpoint=None, ldm_checkpoint=None,
                 map_source="auto", map_id=0, use_ema=True, generation_mode="initial_scene"):
        super().__init__()
        self.token_processor = token_processor
        self.map_source = map_source
        self.map_id = int(map_id)
        self.use_ema = bool(use_ema)
        self.generation_mode = generation_mode
        if generation_mode not in ("initial_scene", "lane_conditioned"):
            raise ValueError("generation_mode must be initial_scene or lane_conditioned")
        if self.map_source not in ("auto", "exact", "tokens") or self.map_id not in (0, 1):
            raise ValueError("Use map_source=auto/exact/tokens and Waymo map_id=0/1")
        ae = _checkpoint(ae_checkpoint or WEIGHTS / "scenario_dreamer_autoencoder_waymo/last.ckpt")
        ldm = _checkpoint(ldm_checkpoint or WEIGHTS / "scenario_dreamer_ldm_large_waymo/last.ckpt")
        if "ema_state_dict" not in ldm:
            raise ValueError("The official LDM checkpoint must include ema_state_dict")
        ae_cfg = ae["hyper_parameters"]["cfg"]
        self.ae_config = OmegaConf.create(OmegaConf.to_container(ae_cfg.model, resolve=True))
        saved = ldm["hyper_parameters"]["cfg"]
        model_cfg = {key: saved.model[key] for key in saved.model
                     if key not in ("autoencoder_run_name", "autoencoder_path")}
        dataset_cfg = {key: value for key, value in OmegaConf.to_container(saved.dataset, resolve=False).items()
                       if isinstance(value, (int, float, bool))}
        dataset_cfg.update(num_map_ids=2, num_agent_types=3, num_lane_types=0)
        train_cfg = {key: saved.train[key] for key in ("loss_type", "lane_weight", "guidance_scale", "ema_decay")}
        self.cfg = OmegaConf.create(dict(model=model_cfg, dataset=dataset_cfg, train=train_cfg, dataset_name="waymo"))
        # The full checkpoint embeds AE tensors as well as LDM tensors.
        self.diff_model = LDM(self.cfg)
        self.autoencoder = AutoEncoder(self.ae_config)
        self.autoencoder.load_state_dict({key.removeprefix("model."): value
                                         for key, value in ae["state_dict"].items()}, strict=True)
        # self.diff_model.load_state_dict({key.removeprefix("diff_model."): value
        #                                 for key, value in ldm["state_dict"].items()
        #                                 if key.startswith("diff_model.")}, strict=True)
        embedded = {key.removeprefix("autoencoder.model."): value for key, value in ldm["state_dict"].items()
                    if key.startswith("autoencoder.model.")}
        self.autoencoder.load_state_dict(embedded, strict=True)
        self.autoencoder.requires_grad_(False).eval()
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)
        # self.ema.load_state_dict(ldm["ema_state_dict"])
        # self.checkpoint_step = int(ldm.get("global_step", 0))

    def train(self, mode=True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def get_extra_state(self):
        # nn.Module state_dict makes EMA part of ordinary SMART checkpoints.
        return {"ema": self.ema.state_dict(), "checkpoint_step": self.checkpoint_step}

    def set_extra_state(self, state):
        self.ema.load_state_dict(state["ema"])
        self.checkpoint_step = state.get("checkpoint_step", 0)

    @torch.no_grad()
    def update_ema(self):
        self.ema.to(next(self.diff_model.parameters()).device)
        self.ema.update()

    def _build_graph(self, agent):
        agent["sd_map_id"] = self.map_id
        return build_graph(
            agent, agent["tokenized_map"], self.cfg.dataset,
            map_source=self.map_source,
        )

    def _encode(self, agent):
        data, rows, centers, angles = self._build_graph(agent)
        with torch.no_grad():
            am, lm, av, lv = self.autoencoder.forward_encoder(data, return_stats=True)
            stats = self.cfg.dataset
            if self.training:
                a = am + (av * 0.5).exp() * torch.randn_like(am)
                l = lm + (lv * 0.5).exp() * torch.randn_like(lm)
            else:
                a, l = am, lm
            data["agent"].latents = (a - stats.agent_latents_mean) / stats.agent_latents_std
            data["lane"].latents = (l - stats.lane_latents_mean) / stats.lane_latents_std
            # LDM.forward infers noise shapes from x, which must now be latent-sized.
            data["agent"].x = data["agent"].latents
            data["lane"].x = data["lane"].latents
        return data, rows, centers, angles

    def forward(self, tokenized_agent):
        tokenized_agent.pop("generated_map", None)
        if not self.training and self.generation_mode == "initial_scene":
            # Joint generation uses counts and graph structure, never GT AE latents.
            data, rows, centers, angles = self._build_graph(tokenized_agent)
            for kind in ("agent", "lane"):
                width = self.cfg.model[f"{kind}_latent_dim"]
                data[kind].x = data[kind].x.new_zeros((data[kind].num_nodes, width))
        else:
            data, rows, centers, angles = self._encode(tokenized_agent)
        if self.training:
            # The same official joint latent-noise objective used by the release.
            return self.diff_model.loss(data)
        from contextlib import nullcontext
        self.ema.to(next(self.diff_model.parameters()).device)
        with torch.no_grad(), self.ema.average_parameters() if self.use_ema else nullcontext():
            a, l = self.diff_model(data, mode=self.generation_mode)
            stats = self.cfg.dataset
            a = a * stats.agent_latents_std + stats.agent_latents_mean
            l = l * stats.lane_latents_std + stats.lane_latents_mean
            states, lanes, types, _, connections = self.autoencoder.forward_decoder(a, l, data)
            keys = ("fov", "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
                    "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y")
            states, lanes = unnormalize_scene(states, lanes, **{key: stats[key] for key in keys})
            if self.generation_mode == "initial_scene":
                # Keep the five-tuple API; carry the generated graph alongside agents.
                # These are physical SD-local coordinates, before SMART's frame transform.
                tokenized_agent["generated_map"] = {
                    "coordinate_frame": "sd_local", "road_points": lanes,
                    "road_connection_types": connections,
                    "edge_index_lane_to_lane": data["lane", "to", "lane"].edge_index,
                    "batch": data["lane"].batch, "lg_type": data.lg_type,
                }
            return self._smart_output(states, types, rows, data["agent"].batch, centers, angles, tokenized_agent)

    def _smart_output(self, states, types, rows, sd_batch, centers, angles, agent):
        # Restore original SMART order, preserving the exact map-reference SE(2).
        world_pos = rotate(states[:, :2].to(centers.dtype), -angles[sd_batch]) + centers[sd_batch]
        world_heading = torch.atan2(states[:, 4], states[:, 3]) - angles[sd_batch]
        world_heading = (world_heading + torch.pi) % (2 * torch.pi) - torch.pi
        velocity = torch.stack((states[:, 2] * world_heading.cos(), states[:, 2] * world_heading.sin()), -1)
        inverse = torch.argsort(rows)
        pos = world_pos[inverse].to(agent["initial_pos"].dtype)
        heading = world_heading[inverse].to(pos.dtype)
        size = states[inverse, 5:7]
        velocity = velocity[inverse].to(pos.dtype)
        types = types[inverse].long()
        agent["type"] = types
        # Generated types need their own token libraries, not the GT type's tokens.
        token_shapes, all_tokens, final_tokens = self.token_processor._get_agent_tokens(types)
        agent.update(token_agent_shape=token_shapes, token_traj=final_tokens, token_traj_all=all_tokens)
        token_velocity = self.token_processor.token_velocity_in_current_frame(
            final_tokens, self.token_processor.shift * 0.1,
        )
        local_velocity = rotate(velocity, -heading)
        index = (token_velocity - local_velocity[:, None]).norm(dim=-1).argmin(-1)
        # Despite the historical name 'initial_local_vel', existing SMART's
        # InitDiffusion output and evaluators require world-frame velocity here.
        return pos[:, None], heading[:, None], index[:, None], size, velocity
