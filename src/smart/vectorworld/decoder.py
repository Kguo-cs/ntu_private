"""VectorWorld VAE and EGR-DiT as a SMART initialization decoder."""
from __future__ import annotations

from contextlib import nullcontext
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch_ema import ExponentialMovingAverage
from omegaconf import OmegaConf

from .core import AutoEncoder, LDM, FlowLDM, MeanFlowLDM
from .core.utils.data_helpers import normalize_latents, unnormalize_latents
from .checkpoints import (ROOT, plain, public_config, resolve_path, load_checkpoint,
                          read_autoencoder, read_generator, validate_stats)
from .data import build_graph, build_generation_graph
from ..scenario_dreamer.core.data_helpers import unnormalize_scene
from ..scenario_dreamer.decoder import ScenarioDreamerInitDecoder
from ..scenario_dreamer.generation import SceneCountPrior, DEFAULT_COUNT_PRIOR

DEFAULT_AE_CONFIG = Path(__file__).with_name("waymo_vae.yaml")
DEFAULT_LDM_CONFIG = Path(__file__).with_name("waymo_flow.yaml")
DEFAULT_AE_CHECKPOINT = ROOT / "src/waymo_data/vectorworld/checkpoints/autoencoder.ckpt"


def _config_override(value):
    if isinstance(value, (str, Path)):
        return OmegaConf.load(resolve_path(value))
    return value or {}


def _build_model(model_class, config, weights=None):
    if weights is None:
        return model_class(config)
    # Avoid allocating a second large randomly initialized model when loading
    # released weights. assign=True retains mmap-backed parameter storage.
    with torch.device("meta"):
        model = model_class(config)
    model.load_state_dict(weights, strict=True, assign=True)
    return model


class VectorWorldInitDecoder(nn.Module):
    use_gan = False
    use_rl = False
    loss_kind = "vectorworld"

    def __init__(self, token_processor, *, training_stage="ldm", training_mode="joint",
                 ae_checkpoint=DEFAULT_AE_CHECKPOINT, ae_config=None, ldm_checkpoint=None,
                 ldm_config=None, map_source="exact", map_id=0, use_ema=True,
                 generation_mode="initial_scene", scene_count_source="input",
                 count_prior_path=None, sampling_seed=0, motion_missing="error",
                 sampling_steps=None, guidance_scale=None):
        super().__init__()
        self.token_processor = token_processor
        if training_stage not in ("autoencoder", "ldm"):
            raise ValueError("training_stage must be autoencoder or ldm")
        if training_mode not in ("joint", "lane_conditioned"):
            raise ValueError("training_mode must be joint or lane_conditioned")
        if generation_mode not in ("initial_scene", "lane_conditioned"):
            raise ValueError("generation_mode must be initial_scene or lane_conditioned")
        if scene_count_source not in ("input", "official_prior"):
            raise ValueError("scene_count_source must be input or official_prior")
        if motion_missing not in ("error", "static_masked"):
            raise ValueError("motion_missing must be error or static_masked")
        if map_source not in ("auto", "exact", "tokens") or int(map_id) not in (0, 1):
            raise ValueError("Use map_source=auto/exact/tokens and Waymo map_id=0/1")
        self.training_stage, self.training_mode = training_stage, training_mode
        self.learn_autoencoder = training_stage == "autoencoder"
        self.generation_mode, self.scene_count_source = generation_mode, scene_count_source
        self.map_source, self.map_id = map_source, int(map_id)
        self.motion_missing = motion_missing
        self.sampling_seed = int(sampling_seed)
        self.use_ema = bool(use_ema) and not self.learn_autoencoder
        self._motion_sources = Counter()
        self._map_sources = Counter()
        self._map_ids = Counter()
        self._placeholder_agents = 0
        self.ae_checkpoint_path = str(resolve_path(ae_checkpoint)) if ae_checkpoint else None
        self.ldm_checkpoint_path = str(resolve_path(ldm_checkpoint)) if ldm_checkpoint else None
        if self.learn_autoencoder and (ldm_checkpoint is not None or ldm_config is not None
                                      or training_mode != "joint"):
            raise ValueError("AE training requires joint mode and no LDM configuration/checkpoint")
        if ae_checkpoint is not None and ae_config is not None:
            raise ValueError("ae_config overrides require ae_checkpoint=null")
        if ldm_checkpoint is not None and ldm_config is not None:
            raise ValueError("Pretrained generators use checkpoint configs; ldm_config requires ldm_checkpoint=null")
        if scene_count_source == "official_prior" and (self.learn_autoencoder
                                                       or generation_mode != "initial_scene"):
            raise ValueError("official_prior requires LDM joint initial_scene generation")

        generator = load_checkpoint(ldm_checkpoint) if ldm_checkpoint is not None else None
        ae_weights, generator_weights, saved_ema = None, None, None
        if generator is not None:
            # Use the embedded AE, whose latents the released generator was trained on.
            self.cfg, self.ae_cfg, generator_weights, ae_weights, saved_ema = read_generator(generator)
        else:
            self.cfg = (OmegaConf.load(resolve_path(ldm_config)) if isinstance(ldm_config, (str, Path))
                        else OmegaConf.merge(OmegaConf.load(DEFAULT_LDM_CONFIG), _config_override(ldm_config)))
            if ae_checkpoint is not None:
                ae = load_checkpoint(ae_checkpoint)
                self.ae_cfg, ae_weights = read_autoencoder(ae)
            else:
                if not self.learn_autoencoder:
                    raise ValueError("Scratch LDM training requires a trained VectorWorld ae_checkpoint")
                self.ae_cfg = OmegaConf.merge(OmegaConf.load(DEFAULT_AE_CONFIG), _config_override(ae_config))
        self.ae_config = self.ae_cfg.model
        self.cfg.dataset.num_points_per_lane = self.ae_config.num_points_per_lane
        self.cfg.dataset.max_num_lanes = self.ae_config.max_num_lanes
        self.autoencoder = _build_model(AutoEncoder, self.ae_config, ae_weights)
        self.diff_model, self.ema, self.count_prior = None, None, None
        self.checkpoint_step = int(generator.get("global_step", 0)) if generator else 0
        if self.learn_autoencoder:
            self.cfg.dataset = OmegaConf.create(plain(self.ae_cfg.dataset))
            return
        for key in ("agent_latent_dim", "lane_latent_dim"):
            if int(self.cfg.model[key]) != int(self.ae_config[key]):
                raise ValueError(f"Generator {key} must match the VectorWorld AE checkpoint")
        validate_stats(self.cfg)
        ldm_type = str(self.cfg.model.ldm_type).lower()
        classes = {"diffusion": LDM, "flow": FlowLDM, "meanflow": MeanFlowLDM, "mf": MeanFlowLDM}
        if ldm_type not in classes:
            raise ValueError("ldm_type must be diffusion, flow or meanflow")
        self.diff_model = _build_model(classes[ldm_type], self.cfg, generator_weights)
        self.autoencoder.requires_grad_(False).eval()
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=float(self.cfg.train.ema_decay))
        if saved_ema is not None:
            self.ema.load_state_dict(saved_ema)
        if guidance_scale is not None:
            self.cfg.train.guidance_scale = float(guidance_scale)
        if sampling_steps is not None:
            if isinstance(sampling_steps, bool) or int(sampling_steps) != sampling_steps or sampling_steps < 1:
                raise ValueError("sampling_steps must be a positive integer")
            if ldm_type in ("meanflow", "mf"):
                self.diff_model.set_num_steps_eval(int(sampling_steps))
                self.cfg.model.meanflow_num_steps_eval = int(sampling_steps)
            elif ldm_type == "flow":
                self.diff_model.n_steps = int(sampling_steps)
                self.cfg.model.flow_num_steps = int(sampling_steps)
            else:
                raise ValueError("DDPM sampling_steps is fixed by its trained diffusion schedule")
        if scene_count_source == "official_prior":
            path = resolve_path(count_prior_path) if count_prior_path else DEFAULT_COUNT_PRIOR
            self.count_prior = SceneCountPrior(path, max_num_agents=int(self.cfg.dataset.max_num_agents),
                                               max_num_lanes=int(self.cfg.dataset.max_num_lanes),
                                               seed=self.sampling_seed)

    def train(self, mode=True):
        super().train(mode)
        if not self.learn_autoencoder:
            self.autoencoder.eval()
        return self

    def get_extra_state(self):
        return {"loss_kind": self.loss_kind, "training_stage": self.training_stage,
                "training_mode": self.training_mode, "checkpoint_step": self.checkpoint_step,
                "ae_config": plain(public_config(self.ae_cfg)),
                "ldm_config": plain(public_config(self.cfg)),
                "ema": self.ema.state_dict() if self.ema is not None else None}

    def set_extra_state(self, state):
        if state.get("loss_kind") != self.loss_kind or state.get("training_stage") != self.training_stage:
            raise ValueError("Resume requires the same VectorWorld training stage")
        if state.get("training_mode", "joint") != self.training_mode:
            raise ValueError("Resume requires the same training_mode; use ldm_checkpoint to change objectives")
        if self.ema is not None:
            if state.get("ema") is None:
                raise ValueError("Missing generator EMA in VectorWorld checkpoint")
            with torch.no_grad():
                self.ema.load_state_dict(state["ema"])
        self.checkpoint_step = int(state.get("checkpoint_step", 0))

    @torch.no_grad()
    def update_ema(self):
        if self.ema is not None:
            p = next(self.diff_model.parameters())
            self.ema.to(device=p.device, dtype=p.dtype)
            self.ema.update()
            self.checkpoint_step += 1

    def reset_sampling(self):
        self._motion_sources.clear()
        self._map_sources.clear()
        self._map_ids.clear()
        self._placeholder_agents = 0
        if self.count_prior is not None:
            self.count_prior.reset()

    def report_options(self):
        report = dict(init_decoder="vectorworld", training_stage=self.training_stage,
                      training_mode=self.training_mode, generation_mode=self.generation_mode,
                      scene_count_source=self.scene_count_source, motion_missing=self.motion_missing,
                      map_source=self.map_source, map_id=self.map_id, use_ema=self.use_ema,
                      ae_checkpoint=self.ae_checkpoint_path, ldm_checkpoint=self.ldm_checkpoint_path)
        if self.diff_model is not None:
            report.update(ldm_type=self.cfg.model.ldm_type,
                          sampling_steps=getattr(self.diff_model, "n_steps",
                                                getattr(self.diff_model, "num_steps_eval",
                                                        getattr(self.diff_model, "n_timesteps", None))),
                          guidance_scale=float(self.cfg.train.guidance_scale))
        return report

    def sampling_report(self):
        report = self.count_prior.report() if self.count_prior is not None else {"scene_count_source": "input"}
        report.update(self.report_options())
        report["ema_num_updates"] = self.ema.num_updates if self.ema else None
        report["motion_encoding_sources"] = dict(self._motion_sources)
        report["motion_placeholder_agents"] = self._placeholder_agents
        report["map_condition_sources"] = dict(self._map_sources)
        report["map_condition_id_counts"] = dict(self._map_ids)
        return report

    def _build_graph(self, agent, *, need_motion=True):
        if "sd_cached_posterior" in agent:
            raise ValueError("Scenario Dreamer latent caches cannot be used with the VectorWorld motion AE")
        agent["sd_map_id"] = self.map_id
        graph_mode = (self.training_mode if self.training else
                      ("lane_conditioned" if self.generation_mode == "lane_conditioned" else "joint"))
        result = build_graph(agent, agent["tokenized_map"], self.cfg.dataset,
                             map_source=self.map_source,
                             motion_dim=int(self.ae_config.get("motion_dim", 0)) if need_motion else 0,
                             motion_missing=self.motion_missing, mode=graph_mode,
                             is_training=self.training or self.learn_autoencoder)
        graph = result[0]
        self._record_map_conditions(graph)
        if need_motion:
            self._motion_sources[graph.vectorworld_motion_source] += int(agent["num_graphs"])
            self._placeholder_agents += int((~graph["agent"].motion_valid_mask).sum())
        return result

    def _record_map_conditions(self, graph):
        ids = graph.map_id.detach().cpu().tolist()
        sources = getattr(graph, "vectorworld_map_sources", None)
        if sources is None:
            sources = [getattr(graph, "vectorworld_map_source", "unknown")] * len(ids)
        self._map_sources.update(sources)
        self._map_ids.update(str(int(value)) for value in ids)

    def autoencoder_loss(self, agent):
        data, _, _, _ = self._build_graph(agent)
        return self.autoencoder.loss(data)

    def _encode(self, agent):
        data, rows, centers, angles = self._build_graph(agent)
        with torch.no_grad():
            am, lm, av, lv = self.autoencoder.forward_encoder(data, return_stats=True)
            if self.training:
                a = am + (0.5 * av).exp() * torch.randn_like(am)
                l = lm + (0.5 * lv).exp() * torch.randn_like(lm)
            else:
                a, l = am, lm
            s = self.cfg.dataset
            a, l = normalize_latents(a, l, s.agent_latents_mean, s.agent_latents_std,
                                    s.lane_latents_mean, s.lane_latents_std)
            data["agent"].latents, data["lane"].latents = a, l
            data["agent"].x, data["lane"].x = a, l
        return data, rows, centers, angles

    _validate_lane_conditioning = ScenarioDreamerInitDecoder._validate_lane_conditioning
    _smart_output = ScenarioDreamerInitDecoder._smart_output

    def _loss(self, data):
        if self.training_mode == "joint":
            return self.diff_model.loss(data)
        # All lanes are clean context. Existing before-partition agent conditions
        # are retained; only the agent objective is optimized.
        data["lane"].partition_mask = torch.ones_like(data["lane"].partition_mask, dtype=torch.bool)
        lane_weight = self.cfg.train.lane_weight
        self.cfg.train.lane_weight = 0.0
        try:
            return self.diff_model.loss(data)
        finally:
            self.cfg.train.lane_weight = lane_weight

    def forward(self, agent):
        agent.pop("generated_map", None)
        if self.learn_autoencoder:
            return self.autoencoder_loss(agent)
        if (self.training and self.training_mode == "lane_conditioned") or (
                not self.training and self.generation_mode == "lane_conditioned"):
            self._validate_lane_conditioning(agent, require_full=not self.training)
        independent_counts = not self.training and self.scene_count_source == "official_prior"
        if self.training:
            data, rows, centers, angles = self._encode(agent)
            return self._loss(data)
        if independent_counts:
            if not agent.get("initial_scene_only", False):
                raise ValueError("official_prior requires direct initial-scene data")
            counts = self.count_prior.sample(int(agent["num_graphs"]))
            data, rows, centers, angles = build_generation_graph(
                counts, agent_latent_dim=int(self.cfg.model.agent_latent_dim),
                lane_latent_dim=int(self.cfg.model.lane_latent_dim),
                device=agent["initial_pos"].device, dtype=agent["initial_pos"].dtype)
            self._record_map_conditions(data)
        elif self.generation_mode == "initial_scene":
            data, rows, centers, angles = self._build_graph(agent, need_motion=False)
            # Unconditioned initialization requests a full scene, even when
            # input counts were taken from a partitioned reference graph.
            data.lg_type = torch.zeros_like(data.lg_type)
            for kind in ("agent", "lane"):
                data[kind].x = data[kind].x.new_zeros((data[kind].num_nodes,
                                                     int(self.cfg.model[f"{kind}_latent_dim"])))
        else:
            data, rows, centers, angles = self._encode(agent)
        p = next(self.diff_model.parameters())
        self.ema.to(device=p.device, dtype=p.dtype)
        with torch.no_grad(), self.ema.average_parameters() if self.use_ema else nullcontext():
            a, l = self.diff_model(data, mode=self.generation_mode)
            s = self.cfg.dataset
            a, l = unnormalize_latents(a, l, s.agent_latents_mean, s.agent_latents_std,
                                      s.lane_latents_mean, s.lane_latents_std)
            states, lanes, types, _, connections = self.autoencoder.forward_decoder_with_motion(a, l, data)
            keys = ("fov", "min_speed", "max_speed", "min_length", "max_length", "min_width", "max_width",
                    "min_lane_x", "max_lane_x", "min_lane_y", "max_lane_y")
            states, lanes = unnormalize_scene(states, lanes, **{key: s[key] for key in keys})
            # Seven initialization metrics use the decoded static state.
            # Motion is retained separately in physical body coordinates.
            motion = states[:, 7:].clone()
            if motion.numel():
                motion[:, 0::2] = (motion[:, 0::2].clamp(-1, 1) - 1) * float(self.ae_config.motion_x_range) / 2
                motion[:, 1::2] *= float(self.ae_config.motion_y_range)
            agent["generated_motion"] = motion[torch.argsort(rows)]
            if self.generation_mode == "initial_scene":
                agent["generated_map"] = {"coordinate_frame": "sd_local", "road_points": lanes,
                    "road_connection_types": connections,
                    "edge_index_lane_to_lane": data["lane", "to", "lane"].edge_index,
                    "batch": data["lane"].batch, "lg_type": torch.zeros_like(data.lg_type)}
            output = self._smart_output(states[:, :7], types, rows, data["agent"].batch, centers, angles, agent)
            if independent_counts:
                pos, heading, _, size, velocity = output
                batch = data["agent"].batch[torch.argsort(rows)]
                ego_mask = torch.zeros_like(batch, dtype=torch.bool)
                ego_mask[data.num_agents.cumsum(0) - 1] = True
                from ..scenario_dreamer.data import rotate
                agent.update(batch=batch, ego_mask=ego_mask, initial_pos=pos[:, 0],
                             initial_heading=heading[:, 0], shape=size, local_vel=rotate(velocity, -heading[:, 0]))
                for key in ("sd_states", "sd_cached_posterior", "sd_map", "vectorworld_motion"):
                    agent.pop(key, None)
                agent["ego_pos2"] = pos[ego_mask].expand(-1, 3, -1)
                agent["ego_heading2"] = heading[ego_mask].expand(-1, 3)
            return output
