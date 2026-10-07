"""Scenario Dreamer as an ordinary SMART init_decoder (no external checkout)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import torch
from torch import nn
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage

from .core.autoencoder import AutoEncoder
from .core.ldm import LDM
from .core.data_helpers import unnormalize_scene
from .data import build_graph, rotate
from .generation import DEFAULT_COUNT_PRIOR, SceneCountPrior, build_generation_graph
from .map_categories import apply_category_index, load_category_keys

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "src/waymo_data/scenario_dreamer/checkpoints"
DEFAULT_LDM_CONFIG = Path(__file__).with_name("waymo_ldm_large.yaml")
DEFAULT_AE_CONFIG = Path(__file__).with_name("waymo_autoencoder.yaml")
DEFAULT_AE_CHECKPOINT = WEIGHTS / "scenario_dreamer_autoencoder_waymo/last.ckpt"


def _checkpoint(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing Scenario Dreamer checkpoint: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Expected a Scenario Dreamer Lightning checkpoint with state_dict: {path}")
    return checkpoint


def _autoencoder_weights(checkpoint):
    """Read either an official AE checkpoint or an AE trained by SMART."""
    state = checkpoint["state_dict"]
    if "cfg" in checkpoint.get("hyper_parameters", {}):
        saved = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])
        config = OmegaConf.create(OmegaConf.to_container(saved.model, resolve=True))
        prefix = "model."
    else:
        extra = state.get("encoder.init_decoder._extra_state", {})
        if "ae_config" not in extra:
            raise ValueError("SMART AE checkpoint must contain encoder.init_decoder._extra_state.ae_config")
        config = OmegaConf.create(extra["ae_config"])
        prefix = "encoder.init_decoder.autoencoder."
    weights = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
    if not weights:
        raise ValueError(f"AE checkpoint contains no weights with prefix {prefix}")
    return config, weights


class ScenarioDreamerInitDecoder(nn.Module):
    """Train AE reconstruction or LDM loss through SMART's initial_logit API.

    LDM inference returns the existing 5-tuple, supporting joint lane/agent generation
    or lane conditioning. Initial-scene counts can come from input scenes or
    the official joint count prior.
    Official checkpoint parameter names and EMA order are retained in the core.
    training_stage selects AE reconstruction or LDM training with a frozen AE.
    training_mode independently selects the joint or lane-conditioned LDM objective.
    With ldm_checkpoint=None, initialize a fresh LDM without reading LDM weights.
    """
    use_gan = False
    learn_autoencoder = False
    use_rl = False
    loss_kind = "scenario_dreamer"

    def __init__(self, token_processor, *, ae_checkpoint=DEFAULT_AE_CHECKPOINT, ae_config=None,
                 ldm_checkpoint=None, ldm_config=None, training_stage="ldm", training_mode="joint",
                 map_source="auto", map_id=0, use_ema=True, generation_mode="initial_scene",
                 scene_count_source="input", count_prior_path=None, sampling_seed=0, map_category_index=None):
        super().__init__()
        self.token_processor = token_processor
        if training_stage not in ("ldm", "autoencoder"):
            raise ValueError("training_stage must be ldm or autoencoder")
        self.training_stage = training_stage
        self.learn_autoencoder = training_stage == "autoencoder"
        if training_mode not in ("joint", "lane_conditioned"):
            raise ValueError("training_mode must be joint or lane_conditioned")
        if self.learn_autoencoder and training_mode != "joint":
            raise ValueError("lane_conditioned training_mode requires training_stage=ldm")
        self.training_mode = training_mode
        self.map_source = map_source
        self.map_id = int(map_id)
        self.map_category_index = self._resolved_checkpoint_path(map_category_index)
        self.map_category_keys = load_category_keys(self.map_category_index) if self.map_category_index else None
        self._map_sources = Counter()
        self._map_ids = Counter()
        self.use_ema = bool(use_ema) and not self.learn_autoencoder
        self.generation_mode = generation_mode
        self.scene_count_source = scene_count_source
        self.sampling_seed = int(sampling_seed)
        self.count_prior = None
        self.ae_checkpoint_path = self._resolved_checkpoint_path(ae_checkpoint)
        self.ldm_checkpoint_path = self._resolved_checkpoint_path(ldm_checkpoint)
        if scene_count_source not in ("input", "official_prior"):
            raise ValueError("scene_count_source must be input or official_prior")
        if scene_count_source == "official_prior" and (self.learn_autoencoder or generation_mode != "initial_scene"):
            raise ValueError("official_prior requires LDM initial_scene generation")
        if generation_mode not in ("initial_scene", "lane_conditioned"):
            raise ValueError("generation_mode must be initial_scene or lane_conditioned")
        if self.map_source not in ("auto", "exact", "tokens") or self.map_id not in (0, 1):
            raise ValueError("Use map_source=auto/exact/tokens and Waymo map_id=0/1")
        if ldm_checkpoint is not None and ldm_config is not None:
            raise ValueError("ldm_config overrides require ldm_checkpoint=null; pretrained models use their saved config")
        if self.learn_autoencoder and (ldm_checkpoint is not None or ldm_config is not None):
            raise ValueError("Autoencoder training requires ldm_checkpoint=null and ldm_config=null")
        if not self.learn_autoencoder and ae_checkpoint is None:
            raise ValueError("LDM training requires a trained ae_checkpoint; use training_stage=autoencoder to train an AE")
        if ae_checkpoint is not None and ae_config is not None:
            raise ValueError("ae_config overrides require ae_checkpoint=null")
        ae = _checkpoint(ae_checkpoint) if ae_checkpoint is not None else None
        ae_weights = None
        if ae is None:
            self.ae_config = OmegaConf.merge(OmegaConf.load(DEFAULT_AE_CONFIG), ae_config or {})
        else:
            self.ae_config, ae_weights = _autoencoder_weights(ae)
        ldm = None
        if ldm_checkpoint is None:
            # This path must not read the large LDM checkpoint, even for configuration.
            self.cfg = OmegaConf.merge(OmegaConf.load(DEFAULT_LDM_CONFIG), ldm_config or {})
        else:
            ldm = _checkpoint(ldm_checkpoint)
            if "ema_state_dict" not in ldm or "cfg" not in ldm.get("hyper_parameters", {}):
                raise ValueError("The official LDM checkpoint must include ema_state_dict and hyper_parameters.cfg")
            saved = ldm["hyper_parameters"]["cfg"]
            model_cfg = {key: saved.model[key] for key in saved.model
                         if key not in ("autoencoder_run_name", "autoencoder_path")}
            dataset_cfg = {key: value for key, value in OmegaConf.to_container(saved.dataset, resolve=False).items()
                           if isinstance(value, (int, float, bool))}
            dataset_cfg.update(num_map_ids=2, num_agent_types=3, num_lane_types=0)
            train_cfg = {key: saved.train[key] for key in ("loss_type", "lane_weight", "guidance_scale", "ema_decay")}
            self.cfg = OmegaConf.create(dict(model=model_cfg, dataset=dataset_cfg, train=train_cfg, dataset_name="waymo"))
        if scene_count_source == "official_prior":
            prior_path = Path(count_prior_path).expanduser() if count_prior_path is not None else DEFAULT_COUNT_PRIOR
            if not prior_path.is_absolute():
                prior_path = ROOT / prior_path
            self.count_prior = SceneCountPrior(prior_path, max_num_agents=self.cfg.dataset.max_num_agents,
                                              max_num_lanes=self.cfg.dataset.max_num_lanes, seed=self.sampling_seed)
        self.autoencoder = AutoEncoder(self.ae_config)
        self._latent_cache_fingerprint = None
        self.autoencoder.register_load_state_dict_post_hook(self._invalidate_latent_cache_fingerprint)
        if ae_weights is not None:
            self.autoencoder.load_state_dict(ae_weights, strict=True)
        self.checkpoint_step = int(ae.get("global_step", 0)) if ae is not None else 0
        self.diff_model = None
        self.ema = None
        if self.learn_autoencoder:
            # AE training needs no diffusion model, checkpoint or EMA allocation.
            self.cfg.dataset.num_points_per_lane = self.ae_config.num_points_per_lane
            self.cfg.dataset.max_num_lanes = self.ae_config.max_num_lanes
            return
        for key in ("agent_latent_dim", "lane_latent_dim"):
            if self.cfg.model[key] != self.ae_config[key]:
                raise ValueError(f"LDM {key} must match the AE checkpoint ({self.ae_config[key]})")
        self.diff_model = LDM(self.cfg)
        if ldm is not None:
            self.diff_model.load_state_dict({key.removeprefix("diff_model."): value
                                             for key, value in ldm["state_dict"].items()
                                             if key.startswith("diff_model.")}, strict=True)
            # Pretrained LDM latents correspond to its embedded AE weights.
            embedded = {key.removeprefix("autoencoder.model."): value for key, value in ldm["state_dict"].items()
                        if key.startswith("autoencoder.model.")}
            self.autoencoder.load_state_dict(embedded, strict=True)
        self.autoencoder.requires_grad_(False).eval()
        self.ema = ExponentialMovingAverage(self.diff_model.parameters(), decay=self.cfg.train.ema_decay)
        self.checkpoint_step = 0
        if ldm is not None:
            self.ema.load_state_dict(ldm["ema_state_dict"])
            self.checkpoint_step = int(ldm.get("global_step", 0))

    @staticmethod
    def _resolved_checkpoint_path(path):
        if path is None:
            return None
        path = Path(path).expanduser()
        return str((path if path.is_absolute() else ROOT / path).resolve())

    def reset_sampling(self):
        """Reset counts and condition reports; diffusion uses the PyTorch RNG."""
        self._map_sources.clear()
        self._map_ids.clear()
        if self.count_prior is not None:
            self.count_prior.reset()

    def sampling_report(self):
        report = self.count_prior.report() if self.count_prior is not None else {"scene_count_source": "input"}
        report.update(ae_checkpoint=self.ae_checkpoint_path, ldm_checkpoint=self.ldm_checkpoint_path,
                      training_mode=self.training_mode, ema_num_updates=self.ema.num_updates if self.ema is not None else None,
                      map_category_index=self.map_category_index, map_id_fallback=self.map_id,
                      map_category_policy=(self.map_category_keys.policy if self.map_category_keys is not None else None),
                      map_condition_sources=dict(self._map_sources),
                      map_condition_id_counts=dict(self._map_ids))
        return report

    def _build_generation_graph(self, agent):
        if not agent.get("initial_scene_only", False) or "sd_states" not in agent:
            raise ValueError("official_prior requires the direct preprocessed AE initial-scene data path")
        counts = self.count_prior.sample(int(agent["num_graphs"]))
        result = build_generation_graph(counts, agent_latent_dim=self.cfg.model.agent_latent_dim,
                                      lane_latent_dim=self.cfg.model.lane_latent_dim,
                                      device=agent["initial_pos"].device, dtype=agent["initial_pos"].dtype)
        self._record_map_conditions(result[0], source="official_count_prior")
        return result

    def train(self, mode=True):
        super().train(mode)
        if not self.learn_autoencoder:
            self.autoencoder.eval()
        return self

    def get_extra_state(self):
        # nn.Module state_dict makes EMA part of ordinary SMART checkpoints.
        return {"ema": self.ema.state_dict() if self.ema is not None else None,
                "checkpoint_step": self.checkpoint_step, "training_stage": self.training_stage,
                "training_mode": self.training_mode,
                "ae_config": OmegaConf.to_container(self.ae_config, resolve=True)}

    def set_extra_state(self, state):
        self._latent_cache_fingerprint = None
        if state.get("training_stage", "ldm") != self.training_stage:
            raise ValueError("Resume requires the same training_stage; pass an AE checkpoint via ae_checkpoint for LDM training")
        saved_mode = state.get("training_mode", "joint")
        if saved_mode != self.training_mode:
            raise ValueError(
                f"Resume requires the same training_mode (checkpoint={saved_mode}, requested={self.training_mode}); "
                "initialize with an official ldm_checkpoint when changing the training objective"
            )
        if self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        self.checkpoint_step = state.get("checkpoint_step", 0)

    @torch.no_grad()
    def update_ema(self):
        if self.ema is None:
            return
        self.ema.to(next(self.diff_model.parameters()).device)
        self.ema.update()

    def _build_graph(self, agent):
        agent["sd_map_id"] = self.map_id
        agent = apply_category_index(agent, self.map_category_keys, model_name="Scenario Dreamer")
        result = build_graph(
            agent, agent["tokenized_map"], self.cfg.dataset,
            map_source=self.map_source,
        )
        self._record_map_conditions(result[0])
        return result

    def _record_map_conditions(self, graph, *, source=None):
        ids = graph.map_id.detach().cpu().tolist()
        sources = [source] * len(ids) if source is not None else graph.map_condition_sources
        self._map_sources.update(sources)
        self._map_ids.update(str(int(value)) for value in ids)

    def _invalidate_latent_cache_fingerprint(self, module, incompatible_keys):
        self._latent_cache_fingerprint = None

    def _check_cached_encoder(self, agent):
        from .latent_cache import encoder_fingerprint
        if self._latent_cache_fingerprint is None:
            self._latent_cache_fingerprint = encoder_fingerprint(self.autoencoder, self.ae_config, self.cfg.dataset)
        identities = agent["sd_cached_posterior"]["encoder_fingerprint"]
        if isinstance(identities, str):
            identities = [identities]
        if len(identities) != int(agent["num_graphs"]) or any(value != self._latent_cache_fingerprint for value in identities):
            raise ValueError("Latent cache does not match the current AE weights/preprocessing; regenerate with this AE")

    def _encode(self, agent):
        data, rows, centers, angles = self._build_graph(agent)
        with torch.no_grad():
            if "sd_cached_posterior" in agent:
                self._check_cached_encoder(agent)
                am, av = data["agent"].posterior_mu, data["agent"].posterior_log_var
                lm, lv = data["lane"].posterior_mu, data["lane"].posterior_log_var
            else:
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

    def _validate_lane_conditioning(self, agent, *, require_full=True):
        """Require exact typed lane graphs; evaluation additionally requires full maps."""
        num_graphs = int(agent["num_graphs"])
        if num_graphs < 1:
            raise ValueError("lane_conditioned training/evaluation requires at least one input scene")
        indices = list(range(num_graphs))
        exact = agent.get("sd_map")
        if self.map_source == "tokens" or exact is None:
            raise ValueError(
                "lane_conditioned training/evaluation requires an exact SD map with explicit lg_type; "
                f"token-map fallback cannot establish graph type (batch indices {indices})"
            )
        metadata = exact.get("lg_type")
        if metadata is None:
            raise ValueError(
                "lane_conditioned training/evaluation requires explicit sd_map.lg_type for every scene; "
                f"missing metadata at batch indices {indices}"
            )
        try:
            kinds = torch.as_tensor(metadata).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(
                f"Invalid lane_conditioned sd_map.lg_type metadata at batch indices {indices}"
            ) from error
        if kinds.numel() != num_graphs:
            raise ValueError(
                "lane_conditioned sd_map.lg_type metadata count must equal num_graphs: "
                f"got {kinds.numel()} for {num_graphs} scenes (batch indices {indices})"
            )
        unsupported = torch.where((kinds != 0) & (kinds != 1))[0].tolist()
        if unsupported:
            raise ValueError(
                "lane_conditioned requires supported lg_type=0 or 1; "
                f"invalid lg_type at batch indices {unsupported}"
            )
        invalid = torch.where(kinds != 0)[0].tolist() if require_full else []
        if invalid:
            raise ValueError(
                "lane_conditioned evaluation requires full non-partitioned lanes (lg_type=0); "
                f"invalid lg_type at batch indices {invalid}"
            )

    def autoencoder_loss(self, tokenized_agent):
        if "sd_cached_posterior" in tokenized_agent:
            raise ValueError("AE training/validation requires raw scenes; disable latent caches for the AE stage")
        data, _, _, _ = self._build_graph(tokenized_agent)
        return self.autoencoder.loss(data)

    def forward(self, tokenized_agent):
        tokenized_agent.pop("generated_map", None)
        if self.learn_autoencoder:
            return self.autoencoder_loss(tokenized_agent)
        if ((self.training and self.training_mode == "lane_conditioned")
                or (not self.training and self.generation_mode == "lane_conditioned")):
            self._validate_lane_conditioning(tokenized_agent, require_full=not self.training)
        independent_counts = not self.training and self.scene_count_source == "official_prior"
        if independent_counts:
            data, rows, centers, angles = self._build_generation_graph(tokenized_agent)
        elif not self.training and self.generation_mode == "initial_scene":
            # Joint generation uses counts and graph structure, never GT AE latents.
            data, rows, centers, angles = self._build_graph(tokenized_agent)
            for kind in ("agent", "lane"):
                width = self.cfg.model[f"{kind}_latent_dim"]
                data[kind].x = data[kind].x.new_zeros((data[kind].num_nodes, width))
        else:
            data, rows, centers, angles = self._encode(tokenized_agent)
        if self.training:
            # Joint remains the released objective; lane conditioning supervises agents only.
            return self.diff_model.loss(data, mode=self.training_mode)
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
            output = self._smart_output(states, types, rows, data["agent"].batch, centers, angles, tokenized_agent)
            if independent_counts:
                pos, heading, _, size, velocity = output
                batch = data["agent"].batch[torch.argsort(rows)]
                ego_mask = torch.zeros_like(batch, dtype=torch.bool)
                ego_mask[data.num_agents.cumsum(0) - 1] = True
                tokenized_agent.update(batch=batch, ego_mask=ego_mask, initial_pos=pos[:, 0],
                                       initial_heading=heading[:, 0], shape=size,
                                       local_vel=rotate(velocity, -heading[:, 0]))
                # Reference tensors have different row counts; remove stale features.
                for key in ("sd_states", "sd_cached_posterior", "sd_map"):
                    tokenized_agent.pop(key, None)
                tokenized_agent["ego_pos2"] = pos[ego_mask].expand(-1, 3, -1)
                tokenized_agent["ego_heading2"] = heading[ego_mask].expand(-1, 3)
            return output

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
