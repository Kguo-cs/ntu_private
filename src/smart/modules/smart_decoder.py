"""Top-level SMART decoder.

This module combines:
    * map encoding;
    * agent-token policy decoding;
    * optional initial-state diffusion;
    * optional GAIL discriminator and value networks.

The public constructor, ``forward`` and ``inference`` signatures are compatible
with the previous implementation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch.nn as nn
from torch import Tensor

from src.smart.diffusion.initial_diffusion import InitDiffusion
from src.smart.layers import MLPLayer
from src.smart.modules.agent_decoder import SMARTAgentDecoder
from src.smart.modules.map_decoder import SMARTMapDecoder


TensorDict = Dict[str, Tensor]


class SMARTDecoder(nn.Module):
    """Compose map, agent, initial-state, and discriminator decoders."""

    def __init__(
        self,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        pl2pl_radius: float,
        time_span: Optional[int],
        pl2a_radius: float,
        a2a_radius: float,
        num_freq_bands: int,
        num_map_layers: int,
        num_agent_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        pt2pt_neighbor: int,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        n_token_agent: int,
        dis_a2a_radius: float,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
        token_processor=None,
        finetune: bool = False,
        init_decoder: str = "flow",
        scenario_dreamer: Optional[dict] = None,
        initial_scene_only: bool = False,
        init_diffusion: Optional[dict] = None,
        sep_map: bool = False,
    ) -> None:
        super().__init__()

        self.token_processor = token_processor
        self.finetune = bool(finetune)


        self.gail = dis_a2a_radius > 0
        self.init_decoder_name = init_decoder
        self.initial_scene_only = bool(initial_scene_only)
        if self.initial_scene_only and not self.token_processor.pred_init:
            raise ValueError("initial_scene_only requires token_processor.pred_init=true")
        self.scenario_dreamer_config = dict(scenario_dreamer or {})
        self.init_diffusion_config = dict(init_diffusion or {})
        if init_decoder not in ("flow", "scenario_dreamer"):
            raise ValueError(f"Unknown init_decoder: {init_decoder}")
        if init_decoder == "scenario_dreamer" and self.gail:
            raise ValueError("Scenario Dreamer supports supervised initialization; GAIL requires its own policy loss")
        self.use_lcf = reward_weight != 0
        self.use_kl_penalty = False
        self.alpha = 0.1

        # External code reads these fields.
        self.pl2a_radius = pl2a_radius
        self.pt2a_neighbor = pt2a_neighbor

        self.map_encoder = self._make_map_encoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_map_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=token_processor,
        )

        self.agent_encoder = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=time_span,
            pl2a_radius=pl2a_radius,
            a2a_radius=a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_agent_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=n_token_agent,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=token_processor,
            alpha=self.alpha,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            use_gail=self.gail,
        )

        # Define optional attributes in every configuration.
        self.init_decoder: Optional[InitDiffusion] = None
        self.init_map_encoder: Optional[SMARTMapDecoder] = None
        self.sep_map = bool(sep_map)
        if self.sep_map and (init_decoder != "flow" or not self.token_processor.pred_init):
            raise ValueError("sep_map requires init_decoder=flow and token_processor.pred_init=true")
        if self.sep_map and self.gail:
            raise ValueError("sep_map currently supports supervised InitDiffusion training, not GAIL")
        self._initial_map_checkpoint_present = False
        self._initial_map_needs_initialization = self.sep_map
        self._initial_map_load_prefix = "init_map_encoder."

        self.discriminator: Optional[SMARTAgentDecoder] = None
        self.value_network: Optional[MLPLayer] = None
        self.nei_value_network: Optional[MLPLayer] = None
        self.init_value_network: Optional[MLPLayer] = None

        if self.token_processor.pred_init:
            self._build_initial_decoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_freq_bands=num_freq_bands,
                pl2pl_radius=pl2pl_radius,
                dropout=dropout,
                head_dim=head_dim,
                pt2pt_neighbor=pt2pt_neighbor,
            )

        if self.gail:
            self._build_gail_modules(
                hidden_dim=hidden_dim,
                num_historical_steps=num_historical_steps,
                num_future_steps=num_future_steps,
                time_span=time_span,
                pl2a_radius=pl2a_radius,
                dis_a2a_radius=dis_a2a_radius,
                num_freq_bands=num_freq_bands,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout=dropout,
                hist_drop_prob=hist_drop_prob,
                pt2a_neighbor=pt2a_neighbor,
                a2a_neighbor=a2a_neighbor,
                dis_weight=dis_weight,
                dist_decay=dist_decay,
                reward_weight=reward_weight,
                reward_decay=reward_decay,
            )
        self.register_load_state_dict_post_hook(self._restore_initial_map_after_load)

    def initialize_initial_map_from_shared(self) -> None:
        """Seed a new initial-map encoder without replacing checkpoint weights."""
        if self.sep_map and self._initial_map_needs_initialization:
            self.init_map_encoder.load_state_dict(self.map_encoder.state_dict())
            self._initial_map_needs_initialization = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._initial_map_load_prefix = prefix + "init_map_encoder."
        self._initial_map_checkpoint_present = any(
            key.startswith(self._initial_map_load_prefix) for key in state_dict
        )
        self._initial_map_needs_initialization = self.sep_map and not self._initial_map_checkpoint_present
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _restore_initial_map_after_load(self, module, incompatible_keys) -> None:
        if self.sep_map and not self._initial_map_checkpoint_present:
            self.initialize_initial_map_from_shared()
            # Shared-map checkpoints predate this optional module. Only exempt
            # its wholly absent keys; partial independent checkpoints stay strict.
            incompatible_keys.missing_keys[:] = [
                key for key in incompatible_keys.missing_keys
                if not key.startswith(self._initial_map_load_prefix)
            ]

    @staticmethod
    def _make_map_encoder(
        *,
        hidden_dim: int,
        pl2pl_radius: float,
        num_freq_bands: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        pt2pt_neighbor: int,
        token_processor,
    ) -> SMARTMapDecoder:
        return SMARTMapDecoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=token_processor,
        )

    def _build_initial_decoder(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_freq_bands: int,
        pl2pl_radius: float,
        dropout: float,
        head_dim: int,
        pt2pt_neighbor: int,
    ) -> None:
        if self.init_decoder_name == "scenario_dreamer":
            from src.smart.scenario_dreamer.decoder import ScenarioDreamerInitDecoder
            self.init_decoder = ScenarioDreamerInitDecoder(self.token_processor, **self.scenario_dreamer_config)
            return
        self.init_decoder = InitDiffusion(
            hidden_dim,
            num_heads,
            num_freq_bands,
            self.token_processor,
            self.gail,
            **self.init_diffusion_config,
        )

        if not self.sep_map:
            return

        self.init_map_encoder = self._make_map_encoder(
            hidden_dim=hidden_dim,
            pl2pl_radius=pl2pl_radius,
            num_freq_bands=num_freq_bands,
            num_layers=self.map_encoder.num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            pt2pt_neighbor=pt2pt_neighbor,
            token_processor=self.token_processor,
        )
        self.initialize_initial_map_from_shared()

    def _build_gail_modules(
        self,
        *,
        hidden_dim: int,
        num_historical_steps: int,
        num_future_steps: int,
        time_span: Optional[int],
        pl2a_radius: float,
        dis_a2a_radius: float,
        num_freq_bands: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        hist_drop_prob: float,
        pt2a_neighbor: int,
        a2a_neighbor: int,
        dis_weight: float,
        dist_decay: float,
        reward_weight: float,
        reward_decay: float,
    ) -> None:
        self.discriminator = SMARTAgentDecoder(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            time_span=10,
            pl2a_radius=pl2a_radius,
            a2a_radius=dis_a2a_radius,
            num_freq_bands=num_freq_bands,
            num_layers=1,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            hist_drop_prob=hist_drop_prob,
            n_token_agent=1,
            pt2a_neighbor=pt2a_neighbor,
            a2a_neighbor=a2a_neighbor,
            token_processor=self.token_processor,
            alpha=self.alpha,
            dis_weight=dis_weight,
            dist_decay=dist_decay,
            reward_weight=reward_weight,
            reward_decay=reward_decay,
            discriminator=True,
        )

        self.value_network = MLPLayer(hidden_dim, hidden_dim * 2, 1)


        if self.token_processor.learn_init:
            self.init_value_network = MLPLayer(
                self.init_decoder.G1.hidden_dim,
                hidden_dim * 2,
                1,
            )

        # Compatibility with existing training code.
        self.agent_encoder.interative_decoder.gail = True

    def _get_map_feature(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
    ):
        map_feature = tokenized_agent.get("map_feature")
        if map_feature is None:
            map_feature = self.map_encoder(tokenized_map)
            tokenized_agent["map_feature"] = map_feature
        return map_feature

    def _prepare_initial_map_feature(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
        map_feature,
    ):
        if not self.token_processor.pred_init:
            return None

        if self.sep_map:
            if self.init_map_encoder is None:
                raise RuntimeError(
                    "sep_map=True but init_map_encoder is missing."
                )
            tokenized_agent["tokenized_map"] = tokenized_map
            # Re-encode for every training forward, including reused batches.
            # A previous projected/raw feature may retain an old autograd graph.
            tokenized_agent.pop("_initial_map_raw_feature", None)
            initial_map_feature = self.init_map_encoder(
                tokenized_map,
                tokenized_agent=tokenized_agent,
            )
            tokenized_agent["initial_map_feature"] = initial_map_feature
            tokenized_agent["_initial_map_feature_is_raw"] = True
            return initial_map_feature
        cached = tokenized_agent.get("initial_map_feature")
        if cached is not None:
            return cached
        tokenized_agent["map_feature"] = map_feature

    def forward(
        self,
        tokenized_map: TensorDict,
        tokenized_agent: TensorDict,
    ) -> TensorDict:
        if self.init_decoder_name == "scenario_dreamer" and self.token_processor.learn_init:
            tokenized_agent["tokenized_map"] = tokenized_map
            return {"initial_logit": self.init_decoder(tokenized_agent)}
        if  self.sep_map and self.token_processor.learn_init and not self.gail:
            map_feature = None
        else:
            map_feature = self._get_map_feature(
                tokenized_map,
                tokenized_agent,
            )

        if self.token_processor.learn_init and not self.gail:
            prediction: TensorDict = {}
        else:
            prediction = self.agent_encoder(
                tokenized_agent,
                map_feature,
            )

        if self.token_processor.learn_init and not self.gail:
            self._prepare_initial_map_feature(
                tokenized_map,
                tokenized_agent,
                map_feature,
            )
            prediction["initial_logit"] = self.init_decoder(
                tokenized_agent
            )

        return prediction

    def inference(
        self,
        tokenized_agent: TensorDict,
        n_step_future_10hz: Optional[int] = None,
    ) -> TensorDict:

        initial_scene_only = getattr(self, "initial_scene_only", False)
        if initial_scene_only and self.init_decoder_name == "flow":
            # Evaluation must identify a complete reference graph. Training token
            # caches need no SD graph metadata and do not enter this inference path.
            sd_map = tokenized_agent.get("sd_map")
            if sd_map is None or sd_map.get("lg_type") is None:
                raise ValueError("Full-lane initial-scene evaluation requires sd_map.lg_type metadata")
            graph_types = sd_map["lg_type"]
            if graph_types.numel() != int(tokenized_agent["num_graphs"]) or (graph_types != 0).any():
                raise ValueError("Full-lane initial-scene evaluation requires lg_type=0 for every scene")
        if getattr(self, "sep_map", False) and "initial_map_feature" not in tokenized_agent:
            self._prepare_initial_map_feature(
                tokenized_agent["tokenized_map"], tokenized_agent, None,
            )
        if initial_scene_only or (self.init_decoder_name == "scenario_dreamer" and tokenized_agent.get("initial_scene_only", False)):
            pos, heading, indices, shape, velocity = self.init_decoder(tokenized_agent)
            # Evaluate continuous initial states directly, without token history
            # reconstruction or autoregressive trajectory generation.
            result = {"pred_traj_10hz": pos, "pred_head_10hz": heading,
                      "pred_z_10hz": pos.new_zeros(pos.shape[:2]),
                      "shape": shape, "initial_local_vel": velocity, "sampled_idx": indices}
            if "generated_map" in tokenized_agent:
                result["generated_map"] = tokenized_agent["generated_map"]
            return result
        return self.agent_encoder.inference(
            self.init_decoder,
            tokenized_agent,
            tokenized_agent["map_feature"],
            n_step_future_10hz=n_step_future_10hz,
        )
