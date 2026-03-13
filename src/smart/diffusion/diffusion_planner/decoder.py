import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp

from .sampling import dpm_sampler
from .sde import SDE, VPSDE_linear
# from .normalizer import ObservationNormalizer, StateNormalizer
# from .mixer import MixerBlock
from .dit import TimestepEmbedder, DiTBlock, FinalLayer


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=None,
            route_encoder=None,
            # route_encoder=RouteEncoder(config.route_num, config.lane_len, drop_path_rate=config.encoder_drop_path_rate,
            #                            hidden_dim=config.hidden_dim),
            depth=3,
            output_dim=4,  # x, y, cos, sin
            hidden_dim=192,
            heads=6,
            dropout=0.1,
            model_type='x_start',
            future_length=None
        )


    @property
    def sde(self):
        return self._sde

    def forward(self, agent_enc, map_enc,ego_enc,diffusion_time):
        if self.training:
            return {
                "score": self.dit(
                    agent_enc,
                    diffusion_time,
                    map_enc,
                    ego_enc,
                )
            }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = (torch.randn(B, self._future_len, 4).to(ego_neighbor_encoding.device) * 0.1)

            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    # "neighbor_current_mask": neighbor_current_mask
                },
                dpm_solver_params={},
                model_wrapper_params={},
            )
            x0 = self._state_normalizer.inverse(x0.reshape(B, -1, 4))
            x0 = torch.cat([
                torch.cumsum(x0[..., :2], dim=-2),
                x0[..., 2:]
            ], dim=-1)

            return {
                "prediction": x0
            }



class DiT(nn.Module):
    def __init__(self, sde: SDE, route_encoder: nn.Module, depth, output_dim, hidden_dim=192, heads=6, dropout=0.1,
                 mlp_ratio=4.0, model_type="x_start", future_length=80):
        super().__init__()

        assert model_type in ["noise", "score", "x_start", "v"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.route_encoder = route_encoder
        #self.agent_embedding = nn.Embedding(future_length, hidden_dim)
        self.preproj = Mlp(in_features=output_dim, hidden_features=512, out_features=hidden_dim, act_layer=nn.GELU,
                           drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, heads, dropout, mlp_ratio) for i in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        # self._sde = sde
        # self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c, route_lanes):
        """
        Forward pass of DiT.
        x: (B, T, output_dim)   -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D)      -> Cross-Attention context
        """
        B, T, _ = x.shape

        x = self.preproj(x)

        # x_embedding = self.agent_embedding.weight[None, :, :].expand(B, -1, -1)  # (B, P, D)
        # x = x + x_embedding

        if self.route_encoder is None:
            x=x+route_lanes
            y=self.t_embedder(t)
        else:
            route_encoding = self.route_encoder(route_lanes)
            y = route_encoding
            y = y + self.t_embedder(t)

        for block in self.blocks:
            x = block(x, cross_c, y)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start" or self._model_type == "noise" or self._model_type == 'v':
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
