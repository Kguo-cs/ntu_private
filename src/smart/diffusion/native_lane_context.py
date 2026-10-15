"""Frozen Scenario Dreamer lane encoding for initialization experiments.

This module is not selected by existing InitDiffusion configurations. It accepts
only saved lane geometry/topology and the reference coordinate transform.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

from src.smart.scenario_dreamer.core.autoencoder import ScenarioDreamerEncoder
from src.smart.scenario_dreamer.data import rotate
from src.smart.scenario_dreamer.decoder import _autoencoder_weights, _checkpoint
from src.smart.utils import wrap_angle


class FrozenSDLaneContext(nn.Module):
    """Use only pretrained lane-to-lane blocks; no agent branch is retained."""

    def __init__(self, checkpoint, *, output_dim=128, query_stride=1, fov=64.):
        super().__init__()
        if isinstance(query_stride, bool) or not isinstance(query_stride, int) or query_stride < 1:
            raise ValueError('query_stride must be a positive integer')
        if fov != 64.:
            raise ValueError('Released Waymo lane encoder requires fov=64m')
        # Adding a frozen context must not alter the generator initialization RNG.
        with torch.random.fork_rng(devices=[]):
            config, weights = _autoencoder_weights(_checkpoint(checkpoint))
            if config.num_lane_types != 0 or config.num_points_per_lane != 20:
                raise ValueError('Expected the released Waymo lane architecture')
            encoder = ScenarioDreamerEncoder(config)
            encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in weights.items()
                                     if k.startswith('encoder.')}, strict=True)
        if isinstance(output_dim, bool) or not isinstance(output_dim, int) or output_dim < config.lane_latent_dim:
            raise ValueError('output_dim must fit the complete lane latent')
        self.output_dim = output_dim
        self.query_stride = query_stride
        self.fov = float(fov)
        self.latent_dim = config.lane_latent_dim
        self.lane_mlp = encoder.lane_mlp
        self.lane_conn_mlp = encoder.lane_conn_mlp
        self.blocks = nn.ModuleList([
            nn.ModuleDict(dict(attention=block.l2l_transformer_layer,
                               downsample=block.downsample_lane_emb,
                               update_edge=block.update_edge_embeddings))
            for block in encoder.encoder_transformer_blocks
        ])
        self.lane_mu = encoder.lane_mu
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen context stays deterministic when the enclosing model trains.
        return super().train(False)

    @torch.no_grad()
    def encode(self, sd_map):
        lanes = sd_map['lanes']
        batch = sd_map['batch'].long()
        edges = sd_map['edges'].long()
        labels = sd_map['types']
        kinds = sd_map['lg_type'].reshape(-1).long()
        if lanes.ndim != 3 or lanes.shape[1:] != (20, 2) or not len(lanes):
            raise ValueError('Native context requires nonempty [L,20,2] exact lanes')
        if batch.shape != (len(lanes),) or not torch.isfinite(lanes).all():
            raise ValueError('Invalid saved lane geometry/batch')
        if kinds.numel() == 0 or not torch.isin(kinds, kinds.new_tensor([0, 1])).all():
            raise ValueError('Unsupported saved lg_type; expected0 or1')
        if edges.ndim != 2 or edges.shape[0] != 2 or not edges.numel():
            raise ValueError('Expected lane connectivity edges')
        if batch.min() < 0 or batch.max() >= len(kinds):
            raise ValueError('Lane batch outside scene metadata')
        if edges.min() < 0 or edges.max() >= len(lanes) or (batch[edges[0]] != batch[edges[1]]).any():
            raise ValueError('Invalid or cross-scene lane connectivity')
        if labels.ndim == 1:
            if (labels.shape[0] != edges.shape[1] or labels.min() < 0 or labels.max() > 5
                    or not torch.equal(labels, labels.long())):
                raise ValueError('Expected six lane relation classes')
            relations = F.one_hot(labels.long(), 6).float()
        elif labels.shape == (edges.shape[1], 6):
            relations = labels.float()
        else:
            raise ValueError('Lane relations must be [E] or[E,6]')
        if (not torch.isfinite(relations).all() or (relations < 0).any()
                or not torch.allclose(relations.sum(-1), relations.new_ones(len(relations)))):
            raise ValueError('Invalid lane relation probabilities')
        normalized = lanes.float() / (self.fov / 2)
        # Exact upstream Waymo partition rule uses point9,y, including y=0.
        before = (kinds[batch] == 1) & (normalized[:, 9, 1] <= 0)
        keep = before[edges[0]] == before[edges[1]]
        edges = edges[:, keep]
        node = self.lane_mlp(normalized.flatten(1))
        edge = self.lane_conn_mlp(relations[keep])
        for block in self.blocks:
            node = block['attention'](node, edge, edges)
            edge = block['update_edge'](block['downsample'](node), edges, edge)
        return self.lane_mu(node)

    @torch.no_grad()
    def forward(self, sd_map, *, scene_pos=None, scene_heading=None):
        latent = self.encode(sd_map)
        points = sd_map['lanes'].to(torch.float64)
        batch = sd_map['batch'].long()
        centers = sd_map['center'].reshape(-1, 2).to(points)
        angles = sd_map['angle'].reshape(-1).to(points)
        if centers.shape[0] != angles.numel() or not torch.isfinite(centers).all() or not torch.isfinite(angles).all():
            raise ValueError('Invalid native map transform')
        delta = points[:, 1:] - points[:, :-1]
        # Last query reuses the preceding segment tangent.
        tangent = torch.cat((delta, delta[:, -1:]), dim=1)
        orientation = torch.atan2(tangent[..., 1], tangent[..., 0])
        if (scene_pos is None) != (scene_heading is None):
            raise ValueError('Supply both ego reference position and heading')
        if scene_pos is None:
            position = rotate(points, -angles[batch, None]) + centers[batch, None]
            orientation = wrap_angle(orientation - angles[batch, None])
        else:
            ego_pos = scene_pos.reshape(-1, 2).to(points)
            ego_head = scene_heading.reshape(-1).to(points)
            if ego_pos.shape != centers.shape or ego_head.shape != angles.shape:
                raise ValueError('Ego transforms must match the saved scene count')
            # Combine transforms in float64, avoiding a large world-coordinate
            # float32 round trip before returning ego-local graph features.
            position = (rotate(points, -angles[batch, None] - ego_head[batch, None])
                        + rotate(centers[batch] - ego_pos[batch], -ego_head[batch])[:, None])
            orientation = wrap_angle(orientation - angles[batch, None] - ego_head[batch, None])
        position = position[:, ::self.query_stride]
        orientation = orientation[:, ::self.query_stride]
        count = position.shape[1]
        feature = F.pad(latent, (0, self.output_dim - self.latent_dim))
        return dict(pt_token=feature.repeat_interleave(count, dim=0),
                    position=position.reshape(-1, 2).float(),
                    orientation=orientation.reshape(-1).float(),
                    batch=batch.repeat_interleave(count))
