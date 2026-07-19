import math
import random
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from src.smart.utils import (
    angle_between_2d_vectors,
    transform_to_global,
    weight_init,
    wrap_angle,
    project_to_local_frame
)
from src.smart.utils.edge_utils import radiusGraphNearest2, radiusGraphNearest
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_scatter import scatter_mean


class _FeatureProjector(nn.Module):
    """Project node/map features to hidden_dim without using LazyLinear.

    If in_dim is provided, this is a learnable Linear projection. If in_dim is
    None, it becomes a safe fallback: same-dim features are used directly,
    larger features are truncated, and smaller features are zero-padded. This
    avoids UninitializedParameter errors during model construction / weight_init.
    """

    def __init__(self, in_dim: Optional[int], out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False) if in_dim is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            return self.proj(x)

        if x.size(-1) == self.out_dim:
            return x
        if x.size(-1) > self.out_dim:
            return x[..., :self.out_dim]

        pad = x.new_zeros(*x.shape[:-1], self.out_dim - x.size(-1))
        return torch.cat([x, pad], dim=-1)


class LearnableTopKEdgeSelector(nn.Module):
    """Fast hard top-k edge router.

    Speed changes compared with the previous version:
      1) score low-dimensional raw edge features before FourierEmbedding;
      2) support global top-ratio selection over all candidate edges, avoiding
         per-agent group top-k;
      3) support adaptive global keep-ratio based on score uncertainty;
      4) support global threshold selection, where the number of kept edges is data-dependent;
      5) return gate values so the caller only embeds selected edges.
    """

    def __init__(
            self,
            hidden_dim: int,
            edge_input_dim: int,
            selector_hidden_dim: Optional[int] = None,
            temperature: float = 1.0,
            use_gumbel: bool = False,
            gate_edges: bool = True,
            distance_prior: bool = True,
            agent_feat_dim: Optional[int] = None,
            map_feat_dim: Optional[int] = None,
            max_dense_groups: int = 200000,
            selection_mode: str = "global_ratio",
            global_keep_ratio: Optional[float] = None,
            min_keep_edges: int = 1,
            adaptive_min_keep_ratio: float = 0.25,
            adaptive_max_keep_ratio: float = 0.75,
            adaptive_score_std_scale: float = 1.0,
            adaptive_strength: float = 1.0,
            threshold_type: str = "zscore",
            score_threshold: float = 0.0,
            prob_threshold: float = 0.5,
            threshold_max_keep_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()
        selector_hidden_dim = selector_hidden_dim or max(hidden_dim // 2, 16)
        self.temperature = max(float(temperature), 1e-4)
        self.use_gumbel = use_gumbel
        self.gate_edges = gate_edges
        self.distance_prior = distance_prior
        self.max_dense_groups = int(max_dense_groups)
        self.selection_mode = selection_mode
        self.global_keep_ratio = global_keep_ratio
        self.min_keep_edges = int(min_keep_edges)
        self.adaptive_min_keep_ratio = float(adaptive_min_keep_ratio)
        self.adaptive_max_keep_ratio = float(adaptive_max_keep_ratio)
        self.adaptive_score_std_scale = max(float(adaptive_score_std_scale), 1e-6)
        self.adaptive_strength = float(adaptive_strength)
        self.threshold_type = threshold_type
        self.score_threshold = float(score_threshold)
        self.prob_threshold = float(prob_threshold)
        self.threshold_max_keep_ratio = threshold_max_keep_ratio

        self.edge_feat_proj = nn.Linear(edge_input_dim, hidden_dim, bias=False)
        self.src_feat_proj = _FeatureProjector(agent_feat_dim, hidden_dim)
        self.dst_feat_proj = _FeatureProjector(agent_feat_dim, hidden_dim)
        self.map_feat_proj = _FeatureProjector(map_feat_dim, hidden_dim)
        self.feature_scale = nn.Parameter(torch.tensor(-2.3025851))  # exp(.) = 0.1
        self.selector_input_norm = nn.LayerNorm(hidden_dim)

        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, selector_hidden_dim),
            nn.GELU(),
            nn.Linear(selector_hidden_dim, 1),
        )
        if distance_prior:
            self.distance_penalty = nn.Parameter(torch.tensor(-2.0))
        else:
            self.register_parameter("distance_penalty", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.edge_feat_proj.weight)
        for module in self.score_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        # Warm start: use distance prior first; learned score initially constant.
        nn.init.zeros_(self.score_net[-1].weight)
        nn.init.constant_(self.score_net[-1].bias, 2.0)

    @staticmethod
    def _sample_gumbel_like(scores: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(scores.dtype).eps
        u = torch.rand_like(scores).clamp(min=eps, max=1.0 - eps)
        return -torch.log(-torch.log(u))

    @staticmethod
    def _max_edges_per_group(group_index: torch.Tensor) -> int:
        if group_index.numel() == 0:
            return 0
        # bincount is much cheaper than sorting when groups are already integer node ids.
        counts = torch.bincount(group_index.clamp_min(0))
        return int(counts.max().item()) if counts.numel() > 0 else 0

    @staticmethod
    def _global_top_ratio_mask(
            scores: torch.Tensor,
            keep_ratio: Optional[float],
            min_keep_edges: int = 1,
    ) -> torch.Tensor:
        """Select a global top fraction over the whole candidate edge pool.

        This is much faster than per-receiver top-k because it needs only one
        torch.topk call and does not build receiver groups.
        """
        E = scores.size(0)
        keep_mask = torch.zeros(E, dtype=torch.bool, device=scores.device)
        if E == 0:
            return keep_mask

        if keep_ratio is None:
            keep_ratio = 1.0
        keep_ratio = float(keep_ratio)
        if keep_ratio >= 1.0:
            keep_mask[:] = True
            return keep_mask
        if keep_ratio <= 0.0:
            return keep_mask

        k = int(math.ceil(E * keep_ratio))
        k = max(int(min_keep_edges), k)
        k = min(k, E)
        if k <= 0:
            return keep_mask

        selected = torch.topk(scores, k=k, largest=True).indices
        keep_mask[selected] = True
        return keep_mask

    def _adaptive_global_keep_ratio(
            self,
            route_scores: torch.Tensor,
            base_keep_ratio: Optional[float],
    ) -> float:
        """Compute one adaptive keep ratio for the whole candidate edge pool.

        Low score spread means the selector is uncertain, so keep more edges.
        High score spread means routing is confident, so keep fewer edges. The
        returned scalar is detached because top-k needs an integer k.
        """
        if route_scores.numel() == 0:
            return 0.0

        if base_keep_ratio is None:
            base_keep_ratio = self.global_keep_ratio
        if base_keep_ratio is None:
            base_keep_ratio = 1.0

        min_ratio = max(0.0, min(1.0, self.adaptive_min_keep_ratio))
        max_ratio = max(0.0, min(1.0, self.adaptive_max_keep_ratio))
        if min_ratio > max_ratio:
            min_ratio, max_ratio = max_ratio, min_ratio

        # Use detached scores: the discrete number of kept edges is not a useful
        # gradient path, while selected edges still train the score/gate network.
        s = route_scores.detach().float()
        if s.numel() <= 1:
            uncertainty = s.new_tensor(1.0)
        else:
            score_std = s.std(unbiased=False)
            confidence = torch.tanh(score_std / self.adaptive_score_std_scale).clamp(0.0, 1.0)
            uncertainty = 1.0 - confidence

        adaptive_ratio = min_ratio + (max_ratio - min_ratio) * uncertainty
        base = adaptive_ratio.new_tensor(float(base_keep_ratio)).clamp(0.0, 1.0)
        strength = adaptive_ratio.new_tensor(self.adaptive_strength).clamp(0.0, 1.0)
        ratio = (1.0 - strength) * base + strength * adaptive_ratio
        return float(ratio.clamp(0.0, 1.0).item())

    def _global_threshold_mask(
            self,
            route_scores: torch.Tensor,
            min_keep_edges: int = 1,
    ) -> torch.Tensor:
        """Select edges whose routing score passes a fixed global threshold.

        This mode does not choose a fixed keep ratio. The kept edge count is
        determined by the learned score distribution and the threshold.

        threshold_type:
          - "score": raw selector score >= score_threshold;
          - "prob": sigmoid(score / temperature) >= prob_threshold;
          - "zscore": normalized score z >= score_threshold. This is usually
            the most stable option because raw scores may drift during training.
        """
        E = route_scores.size(0)
        keep_mask = torch.zeros(E, dtype=torch.bool, device=route_scores.device)
        if E == 0:
            return keep_mask

        threshold_type = str(self.threshold_type).lower()
        if threshold_type == "score":
            values = route_scores
            threshold = route_scores.new_tensor(self.score_threshold)
        elif threshold_type == "prob":
            values = torch.sigmoid(route_scores / self.temperature)
            threshold = route_scores.new_tensor(self.prob_threshold)
        elif threshold_type == "zscore":
            if E <= 1:
                values = torch.zeros_like(route_scores)
            else:
                s = route_scores.float()
                values = ((s - s.mean()) / s.std(unbiased=False).clamp_min(1e-6)).to(route_scores.dtype)
            threshold = route_scores.new_tensor(self.score_threshold)
        else:
            raise ValueError(
                f"Unknown threshold_type={self.threshold_type!r}. "
                "Use 'score', 'prob', or 'zscore'."
            )

        keep_mask = values >= threshold

        # Safety 1: never return an empty edge set unless explicitly requested.
        min_keep_edges = max(int(min_keep_edges), 0)
        if min_keep_edges > 0 and int(keep_mask.sum().item()) < min_keep_edges:
            k = min(min_keep_edges, E)
            fallback = torch.topk(route_scores, k=k, largest=True).indices
            keep_mask[fallback] = True

        # Safety 2: optional cap to avoid a threshold that keeps almost all
        # candidate edges. This is only a cap; it is not the selection rule.
        if self.threshold_max_keep_ratio is not None:
            max_keep_ratio = float(self.threshold_max_keep_ratio)
            if max_keep_ratio < 1.0:
                max_keep = max(min_keep_edges, int(math.ceil(E * max_keep_ratio)))
                max_keep = min(max_keep, E)
                if int(keep_mask.sum().item()) > max_keep:
                    # Keep the highest-scoring edges among all candidates.
                    selected = torch.topk(route_scores, k=max_keep, largest=True).indices
                    capped_mask = torch.zeros_like(keep_mask)
                    capped_mask[selected] = True
                    keep_mask = capped_mask

        return keep_mask

    @staticmethod
    def _group_topk_mask_dense(
            scores: torch.Tensor,
            group_index: torch.Tensor,
            topk: int,
            max_dense_groups: int = 200000,
    ) -> torch.Tensor:
        """Vectorized top-k per group.

        This avoids looping over every receiver node. It sorts edges once by
        receiver, writes scores to a dense [num_groups, max_edges_per_group]
        tensor, and runs one torch.topk call.
        """
        E = scores.size(0)
        keep_mask = torch.zeros(E, dtype=torch.bool, device=scores.device)
        if E == 0 or topk is None or topk <= 0:
            return keep_mask

        sort_idx = torch.argsort(group_index)
        group_sorted = group_index[sort_idx]
        scores_sorted = scores[sort_idx]
        unique_group, counts = torch.unique_consecutive(group_sorted, return_counts=True)
        num_groups = int(unique_group.numel())
        if num_groups == 0:
            return keep_mask

        max_count = int(counts.max().item())
        k = min(int(topk), max_count)
        if k <= 0:
            return keep_mask

        # Avoid pathological dense allocation. This fallback is rarely used if
        # candidate edges come from knn_graph, where max_count is bounded.
        if num_groups * max_count > max_dense_groups:
            offset = torch.cumsum(counts, dim=0) - counts
            for g in range(num_groups):
                st = int(offset[g].item())
                ed = st + int(counts[g].item())
                local_k = min(k, ed - st)
                if local_k > 0:
                    local = torch.topk(scores_sorted[st:ed], k=local_k, largest=True).indices
                    keep_mask[sort_idx[st + local]] = True
            return keep_mask

        group_row = torch.repeat_interleave(torch.arange(num_groups, device=scores.device), counts)
        group_start = torch.repeat_interleave(torch.cumsum(counts, dim=0) - counts, counts)
        col = torch.arange(E, device=scores.device) - group_start

        dense_scores = scores.new_full((num_groups, max_count), -torch.inf)
        dense_edges = torch.full((num_groups, max_count), -1, dtype=torch.long, device=scores.device)
        dense_scores[group_row, col] = scores_sorted
        dense_edges[group_row, col] = sort_idx

        top_col = torch.topk(dense_scores, k=k, dim=1, largest=True).indices
        selected = dense_edges.gather(1, top_col).reshape(-1)
        selected = selected[selected >= 0]
        keep_mask[selected] = True
        return keep_mask

    def _fuse_selector_input(
            self,
            edge_input: torch.Tensor,
            src_feat: Optional[torch.Tensor] = None,
            dst_feat: Optional[torch.Tensor] = None,
            map_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        selector_input = self.edge_feat_proj(edge_input)
        semantic_terms = []

        if src_feat is not None:
            if src_feat.size(0) != edge_input.size(0):
                raise ValueError(f"src_feat must be edge-wise [E, C], got {tuple(src_feat.shape)} for E={edge_input.size(0)}")
            semantic_terms.append(self.src_feat_proj(src_feat))

        if dst_feat is not None:
            if dst_feat.size(0) != edge_input.size(0):
                raise ValueError(f"dst_feat must be edge-wise [E, C], got {tuple(dst_feat.shape)} for E={edge_input.size(0)}")
            semantic_terms.append(self.dst_feat_proj(dst_feat))

        if map_feat is not None:
            if map_feat.size(0) != edge_input.size(0):
                raise ValueError(f"map_feat must be edge-wise [E, C], got {tuple(map_feat.shape)} for E={edge_input.size(0)}")
            semantic_terms.append(self.map_feat_proj(map_feat))

        if semantic_terms:
            semantic = torch.stack(semantic_terms, dim=0).sum(dim=0)
            selector_input = selector_input + self.feature_scale.exp().clamp(max=1.0) * semantic

        return self.selector_input_norm(selector_input)

    def forward(
            self,
            edge_index: torch.Tensor,
            edge_input: torch.Tensor,
            topk: Optional[int],
            keep_ratio: Optional[float] = None,
            dst_index: Optional[torch.Tensor] = None,
            dist: Optional[torch.Tensor] = None,
            dist_norm: Optional[float] = None,
            src_feat: Optional[torch.Tensor] = None,
            dst_feat: Optional[torch.Tensor] = None,
            map_feat: Optional[torch.Tensor] = None,
    ):
        if edge_input.size(0) == 0:
            keep_mask = torch.zeros(0, dtype=torch.bool, device=edge_input.device)
            info = {"score": edge_input.new_zeros((0,)), "gate": edge_input.new_zeros((0,)), "keep_mask": keep_mask}
            return edge_index, edge_input, keep_mask, info

        if dst_index is None:
            dst_index = edge_index[1]

        selector_input = self._fuse_selector_input(
            edge_input=edge_input,
            src_feat=src_feat,
            dst_feat=dst_feat,
            map_feat=map_feat,
        )
        scores = self.score_net(selector_input).squeeze(-1)

        if self.distance_prior and dist is not None:
            if dist_norm is None:
                norm = dist.detach().mean().clamp_min(1.0)
            else:
                norm = torch.as_tensor(dist_norm, device=dist.device, dtype=dist.dtype).clamp_min(1e-6)
            scores = scores - F.softplus(self.distance_penalty) * dist / norm

        route_scores = scores
        if self.training and self.use_gumbel:
            route_scores = route_scores + self._sample_gumbel_like(route_scores)

        if self.selection_mode == "global_ratio":
            keep_mask = self._global_top_ratio_mask(
                route_scores,
                keep_ratio=keep_ratio if keep_ratio is not None else self.global_keep_ratio,
                min_keep_edges=self.min_keep_edges,
            )
        elif self.selection_mode == "adaptive_global_ratio":
            adaptive_keep_ratio = self._adaptive_global_keep_ratio(
                route_scores,
                base_keep_ratio=keep_ratio if keep_ratio is not None else self.global_keep_ratio,
            )
            keep_mask = self._global_top_ratio_mask(
                route_scores,
                keep_ratio=adaptive_keep_ratio,
                min_keep_edges=self.min_keep_edges,
            )
        elif self.selection_mode == "global_threshold":
            keep_mask = self._global_threshold_mask(
                route_scores,
                min_keep_edges=self.min_keep_edges,
            )
        elif self.selection_mode == "per_receiver":
            # Fallback compatible with the older behavior: top-k incoming edges
            # for every receiver/center node.
            max_per_group = self._max_edges_per_group(dst_index)
            if topk is None or int(topk) >= max_per_group:
                keep_mask = torch.ones(edge_input.size(0), dtype=torch.bool, device=edge_input.device)
            else:
                keep_mask = self._group_topk_mask_dense(
                    route_scores,
                    dst_index,
                    topk=int(topk),
                    max_dense_groups=self.max_dense_groups,
                )
        else:
            raise ValueError(
                f"Unknown selection_mode={self.selection_mode!r}. "
                "Use 'global_ratio', 'adaptive_global_ratio', 'global_threshold', or 'per_receiver'."
            )

        selected_edge_index = edge_index[:, keep_mask]
        selected_edge_input = edge_input[keep_mask]
        selected_scores = scores[keep_mask]

        if self.gate_edges and selected_edge_input.size(0) > 0:
            gate = torch.sigmoid(selected_scores / self.temperature)
        else:
            gate = torch.ones_like(selected_scores)

        info = {
            "score": selected_scores,
            "gate": gate,
            "keep_mask": keep_mask,
            "num_candidates": int(edge_input.size(0)),
            "num_kept": int(selected_edge_input.size(0)),
            "selection_mode": self.selection_mode,
            "threshold_type": self.threshold_type if self.selection_mode == "global_threshold" else None,
        }
        return selected_edge_index, selected_edge_input, keep_mask, info




class EdgeEncoder(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_freq_bands:int,
            hist_drop_prob=0.0,
            time_span=30,
            shift=0,
            discriminator=False,
            use_bird=False,
            use_pl2a=False,
            use_a2a=False,
            use_t2t=False,
            differentiable_edge=True,
            learnable_edge_selector=True,
            learnable_pl2a_selector: Optional[bool] = None,
            selector_topk: Optional[int] = None,
            selector_candidate_factor: int = 4,
            selector_temperature: float = 1.0,
            selector_use_gumbel: bool = False,
            selector_gate_edges: bool = True,
            selector_distance_prior: bool = True,
            selector_agent_feat_dim: Optional[int] = None,
            selector_map_feat_dim: Optional[int] = None,
            selector_selection_mode: str = "global_ratio",
            selector_global_keep_ratio: Optional[float] = None,
            selector_min_keep_edges: int = 1,
            selector_adaptive_min_keep_ratio: float = 0.25,
            selector_adaptive_max_keep_ratio: float = 0.75,
            selector_adaptive_score_std_scale: float = 1.0,
            selector_adaptive_strength: float = 1.0,
            selector_threshold_type: str = "zscore",
            selector_score_threshold: float = 0.0,
            selector_prob_threshold: float = 0.25,
            selector_threshold_max_keep_ratio: Optional[float] = None,
            return_selector_info: bool = False,
    ) -> None:
        super(EdgeEncoder, self).__init__()

        self.differentiable_edge=differentiable_edge

        self.rollout_traj=False

        self.hist_drop_prob = hist_drop_prob
        self.time_span = time_span
        self.shift = shift
        self.use_t2t=use_t2t

        self.learnable_edge_selector = learnable_edge_selector
        self.selector_topk = selector_topk
        self.selector_candidate_factor = max(int(selector_candidate_factor), 1)
        self.selector_selection_mode = selector_selection_mode
        self.selector_global_keep_ratio = selector_global_keep_ratio
        self.selector_min_keep_edges = int(selector_min_keep_edges)
        self.selector_adaptive_min_keep_ratio = float(selector_adaptive_min_keep_ratio)
        self.selector_adaptive_max_keep_ratio = float(selector_adaptive_max_keep_ratio)
        self.selector_adaptive_score_std_scale = float(selector_adaptive_score_std_scale)
        self.selector_adaptive_strength = float(selector_adaptive_strength)
        self.selector_threshold_type = selector_threshold_type
        self.selector_score_threshold = float(selector_score_threshold)
        self.selector_prob_threshold = float(selector_prob_threshold)
        self.selector_threshold_max_keep_ratio = selector_threshold_max_keep_ratio
        self.return_selector_info = return_selector_info
        if learnable_pl2a_selector is None:
            learnable_pl2a_selector = learnable_edge_selector
        self.learnable_pl2a_selector = learnable_pl2a_selector

        if not use_bird:
            input_dim = 3
        else:
            input_dim = 4

        if use_pl2a:
            self.r_pt2a_emb = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

        if use_a2a:
            self.tokenized_pos=False

            self.r_a2a_emb = FourierEmbedding(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

        if use_t2t:
            self.r_t_emb = FourierEmbedding(
                input_dim=input_dim+1,
                hidden_dim=hidden_dim,
                num_freq_bands=num_freq_bands,
            )

        self.a2a_selector = (
            LearnableTopKEdgeSelector(
                hidden_dim=hidden_dim,
                edge_input_dim=input_dim,
                temperature=selector_temperature,
                use_gumbel=selector_use_gumbel,
                gate_edges=selector_gate_edges,
                distance_prior=selector_distance_prior,
                agent_feat_dim=selector_agent_feat_dim,
                map_feat_dim=selector_map_feat_dim,
                selection_mode=selector_selection_mode,
                global_keep_ratio=selector_global_keep_ratio,
                min_keep_edges=selector_min_keep_edges,
                adaptive_min_keep_ratio=selector_adaptive_min_keep_ratio,
                adaptive_max_keep_ratio=selector_adaptive_max_keep_ratio,
                adaptive_score_std_scale=selector_adaptive_score_std_scale,
                adaptive_strength=selector_adaptive_strength,
                threshold_type=selector_threshold_type,
                score_threshold=selector_score_threshold,
                prob_threshold=selector_prob_threshold,
                threshold_max_keep_ratio=selector_threshold_max_keep_ratio,
            )
            if learnable_edge_selector and use_a2a
            else None
        )
        self.pl2a_selector = (
            LearnableTopKEdgeSelector(
                hidden_dim=hidden_dim,
                edge_input_dim=input_dim,
                temperature=selector_temperature,
                use_gumbel=selector_use_gumbel,
                gate_edges=selector_gate_edges,
                distance_prior=selector_distance_prior,
                agent_feat_dim=selector_agent_feat_dim,
                map_feat_dim=selector_map_feat_dim,
                selection_mode=selector_selection_mode,
                global_keep_ratio=selector_global_keep_ratio,
                min_keep_edges=selector_min_keep_edges,
                adaptive_min_keep_ratio=selector_adaptive_min_keep_ratio,
                adaptive_max_keep_ratio=selector_adaptive_max_keep_ratio,
                adaptive_score_std_scale=selector_adaptive_score_std_scale,
                adaptive_strength=selector_adaptive_strength,
                threshold_type=selector_threshold_type,
                score_threshold=selector_score_threshold,
                prob_threshold=selector_prob_threshold,
                threshold_max_keep_ratio=selector_threshold_max_keep_ratio,
            )
            if learnable_edge_selector and learnable_pl2a_selector and use_pl2a
            else None
        )

    def _selector_candidate_k(self, final_k: int) -> int:
        if not self.learnable_edge_selector:
            return final_k
        return max(int(final_k), int(final_k) * self.selector_candidate_factor)

    def _selector_topk(self, fallback_k: int) -> int:
        return int(self.selector_topk) if self.selector_topk is not None else int(fallback_k)

    def _selector_keep_ratio(self) -> float:
        if self.selector_global_keep_ratio is not None:
            return float(self.selector_global_keep_ratio)
        # Candidate edges are generated with final_k * candidate_factor. Keeping
        # 1 / candidate_factor approximately preserves the original edge budget,
        # but allows the selector to allocate edges unevenly across agents.
        return 1.0 / float(self.selector_candidate_factor)

    def _apply_edge_selector(
            self,
            selector: Optional[LearnableTopKEdgeSelector],
            edge_index: torch.Tensor,
            edge_input: torch.Tensor,
            final_topk: int,
            keep_ratio: Optional[float] = None,
            dst_index: Optional[torch.Tensor] = None,
            dist: Optional[torch.Tensor] = None,
            dist_norm: Optional[float] = None,
            src_feat: Optional[torch.Tensor] = None,
            dst_feat: Optional[torch.Tensor] = None,
            map_feat: Optional[torch.Tensor] = None,
    ):
        if selector is None:
            keep_mask = torch.ones(edge_input.size(0), dtype=torch.bool, device=edge_input.device)
            return edge_index, edge_input, keep_mask, None

        return selector(
            edge_index=edge_index,
            edge_input=edge_input,
            topk=self._selector_topk(final_topk),
            keep_ratio=keep_ratio if keep_ratio is not None else self._selector_keep_ratio(),
            dst_index=dst_index,
            dist=dist,
            dist_norm=dist_norm,
            src_feat=src_feat,
            dst_feat=dst_feat,
            map_feat=map_feat,
        )

    def build_temporal_edge(
            self,
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2],
            mask,  # [n_agent, n_step]
            inference_mask=None,  # [n_agent, n_step]
            agent_train_mask=None
    ):
        if agent_train_mask is not None:
            pos_a=pos_a[agent_train_mask]
            head_a=head_a[agent_train_mask]
            head_vector_a=head_vector_a[agent_train_mask]
            mask=mask[agent_train_mask]

        pos_t = pos_a.flatten(0, 1)
        head_t = head_a.flatten(0, 1)
        head_vector_t = head_vector_a.flatten(0, 1)

        flat_mask = mask.transpose(0, 1).flatten(0, 1)

        if self.hist_drop_prob > 0 and self.training:
            _mask_keep = torch.bernoulli(
                torch.ones_like(mask) * (1 - self.hist_drop_prob)
            )
            mask = mask & _mask_keep

        if inference_mask is not None:
            mask_t = mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = mask.unsqueeze(2) & mask.unsqueeze(1)

        if self.shift <= 0:
            raise ValueError("shift must be positive when temporal edges are enabled.")

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[
            :, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift
        ]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])

        feat_a=project_to_local_frame(rel_pos_t,head_vector_t[edge_index_t[1]],self.differentiable_edge)

        r_t = torch.cat(
            [
                feat_a,
                rel_head_t[:,None],
                (edge_index_t[0] - edge_index_t[1])[:,None],
            ],
            dim=-1,
        )

        n_agent, n_step = mask.shape

        edge_index_t = (edge_index_t % n_step) * n_agent + edge_index_t // n_step

        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)

        if torch.any(flat_mask==False):

            N_total = n_step * n_agent  # total nodes in transposed ordering

            kept_nodes = torch.nonzero(flat_mask, as_tuple=True)[0]  # shape [M]
            map_to_compact = torch.full((N_total,), -1, dtype=torch.long, device=kept_nodes.device)
            map_to_compact[kept_nodes] = torch.arange(kept_nodes.size(0), device=kept_nodes.device, dtype=torch.long)

            edge_index_t = map_to_compact[edge_index_t]

        return edge_index_t, r_t

    @staticmethod
    def _compact_feature_by_mask(feature: Optional[torch.Tensor], mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if feature is None:
            return None
        if mask is None:
            if feature.dim() > 2:
                return feature.reshape(-1, feature.size(-1))
            return feature
        if feature.shape[:mask.dim()] == mask.shape:
            return feature[mask]
        if feature.size(0) == mask.numel():
            return feature[mask.reshape(-1)]
        raise ValueError(
            f"Feature shape {tuple(feature.shape)} is incompatible with mask shape {tuple(mask.shape)}."
        )

    def build_interaction_edge(
            self,
            pos_s,  # [n_agent, n_step, 2]
            head_s,  # [n_agent, n_step]
            head_vector_s,  # [n_agent, n_step, 2]
            batch_s,  # [n_agent*n_step]
            mask,  # [n_agent, n_step]
            max_num_neighbors,
            max_radius,
            agent_train_mask=None,
            layer_num=1,
            counter_feat_a=None,
            dis_edge_mask=None,
            a2a_edge_index=None,
            feat_a: Optional[torch.Tensor] = None,  # agent/token feature, [n_agent, n_step, C] or compact [N, C]
        ):
        agent_feat_s = self._compact_feature_by_mask(feat_a, mask)

        if mask is not None:
            pos_s = pos_s[mask]
            head_s = head_s[mask]
            head_vector_s = head_vector_s[mask]
            batch_s = batch_s[mask]

        if a2a_edge_index is None:
            edge_index_a2a = radiusGraphNearest(x=pos_s,
                                                r=max_radius,
                                                batch=batch_s,
                                                loop=False,
                                                max_num_neighbors=self._selector_candidate_k(max_num_neighbors))
        else:
            edge_index_a2a = a2a_edge_index

        if agent_train_mask is not None and layer_num==1:
            edge_index_a2a = edge_index_a2a[:, agent_train_mask[edge_index_a2a[1]]]

        # if mask is not None:
        #     edge_index_a2a = subgraph(subset=mask, edge_index=edge_index_a2a)[0]

        # if self.training:
        #     keep_mask = torch.rand(len(edge_index_a2a[0])) > 0.1
        #     edge_index_a2a = edge_index_a2a[:, keep_mask]
        if dis_edge_mask is not None:
            dis_edge_mask=dis_edge_mask[edge_index_a2a[1]]
            edge_index_a2a=edge_index_a2a[:,dis_edge_mask]

        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])

        dist=torch.norm(rel_pos_a2a, p=2, dim=-1)

        rel_feat_a=project_to_local_frame(rel_pos_a2a,head_vector_s[edge_index_a2a[1]],self.differentiable_edge)

        raw_a2a = torch.cat(
            [
                rel_feat_a,
                rel_head_a2a[:, None],
            ],
            dim=-1,
        )

        # Fast path: select on cheap raw geometry + semantic features first,
        # then run FourierEmbedding only for the selected top-k edges.
        edge_index_a2a, raw_a2a, keep_mask_a2a, selector_info = self._apply_edge_selector(
            selector=self.a2a_selector,
            edge_index=edge_index_a2a,
            edge_input=raw_a2a,
            final_topk=max_num_neighbors,
            dst_index=edge_index_a2a[1],
            dist=dist,
            dist_norm=max_radius,
            src_feat=(agent_feat_s[edge_index_a2a[0]] if agent_feat_s is not None else None),
            dst_feat=(agent_feat_s[edge_index_a2a[1]] if agent_feat_s is not None else None),
        )
        dist = dist[keep_mask_a2a]

        r_a2a = self.r_a2a_emb(continuous_inputs=raw_a2a, categorical_embs=None)
        if selector_info is not None and self.a2a_selector is not None and self.a2a_selector.gate_edges:
            r_a2a = r_a2a * selector_info["gate"].unsqueeze(-1)

        if counter_feat_a is not None:
            start_index = edge_index_a2a[0]
            end_index = edge_index_a2a[1]

            start_pos = pos_s[start_index]

            start_heading = head_s[start_index]

            center_nei_pos = scatter_mean(start_pos, end_index, dim=0, dim_size=len(pos_s))

            center_nei_heading= scatter_mean(start_heading, end_index, dim=0, dim_size=len(head_s))

            rel_pos_a2a = start_pos - center_nei_pos[end_index]
            rel_head_a2a = wrap_angle(start_heading - center_nei_heading[end_index])

            r_a2a_nei = torch.stack(
                [
                    torch.norm(rel_pos_a2a, p=2, dim=-1),
                    angle_between_2d_vectors(
                        ctr_vector=head_vector_s[edge_index_a2a[1]],
                        nbr_vector=rel_pos_a2a[:, :2],
                    ),
                    rel_head_a2a
                ],
                dim=-1,
            )

            r_a2a_nei = torch.cat([r_a2a_nei, rel_pos_a2a[:, 2:]], dim=-1)

            r_a2a_nei = self.r_a2a_emb(continuous_inputs=r_a2a_nei, categorical_embs=None)
        else:
            r_a2a_nei=center_nei_pos=center_nei_heading=None

        return edge_index_a2a, r_a2a,dist,(selector_info if self.return_selector_info else None),r_a2a_nei,center_nei_pos,center_nei_heading

    def build_map2map_edge(self,
                           pos_pl,  # [n_pl, 2]
                           orient_pl,  # [n_pl]
                           pos_s,  # [n_agent, n_step, 2]
                           head_s,  # [n_agent, n_step]
                           head_vector_s,  # [n_agent, n_step, 2]
                           batch_s,  # [n_agent*n_step]
                           batch_pl,  # [n_pl*n_step]
                           pl2a_radius,
                           max_num_neighbors,
                           l2l_edge_index=None,
                           l2l_feature=None,
                           feat_a: Optional[torch.Tensor] = None,
                           feat_map: Optional[torch.Tensor] = None
                           ):

        agent_feat_s = feat_a

        if l2l_edge_index is None:
            edge_index_pl2pl = radiusGraphNearest2(x=pos_s,
                                                  y=pos_pl,
                                                  r=pl2a_radius,
                                                  batch_x=batch_s,
                                                  batch_y=batch_pl,
                                                  max_num_neighbors=self._selector_candidate_k(max_num_neighbors))
        else:
            edge_index_pl2pl=l2l_edge_index

        # #edge_index[0] → indices in y (query points)            edge_index[1] → indices in x (neighbor points)
        rel_pos_pl2a = pos_pl[edge_index_pl2pl[0]] - pos_s[edge_index_pl2pl[1]]   #src, dst
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2pl[0]] - head_s[edge_index_pl2pl[1]]
        )
        dist_pl2a = torch.norm(rel_pos_pl2a, p=2, dim=-1)

        rel_feat_a=project_to_local_frame(rel_pos_pl2a,head_vector_s[edge_index_pl2pl[1]],self.differentiable_edge)


        raw_pl2a = torch.cat(
            [
                rel_feat_a,
                rel_orient_pl2a[:, None],
            ],
            dim=-1,
        )

        edge_index_pl2pl, raw_pl2a, keep_mask_pl2a, selector_info = self._apply_edge_selector(
            selector=self.pl2a_selector,
            edge_index=edge_index_pl2pl,
            edge_input=raw_pl2a,
            final_topk=max_num_neighbors,
            dst_index=edge_index_pl2pl[1],
            dist=dist_pl2a,
            dist_norm=pl2a_radius,
            src_feat=(feat_map[edge_index_pl2pl[0]] if feat_map is not None else None),
            dst_feat=(agent_feat_s[edge_index_pl2pl[1]] if agent_feat_s is not None else None),
            map_feat=(feat_map[edge_index_pl2pl[0]] if feat_map is not None else None),
        )

        if l2l_feature is not None:
            l2l_feature = l2l_feature[keep_mask_pl2a]
        r_pl2a = self.r_pt2a_emb(continuous_inputs=raw_pl2a, categorical_embs=l2l_feature)
        if selector_info is not None and self.pl2a_selector is not None and self.pl2a_selector.gate_edges:
            r_pl2a = r_pl2a * selector_info["gate"].unsqueeze(-1)

        return edge_index_pl2pl, r_pl2a


    def build_map2agent_edge(
            self,
            pos_pl,  # [n_pl, 2]
            orient_pl,  # [n_pl]
            pos_a,  # [n_agent, n_step, 2]
            head_a,  # [n_agent, n_step]
            head_vector_a,  # [n_agent, n_step, 2]
            mask,  # [n_agent, n_step]
            batch_s,  # [n_agent*n_step]
            batch_pl,  # [n_pl*n_step]
            pl2a_radius,
            max_num_neighbors,
            mask_pl=None,
            agent_train_mask=None,
            use_counterfactual=False,
            route_map_index=None,
            layer_num=1,
            l2a_edge_index=None,
            feat_a: Optional[torch.Tensor] = None,
            feat_map: Optional[torch.Tensor] = None
    ):

        if agent_train_mask is not None and layer_num==1:
            mask = mask & agent_train_mask[:,None]

        agent_feat_s = self._compact_feature_by_mask(feat_a, mask)

        if mask is not None:
            n_agent, n_step = mask.shape

            pos_s=pos_a[mask]
            head_s=head_a[mask]
            head_vector_s=head_vector_a[mask]
            batch_s=batch_s[mask]
        else:
            pos_s=pos_a
            head_s=head_a
            head_vector_s=head_vector_a
            batch_s=batch_s
            n_step=1


        if l2a_edge_index is None:
            edge_index_pl2a = radiusGraphNearest2(x=pos_s,
                                                  y=pos_pl,
                                                  r=pl2a_radius,
                                                  batch_x=batch_s,
                                                  batch_y=batch_pl,
                                                  max_num_neighbors=self._selector_candidate_k(max_num_neighbors))

        else:
            edge_index_pl2a=l2a_edge_index

        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(
            orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]]
        )
        dist_pl2a = torch.norm(rel_pos_pl2a, p=2, dim=-1)

        rel_feat_a=project_to_local_frame(rel_pos_pl2a,head_vector_s[edge_index_pl2a[1]],self.differentiable_edge)

        raw_pl2a = torch.cat(
            [
                rel_feat_a,
                rel_orient_pl2a[:, None],
            ],
            dim=-1,
        )

        edge_index_pl2a, raw_pl2a, keep_mask_pl2a, selector_info = self._apply_edge_selector(
            selector=self.pl2a_selector,
            edge_index=edge_index_pl2a,
            edge_input=raw_pl2a,
            final_topk=max_num_neighbors,
            dst_index=edge_index_pl2a[1],
            dist=dist_pl2a,
            dist_norm=pl2a_radius,
            src_feat=(feat_map[edge_index_pl2a[0]] if feat_map is not None else None),
            dst_feat=(agent_feat_s[edge_index_pl2a[1]] if agent_feat_s is not None else None),
            map_feat=(feat_map[edge_index_pl2a[0]] if feat_map is not None else None),
        )

        r_pl2a = self.r_pt2a_emb(continuous_inputs=raw_pl2a, categorical_embs=None)
        if selector_info is not None and self.pl2a_selector is not None and self.pl2a_selector.gate_edges:
            r_pl2a = r_pl2a * selector_info["gate"].unsqueeze(-1)

        if n_step>1:
            N_total = n_agent * n_step

            # 1) Kept global indices in both orderings
            flat_mask_agent = mask.flatten(0, 1)  # agent-major
            flat_mask_time = mask.transpose(0, 1).flatten(0, 1)  # time-major

            kept_agent = torch.nonzero(flat_mask_agent, as_tuple=False).squeeze(1)  # [M], global idx
            kept_time = torch.nonzero(flat_mask_time, as_tuple=False).squeeze(1)  # [M], global idx

            map_global_to_compact_time = torch.full((N_total,), -1, dtype=torch.long, device=mask.device)
            map_global_to_compact_time[kept_time] = torch.arange(kept_time.numel(), device=mask.device)

            # 3) Convert compact agent-major indices -> global -> compact time-major indices
            dst_compact_agent = edge_index_pl2a[1]  # indices into pos_a[mask]
            dst_global = kept_agent[dst_compact_agent]  # global flattened indices

            dst_global=(dst_global % n_step) * n_agent + dst_global // n_step

            new_dst = map_global_to_compact_time[dst_global]
            edge_index_pl2a = torch.stack([edge_index_pl2a[0], new_dst], dim=0)

        return edge_index_pl2a, r_pl2a

