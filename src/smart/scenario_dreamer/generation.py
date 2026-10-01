"""Official Waymo initial-scene count prior and unconditioned generation graphs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from .core.data_container import ScenarioDreamerData
from .core.pyg_helpers import get_edge_index_complete_graph, get_edge_index_bipartite


DEFAULT_COUNT_PRIOR = Path(__file__).with_name("assets") / "initial_prob_matrix_waymo.npz"
MAP_ID_PROBABILITIES = (0.62, 0.38)


class SceneCountPrior:
    """Sample map category and its joint (lane count, agent count) distribution.

    The official tensor has shape [2, 101, 31]. Array indices are counts,
    including zero-probability entries for zero lanes or agents. NPZ is read
    without pickle support; JSON can supply an equivalent ``probabilities`` array.
    An independent CPU Torch generator controls counts only; diffusion noise
    uses PyTorch's global seeded RNG, as in the model core.
    """

    def __init__(self, path, *, max_num_lanes, max_num_agents, seed=0):
        self.path = Path(path).expanduser().resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.path.suffix == ".npz":
            with np.load(self.path, allow_pickle=False) as source:
                probabilities = np.asarray(source["probabilities"], dtype=np.float32)
        elif self.path.suffix == ".json":
            probabilities = np.asarray(json.loads(self.path.read_text())["probabilities"], dtype=np.float32)
        else:
            raise ValueError("Count prior must be a .npz or .json file with a probabilities array")
        if (probabilities.ndim != 3 or probabilities.shape[0] != 2
                or not 2 <= probabilities.shape[1] <= max_num_lanes + 1
                or not 2 <= probabilities.shape[2] <= max_num_agents + 1):
            raise ValueError("Count prior must have shape [2, lane counts, agent counts] within model limits")
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError("Count prior probabilities must be finite and nonnegative")
        if probabilities[:, 0, :].any() or probabilities[:, :, 0].any():
            raise ValueError("Count prior must assign zero probability to zero lanes or zero agents")
        flat = probabilities.reshape(2, -1)
        totals = flat.sum(axis=1, keepdims=True)
        if (totals <= 0).any():
            raise ValueError("Each map category must have a nonempty count distribution")
        self.probabilities = torch.from_numpy(flat.copy())
        self.agent_axis_size = probabilities.shape[2]
        self.seed = int(seed)
        self.reset()

    def reset(self):
        self.rng = torch.Generator(device="cpu").manual_seed(self.seed)
        self.joint_counts = Counter()

    def sample(self, num_scenes):
        if not isinstance(num_scenes, (int, np.integer)) or num_scenes < 1:
            raise ValueError("Generation requires a positive integer scene count")
        sampled = []
        for _ in range(num_scenes):
            map_id = int(torch.multinomial(torch.tensor(MAP_ID_PROBABILITIES), 1, generator=self.rng))
            index = int(torch.multinomial(self.probabilities[map_id], 1, generator=self.rng))
            lanes, agents = divmod(index, self.agent_axis_size)
            triple = (map_id, lanes, agents)
            self.joint_counts[triple] += 1
            sampled.append(triple)
        return np.asarray(sampled, dtype=np.int64)

    def report(self):
        return {
            "scene_count_source": "official_prior", "sampling_seed": self.seed,
            "count_sampling_rng": "independent CPU torch.Generator; interleaved torch.multinomial",
            "count_prior_path": str(self.path), "count_prior_sha256": self.sha256,
            "map_id_probabilities": list(MAP_ID_PROBABILITIES),
            "map_id_counts": {str(i): sum(n for (m, _, _), n in self.joint_counts.items() if m == i)
                              for i in range(2)},
            "num_scenes": sum(self.joint_counts.values()),
            "num_agents": sum(a * n for (_, _, a), n in self.joint_counts.items()),
            "num_lanes": sum(l * n for (_, l, _), n in self.joint_counts.items()),
            "joint_counts": [{"map_id": m, "num_lanes": l, "num_agents": a, "num_scenes": n}
                             for (m, l, a), n in sorted(self.joint_counts.items())],
        }


def build_generation_graph(counts, *, agent_latent_dim, lane_latent_dim, device, dtype):
    """Build complete graphs without accessing reference geometry or states.

    SD keeps ego first; ``rows`` maps its rows to SMART's ego-last order.
    The generated coordinate frame is SD-local (the transforms are identities).
    """
    graphs, rows = [], []
    offset = 0
    for map_id, lanes, agents in counts:
        map_id, lanes, agents = int(map_id), int(lanes), int(agents)
        if map_id not in (0, 1) or lanes < 1 or agents < 1:
            raise ValueError("Generated graphs require map_id=0/1 and positive node counts")
        data = ScenarioDreamerData()
        data.num_agents, data.num_lanes = agents, lanes
        data.map_id, data.lg_type, data.num_lanes_after_origin = map_id, 0, 0
        data["agent"].x = torch.zeros((agents, agent_latent_dim), dtype=dtype)
        data["lane"].x = torch.zeros((lanes, lane_latent_dim), dtype=dtype)
        for source, target, edge in (
            ("agent", "agent", get_edge_index_complete_graph(agents)),
            ("lane", "lane", get_edge_index_complete_graph(lanes)),
            ("lane", "agent", get_edge_index_bipartite(lanes, agents)),
        ):
            data[source, "to", target].edge_index = edge
        graphs.append(data)
        rows.append(torch.cat((torch.tensor([offset + agents - 1]), torch.arange(offset, offset + agents - 1))))
        offset += agents
    if not graphs:
        raise ValueError("Generation requires at least one scene")
    return (Batch.from_data_list(graphs).to(device), torch.cat(rows).to(device),
            torch.zeros((len(graphs), 2), device=device, dtype=dtype),
            torch.zeros(len(graphs), device=device, dtype=dtype))
