"""Frozen Scenario Dreamer numerical core.

Function bodies below are copied from the THREE USER-SUPPLIED source files.
They deliberately retain upstream ordering, thresholds, and malformed-map
behaviour. No Hydra/PyG dependency is needed for these numerical methods.
See source_manifest.json and reference_sources/ for provenance.

Only the small SE(2) helpers were retrieved from the upstream geometry.py.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import copy
import numpy as np


def rotate_and_normalize_angles(current_angles, rotation_angle):
    new_angles = current_angles + rotation_angle
    normalized_angles = (new_angles + np.pi) % (2 * np.pi) - np.pi
    return normalized_angles


def make_2d_rotation_matrix(angle_in_radians):
    return np.array([[np.cos(angle_in_radians), -np.sin(angle_in_radians)],
                     [np.sin(angle_in_radians), np.cos(angle_in_radians)]])


def apply_se2_transform(coordinates, translation, yaw):
    coordinates = coordinates - translation
    transform = make_2d_rotation_matrix(angle_in_radians=yaw)
    if len(coordinates.shape) > 2:
        coord_shape = coordinates.shape
        return np.dot(transform, coordinates.reshape((-1, 2)).T).T.reshape(*coord_shape)
    return np.dot(transform, coordinates.T).T[:, :2]

def get_compact_lane_graph(data):
    """Apply lane graph compression algorithm (merging lanes that connect with node degree=2).

    The resulting compact graph uses *lane-group* identifiers where
    contiguous segments have been concatenated.  All
    connection dictionaries (``pre``, ``succ``, ``left``, ``right``)
    are updated to reference the new identifiers.

    Parameters
    ----------
    data
        Dictionary from a raw Waymo pickle.  Requires the key
        ``"lane_graph"``.

    Returns
    -------
    compact_lane_graph
        A dict with the same layout as the original Waymo lane graph
        but using merged lanes.
    """
    
    lane_ids = data['lane_graph']['lanes'].keys()
    pre_pairs = data['lane_graph']['pre_pairs']
    suc_pairs = data['lane_graph']['suc_pairs']
    left_pairs = data['lane_graph']['left_pairs']
    right_pairs = data['lane_graph']['right_pairs']

    # Remove dangling references ------------------------------------
    for lid in pre_pairs.keys():
        lid1s = pre_pairs[lid]
        for lid1 in lid1s:
            if lid1 not in lane_ids:
                pre_pairs[lid].remove(lid1)

    for lid in suc_pairs.keys():
        lid1s = suc_pairs[lid]
        for lid1 in lid1s:
            if lid1 not in lane_ids:
                suc_pairs[lid].remove(lid1)

    for lid in left_pairs.keys():
        lid1s = left_pairs[lid]
        for lid1 in lid1s:
            if lid1 not in lane_ids:
                left_pairs[lid].remove(lid1)

    for lid in right_pairs.keys():
        lid1s = right_pairs[lid]
        for lid1 in lid1s:
            if lid1 not in lane_ids:
                right_pairs[lid].remove(lid1)

    # Ensure every lane appears as a key ----------------------------
    for lane_id in lane_ids:
        if lane_id not in pre_pairs:
            pre_pairs[lane_id] = []
        if lane_id not in suc_pairs:
            suc_pairs[lane_id] = []
        if lane_id not in left_pairs:
            left_pairs[lane_id] = []
        if lane_id not in right_pairs:
            right_pairs[lane_id] = []

    lane_groups = find_lane_groups(pre_pairs, suc_pairs)       
    
    compact_lanes = {}
    compact_pre_pairs = {}
    compact_suc_pairs = {}
    compact_left_pairs = {}
    compact_right_pairs = {}
    
    for lane_group_id in lane_groups:
        compact_lane = []
        compact_pre_pair = []
        compact_suc_pair = []
        compact_left_pair = []
        compact_right_pair = []
        for i, lane_id in enumerate(lane_groups[lane_group_id]):
            # first lane in group is used to find predecessor lane group
            if i == 0:
                compact_lane.append(data['lane_graph']['lanes'][lane_id])
                
                if len(pre_pairs[lane_id]) > 0:
                    for pre_lane_id in pre_pairs[lane_id]:
                        compact_pre_pair.append(find_lane_group_id(pre_lane_id, lane_groups))
            else:
                # avoid duplicate coordinates
                compact_lane.append(data['lane_graph']['lanes'][lane_id][1:])

            if len(left_pairs[lane_id]) > 0:
                for left_lane_id in left_pairs[lane_id]:
                    to_append = find_lane_group_id(left_lane_id, lane_groups)
                    if to_append not in compact_left_pair:
                        compact_left_pair.append(to_append)
            
            if len(right_pairs[lane_id]) > 0:
                for right_lane_id in right_pairs[lane_id]:
                    to_append = find_lane_group_id(right_lane_id, lane_groups)
                    if to_append not in compact_right_pair:
                        compact_right_pair.append(to_append)

            # last lane in group is used to find successor lane group
            if i == len(lane_groups[lane_group_id]) - 1:
                if len(suc_pairs[lane_id]) > 0:
                    for suc_lane_id in suc_pairs[lane_id]:
                        compact_suc_pair.append(find_lane_group_id(suc_lane_id, lane_groups))

        compact_lane = np.concatenate(compact_lane, axis=0)
        compact_lanes[lane_group_id] = compact_lane
        compact_pre_pairs[lane_group_id] = compact_pre_pair 
        compact_suc_pairs[lane_group_id] = compact_suc_pair 
        compact_left_pairs[lane_group_id] = compact_left_pair
        compact_right_pairs[lane_group_id] = compact_right_pair
    
    compact_lane_graph = {
        'lanes': compact_lanes,
        'pre_pairs': compact_pre_pairs,
        'suc_pairs': compact_suc_pairs,
        'left_pairs': compact_left_pairs,
        'right_pairs': compact_right_pairs
    }

    return compact_lane_graph


def find_lane_groups(pre_pairs, suc_pairs):
    """Group lane IDs into compact lanes based on lane compression algorithm originally described here: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00490-supp.pdf"""
    def dfs(lane_id, group):
        visited.add(lane_id)
        group.append(lane_id)
        
        if len(suc_pairs[lane_id]) == 1:
            next_lane = suc_pairs[lane_id][0]
            if next_lane not in visited and len(pre_pairs[next_lane]) == 1:
                dfs(next_lane, group)

    lane_groups = []
    visited = set()

    def find_starting_lanes(pre_pairs, suc_pairs):
        starting_lanes = set()
        all_lanes = set(pre_pairs.keys()).union(set(suc_pairs.keys()))
        
        for lane_id in all_lanes:
            if len(pre_pairs[lane_id]) == 1:
                if len(suc_pairs[pre_pairs[lane_id][0]]) == 1:
                    continue
            
            starting_lanes.add(lane_id)
        
        return starting_lanes

    starting_lanes = find_starting_lanes(pre_pairs, suc_pairs)

    for lane_id in starting_lanes:
        if lane_id not in visited:
            group = []
            dfs(lane_id, group)
            if group:
                lane_groups.append(group)

    # edge case: get the centerline cycles that are fully closed (all degree 2 lane segments)
    num_lane_ids = sum([len(lane_groups[i]) for i in range(len(lane_groups))])
    while num_lane_ids != len(pre_pairs.keys()):
        unaccounted_ids = list(set(list(pre_pairs.keys())).difference(set([x for xs in lane_groups for x in xs])))
        
        lane_id = unaccounted_ids[0]
        if lane_id not in visited:
            group = []
            dfs(lane_id, group)
            if group:
                lane_groups.append(group)

        num_lane_ids = sum([len(lane_groups[i]) for i in range(len(lane_groups))])

    lane_groups_dict = {}
    for i in range(len(lane_groups)):
        lane_groups_dict[i] = lane_groups[i]

    return lane_groups_dict


def find_lane_group_id(lane_id, lane_groups):
    """Return the ID of the lane-group containing `lane_id`"""
    for lane_group_id in lane_groups:
        if lane_id in lane_groups[lane_group_id]:
            return lane_group_id


def resample_polyline(points, num_points=20):
    """Resample a polyline to `num_points` equally spaced points along its arc-length."""
    # Calculate the cumulative distances along the polyline
    distances = np.sqrt(((points[1:] - points[:-1])**2).sum(axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    
    # Create an array of 20 evenly spaced distance values along the polyline
    target_distances = np.linspace(0, cumulative_distances[-1], num=num_points)
    
    # Interpolate to find x and y values at these target distances
    x_new = np.interp(target_distances, cumulative_distances, points[:, 0])
    y_new = np.interp(target_distances, cumulative_distances, points[:, 1])
    
    # Combine x and y coordinates into a single array
    new_points = np.stack((x_new, y_new), axis=-1)
    
    return new_points


def extract_raw_waymo_data(agents_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the list-of-dict agent format from Waymo to flat arrays.

    Parameters
    ----------
    agents_data
        List where each element corresponds to a single agent and
        replicates Waymo's *per-time-step* trajectory dictionaries.

    Returns
    -------
    agent_data
        Array with shape ``(num_agents, T, 8)`` containing position
        ``(x, y)``, velocity ``(vx, vy)``, heading *(rad)*, length,
        width and existence mask for each time-step ``T``.
    agent_types
        One-hot encoded array of shape ``(num_agents, 5)`` for
        ``{"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}``.
    """
    
    # Get indices of non-parked cars and cars that exist for the entire episode
    agent_data = []
    agent_types = []

    for n in range(len(agents_data)):
        # Position ---------------------------------------------------
        ag_position = agents_data[n]['position']
        x_values = [entry['x'] for entry in ag_position]
        y_values = [entry['y'] for entry in ag_position]
        ag_position = np.column_stack((x_values, y_values))
        
        # Heading (unwrap to (‑pi, pi]) ------------------------------
        ag_heading = np.radians(np.array(agents_data[n]['heading']).reshape((-1, 1)))
        ag_heading = np.mod(ag_heading + np.pi, 2 * np.pi) - np.pi
        
        # Velocity ---------------------------------------------------
        ag_velocity = agents_data[n]['velocity']
        x_values = [entry['x'] for entry in ag_velocity]
        y_values = [entry['y'] for entry in ag_velocity]
        ag_velocity = np.column_stack((x_values, y_values))
        
        # Existence & size -----------------------------------------
        ag_existence = np.array(agents_data[n]['valid']).reshape((-1, 1))
        ag_length = np.ones((len(ag_position), 1)) * agents_data[n]['length']
        ag_width = np.ones((len(ag_position), 1)) * agents_data[n]['width']
        
        # Pack -------------------------------------------------------
        agent_type = get_object_type_onehot_waymo(agents_data[n]['type'])
        ag_state = np.concatenate((ag_position, ag_velocity, ag_heading, ag_length, ag_width, ag_existence), axis=-1)
        agent_data.append(ag_state)
        agent_types.append(agent_type)
    
    # convert to numpy array
    agent_data = np.array(agent_data)
    agent_types = np.array(agent_types)
    
    return agent_data, agent_types


def get_object_type_onehot_waymo(agent_type):
    """Return the one-hot NumPy vector encoding of an agent type."""
    agent_types = {"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}
    return np.eye(len(agent_types))[agent_types[agent_type]]


def modify_agent_states(agent_states):
    """Canonicalise velocity & heading for neural consumption. All remaining trailing columns (if any) are copied verbatim.

    Parameters
    ----------
    agent_states : np.ndarray
        Float32 array of shape ``(N, D)`` where columns ``2-4`` are
        ``vx``, ``vy``, and ``yaw`` respectively.

    Returns
    -------
    new_agent_states : np.ndarray
        Array with the *same* shape ``(N, D)`` where columns ``2-4``
        have been replaced by ``speed``, ``cosθ``, ``sinθ``.
    """
    new_agent_states = np.zeros_like(agent_states)
    new_agent_states[:, :2] = agent_states[:, :2]
    new_agent_states[:, 5:] = agent_states[:, 5:]
    new_agent_states[:, 2] = np.sqrt(agent_states[:, 2] ** 2 + agent_states[:, 3] ** 2)
    new_agent_states[:, 3] = np.cos(agent_states[:, 4])
    new_agent_states[:, 4] = np.sin(agent_states[:, 4])

    return new_agent_states


class ReferenceSceneOps:
    def partition_compact_lane_graph(self, compact_lane_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Split lanes that cross the scene's x-axis (``y = 0``).

        The coordinate frame places the ego at ``(0, 0)``.
        To simplify conditional generation (in-painting), we partition
        any merged *compact* lane that crosses ``y = 0`` into multiple
        *sub-lanes* so that the origin acts as a semantic divider.

        Parameters
        ----------
        compact_lane_graph
            The *compact* lane graph returned by
            :meth:`get_compact_lane_graph`.

        Returns
        -------
        partitioned_lane_graph
            A deep-copy of *compact_lane_graph* where lanes have been
            further split and edge dictionaries updated so that no lane
            segment itself crosses ``y = 0``.
        """
        max_lane_id = max(list(compact_lane_graph['lanes'].keys()))
        next_lane_id = max_lane_id + 1

        lane_ids = list(compact_lane_graph['lanes'].keys())
        for lane_id in lane_ids:
            lane = compact_lane_graph['lanes'][lane_id]
            
            # Get y-values of the lane and find where it crosses or is near y = 0
            y_values = lane[:, 1]  # Assuming lane is [x, y] points
            sign_diff = np.insert(np.diff(np.signbit(y_values)), 0, 0)
            zero_crossings = np.where(sign_diff)[0]  # Indices where lane crosses y = 0
            
            if len(zero_crossings) == 0:  # If no crossings, skip this lane
                continue
            
            # Add artificial partitions at y = 0 crossings
            new_lanes = {}
            start_index = 0
            for crossing in zero_crossings:
                end_index = crossing + 1  # Create a partition from start to crossing
                new_lanes[next_lane_id] = lane[start_index:end_index]
                start_index = crossing  # Update start index for the next partition
                next_lane_id += 1
            
            # Handle the remaining part of the lane after the last crossing
            if zero_crossings[-1] < len(y_values) - 1:
                new_lanes[next_lane_id] = lane[start_index:]
                next_lane_id += 1
            
            # Update the compact_lane_graph with new lanes
            num_new_lanes = len(new_lanes)
            if num_new_lanes == 1:
                continue
            
            for j, new_lane_id in enumerate(new_lanes.keys()):
                compact_lane_graph['lanes'][new_lane_id] = new_lanes[new_lane_id]
                if j == 0:
                    compact_lane_graph['pre_pairs'][new_lane_id] = compact_lane_graph['pre_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace successors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['pre_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['suc_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['suc_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1] # by way we defined new lane ids
                
                elif j == num_new_lanes - 1:
                    compact_lane_graph['suc_pairs'][new_lane_id] = compact_lane_graph['suc_pairs'][lane_id]
                    # leveraging bijection between suc/pre
                    # replace predecessors of other lanes with new lane
                    for other_lane_id in compact_lane_graph['suc_pairs'][lane_id]:
                        if other_lane_id is not None:
                            compact_lane_graph['pre_pairs'][other_lane_id].remove(lane_id)
                            compact_lane_graph['pre_pairs'][other_lane_id].append(new_lane_id)
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1] # by way we define new lane ids
                
                else:
                    compact_lane_graph['pre_pairs'][new_lane_id] = [new_lane_id - 1]
                    compact_lane_graph['suc_pairs'][new_lane_id] = [new_lane_id + 1]

                compact_lane_graph['left_pairs'][new_lane_id] = compact_lane_graph['left_pairs'][lane_id]
                compact_lane_graph['right_pairs'][new_lane_id] = compact_lane_graph['right_pairs'][lane_id]

            for other_lane_id in compact_lane_graph['right_pairs']:
                if lane_id in compact_lane_graph['right_pairs'][other_lane_id]:
                    compact_lane_graph['right_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['right_pairs'][other_lane_id].append(new_lane_id)

            for other_lane_id in compact_lane_graph['left_pairs']:
                if lane_id in compact_lane_graph['left_pairs'][other_lane_id]:
                    compact_lane_graph['left_pairs'][other_lane_id].remove(lane_id)
                    for new_lane_id in new_lanes.keys():
                        compact_lane_graph['left_pairs'][other_lane_id].append(new_lane_id)

            # remove old (now partitioned) lane from lane graph
            del compact_lane_graph['lanes'][lane_id]
            del compact_lane_graph['pre_pairs'][lane_id]
            del compact_lane_graph['suc_pairs'][lane_id]
            del compact_lane_graph['left_pairs'][lane_id]
            del compact_lane_graph['right_pairs'][lane_id]

        return compact_lane_graph

    def normalize_compact_lane_graph(self, lane_graph: Dict[str, Any], normalize_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Translate & rotate lanes so that the AV sits at the origin.

        Parameters
        ----------
        lane_graph
            *Compact* or *partitioned* lane graph in global Waymo
            coordinates.
        normalize_dict
            Dictionary with keys ``{"center", "yaw"}`` describing the
            ego vehicle's position and heading at the sampling
            time-step.

        Returns
        -------
        lane_graph
            The *same* input dict, modified *in-place* so that every lane
            point is expressed in the AV-centric coordinate frame.
        """
        lane_ids = lane_graph['lanes'].keys()
        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        # normalize lanes to ego
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            lane = apply_se2_transform(coordinates=lane[:, np.newaxis, :],
                                       translation=center,
                                       yaw=angle_of_rotation)[:, 0]
            # overwrite with normalized lane (centered + rotated on AV)
            lane_graph['lanes'][lane_id] = lane
        
        return lane_graph

    def get_lane_graph_within_fov(self, lane_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Return only those lanes that intersect the square *field-of-view*.

        The coordinate frame is converted to an ego-centred frame
        earlier in the pipeline, so the autonomous vehicle (AV) is at
        the origin.  A lane point is considered *in view* when both its
        absolute X *and* Y coordinates are strictly smaller than
        ``cfg_dataset.fov / 2``.  Each retained lane is then resampled
        to a fixed number of points.

        Parameters
        ----------
        lane_graph : Dict[str, Any]
            A *compact* or *partitioned* lane-graph with the standard
            keys ``{"lanes", "pre_pairs", "suc_pairs", "left_pairs",
            "right_pairs"}``.  All coordinates must already be expressed
            in the AV-centric frame.

        Returns
        -------
        lane_graph_within_fov: Dict[str, Any]
            A new lane-graph containing only lanes that intersect the
            configured field-of-view.  Connection dictionaries are
            pruned so they reference *in-FOV* lanes exclusively, and each
            lane polyline has exactly
            ``cfg_dataset.upsample_lane_num_points`` points.
        """
        lane_ids = lane_graph['lanes'].keys()
        pre_pairs = lane_graph['pre_pairs']
        suc_pairs = lane_graph['suc_pairs']
        left_pairs = lane_graph['left_pairs']
        right_pairs = lane_graph['right_pairs']
        
        # ── Identify lanes that intersect the square FOV ──────────────
        lane_ids_within_fov = []
        valid_pts = {}
        for lane_id in lane_ids:
            lane = lane_graph['lanes'][lane_id]
            points_in_fov_x = np.abs(lane[:, 0]) < (self.cfg.fov / 2)
            points_in_fov_y = np.abs(lane[:, 1]) < (self.cfg.fov / 2)
            points_in_fov = points_in_fov_x * points_in_fov_y
            
            if np.any(points_in_fov):
                lane_ids_within_fov.append(lane_id)
                valid_pts[lane_id] = points_in_fov

        lanes_within_fov = {}
        pre_pairs_within_fov = {}
        suc_pairs_within_fov = {}
        left_pairs_within_fov = {}
        right_pairs_within_fov = {}
        
        # ── Prune connection dictionaries and resample polylines ─────────────────────────────
        for lane_id in lane_ids_within_fov:
            if lane_id in lane_ids:
                lane = lane_graph['lanes'][lane_id][valid_pts[lane_id]]
                # why upsample here instead of resample to self.cfg.num_points_per_lane?
                # these lanes may need to be partitioned later, so we want to ensure high lane resolution
                # for accurate partitioning. We resample to self.cfg.num_points_per_lane in get_road_points_adj
                resampled_lane = resample_polyline(lane, num_points=self.cfg.upsample_lane_num_points)
                lanes_within_fov[lane_id] = resampled_lane
            
            if lane_id in pre_pairs:
                pre_pairs_within_fov[lane_id] = [l for l in pre_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                pre_pairs_within_fov[lane_id] = []
            
            if lane_id in suc_pairs:
                suc_pairs_within_fov[lane_id] = [l for l in suc_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                suc_pairs_within_fov[lane_id] = [] 

            if lane_id in left_pairs:
                left_pairs_within_fov[lane_id] = [l for l in left_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                left_pairs_within_fov[lane_id] = []
            
            if lane_id in right_pairs:
                right_pairs_within_fov[lane_id] = [l for l in right_pairs[lane_id] if l in lane_ids_within_fov]
            else:
                right_pairs_within_fov[lane_id] = []
        
        lane_graph_within_fov = {
            'lanes': lanes_within_fov,
            'pre_pairs': pre_pairs_within_fov,
            'suc_pairs': suc_pairs_within_fov,
            'left_pairs': left_pairs_within_fov,
            'right_pairs': right_pairs_within_fov
        }
        
        return lane_graph_within_fov

    def get_road_points_adj(
        self,
        compact_lane_graph: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """This helper converts the *sparse*, dictionary-based lane graph
        representation that comes out of
        :meth:`get_compact_lane_graph` / :meth:`partition_compact_lane_graph`
        into adjacency matrices and resamples lanes to num_points_per_lane points.

        Parameters
        ----------
        compact_lane_graph : Dict[str, Any]
            Lane graph already translated & rotated into the
            ego-centric frame.  Must contain keys
            ``{"lanes", "pre_pairs", "suc_pairs", "left_pairs", "right_pairs"}``.

        Returns
        -------
        road_points : np.ndarray
            Float32 tensor of shape ``(L, P, 2)`` where ``P`` is
            ``cfg_dataset.num_points_per_lane`` and ``L`` ≤
            ``cfg_dataset.max_num_lanes``.
        pre_adj, suc_adj, left_adj, right_adj : np.ndarray
            Four dense binary adjacency matrices of shape ``(L, L)``
            corresponding to predecessor, successor, left and
            right relationships respectively.
        num_lanes : int
            The number of lanes actually retained
        """
        
        # ── Step 1: resample every lane to fixed P points ──────────────
        resampled_lanes = []
        idx_to_id = {}
        id_to_idx = {}
        i = 0
        for lane_id in compact_lane_graph['lanes']:
            lane = compact_lane_graph['lanes'][lane_id]
            resampled_lane = resample_polyline(lane, num_points=self.cfg.num_points_per_lane)
            resampled_lanes.append(resampled_lane)
            idx_to_id[i] = lane_id
            id_to_idx[lane_id] = i
            
            i += 1
        
        # ── Step 2: keep the max_num_lanes closest to the origin ───────
        resampled_lanes = np.array(resampled_lanes)
        num_lanes = min(len(resampled_lanes), self.cfg.max_num_lanes)
        dist_to_origin = np.linalg.norm(resampled_lanes, axis=-1).min(1)
        closest_lane_ids = np.argsort(dist_to_origin)[:num_lanes]
        resampled_lanes = resampled_lanes[closest_lane_ids]

        # mapping from old idx to new index after ordering by distance
        idx_to_new_idx = {}
        new_idx_to_idx = {}
        for i, j in enumerate(closest_lane_ids):
            idx_to_new_idx[j] = i 
            new_idx_to_idx[i] = j

        # Pre‑allocate adjacency matrices ------------------------------
        pre_road_adj = np.zeros((num_lanes, num_lanes))
        suc_road_adj = np.zeros((num_lanes, num_lanes))
        left_road_adj = np.zeros((num_lanes, num_lanes))
        right_road_adj = np.zeros((num_lanes, num_lanes))
        
        
        # ── Step 3: populate the matrices ──────────────────────────────
        for new_idx_i in range(num_lanes):
            for id_j in compact_lane_graph['pre_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    pre_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1 

            for id_j in compact_lane_graph['suc_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    suc_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['left_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    left_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1

            for id_j in compact_lane_graph['right_pairs'][idx_to_id[new_idx_to_idx[new_idx_i]]]:
                if id_to_idx[id_j] in closest_lane_ids:
                    right_road_adj[new_idx_i, idx_to_new_idx[id_to_idx[id_j]]] = 1
        
        return resampled_lanes, pre_road_adj, suc_road_adj, left_road_adj, right_road_adj, num_lanes

    def get_agents_within_fov(
        self,
        agent_states: np.ndarray,
        agent_types: np.ndarray,
        normalize_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Translate agent states into the AV frame and retain only those in view.

        Parameters
        ----------
        agent_states : np.ndarray
            Float32 array of shape ``(N, D)`` where the first 5 columns
            follow Waymo's convention ``[x, y, vx, vy, yaw]`` and the
            remaining columns hold size/existence meta-data.  Coordinates
            are in the *scenario* frame **before** any ego alignment.
        agent_types : np.ndarray
            One-hot encoded array of shape ``(N, 5)`` with indices
            ``{"unset": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3, "other": 4}``.
        normalize_dict : Dict[str, np.ndarray]
            Mapping with keys:
                * ``"center"`` - the ego position ``(x, y)`` used for
                  translation.
                * ``"yaw"`` - the ego heading (radians) used for rotation.

        Returns
        -------
        agent_states_fov : np.ndarray
            Transformed *and* cropped agent state array with shape
            ``(M, D)`` where ``M`` ≤ ``cfg_dataset.max_num_agents``.
        agent_types_fov : np.ndarray
            Corresponding one-hot type matrix with shape ``(M, 5)``.
        """

        center = normalize_dict['center']
        angle_of_rotation = (np.pi / 2) + np.sign(-normalize_dict['yaw']) * np.abs(normalize_dict['yaw'])
        center = center[np.newaxis, np.newaxis, :]

        agent_states[:, :2] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, :2],
                                    translation=center,
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 2:4] = apply_se2_transform(coordinates=agent_states[:, np.newaxis, 2:4],
                                    translation=np.zeros_like(center),
                                    yaw=angle_of_rotation)[:, 0]
        agent_states[:, 4] = rotate_and_normalize_angles(agent_states[:, 4], angle_of_rotation)

        agents_in_fov_x = np.abs(agent_states[:, 0]) < (self.cfg.fov / 2)
        agents_in_fov_y = np.abs(agent_states[:, 1]) < (self.cfg.fov / 2)
        agents_in_fov_mask = agents_in_fov_x * agents_in_fov_y
        valid_agents = np.where(agents_in_fov_mask > 0)[0]
        
        dist_to_origin = np.linalg.norm(agent_states[:, :2], axis=-1)
        # up to max_num_agents agents
        closest_ag_ids = np.argsort(dist_to_origin)[:self.cfg.max_num_agents]
        closest_ag_ids = closest_ag_ids[np.in1d(closest_ag_ids, valid_agents)]

        new_agent_states = agent_states[closest_ag_ids]
        dist_to_origin = np.linalg.norm(new_agent_states[:, :2], axis=-1)

        return agent_states[closest_ag_ids], agent_types[closest_ag_ids]

    def remove_offroad_agents(
        self,
        agent_states: np.ndarray,
        agent_types: np.ndarray,
        lane_dict: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Drop *vehicle* agents whose centres lie off the centerline map.

        Parameters
        ----------
        agent_states : np.ndarray
            Array of shape ``(N, D)`` holding modified agent states (see
            :meth:`modify_agent_states`).  *Row 0* is assumed to be the
            ego vehicle.
        agent_types : np.ndarray
            One-hot matrix of shape ``(N, 5)``.
        lane_dict : Dict[int, np.ndarray]
            Mapping *lane id → (1000 x 2) polyline*

        Returns
        -------
        filtered_states : np.ndarray
            Same layout as ``agent_states`` but with off-road vehicles
            removed; ego is guaranteed to remain row 0.
        filtered_types : np.ndarray
            Corresponding one-hot type matrix.
        """
        
        # keep the ego vehicle always
        non_ego_agent_states = agent_states[1:]
        non_ego_agent_types = agent_types[1:]
        
        road_pts = []
        for lane_id in lane_dict:
            road_pts.append(lane_dict[lane_id])
        road_pts = np.concatenate(road_pts, axis=0)

        agent_road_dist = np.linalg.norm(non_ego_agent_states[:, np.newaxis, :2] - road_pts[np.newaxis, :, :], axis=-1).min(1)
        offroad_mask = agent_road_dist > self.cfg.offroad_threshold
        vehicle_mask = non_ego_agent_types[:, 1].astype(bool)
        offroad_vehicle_mask = offroad_mask * vehicle_mask

        onroad_agents = np.where(~offroad_vehicle_mask)[0]

        filtered_states = np.concatenate([agent_states[:1], non_ego_agent_states[onroad_agents]], axis=0)
        filtered_types = np.concatenate([agent_types[:1], non_ego_agent_types[onroad_agents]], axis=0)

        return filtered_states, filtered_types

