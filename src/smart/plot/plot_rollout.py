
from src.utils.vis_waymo import VisWaymo,get_map_features
import torch


COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_CYAN = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_VIOLET = (170, 0, 255)
COLOR_BUTTER = (252, 233, 79)
COLOR_ORANGE = (209, 92, 0)
COLOR_CHOCOLATE = (143, 89, 2)
COLOR_CHAMELEON = (78, 154, 6)
COLOR_SKY_BLUE_0 = (114, 159, 207)
COLOR_SKY_BLUE_1 = (32, 74, 135)
COLOR_PLUM = (92, 53, 102)
COLOR_SCARLET_RED = (164, 0, 0)
COLOR_ALUMINIUM_0 = (238, 238, 236)
COLOR_ALUMINIUM_1 = (211, 215, 207)
COLOR_ALUMINIUM_2 = (66, 62, 64)

lane_style = [
    (COLOR_WHITE, 6),  # FREEWAY = 0
    (COLOR_ALUMINIUM_2, 6),  # SURFACE_STREET = 1
    (COLOR_ORANGE, 6),  # STOP_SIGN = 2
    (COLOR_CHOCOLATE, 6),  # BIKE_LANE = 3
    (COLOR_SKY_BLUE_1, 4),  # TYPE_ROAD_EDGE_BOUNDARY = 4
    (COLOR_PLUM, 4),  # TYPE_ROAD_EDGE_MEDIAN = 5
    (COLOR_BUTTER, 2),  # BROKEN = 6
    (COLOR_MAGENTA, 2),  # SOLID_SINGLE = 7
    (COLOR_SCARLET_RED, 2),  # DOUBLE = 8
    (COLOR_CHAMELEON, 4),  # SPEED_BUMP = 9
    (COLOR_SKY_BLUE_0, 4),  # CROSSWALK = 10
]


# def plot_boxes_and_trajs(tokenized_agent, t_box: int = 1, history_horizon: int = 11, future_horizon: int = 80):
def plot_rollout_frames(
    tokenized_agent,
    scenario_path,
    disc_val,
    pred,
    frames=(30, 50, 70, 90),
    ego_index=None,
    ag_role=None,                 # Optional: [N, R] bool or 0/1
    agent_role_style=None,        # Optional: dict{role_idx: (R,G,B)} using your COLOR_* tuples
    arrow_len=1.5,                # meters
    radius_m=60.0,                # crop around ego
):
    """
    Render map + predicted agent boxes at specific frames in one horizontal figure.
    - Real (GT/history) agents: alpha = 0.5
    - Simulated agents: alpha = 1.0 (except frames f < 11 -> alpha = 0.5)
    - Axes background black, figure background white.
    - Panels centered on ego, only agents within radius_m.

    Assumes the following globals are defined in your module:
      lane_style, COLOR_* constants, get_map_features(...)

    Required keys:
      tokenized_agent:
        - "shape" [N,2]
        - "pred_traj_10hz" [N,Th,2]  (used here as GT/history layer)
        - "pred_head_10hz" [N,Th]
        - "all_valid" [N,Th] (bool)  for the GT/history layer
        - optional "ego_mask" [N] bool
      pred:
        - "pred_traj_10hz" [N,Tp,2]
        - "pred_head_10hz" [N,Tp]
        - optional "all_valid" [N,Tp] (bool)  for the sim layer
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import math
    import torch
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    eps = 1e-6
    scores_all=np.asarray(disc_val, dtype=float)
    # scores_all = np.maximum(eps, np.asarray(disc_val, dtype=float))  # [N, K] or flat
    vmin = np.nanpercentile(scores_all, 5)  # robust low
    vmax = np.nanpercentile(scores_all, 95)  # robust high


    print(vmin, vmax)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("RdYlGn")  # low=red, high=green

    # ---------- load scenario proto ----------
    scenario = scenario_pb2.Scenario()
    for data in tf.data.TFRecordDataset([scenario_path], compression_type=""):
        scenario.ParseFromString(bytes(data.numpy()))
        break

    # ---------- colors ----------
    def rgb01(c255):
        import numpy as _np
        return tuple(_np.array(c255, dtype=float) / 255.0)

    lane_rgba = [rgb01(rgb) for (rgb, _) in lane_style]

    # ---------- map features ----------
    mp_xyz, mp_id, mp_type = get_map_features(scenario.map_features)
    mp_type = np.asarray(mp_type)
    mp_type = np.minimum(mp_type, 9)

    # ---------- helpers ----------
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def oriented_box_corners(center_xy, heading, length, width):
        c, s = math.cos(heading), math.sin(heading)
        hx, hy = length * 0.5, width * 0.5
        local = np.array([[ hx,  hy],
                          [ hx, -hy],
                          [-hx, -hy],
                          [-hx,  hy]], dtype=np.float32)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (local @ R.T) + center_xy[None, :]

    # ---------- tensors ----------
    # History/GT layer (drawn with alpha 0.5 when frame exists)
    real_pos  = to_np(tokenized_agent["pred_traj_10hz"])         # [N, Th, 2]
    real_head = to_np(tokenized_agent["pred_head_10hz"])         # [N, Th]
    real_val  = to_np(tokenized_agent.get("all_valid",
                  np.ones(real_pos.shape[:2], dtype=bool))).astype(bool)

    # Sim layer: concat first 11 history frames + predicted future
    hist_T = min(11, real_pos.shape[1])
    sim_pos  = to_np(torch.cat(
        (torch.as_tensor(real_pos[:, :hist_T]).cuda(), pred["pred_traj_10hz"]), dim=1))  # [N, hist_T+Tp, 2]
    sim_head = to_np(torch.cat(
        (torch.as_tensor(real_head[:, :hist_T]).cuda(), pred["pred_head_10hz"]), dim=1))  # [N, hist_T+Tp]
    sim_val  = to_np(pred.get("all_valid",
                  np.ones(sim_pos.shape[:2], dtype=bool))).astype(bool)
    # make the first hist_T frames valid by default
    if sim_val.shape[1] < sim_pos.shape[1]:
        # pad val with True for the history portion
        right = np.ones((sim_val.shape[0], sim_pos.shape[1] - sim_val.shape[1]), dtype=bool)
        sim_val = np.concatenate([right, sim_val], axis=1)  # (this order won't matter if frames are clamped)

    N, T, _ = sim_pos.shape

    # sizes
    if "shape" in tokenized_agent:
        shp = to_np(tokenized_agent["shape"])[:, :2]
    else:
        shp = np.tile(np.array([4.5, 2.0], dtype=np.float32), (N, 1))


    # ego
    ego_index = torch.where(tokenized_agent["ego_mask"])[0][0].cpu()

    # clamp frames
    frames = [int(max(0, min(f, T - 1))) for f in frames]

    # ---------- figure ----------
    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), sharex=False, sharey=False, constrained_layout=True)
    fig.patch.set_facecolor("white")  # white figure canvas (outside axes)

    if n == 1:
        axes = [axes]

    for ax, f in zip(axes, frames):
        # black axes background
        ax.set_facecolor("black")
        ax.patch.set_visible(True)
        ax.patch.set_alpha(1.0)
        # simplify frame
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])

        f0 = f
        ego_xy = sim_pos[ego_index, f0]

        # ---- map ----
        for i in range(len(mp_xyz)):
            xy = np.asarray(mp_xyz[i])
            idx = int(mp_type[i])
            if idx in [0, 1, 2, 3, 4, 5, 9, 10]:
                _, width = lane_style[idx]
                ax.plot(
                    xy[:, 0], xy[:, 1],
                    color=lane_rgba[idx],
                    linewidth=max(1, width // 2),
                    alpha=1.0,
                    zorder=1,
                )

        # ---- within radius (by sim positions) ----
        mask_sim = sim_val[:, f0]
        if mask_sim.any():
            dxy = sim_pos[mask_sim, f0] - ego_xy[None, :]
            dist = np.hypot(dxy[:, 0], dxy[:, 1])
            local_sel_idx = np.where(mask_sim)[0][dist <= radius_m]
        else:
            local_sel_idx = np.array([], dtype=int)

        # ensure ego included
        if sim_val[ego_index, f0] and ego_index not in local_sel_idx:
            local_sel_idx = np.concatenate([local_sel_idx, [ego_index]])

        # ============================================================
        # 1) REAL LAYER (alpha 0.5)  — only if that frame exists in GT
        # ============================================================
        if f0 < real_pos.shape[1]:
            mask_real = real_val[:, f0]
            real_idx = np.intersect1d(local_sel_idx, np.where(mask_real)[0], assume_unique=False)

            if len(real_idx) > 0:
                r_patches, r_edges, r_faces = [], [], []
                r_arrow_starts, r_arrow_ends = [], []

                for i in real_idx:
                    center = real_pos[i, f0]
                    theta  = float(real_head[i, f0])
                    L, W   = float(shp[i, 0]), float(shp[i, 1])
                    corners = oriented_box_corners(center, theta, L, W)
                    r_patches.append(Polygon(corners, closed=True))
                    # edges light gray on black
                    r_edges.append((0.8, 0.8, 0.8, 1.0))
                    # face color: ego cyan; others aluminum
                    fc =rgb01(COLOR_ALUMINIUM_0) # rgb01(COLOR_CYAN) if i == ego_index else
                    r_faces.append((*fc[:3], 1.0))
                    # arrows (alpha 0.5)
                    r_arrow_starts.append(center)
                    r_arrow_ends.append(center + arrow_len * np.array([math.cos(theta), math.sin(theta)], dtype=float))

                rpc = PatchCollection(
                    r_patches,
                    facecolors=r_faces,
                    #edgecolors=r_edges,
                    linewidths=0.8,
                    alpha=0.5,           # <-- GT boxes at 0.5
                    zorder=4,
                )
                ax.add_collection(rpc)

                # GT arrows at alpha 0.5 (light gray/white)
                for s, e in zip(r_arrow_starts, r_arrow_ends):
                    ax.annotate(
                        "", xy=(e[0], e[1]), xytext=(s[0], s[1]),
                        arrowprops=dict(
                            arrowstyle="-|>", lw=1.8,
                            color="black",  # RGBA white with alpha
                            alpha=0.5,
                            shrinkA=0, shrinkB=0
                        ),
                        zorder=5,
                    )


        if len(local_sel_idx) > 0 and f>10:
            s_patches, s_edges, s_faces = [], [], []
            s_arrow_starts, s_arrow_ends = [], []

            # alpha rule for sim layer
            sim_alpha = 1.0 if f0 >= hist_T else 0.5

            for i in local_sel_idx:
                center = sim_pos[i, f0]
                theta  = float(sim_head[i, f0])
                L, W   = float(shp[i, 0]), float(shp[i, 1])
                corners = oriented_box_corners(center, theta, L, W)
                s_patches.append(Polygon(corners, closed=True))
                # edges white on black
                s_edges.append((1.0, 1.0, 1.0, 1.0))
                # face color: ego cyan; others aluminum
                if i == ego_index:
                    fc = rgb01(COLOR_CYAN)
                else:
                    fc = rgb01(COLOR_RED)
                    idx_score = (f - 10) // 5 - 1

                    s_val = float(disc_val[i][idx_score])
                    r, g, b, _ = cmap(norm(s_val))
                    fc = (r, g, b)

                #s_faces.append((fc[0], fc[1], fc[2], 1.0))

                s_faces.append((*fc[:3], 1.0))

                # heading arrow (sim only, in white)
                s_arrow_starts.append(center)
                s_arrow_ends.append(center + arrow_len * np.array([math.cos(theta), math.sin(theta)], dtype=float))

            spc = PatchCollection(
                s_patches,
                facecolors=s_faces,
                #edgecolors=s_edges,
                linewidths=0.9,
                alpha=sim_alpha,        # <-- 1.0 normally, 0.5 for warm-up frames
                zorder=6,
            )
            ax.add_collection(spc)

            # arrows for sim
            for s, e in zip(s_arrow_starts, s_arrow_ends):
                ax.annotate(
                    "", xy=(e[0], e[1]), xytext=(s[0], s[1]),
                    arrowprops=dict(arrowstyle="-|>", lw=2.0, color="black", shrinkA=0, shrinkB=0),
                    zorder=7,
                )

        # ---- view & title ----
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(sim_pos[ego_index, f][0] - 60, sim_pos[ego_index, f][0] + 60)
        ax.set_ylim(sim_pos[ego_index, f][1] - 60, sim_pos[ego_index, f][1] + 60)
        ax.text(
            0.02, 0.98, f"Frame {f0}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=15, color="white", fontweight="bold", zorder=10,
        )

        # if f==70:
        #     x0 = ego_xy[0] - 45
        #     y0 = ego_xy[1] - 10
        #     rect = Rectangle((x0, y0), 10, 10,
        #                      linewidth=2, edgecolor="red", facecolor="none",
        #                      zorder=9)
        #     ax.add_patch(rect)

    # shared colorbar on the right for all subplots
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for Matplotlib < 3.8
    cbar = fig.colorbar(
        sm, ax=axes, orientation="vertical",
        fraction=0.05, pad=0.02  # tweak to taste
    )
    cbar.set_label("Reward", rotation=90)
   # plt.savefig("comparison_plots_largefont.pdf", format="pdf")

    plt.show()
    return fig
