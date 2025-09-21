
from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors
)





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


def plot_rollout(tokenized_agent,tokenized_map,token_processor,pred):
    # global_edge = tokenized_map["global_edge"]
    # import matplotlib as mpl
    #
    # mpl.rcParams['toolbar'] = 'None'

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import math

    position=tokenized_map["position"]
    token_idx=tokenized_map['token_idx']
    orientation=tokenized_map['orientation']
    map_type=tokenized_map['type']
    map_type[map_type > 9] = 9

    local_traj = token_processor.map_token_traj_src[token_idx]

    global_edge, _ = transform_to_global(pos_local=local_traj.reshape(-1, 11, 2), head_local=None, pos_now=position,
                                         head_now=orientation)

    ge = global_edge.cpu().detach().numpy()
    mt = map_type.detach().cpu().numpy() if hasattr(map_type, "detach") else np.asarray(map_type)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Collect starting points of each line
    starts = ge[:, 0, :2]  # shape [num_lines, 2]

    # # For each line, connect to neighbors within 20 m
    # thresh = 20.0
    # for i in range(len(starts)):
    #     for j in range(i + 1, len(starts)):
    #         dx, dy = starts[i] - starts[j]
    #         dist = np.hypot(dx, dy)
    #         if dist < thresh:
    #             ax.plot(
    #                 [starts[i, 0], starts[j, 0]],
    #                 [starts[i, 1], starts[j, 1]],
    #                 linestyle="--",
    #                 color="gray",
    #                 linewidth=0.8,
    #                 alpha=0.6,
    #                 zorder=0.5,  # behind lane lines
    #             )

    for i in range(ge.shape[0]):
        x = ge[i, :, 0]
        y = ge[i, :, 1]

        idx = int(mt[i])

        color_255, width = lane_style[idx]
        color = tuple(np.array(color_255) / 255.0)

        ax.plot(x, y, color=color, linewidth=2,alpha=0.5,zorder=1)

        #mid = len(x) // 2
        # plt.annotate(
        #     '', xy=(x[mid + 1], y[mid + 1]), xytext=(x[mid], y[mid]),
        #     arrowprops=dict(arrowstyle='->', lw=0.9)
        # )
    # #plot_boxes_and_trajs(tokenized_agent, t_box=11, history_horizon=11, future_horizon=80)

    #plt.show()

    t_box = 1
    history_horizon = 2#*5
    future_horizon = 16
    #
    def to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    #
    # pos = to_np(tokenized_agent["pred_traj_10hz"])        # [N, T, 2]
    # head = to_np(tokenized_agent["pred_head_10hz"])   # [N, T] radians
    # valid = to_np(tokenized_agent["all_valid"]).astype(bool)  # [N, T]
    pos = to_np(tokenized_agent["sampled_pos"])        # [N, T, 2]
    head = to_np(tokenized_agent["sampled_heading"])   # [N, T] radians
    valid = to_np(tokenized_agent["valid_mask"]).astype(bool)  # [N, T]
    #
    #
    N, T, _ = pos.shape
    t_box = max(0, min(t_box, T - 1))

    # Vehicle size
    if "shape" in tokenized_agent:
        shp = to_np(tokenized_agent["shape"])[:, :2]
    else:
        shp = np.tile(np.array([4.5, 2.0]), (N, 1))

    # Helpers
    def oriented_box_corners(center_xy, heading, length, width):
        c, s = math.cos(heading), math.sin(heading)
        hx, hy = length * 0.5, width * 0.5
        local = np.array([[ hx,  hy],
                          [ hx, -hy],
                          [-hx, -hy],
                          [-hx,  hy]], dtype=np.float32)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return (local @ R.T) + center_xy[None, :]

    t_now = t_box  # or any timestep you want (e.g., 1)
    thresh = 60.0

    # Get valid agent positions at t_now
    mask = valid[:, t_now]
    curr_pos = pos[mask, t_now]  # [M, 2] where M <= N
    #
    # M = len(curr_pos)
    # for i in range(M):
    #     for j in range(i + 1, M):
    #         dx, dy = curr_pos[i] - curr_pos[j]
    #         dist = np.hypot(dx, dy)
    #         if dist < thresh:
    #             ax.plot(
    #                 [curr_pos[i, 0], curr_pos[j, 0]],
    #                 [curr_pos[i, 1], curr_pos[j, 1]],
    #                 linestyle="--",
    #                 color="grey",
    #                 linewidth=0.8,
    #                 alpha=0.6,
    #                 zorder=2,
    #             )

    # --- History points (red→blue) ---
    # hist_steps = min(history_horizon, T)
    # cmap_hist = plt.get_cmap("coolwarm")  # red→blue
    # for t in range(hist_steps):
    #     mask_t = valid[:, t]
    #     pts = pos[mask_t, t]
    #     if len(pts) > 0:
    #         u = t / max(1, hist_steps - 1)
    #         ax.scatter(pts[:, 0], pts[:, 1], s=16, color=cmap_hist(u), alpha=0.9,
    #                    label="history" if t == 0 else None)
    #
    # # --- Future points (blue→cyan) ---

    #--- Boxes at t_box ---
    # patches = []
    # for i in range(N):
    #     if not valid[i, t_box]:
    #         continue
    #     center = pos[i, t_box]
    #     theta = float(head[i, t_box])
    #     L, W = float(shp[i, 0]), float(shp[i, 1])
    #     corners = oriented_box_corners(center, theta, L, W)
    #     patches.append(Polygon(corners, closed=True))
    #
    # if patches:
    #     pc = PatchCollection(patches, facecolors=(0.5, 0.5, 0.5, 1.0),
    #                          edgecolors=(0, 0, 0, 0.9), linewidths=0.8)
    #     ax.add_collection(pc)

    # --- History boxes (color = #ffe6cc) ---
    hist_steps = min(history_horizon, T)
    # history_color = (255 / 255, 230 / 255, 204 / 255, 1.0)  # rgba for #ffe6cc
    history_color = (215 / 255, 155 / 255, 0 / 255, 1.0)  # rgba for #d79b00

    for t in range(1,hist_steps):
        mask_t = valid[:, t]
        if not np.any(mask_t):
            continue

        patches = []
        for i in range(N):
            if not mask_t[i]:
                continue
            center = pos[i, t]
            theta = float(head[i, t])
            L, W = float(shp[i, 0]), float(shp[i, 1])
            corners = oriented_box_corners(center, theta, L, W)
            patches.append(Polygon(corners, closed=True))

        if patches:
            pc = PatchCollection(
                patches,
                facecolors=history_color,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.5+t*0.5,
                zorder=4,
            )
            ax.add_collection(pc)
    # --- Future boxes (blue→cyan) ---
    if history_horizon < T:
        future_steps = min(future_horizon, T - history_horizon)
        cmap_future = plt.get_cmap("winter")  # dark blue→cyan
        for k in range(0,future_steps,2):
            t = history_horizon + k
            mask_t = valid[:, t]
            if not np.any(mask_t):
                continue
            u = k / max(1, future_steps - 1)
            color = cmap_future(u)  # RGBA

            patches = []
            for i in range(N):
                if not mask_t[i]:
                    continue
                center = pos[i, t]
                theta = float(head[i, t])
                L, W = float(shp[i, 0]), float(shp[i, 1])
                corners = oriented_box_corners(center, theta, L, W)
                patches.append(Polygon(corners, closed=True))

            if patches:
                pc = PatchCollection(
                    patches,
                    facecolors=(color[0], color[1], color[2], 0.3),  # translucent fill
                    edgecolors=color,
                    linewidths=0.6,
                   # zorder=5,
                )
                ax.add_collection(pc)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    #handles, labels = ax.get_legend_handles_labels()
    # if labels:
    #     seen, new_h, new_l = set(), [], []
    #     for h, l in zip(handles, labels):
    #         if l not in seen:
    #             new_h.append(h); new_l.append(l); seen.add(l)
    #     ax.legend(new_h, new_l, loc="upper right", frameon=False)
    plt.tight_layout()


    valid_pos=pos[valid]

    print( max(valid_pos[:, 0])-min(valid_pos[:, 0]))

    print( max(valid_pos[:, 1])-min(valid_pos[:, 1]))

    #plt.xlim(min(valid_pos[:, 0])+44.5, max(valid_pos[:, 0])-45)
    #plt.ylim(min(valid_pos[:, 1])-30, max(valid_pos[:, 1])+30)
    plt.show()

    # pred_traj_10hz=tokenized_agent["pred_traj_10hz"]
    # pred_head_10hz=tokenized_agent["pred_head_10hz"]
    # all_valid=tokenized_agent["pred_head_10hz"]


    # plt.show()

