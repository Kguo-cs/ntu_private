
def plot_map(tokenized_agent,tokenized_map,train_mask,near_dist):
    global_edge = tokenized_map["global_edge"]
    import matplotlib as mpl

    mpl.rcParams['toolbar'] = 'None'

    import matplotlib.pyplot as plt
    import numpy as np

    ge = global_edge.cpu().detach().numpy()
    point = tokenized_agent["sampled_pos"].cpu().detach().numpy()[:, 2:]
    # for i in range(len(global_edge)):
    #     plt.plot(global_edge[i, :, 0], global_edge[i, :, 1], 'r')
    for i in range(ge.shape[0]):
        x = ge[i, :, 0]
        y = ge[i, :, 1]
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) < 2: continue

        plt.plot(x, y, 'r', linewidth=0.8)
        mid = len(x) // 2
        plt.annotate(
            '', xy=(x[mid + 1], y[mid + 1]), xytext=(x[mid], y[mid]),
            arrowprops=dict(arrowstyle='->', lw=0.9)
        )

    # plt.show()

    mask = (near_dist < 0)[train_mask]
    mask1 = (near_dist > 0)[train_mask]

    mask = mask.cpu().detach().numpy()
    mask1 = mask1.cpu().detach().numpy()
    from matplotlib.collections import PolyCollection
    import numpy as np

    # --- Get a single timestep arrays (squeeze the singleton time dim if present) ---
    pos_np = tokenized_agent["sampled_pos"].detach().cpu().numpy()[:, 2:][
        train_mask.cpu().detach().numpy()]  # [N,1,2] or [N,2]
    hd_np = tokenized_agent["sampled_heading"].detach().cpu().numpy()[:, 2:][
        train_mask.cpu().detach().numpy()]  # [N,1] or [N]
    shape_np = tokenized_agent["shape"].detach().cpu().numpy()[:, :2][:, None].repeat(16, axis=1)[
        train_mask.cpu().detach().numpy()]  # [N,2] (L,W)

    if pos_np.ndim == 3:  # [N,1,2] -> [N,2]
        pos_np = pos_np[:, 0, :]
    if hd_np.ndim == 2:  # [N,1] -> [N]
        hd_np = hd_np[:, 0]

    # --- Build OBB corners (order: (+,+),(+,-),(-,-),(-,+)) ---
    L = shape_np[:, 0]
    W = shape_np[:, 1]
    hl, hw = 0.5 * L, 0.5 * W

    c = np.cos(hd_np)
    s = np.sin(hd_np)
    u = np.stack([c, s], axis=1)  # forward axis
    v = np.stack([-s, c], axis=1)  # left axis

    offs = np.stack([
        np.stack([hl, hw], axis=1),
        np.stack([hl, -hw], axis=1),
        np.stack([-hl, -hw], axis=1),
        np.stack([-hl, hw], axis=1),
    ], axis=1)  # [N,4,2] in local (u,v) coords

    # world corners: p + off_u*u + off_v*v  -> [N,4,2]
    corners = pos_np[:, None, :] + offs[:, :, :1] * u[:, None, :] + offs[:, :, 1:] * v[:, None, :]

    # --- Colors per box from your masks ---
    mi = mask.ravel()  # inside (near_dist < 0) -> green
    mo = mask1.ravel()  # outside (near_dist > 0) -> blue
    colors = np.full((corners.shape[0],), '0.6', dtype=object)  # default gray if you also have unlabeled points
    colors[mi] = 'b'
    colors[mo] = 'g'

    # colors=colors[mi]
    # corners=corners[mi]

    # --- Add boxes to plot efficiently ---
    ax = plt.gca()
    pc = PolyCollection([corners[i] for i in range(corners.shape[0])],
                        facecolors=colors, edgecolors='k', linewidths=0.6, alpha=0.25)
    ax.add_collection(pc)

    # (optional) also draw the agent center points as you already do
    # plt.scatter(point[mi,0], point[mi,1], s=10, c='g')  # inside
    # plt.scatter(point[mo,0], point[mo,1], s=10, c='b')  # outside

    ax.set_aspect('equal', 'box')
    ax.autoscale_view()
    plt.show()

# plt.scatter(point[mask][:,0], point[mask][:,1],s=10, c='g')
# plt.scatter(point[mask1][:,0], point[mask1][:,1],s=10, c='blue')
#
# for i in range(len(near_dist)):
#     for j in range(len(near_dist[i])):
#         if train_mask[i][j]:
#             if near_dist[i][j]>0:
#                 plt.scatter(point[i][j][0],point[i][j][1],'g')
#             else:
#                 plt.scatter(point[i][j][0],point[i][j][1],'blue')


# # global_edge = tokenized_map["global_edge"]
# import matplotlib.pyplot as plt
#
# #
# # global_edge = global_edge.cpu().detach().numpy()
# # point = tokenized_agent["sampled_pos"].cpu().detach().numpy()[:, 2:]
# # for i in range(len(global_edge)):
# #     plt.plot(global_edge[i, :, 0], global_edge[i, :, 1], 'r')
#
# mask = col_flag[train_mask]
# mask1 = (~col_flag)[train_mask]
#
# print(mask.float().mean())
#
# mask = mask.cpu().detach().numpy()
# mask1 = mask1.cpu().detach().numpy()
# from matplotlib.collections import PolyCollection
# import numpy as np
#
# # --- Get a single timestep arrays (squeeze the singleton time dim if present) ---
# pos_np = tokenized_agent["sampled_pos"].detach().cpu().numpy()[:, 2:][
#     train_mask.cpu().detach().numpy()]  # [N,1,2] or [N,2]
# hd_np = tokenized_agent["sampled_heading"].detach().cpu().numpy()[:, 2:][
#     train_mask.cpu().detach().numpy()]  # [N,1] or [N]
# shape_np = tokenized_agent["shape"].detach().cpu().numpy()[:, :2][:, None].repeat(16, axis=1)[
#     train_mask.cpu().detach().numpy()]  # [N,2] (L,W)
#
# if pos_np.ndim == 3:  # [N,1,2] -> [N,2]
#     pos_np = pos_np[:, 0, :]
# if hd_np.ndim == 2:  # [N,1] -> [N]
#     hd_np = hd_np[:, 0]
#
# # --- Build OBB corners (order: (+,+),(+,-),(-,-),(-,+)) ---
# L = shape_np[:, 0]
# W = shape_np[:, 1]
# hl, hw = 0.5 * L, 0.5 * W
#
# c = np.cos(hd_np)
# s = np.sin(hd_np)
# u = np.stack([c, s], axis=1)  # forward axis
# v = np.stack([-s, c], axis=1)  # left axis
#
# offs = np.stack([
#     np.stack([hl, hw], axis=1),
#     np.stack([hl, -hw], axis=1),
#     np.stack([-hl, -hw], axis=1),
#     np.stack([-hl, hw], axis=1),
# ], axis=1)  # [N,4,2] in local (u,v) coords
#
# # world corners: p + off_u*u + off_v*v  -> [N,4,2]
# corners = pos_np[:, None, :] + offs[:, :, :1] * u[:, None, :] + offs[:, :, 1:] * v[:, None, :]
#
# # --- Colors per box from your masks ---
# mi = mask.ravel()  # inside (near_dist < 0) -> green
# mo = mask1.ravel()  # outside (near_dist > 0) -> blue
# colors = np.full((corners.shape[0],), '0.6', dtype=object)  # default gray if you also have unlabeled points
# colors[mi] = 'g'
# colors[mo] = 'b'
#
# # --- Add boxes to plot efficiently ---
# ax = plt.gca()
# pc = PolyCollection([corners[i] for i in range(corners.shape[0])],
#                     facecolors=colors, edgecolors='k', linewidths=0.6, alpha=0.25)
# ax.add_collection(pc)
#
# # (optional) also draw the agent center points as you already do
# # plt.scatter(point[mi,0], point[mi,1], s=10, c='g')  # inside
# # plt.scatter(point[mo,0], point[mo,1], s=10, c='b')  # outside
#
# ax.set_aspect('equal', 'box')
# ax.autoscale_view()
# plt.show()
