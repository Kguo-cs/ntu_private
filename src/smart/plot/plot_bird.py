import numpy as np
import torch

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _apply_mask_nan(traj_xyz, valid):
    """
    traj_xyz: (T,3), valid: (T,) bool
    Returns copy where invalid steps are NaN so lines break.
    """
    out = traj_xyz.copy()
    m = valid.astype(bool)
    out[~m] = np.nan
    return out

def plot_bird_from_tensors(pred_traj, sampled_pos, gt_pos_raw, gt_valid_raw,
                          title="Predicted vs Tokenized vs GT (3D)",
                          max_rollouts_per_agent=None, alpha_pred=0.35,
                          lw_pred=0.8, lw_ref=2.0, show=True, save_path=None):
    """
    pred_traj:  (A,K,T,2/3) or (K,T,2/3) or (T,2/3)
    sampled_pos:(A,T,3)   tokenized
    gt_pos_raw: (A,T,3)   ground truth
    gt_valid_raw:(A,T)    bool mask (applied to tokenized & GT)
    """
    P = _to_np(pred_traj)
    #S = _to_np(sampled_pos)     # (A,T,3)
    G = _to_np(gt_pos_raw)      # (A,T,3)
    M = _to_np(gt_valid_raw)    # (A,T)

    #S=S[:,2:]
    # G=G[:,2:]
    # M=M[:,2:]
    # P=P[:,:,:,2:]

    # assert S.ndim == 3 and S.shape[-1] == 3, f"sampled_pos must be (A,T,3), got {S.shape}"
    # assert G.ndim == 3 and G.shape[-1] == 3, f"gt_pos_raw must be (A,T,3), got {G.shape}"
    # assert M.shape[:2] == S.shape[:2], f"gt_valid_raw must match (A,T), got {M.shape}"

    # Normalize P to (A,K,T,Dp)
    A, K, T, Dp = P.shape

    if max_rollouts_per_agent is not None:
        K = min(K, max_rollouts_per_agent)
        P = P[:, :K]


    # Build masked tokenized & GT with NaNs for invalid
#    S_masked = np.stack([_apply_mask_nan(S[a], M[a]) for a in range(S.shape[0])], axis=0)  # (A,T,3)
    G_masked = np.stack([_apply_mask_nan(G[a], M[a]) for a in range(G.shape[0])], axis=0)  # (A,T,3)



    all_gt   = G_masked.reshape(-1, 3)
    stack = all_gt
    # ignore NaNs
    mins = np.nanmin(stack, axis=0)
    maxs = np.nanmax(stack, axis=0)
    pads = 0.03 * np.maximum(1e-6, maxs - mins)

    import matplotlib.pyplot as plt

    # Plot
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection='3d')

    nan_mask= (P==0).all(axis=-1)

    nan_mask=nan_mask | ~M[:,None]

    P[nan_mask] = np.nan


    # print(np.linalg.norm(P[:,0]-np.array([30,-40,20])[None,None],axis=-1))#30,-40,20


    # Predicted rollouts (ALL agents × rollouts)

    first_rollout_label = True

    for a in range(A):
       for k in range(P.shape[1]):
            traj = P[a, k]  # (T, 3)

            # Plot trajectory
            ax.plot(
                traj[:, 0], traj[:, 1], traj[:, 2],
                alpha=alpha_pred, lw=lw_pred, color="tab:blue",
                label="rollout" if first_rollout_label else None
            )
            first_rollout_label = False
    # #valid = ~nan_mask[a,k]
    #
    # # Skip if no valid positions
    # if not valid.any():
    #     continue

    #
        # # --- show agent ID at its first valid position --- #
        # first_idx = np.where(valid)[0][0].item()
        # x0, y0, z0 = traj[first_idx].tolist()
        #
        # ax.text(
        #     x0, y0, z0,
        #     f"{int(a)}",  # display agent ID
        #     fontsize=6,
        #     color="black",
        #     ha="center",
        #     va="center",
        #     backgroundcolor="white",
        # )

    # Tokenized reference (masked)
    # first_token_label = True
    # for a in range(S_masked.shape[0]):
    #     ref = S_masked[a]
    #     ax.plot(ref[:,0], ref[:,1], ref[:,2], lw=lw_ref, alpha=0.95,
    #             color="tab:orange", label="tokenized" if first_token_label else None)
    #     first_token_label = False

    # Ground truth (masked, dashed)
    first_gt_label = True
    for a in range(G_masked.shape[0]):
        gt = G_masked[a]
        ax.plot(gt[:,0], gt[:,1], gt[:,2], lw=lw_ref, ls="--", alpha=0.95,
                color="tab:green", label="ground truth" if first_gt_label else None)
        first_gt_label = False

    ax.set_xlim(mins[0]-pads[0], maxs[0]+pads[0])
    ax.set_ylim(mins[1]-pads[1], maxs[1]+pads[1])
    ax.set_zlim(mins[2]-pads[2], maxs[2]+pads[2])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)

    import matplotlib.animation as animation

    fig = plt.figure(figsize=(12, 6))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pred = fig.add_subplot(1, 2, 2, projection='3d')

    ax_gt.set_title("Ground Truth")
    ax_gt.set_xlabel("X");
    ax_gt.set_ylabel("Y");
    ax_gt.set_zlabel("Z")

    ax_pred.set_title("Prediction")
    ax_pred.set_xlabel("X");
    ax_pred.set_ylabel("Y");
    ax_pred.set_zlabel("Z")

    fps = 29.97

    # --- compute shared axis limits from all coords (safe conversion) --- #
    all_coords = gt_pos_raw.reshape(-1, 3)
    mins = all_coords.min(0).values.cpu().numpy()
    maxs = all_coords.max(0).values.cpu().numpy()
    for i, label in enumerate(["x", "y", "z"]):
        getattr(ax_gt, f"set_{label}lim")(mins[i], maxs[i])
        getattr(ax_pred, f"set_{label}lim")(mins[i], maxs[i])

    # --- pre-convert tensors to numpy arrays (faster in update) --- #
    P_np = P # shape (A, K, T, 3) or (A, T, 3) depending on your variable

    G_np = G_masked  # (A, T, 3)

    A = P_np.shape[0]
    T = P_np.shape[2]

    # --- create line artists: one per agent per axis --- #
    lines_gt = []
    lines_pred = []
    alpha_gt = 0.5
    alpha_pred = 1.0

    for a in range(A):
        (lg,) = ax_gt.plot([], [], [], color="tab:green", lw=lw_ref, alpha=alpha_gt)
        (lp,) = ax_pred.plot([], [], [], color="tab:blue", lw=lw_pred, alpha=alpha_pred)
        lines_gt.append(lg)
        lines_pred.append(lp)

    # --- robust helper to set 3D line from finite points only --- #
    def set_3d_line_from_traj(line_artist, traj_slice):
        """
        traj_slice: (N,3) numpy array (may contain NaN)
        if there are any finite rows, set the line to those points (in order).
        otherwise clear the line.
        """
        if traj_slice.size == 0:
            line_artist.set_data([], [])
            line_artist.set_3d_properties([])
            return

        finite_mask = np.isfinite(traj_slice).all(axis=1)
        if finite_mask.any():
            pts = traj_slice[finite_mask]
            line_artist.set_data(pts[:, 0], pts[:, 1])
            line_artist.set_3d_properties(pts[:, 2])
        else:
            line_artist.set_data([], [])
            line_artist.set_3d_properties([])

    # --- update function updates both subplots each frame --- #
    def update(frame_idx):
        artists = []
        for a in range(A):
            traj_pred = P_np[a, 0]  # choose rollout 0 (shape (T,3))
            traj_gt = G_np[a]

            # slice up to and including current frame
            endp = min(frame_idx + 1, traj_pred.shape[0])
            endg = min(frame_idx + 1, traj_gt.shape[0])

            set_3d_line_from_traj(lines_pred[a], traj_pred[max(endp-10,0):endp])
            set_3d_line_from_traj(lines_gt[a], traj_gt[max(endg-10,0):endg])

            artists.append(lines_pred[a])
            artists.append(lines_gt[a])

        ax_gt.set_title(f"Ground Truth\nFrame {frame_idx}/{T - 1}")
        ax_pred.set_title(f"Prediction\nFrame {frame_idx}/{T - 1}")

        # return list of artists to animate
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000/fps, blit=False
    )

    # Save to video
    if save_path:
        ani.save(save_path.with_suffix(".mp4"), writer="ffmpeg", fps=fps)
        print(f"✅ Saved video to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


