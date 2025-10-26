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
    S = _to_np(sampled_pos)     # (A,T,3)
    G = _to_np(gt_pos_raw)      # (A,T,3)
    M = _to_np(gt_valid_raw)    # (A,T)

    S=S[:,2:]
    G=G[:,2:]
    M=M[:,2:]
    P=P[:,:,4::5]

    assert S.ndim == 3 and S.shape[-1] == 3, f"sampled_pos must be (A,T,3), got {S.shape}"
    assert G.ndim == 3 and G.shape[-1] == 3, f"gt_pos_raw must be (A,T,3), got {G.shape}"
    assert M.shape[:2] == S.shape[:2], f"gt_valid_raw must match (A,T), got {M.shape}"

    # Normalize P to (A,K,T,Dp)
    if P.ndim == 4:
        A, K, T, Dp = P.shape
    elif P.ndim == 3:
        K, T, Dp = P.shape
        A = 1
        P = P[None, ...]
    elif P.ndim == 2:
        T, Dp = P.shape
        A, K = 1, 1
        P = P[None, None, ...]
    else:
        raise ValueError(f"Unsupported pred_traj shape {P.shape}")

    if max_rollouts_per_agent is not None:
        K = min(K, max_rollouts_per_agent)
        P = P[:, :K]

    # Ensure 3D for predictions: if 2D, lift Z=0
    if Dp == 2:
        P = np.concatenate([P, np.zeros((*P.shape[:3], 1), dtype=P.dtype)], axis=-1)  # (A,K,T,3)

    # Build masked tokenized & GT with NaNs for invalid
    S_masked = np.stack([_apply_mask_nan(S[a], M[a]) for a in range(S.shape[0])], axis=0)  # (A,T,3)
    G_masked = np.stack([_apply_mask_nan(G[a], M[a]) for a in range(G.shape[0])], axis=0)  # (A,T,3)



    # Global bounds
    all_pred = P.reshape(-1, 3)
    all_tok  = S_masked.reshape(-1, 3)
    all_gt   = G_masked.reshape(-1, 3)
    stack = np.vstack([all_pred, all_tok, all_gt])
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
    first_token_label = True
    for a in range(S_masked.shape[0]):
        ref = S_masked[a]
        ax.plot(ref[:,0], ref[:,1], ref[:,2], lw=lw_ref, alpha=0.95,
                color="tab:orange", label="tokenized" if first_token_label else None)
        first_token_label = False

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

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    rollout_idx = torch.randint(0, K, (A,)) if max_rollouts_per_agent is None else \
                  torch.arange(min(K, max_rollouts_per_agent))

    lines_pred, lines_gt, lines_sampled = [], [], []

    alpha_gt=0.5

    alpha_pred=1

    fps=29.97/5

    # Initialize line objects
    for a in range(A):
        (lp,) = ax.plot([], [], [], color="tab:blue", lw=lw_pred, alpha=alpha_pred)
        (lg,) = ax.plot([], [], [], color="tab:green", lw=lw_ref, alpha=alpha_gt)
        lines_pred.append(lp)
        lines_gt.append(lg)

    # Set axis limits
    all_coords = torch.cat([pred_traj.reshape(-1, 3),
                            gt_pos_raw.reshape(-1, 3),
                            sampled_pos.reshape(-1, 3)], dim=0)
    mins = all_coords.min(0).values.cpu().numpy()
    maxs = all_coords.max(0).values.cpu().numpy()
    for i, label in enumerate(["x", "y", "z"]):
        getattr(ax, f"set_{label}lim")(mins[i], maxs[i])

    # Animation update function
    def update(frame_idx):
        for a in range(A):

            traj_pred = pred_traj[a, rollout_idx[a] if rollout_idx.ndim == 1 else a % len(rollout_idx)].cpu().numpy()
            traj_gt = gt_pos_raw[a].cpu().numpy()

            lines_pred[a].set_data(traj_pred[:frame_idx, 0], traj_pred[:frame_idx, 1])
            lines_pred[a].set_3d_properties(traj_pred[:frame_idx, 2])

            lines_gt[a].set_data(traj_gt[:frame_idx, 0], traj_gt[:frame_idx, 1])
            lines_gt[a].set_3d_properties(traj_gt[:frame_idx, 2])

        ax.set_title(f"{title}\nFrame {frame_idx}/{T}")
        return lines_pred + lines_gt

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=1000/fps, blit=False
    )

    # Save to video
    if save_path:
        ani.save(save_path, writer="ffmpeg", fps=fps)
        print(f"✅ Saved video to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


