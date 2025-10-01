import torch

def _compute_corners_3d_like_ref(boxes):
    # boxes: (1, N, 7)  [x,y,z,l,w,h,yaw]
    T, N = boxes.shape[:2]
    device = boxes.device
    dtype = boxes.dtype

    template = torch.tensor([
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5,  0.5],
        [-0.5, 0.5,  0.5],
        [-0.5, -0.5,  0.5],
        [0.5, -0.5,  0.5]
    ], device=device, dtype=dtype)  # (8,3)

    dims = boxes[:, :, 3:6].unsqueeze(2)   # (1,N,1,3)
    corners = template[None, None] * dims  # (1,N,8,3)

    yaws = boxes[:, :, 6]                  # (1,N)
    c, s = torch.cos(yaws), torch.sin(yaws)
    zeros = torch.zeros_like(c); ones = torch.ones_like(c)
    R = torch.stack([
        torch.stack([c, -s, zeros], dim=-1),
        torch.stack([s,  c, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)                             # (1,N,3,3)

    corners = corners @ R.transpose(2, 3)  # (1,N,8,3)
    centers = boxes[:, :, :3].unsqueeze(2)
    return corners + centers               # (1,N,8,3)

def _project_points_multi_cam_like_ref(corners, cams):
    """
    corners: (1,N,8,3)
    cams:    (K,4,4) lidar->img
    returns:
      xy:    (K,N,8,2)
      depth: (K,N,8)
    """
    K = cams.shape[0]
    device = corners.device
    dtype = corners.dtype
    T, N = corners.shape[:2]

    ones = torch.ones_like(corners[..., :1])
    homo = torch.cat([corners, ones], dim=-1)   # (1,N,8,4)

    homoK = homo.expand(K, N, 8, 4).contiguous()
    camsT = cams.transpose(-1, -2)              # (K,4,4)

    proj = torch.einsum('knqf,kfF->knqF', homoK, camsT)   # (K,N,8,4)
    depth = proj[..., 2]
    xy = proj[..., :2] / torch.clamp(depth.unsqueeze(-1), min=1e-6)
    return xy, depth

def _raster_scene_multi_cam_like_ref(boxes, cams, image_size, tie_break_first=False):
    """
    Vectorized version that mimics your per-camera loop.
    boxes: (1,N,7)
    cams:  (K,4,4)
    image_size: (H, W)
    Returns:
      image_box_ids: (K,H,W) int64 in [-1..N-1]
      corners2d:     (K,N,8,2)
    """
    assert boxes.shape[0] == 1, "Assume T=1 for now."
    device = boxes.device
    dtype = boxes.dtype
    K = cams.shape[0]
    H, W = image_size
    _, N, _ = boxes.shape

    corners3d = _compute_corners_3d_like_ref(boxes)         # (1,N,8,3)
    corners2d, depths = _project_points_multi_cam_like_ref(corners3d, cams)  # (K,N,8,2),(K,N,8)

    avg_depths = depths.mean(dim=-1)                        # (K,N)

    # AABBs like in your code: floor both ends after clamp
    xy_min = corners2d.min(dim=2).values   # (K,N,2)
    xy_max = corners2d.max(dim=2).values   # (K,N,2)

    # clamp to [0, W-1] / [0, H-1], then floor (int cast in your code)
    x1 = torch.clamp(xy_min[..., 0], 0, W - 1).floor().long()  # (K,N)
    y1 = torch.clamp(xy_min[..., 1], 0, H - 1).floor().long()
    x2 = torch.clamp(xy_max[..., 0], 0, W - 1).floor().long()
    y2 = torch.clamp(xy_max[..., 1], 0, H - 1).floor().long()

    # validity (front of cam & non-empty window)
    valid = (avg_depths >= 0) & (x2 > x1) & (y2 > y1)          # (K,N)

    # Build mask (K,N,H,W) with inclusive bounds (your loop writes y1:y2+1, x1:x2+1)
    yi = torch.arange(H, device=device).view(1, 1, H, 1)
    xi = torch.arange(W, device=device).view(1, 1, 1, W)

    x1e = x1.view(K, N, 1, 1); x2e = x2.view(K, N, 1, 1)
    y1e = y1.view(K, N, 1, 1); y2e = y2.view(K, N, 1, 1)

    in_x = (xi >= x1e) & (xi <= x2e)                           # (K,N,1,W)
    in_y = (yi >= y1e) & (yi <= y2e)                           # (K,N,H,1)
    mask = (in_x & in_y) & valid.view(K, N, 1, 1)              # (K,N,H,W)

    # Depth volume like your "avg_depth per box fill"
    depth_vals = avg_depths.view(K, N, 1, 1).expand(K, N, H, W)
    depth_vol = torch.full_like(depth_vals, float('inf'))
    if tie_break_first:
        # bias earlier (smaller rank) to win ties: subtract tiny eps * rank
        # We emulate your sort asc then draw in that order => first wins on equal depth.
        # Build ranks per camera by sorting avg_depths along N:
        _, draw_order = torch.sort(avg_depths, dim=1)  # (K,N), ascending depth
        rank = torch.empty_like(draw_order, dtype=depth_vals.dtype)
        # assign rank 0..N-1 per camera
        rank.scatter_(1, draw_order, torch.arange(N, device=device, dtype=depth_vals.dtype).unsqueeze(0).expand(K, -1))
        rank = rank.view(K, N, 1, 1).expand_as(depth_vals)
        eps = 1e-6
        depth_vals = depth_vals - eps * rank

    depth_vol = torch.where(mask, depth_vals, depth_vol)
    min_depth, argmin_idx = depth_vol.min(dim=1)               # (K,H,W)
    image_box_ids = argmin_idx.to(torch.long)
    image_box_ids[min_depth.isinf()] = -1

    return image_box_ids, corners2d

def check_occlusion_multi_cam(boxes, cams, image_size, tie_break_first=False):
    """
    boxes: (1,N,7)
    cams:  (K,4,4)
    image_size: (H,W) common to all cameras
    Returns:
      visible_any: (1,N) bool — visible in ANY camera (matches your loop semantics)
    """
    device = boxes.device
    H, W = image_size
    _, N, _ = boxes.shape

    image_box_ids, corners2d = _raster_scene_multi_cam_like_ref(
        boxes, cams, image_size, tie_break_first=tie_break_first
    )  # (K,H,W), (K,N,8,2)

    # sample corner pixels exactly like your code: clamp then int() (== floor on [0..])
    px = torch.clamp(corners2d[..., 0], 0, W - 1).floor().long()  # (K,N,8)
    py = torch.clamp(corners2d[..., 1], 0, H - 1).floor().long()  # (K,N,8)

    # winner id at each corner
    K = image_box_ids.shape[0]
    ids_at = image_box_ids[torch.arange(K, device=device).view(K, 1, 1), py, px]  # (K,N,8)
    idxs = torch.arange(N, device=device).view(1, N, 1)
    vis_per_cam = (ids_at == idxs).any(dim=-1)   # (K,N)
    visible_any = vis_per_cam.any(dim=0, keepdim=True)  # (1,N)
    return visible_any
