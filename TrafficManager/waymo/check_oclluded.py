import torch
import math
import matplotlib.pyplot as plt

def compute_corners_3d_torch(boxes):
    T, N = boxes.shape[:2]
    device = boxes.device

    corner_template = torch.tensor([
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5]
    ], device=device)  # (8, 3)

    dims = boxes[:, :, 3:6].unsqueeze(2)  # (T, N, 1, 3)
    corners = corner_template[None, None, :, :] * dims  # (T, N, 8, 3)

    yaws = boxes[:, :, 6]
    c, s = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(c), torch.ones_like(c)
    R = torch.stack([
        torch.stack([c, -s, zeros], dim=-1),
        torch.stack([s,  c, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)  # (T, N, 3, 3)

    corners = torch.matmul(corners, R.transpose(2, 3))  # (T, N, 8, 3)
    centers = boxes[:, :, :3].unsqueeze(2)
    return corners + centers  # (T, N, 8, 3)

def project_points(corners, lidar2img):
    T, N = corners.shape[:2]
    ones = torch.ones_like(corners[..., :1])
    homo = torch.cat([corners, ones], dim=-1)  # (T,N,8,4)
    lidar2img = lidar2img.view(T, 1, 1, 4, 4)
    proj = torch.matmul(homo.unsqueeze(-2), lidar2img.transpose(-1, -2))  # (T,N,8,1,4)
    proj = proj.squeeze(-2).squeeze(-2)  # (T,N,8,4)

    depth = proj[..., 2]
    xy = proj[..., :2] / torch.clamp(depth.unsqueeze(-1), min=1e-6)
    return xy, depth  # (T,N,8,2), (T,N,8)

def raster_scene(boxes, lidar2img, image_size):
    T, N = boxes.shape[:2]
    H, W  = image_size
    device = boxes.device

    corners3d = compute_corners_3d_torch(boxes)         # (T,N,8,3)
    corners2d, depths = project_points(corners3d, lidar2img)  # (T,N,8,2), (T,N,8)
    avg_depths = depths.mean(dim=-1)                    # (T,N)

    image_box_ids = torch.full((T, H, W), fill_value=-1, device=device, dtype=torch.long)
    image_depths = torch.full((T, H, W), fill_value=float('inf'), device=device)

    boxes2d = torch.cat([
        corners2d.min(dim=2).values,
        corners2d.max(dim=2).values
    ], dim=-1)  # (T,N,4) — [x1, y1, x2, y2]

    for t in range(T):
        # 🔀 Sort by increasing depth (closest first)
        sort_idx = avg_depths[t].argsort()
        for rank, i in enumerate(sort_idx):
            d = avg_depths[t, i].item()

            if d<0:
                continue

            x1, y1, x2, y2 = boxes2d[t, i]
            x1 = int(torch.clamp(x1, 0, W - 1))
            y1 = int(torch.clamp(y1, 0, H - 1))
            x2 = int(torch.clamp(x2, 0, W - 1))
            y2 = int(torch.clamp(y2, 0, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # patch = image_depths[t, y1:y2+1, x1:x2+1]
            # mask = (patch > d)
            # patch[mask] = d
            # image_depths[t, y1:y2 + 1, x1:x2 + 1]=
            # image_box_ids[t, y1:y2+1, x1:x2+1][mask] = i
            patch = image_depths[t, y1:y2+1, x1:x2+1]
            box_patch = image_box_ids[t, y1:y2+1, x1:x2+1]
            mask = (patch > d)
            patch[mask] = d
            box_patch[mask] = i.item()

            image_depths[t, y1:y2 + 1, x1:x2 + 1]=patch
            image_box_ids[t, y1:y2 + 1, x1:x2 + 1]=box_patch

    #plt.imshow(image_box_ids[0].cpu().numpy())
    #plt.imshow(image_depths[0].cpu().numpy())

    #plt.show()

    return image_box_ids, corners2d


def check_occlusion_fully_batched(boxes, lidar2img, image_size):
    T, N = boxes.shape[:2]
    H, W  = image_size

    image_box_ids, corners2d = raster_scene(boxes, lidar2img, image_size=image_size)
    occluded = torch.ones((T, N), dtype=torch.bool, device=boxes.device)

    for t in range(T):
        for i in range(N):
            for c in range(8):
                x, y = corners2d[t, i, c]
                px = int(torch.clamp(x, 0, W-1).item())
                py = int(torch.clamp(y, 0, H-1).item())
                if image_box_ids[t, py, px] == i:
                    occluded[t, i] = False
                    break

    return ~occluded

# T, N = 1, 3
# boxes = torch.tensor([[
#     [0, 0, 0, 2, 2, 2, 0],        # Box 0
#     [0, 1.5, 0, 2, 2, 2, 0],      # Box 1 in front
#     [0, 0.7, 0, 1, 1, 2, 0]       # Box 2 overlaps with 1, together occlude 0
# ]], dtype=torch.float32).cuda()
#
# lidar2img = torch.tensor([[
#     [700, 0, 640, 0],
#     [0, 700, 360, 0],
#     [0,   0,   1, 0],
#     [0,   0,   0, 1]
# ]], dtype=torch.float32).cuda()
#
# occluded = check_occlusion_fully_batched(boxes, lidar2img, image_size=(1280, 720))
# print(occluded.cpu()) #get tensor([[False,  True,  True]])


#tensor([[ True, False, False]])
