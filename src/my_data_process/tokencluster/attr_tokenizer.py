import torch
import torch.nn as nn
import numpy as np


from src.smart.utils import (
    cal_polygon_contour,
    transform_to_global,
    transform_to_local,
    wrap_angle,
    angle_between_2d_vectors
)



def compute_goal( tokenized_agent):
    sampled_pos = tokenized_agent["sampled_pos"]
    sampled_heading = tokenized_agent["sampled_heading"]
    valid_mask = tokenized_agent["valid_mask"]

    A, T, _ = sampled_pos.shape

    # Convert heading → unit direction (XY)
    dir_xy = torch.stack([torch.cos(sampled_heading), torch.sin(sampled_heading)], dim=-1)  # (A,T,2)

    # Find index of last valid step for each agent
    valid_mask = valid_mask.bool()
    # We want *last* valid, not first
    last_idx = (valid_mask.float() * torch.arange(T, device=sampled_pos.device).float()).max(dim=1).indices

    # Gather last valid pos and heading
    idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)  # shape (A,1,3)
    last_pos = sampled_pos.gather(1, idx).squeeze(1)  # (A,3)
    last_dir = dir_xy.gather(1, idx).squeeze(1)  # (A,2)

    # Sample random extrapolation distances [0, max_extend)
    goal_dist = torch.rand((A,), device=sampled_pos.device) * 50

    goal_dist[np.random.random(A) < 0.5] = 0

    # Compute goal position = last_pos + dist * direction
    goal_pos = last_pos + goal_dist[:, None] * last_dir

    tokenized_agent["goal_pos"] = goal_pos

    batch_idx = tokenized_agent["batch"]

    rand_idx = torch.randint(low=0, high=2, size=(max(batch_idx) + 1, 1), device=batch_idx.device)

    goal_mask = rand_idx[batch_idx] < 1

    goal_mask[np.random.random(len(goal_mask)) < 0.5] = True

    tokenized_agent["goal_mask"] = goal_mask[:, 0]


class Attr_Tokenizer(nn.Module):

    def __init__(self, grid_range, grid_interval, radius, angle_interval):
        super().__init__()
        self.grid_range = grid_range
        self.grid_interval = grid_interval
        self.radius = radius
        self.angle_interval = angle_interval
        self.heading = 0#torch.pi / 2
        self._prepare_grid()

        self.grid_size = self.grid.shape[0]
        self.angle_size = int(360. / self.angle_interval)

        assert torch.all(self.grid[self.grid_size // 2] == 0.)

    def _prepare_grid(self):
        num_grid = int(self.grid_range / self.grid_interval) + 1 # Do not use '//'     #-75,75 step interval is 3m ,within distance 75 m, the left grid is  [1961, 2] shape

        x = torch.linspace(0, num_grid - 1, steps=num_grid)
        y = torch.linspace(0, num_grid - 1, steps=num_grid)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (n^2, 2)
        grid = grid.reshape(num_grid, num_grid, 2).flip(dims=[0]).reshape(-1, 2)
        grid = (grid - x.shape[0] // 2) * self.grid_interval

        distance = (grid ** 2).sum(-1).sqrt()
        square_mask = ((distance <= self.radius) & (distance >= 0.)) | (distance == 0.)
        self.register_buffer('grid', grid[square_mask])
        self.register_buffer('dist', torch.norm(self.grid, p=2, dim=-1))
        head_vector = torch.stack([torch.tensor(self.heading).cos(), torch.tensor(self.heading).sin()])
        self.register_buffer('dir', angle_between_2d_vectors(ctr_vector=head_vector.unsqueeze(0),
                                            nbr_vector=self.grid))  # (-pi, pi]

        self.num_grid = num_grid
        self.square_mask = square_mask.numpy()

    def _apply_rot(self, x, theta):
        # x: (b, l, 2) e.g. (num_step, num_agent, 2)
        # theta: (b,) e.g. (num_step,)
        cos, sin = theta.cos(), theta.sin()
        rot_mat = torch.zeros((theta.shape[0], 2, 2), device=theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        x = torch.bmm(x, rot_mat)
        return x

    def pad_square(self, prob, indices=None):
        # square_mask: bool array of shape (n^2,)
        # prob: float array of shape (num_step, m)
        pad_prob = np.zeros((*prob.shape[:-1], self.square_mask.shape[0]))
        pad_prob[..., self.square_mask] = prob

        square_indices = np.arange(self.square_mask.shape[0])
        circle_indices = np.concatenate([square_indices[self.square_mask], [-1]])
        if indices is not None:
            indices = circle_indices[indices]

        return pad_prob, indices

    def get_grid(self, x, theta=None):
        x = x.reshape(-1, 2)
        grid = self.grid[None, ...].to(x.device)
        if theta is not None:
            grid = self._apply_rot(grid, (theta - self.heading).expand(x.shape[0]))
        return x[:, None] + grid

    def encode_pos(self, x, y, theta_y=None):
        # assert x.dim() == y.dim() and x.shape[-1] == 2 and y.shape[-1] == 2, \
        #             f"Invalid input shape x: {x.shape}, y: {y.shape}."
        centered_x = x - y
        if theta_y is not None:
            centered_x = self._apply_rot(centered_x[:, None], -(theta_y - self.heading).expand(x.shape[0]))[:, 0]
        distance = ((centered_x[:, None] - self.grid.to(x.device)[None, ...]) ** 2).sum(-1).sqrt()
        index = torch.argmin(distance, dim=-1)

        grid_xy = self.grid[index]
        offset_xy = centered_x - grid_xy

        return index.long(), offset_xy

    def decode_pos(self, index, offset_xy=None,y=None, theta_y=None):
        assert torch.all((index >= 0) & (index < self.grid_size))
        centered_x = self.grid.to(index.device)[index.long()]
        if y is not None:
            if offset_xy is not None:
                centered_x=offset_xy+centered_x
            if theta_y is not None:
                centered_x = self._apply_rot(centered_x[:, None], (theta_y - self.heading).expand(centered_x.shape[0]))[:, 0]
            x = centered_x + y
            return x.float()
        return centered_x.float()

    def encode_heading(self, heading):
        heading = (wrap_angle(heading) + torch.pi) / (2 * torch.pi) * 360
        index = heading // self.angle_interval
        return index.long()

    def decode_heading(self, index):
        assert torch.all(index >= 0) and torch.all(index < (360 / self.angle_interval))
        angles = index * self.angle_interval - 180
        angles = angles / 360 * (2 * torch.pi)
        return angles.float()




    def batched_nn_chain(
            self,
            initial_pos,  # (N, D)
            batch,  # (N,)
            type,  # (N,)
            ego_mask,  # (N,) bool
            batch_num,
            num_types=3,
    ):
        lengths = torch.bincount(batch,minlength=batch_num)

        padding_pos_a = padding(initial_pos, lengths.tolist(), padding_value=torch.inf)  # b, n, d

        type_b = padding(type.to(torch.float32), lengths.tolist(), padding_value=torch.inf)  # b, n, d

        mask=padding_pos_a[:,:,0]!=torch.inf

        sort_idx = torch.zeros_like(padding_pos_a[:,:,0]).to(torch.int64)-1

        last_pos = initial_pos[ego_mask]

        last_type=type[ego_mask]

        for i in range(padding_pos_a.shape[1]):

            same_type=torch.abs(type_b-last_type[:,None])

            dist=torch.linalg.norm(padding_pos_a-last_pos[:,None],dim=-1)

            nearest_last=(same_type*1000+ dist).argmin(-1)

            sort_idx[:,i] =nearest_last

            last_pos = padding_pos_a[torch.arange(batch_num),nearest_last]
            last_type = type_b[torch.arange(batch_num),nearest_last]

            padding_pos_a[torch.arange(batch_num),nearest_last]=torch.inf

        final_idx=sort_idx[mask]

        offsets = torch.cumsum(lengths, 0) - lengths

        final_idx=final_idx+offsets[batch]

        return final_idx


    def chained_sort(self,initial_pos, batch, type, ego_mask, type_order=(0, 1, 2)):
        device = initial_pos.device
        N = initial_pos.size(0)
        sort_idx_all = []

        for b in batch.unique(sorted=True):
            batch_mask = batch == b
            batch_idx = torch.where(batch_mask)[0]

            # ego index
            ego_idx = batch_idx[ego_mask[batch_mask]][0]
            last_pos = initial_pos[ego_idx]

            sort_idx_all.append(ego_idx)

            for t in type_order:
                type_mask = batch_mask & (type == t)
                candidates = torch.where(type_mask)[0]

                # remove ego if ego is car
                candidates = candidates[candidates != ego_idx]

                if candidates.numel() == 0:
                    continue

                remaining = candidates.clone()

                while remaining.numel() > 0:
                    d = torch.norm(initial_pos[remaining] - last_pos, dim=-1)
                    nn = torch.argmin(d)
                    chosen = remaining[nn]

                    sort_idx_all.append(chosen)
                    last_pos = initial_pos[chosen]

                    remaining = torch.cat([remaining[:nn], remaining[nn + 1:]])

        return torch.stack(sort_idx_all)