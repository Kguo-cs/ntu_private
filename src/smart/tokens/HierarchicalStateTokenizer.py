import math
import torch
import torch.nn as nn


class HierarchicalStateTokenizer(nn.Module):
    """
    Hierarchical tokenizer where ALL dims (x, y, z, heading) use base = 3.

    - num_levels = 4
    - each level ℓ:
        (ix_ℓ, iy_ℓ, iz_ℓ, ih_ℓ), each ∈ {0,1,2}
        token_ℓ ∈ {0..80}   because 3*3*3*3 = 81

    Encode:
        tokens = tok(pos, heading)         # (..., 4)

    Decode (slice-aware):
        pos, heading = tok.decode_tokens_to_state(tokens[:, :l])
        # l can be 0,1,2,3,4
    """

    def __init__(
        self,
        x_range=(-75, 45),
        y_range=(-20 ,90),
        z_range=(0  ,60),
        h_range=(-math.pi, math.pi),
        num_levels=3,
        base=5,
    ):
        super().__init__()

        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.h_min, self.h_max = float(h_range[0]), float(h_range[1])

        self.num_levels = num_levels
        self.base = base
        self.n_total = base ** 4     # = 81 bins per dim at finest level

        self.digit_total=base ** num_levels

    # ---------------- Util functions ---------------- #

    @staticmethod
    def _continuous_to_bins(x, xmin, xmax, nbins):
        eps = 1e-6
        xn = (x - xmin) / (xmax - xmin + eps)
        xn = xn.clamp(0, 1 - eps)
        return (xn * nbins).floor().long()

    @staticmethod
    def _to_digits(idx, base, L):
        """
        idx → digits[L], coarse→fine
        """
        out = []
        tmp = idx
        for _ in range(L):
            out.append(tmp % base)
            tmp //= base
        out = torch.stack(out, dim=-1)
        return torch.flip(out, dims=[-1])   # coarse → fine

    @staticmethod
    def encode_digits(ix, iy, iz, ih, base):
        return (((ix * base + iy) * base + iz) * base + ih).long()

    @staticmethod
    def decode_token(token, base):
        ih = token % base
        token =token// base
        iz = token % base
        token =token //base
        iy = token % base
        token= token//base
        ix = token
        return ix, iy, iz, ih

    @staticmethod
    def _bins_to_center(idx, vmin, vmax, nbins):
        w = (vmax - vmin) / nbins
        return vmin + (idx.to(torch.float32) + 0.5) * w

    # ---------------- Encode ---------------- #

    def forward(self, pos, heading):
        """
        pos:     (..., 3)
        heading: (...)

        returns:
            tokens: (..., 4)  (per-level tokens)
        """
        orig_shape = pos.shape[:-1]

        pos_flat = pos.reshape(-1, 3)
        heading_flat = heading.reshape(-1)

        # wrap heading into [h_min, h_max]
        heading_flat = ((heading_flat - self.h_min) % (self.h_max - self.h_min)) + self.h_min

        ix = self._continuous_to_bins(pos_flat[:, 0], self.x_min, self.x_max, self.digit_total)
        iy = self._continuous_to_bins(pos_flat[:, 1], self.y_min, self.y_max, self.digit_total)
        iz = self._continuous_to_bins(pos_flat[:, 2], self.z_min, self.z_max, self.digit_total)
        ih = self._continuous_to_bins(heading_flat, self.h_min, self.h_max, self.digit_total)

        ix_d = self._to_digits(ix, self.base, self.num_levels)
        iy_d = self._to_digits(iy, self.base, self.num_levels)
        iz_d = self._to_digits(iz, self.base, self.num_levels)
        ih_d = self._to_digits(ih, self.base, self.num_levels)

        tokens = self.encode_digits(ix_d, iy_d, iz_d, ih_d, self.base)
        return tokens.view(*orig_shape, self.num_levels)

    # ---------------- Decode (slice-aware) ---------------- #

    @torch.no_grad()
    def decode_tokens_to_state(self, tokens):
        """
        tokens: (..., l), where l ∈ {0,1,2,3,4}

        - l = 0 → global center of the whole range
        - l > 0 → decode using first l hierarchical tokens

        Returns:
            pos:     (..., 3)
            heading: (...)
        """
        prefix_shape = tokens.shape[:-1]
        l = tokens.shape[-1]

        # l = 0: no tokens → just global center of the range
        if l == 0:
            xc = 0.5 * (self.x_min + self.x_max)
            yc = 0.5 * (self.y_min + self.y_max)
            zc = 0.5 * (self.z_min + self.z_max)
            hc = 0.5 * (self.h_min + self.h_max)

            center_xyz = torch.tensor(
                [xc, yc, zc],
                device=tokens.device,
                dtype=torch.float32,
            )
            pos = center_xyz.expand(*prefix_shape, 3)
            heading = torch.full(
                prefix_shape,
                hc,
                device=tokens.device,
                dtype=torch.float32,
            )
            return pos, heading

        # l > 0: use first l levels
        tokens_flat = tokens.reshape(-1, l)
        device = tokens.device

        ix_list, iy_list, iz_list, ih_list = [], [], [], []
        for i in range(l):
            ix, iy, iz, ih = self.decode_token(tokens_flat[:, i], self.base)
            ix_list.append(ix.unsqueeze(-1))
            iy_list.append(iy.unsqueeze(-1))
            iz_list.append(iz.unsqueeze(-1))
            ih_list.append(ih.unsqueeze(-1))

        ix_d = torch.cat(ix_list, dim=-1)   # [N, l]
        iy_d = torch.cat(iy_list, dim=-1)
        iz_d = torch.cat(iz_list, dim=-1)
        ih_d = torch.cat(ih_list, dim=-1)

        nbins = self.base ** l
        powers = torch.tensor(
            [self.base ** (l - 1 - j) for j in range(l)],
            device=device,
            dtype=torch.long,
        )  # [l]125

        ix_idx = (ix_d * powers).sum(dim=-1)
        iy_idx = (iy_d * powers).sum(dim=-1)
        iz_idx = (iz_d * powers).sum(dim=-1)
        ih_idx = (ih_d * powers).sum(dim=-1)

        x = self._bins_to_center(ix_idx, self.x_min, self.x_max, nbins)
        y = self._bins_to_center(iy_idx, self.y_min, self.y_max, nbins)
        z = self._bins_to_center(iz_idx, self.z_min, self.z_max, nbins)
        h = self._bins_to_center(ih_idx, self.h_min, self.h_max, nbins)

        pos = torch.stack([x, y, z], dim=-1).view(*prefix_shape, 3)
        heading = h.view(*prefix_shape)
        return pos, heading


# ---------------- Quick sanity check ----------------
if __name__ == "__main__":
    tok = HierarchicalStateTokenizer()

    pos = torch.tensor([[0., 0., 10.], [-50., 80., 30.]])
    heading = torch.tensor([0.1, -2.0])

    tokens = tok(pos, heading)
    print("tokens:", tokens)


    for l in range(5):
        partial = tokens[:, :l]
        pos_l, h_l = tok.decode_tokens_to_state(partial)
        print(f"\nl={l}, tokens[:, :{l}].shape = {partial.shape}")
        print("pos:\n", pos_l)
        print("heading:\n", h_l)


# tokens: tensor([[28, 67, 49, 47],
#         [21, 40, 64, 77]])
