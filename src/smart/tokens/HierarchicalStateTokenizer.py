import math
import torch
import torch.nn as nn


class HierarchicalStateTokenizer(nn.Module):
    """
    Hierarchical tokenizer where ALL dims (x, y, z, [heading]) use base = self.base.

    - num_levels = 4
    - each level ℓ:
        (ix_ℓ, iy_ℓ, iz_ℓ, ih_ℓ), each ∈ {0,1,2} if base=3
        token_ℓ ∈ {0..base^4-1} (with heading) or {0..base^3-1} (position_only)

    Encode:
        tokens = tok(pos, heading)         # (..., num_levels)  (if position_only=False)
        tokens = tok(pos)                  # (..., num_levels)  (if position_only=True)

    Decode (slice-aware):
        pos, heading = tok.decode_tokens_to_state(tokens[..., :l])
        # l can be 0,1,...,num_levels
    """

    def __init__(
        self,
        # x_range=(-75, 45),
        # y_range=(-20 ,90),
        # z_range=(0  ,60),
        x_range=(-78, 50),
        y_range=(-28, 100),
        z_range=(0, 64),
        h_range=(-math.pi, math.pi),
        num_levels=3,
        base=5,
        position_only: bool = False,
    ):
        super().__init__()

        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        # Still keep heading range (used for dummy heading / wrapping)
        self.h_min, self.h_max = float(h_range[0]), float(h_range[1])

        self.num_levels = num_levels
        self.base = base
        self.position_only = position_only

        # total number of "digits" bins along each axis at finest level
        self.digit_total = base ** num_levels

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

    # ---- encoding / decoding of one-level token ---- #

    @staticmethod
    def encode_digits_xyzh(ix, iy, iz, ih, base):
        """Pack 4 digits (x,y,z,h) into one token."""
        return (((ix * base + iy) * base + iz) * base + ih).long()

    @staticmethod
    def encode_digits_xyz(ix, iy, iz, base):
        """Pack 3 digits (x,y,z) into one token (position-only)."""
        return ((ix * base + iy) * base + iz).long()

    @staticmethod
    def decode_token_xyzh(token, base):
        """Unpack 4 digits (x,y,z,h) from one token."""
        ih = token % base
        token = token // base
        iz = token % base
        token = token // base
        iy = token % base
        token = token // base
        ix = token
        return ix, iy, iz, ih

    @staticmethod
    def decode_token_xyz(token, base):
        """Unpack 3 digits (x,y,z) from one token (position-only)."""
        iz = token % base
        token = token // base
        iy = token % base
        token = token // base
        ix = token
        return ix, iy, iz

    @staticmethod
    def _bins_to_center(idx, vmin, vmax, nbins):
        w = (vmax - vmin) / nbins
        return vmin + (idx.to(torch.float32) + 0.5) * w

    # ---------------- Encode ---------------- #

    def forward(self, pos, heading=None):
        """
        pos:     (..., 3)
        heading: (...) or None (ignored if position_only=True)

        returns:
            tokens: (..., num_levels)  (per-level tokens)
        """
        orig_shape = pos.shape[:-1]

        pos_flat = pos.reshape(-1, 3)

        ix = self._continuous_to_bins(
            pos_flat[:, 0], self.x_min, self.x_max, self.digit_total
        )
        iy = self._continuous_to_bins(
            pos_flat[:, 1], self.y_min, self.y_max, self.digit_total
        )
        iz = self._continuous_to_bins(
            pos_flat[:, 2], self.z_min, self.z_max, self.digit_total
        )

        ix_d = self._to_digits(ix, self.base, self.num_levels)
        iy_d = self._to_digits(iy, self.base, self.num_levels)
        iz_d = self._to_digits(iz, self.base, self.num_levels)

        if not self.position_only:
            if heading is None:
                raise ValueError(
                    "heading must be provided when position_only=False."
                )
            heading_flat = heading.reshape(-1)

            # wrap heading into [h_min, h_max]
            heading_flat = (
                (heading_flat - self.h_min)
                % (self.h_max - self.h_min)
            ) + self.h_min

            ih = self._continuous_to_bins(
                heading_flat, self.h_min, self.h_max, self.digit_total
            )
            ih_d = self._to_digits(ih, self.base, self.num_levels)

            tokens = self.encode_digits_xyzh(ix_d, iy_d, iz_d, ih_d, self.base)
        else:
            # position-only: no heading quantization
            tokens = self.encode_digits_xyz(ix_d, iy_d, iz_d, self.base)

        return tokens.view(*orig_shape, self.num_levels)

    # ---------------- Decode (slice-aware) ---------------- #

    @torch.no_grad()
    def decode_tokens_to_state(self, tokens):
        """
        tokens: (..., l), where l ∈ {0,1,...,num_levels}

        - l = 0 → global center of the whole range
        - l > 0 → decode using first l hierarchical tokens

        Returns:
            pos:     (..., 3)
            heading: (...)   (dummy constant if position_only=True)
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

        ix_list, iy_list, iz_list = [], [], []

        if not self.position_only:
            ih_list = []

            for i in range(l):
                ix, iy, iz, ih = self.decode_token_xyzh(
                    tokens_flat[:, i], self.base
                )
                ix_list.append(ix.unsqueeze(-1))
                iy_list.append(iy.unsqueeze(-1))
                iz_list.append(iz.unsqueeze(-1))
                ih_list.append(ih.unsqueeze(-1))

            ix_d = torch.cat(ix_list, dim=-1)  # [N, l]
            iy_d = torch.cat(iy_list, dim=-1)
            iz_d = torch.cat(iz_list, dim=-1)
            ih_d = torch.cat(ih_list, dim=-1)
        else:
            for i in range(l):
                ix, iy, iz = self.decode_token_xyz(
                    tokens_flat[:, i], self.base
                )
                ix_list.append(ix.unsqueeze(-1))
                iy_list.append(iy.unsqueeze(-1))
                iz_list.append(iz.unsqueeze(-1))

            ix_d = torch.cat(ix_list, dim=-1)  # [N, l]
            iy_d = torch.cat(iy_list, dim=-1)
            iz_d = torch.cat(iz_list, dim=-1)

        nbins = self.base ** l
        powers = torch.tensor(
            [self.base ** (l - 1 - j) for j in range(l)],
            device=device,
            dtype=torch.long,
        )  # [l]

        ix_idx = (ix_d * powers).sum(dim=-1)
        iy_idx = (iy_d * powers).sum(dim=-1)
        iz_idx = (iz_d * powers).sum(dim=-1)

        x = self._bins_to_center(ix_idx, self.x_min, self.x_max, nbins)
        y = self._bins_to_center(iy_idx, self.y_min, self.y_max, nbins)
        z = self._bins_to_center(iz_idx, self.z_min, self.z_max, nbins)

        pos = torch.stack([x, y, z], dim=-1).view(*prefix_shape, 3)

        if not self.position_only:
            ih_idx = (ih_d * powers).sum(dim=-1)
            h = self._bins_to_center(ih_idx, self.h_min, self.h_max, nbins)
            heading = h.view(*prefix_shape)
        else:
            # heading not encoded: just return constant center value
            hc = 0.5 * (self.h_min + self.h_max)
            heading = torch.full(
                prefix_shape,
                hc,
                device=device,
                dtype=torch.float32,
            )

        return pos, heading


# ---------------- Quick sanity check ----------------
if __name__ == "__main__":
    # original behavior (includes heading)
    tok_full = HierarchicalStateTokenizer(position_only=False)

    pos = torch.tensor([[0., 0., 10.], [-50., 80., 30.]])
    heading = torch.tensor([0.1, -2.0])

    tokens_full = tok_full(pos, heading)
    print("tokens (full):", tokens_full)

    for l in range(0, tok_full.num_levels + 1):
        partial = tokens_full[:, :l]
        pos_l, h_l = tok_full.decode_tokens_to_state(partial)
        print(f"\n[full] l={l}, tokens[:, :{l}].shape = {partial.shape}")
        print("pos:\n", pos_l)
        print("heading:\n", h_l)

    # position-only behavior
    tok_pos = HierarchicalStateTokenizer(position_only=True)

    tokens_pos = tok_pos(pos)  # heading omitted
    print("\ntokens (position-only):", tokens_pos)

    for l in range(0, tok_pos.num_levels + 1):
        partial = tokens_pos[:, :l]
        pos_l, h_l = tok_pos.decode_tokens_to_state(partial)
        print(f"\n[pos-only] l={l}, tokens[:, :{l}].shape = {partial.shape}")
        print("pos:\n", pos_l)
        print("heading (dummy):\n", h_l)
