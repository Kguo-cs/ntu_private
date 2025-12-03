import math
import torch
import torch.nn as nn


class HierarchicalStateTokenizer(nn.Module):
    """
    Hierarchical tokenizer where ALL dims (x,y,z,heading) use base=3.

    - num_levels = 4
    - each level ℓ:
        (ix_ℓ, iy_ℓ, iz_ℓ, ih_ℓ), each ∈ {0,1,2}
        token_ℓ ∈ {0..80}   because 3*3*3*3 = 81

    - finest resolution bins:
        x,y,z,h each use 3^4 = 81 bins

    Returned tokens: (..., 4) int64 in [0,80]
    """

    def __init__(
        self,
        x_range=(-75.0, 50.0),
        y_range=(-20.0, 90.0),
        z_range=(0.0, 60.0),
        h_range=(-math.pi, math.pi),
        num_levels=4,
        base=3,
    ):
        super().__init__()

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.h_min, self.h_max = h_range

        self.num_levels = num_levels
        self.base = base

        # finest bin count for each dimension
        self.n_total = base ** num_levels   # = 3^4 = 81

        # mixed radix powers (digit reconstruction)
        pow_b = torch.tensor(
            [base ** (num_levels - 1 - l) for l in range(num_levels)],
            dtype=torch.long,
        )
        self.register_buffer("pow_b", pow_b, persistent=False)

        # For token = (((ix * 3 + iy) * 3 + iz) * 3 + ih)
        self.token_base = base ** 4    # = 81

    # -------------------- Helpers --------------------

    @staticmethod
    def _continuous_to_bins(x, x_min, x_max, n_bins):
        """Map continuous → integer bin index [0..n_bins-1]."""
        eps = 1e-6
        x_norm = (x - x_min) / (x_max - x_min + eps)
        x_norm = x_norm.clamp(0.0, 1.0 - eps)
        return (x_norm * n_bins).floor().long()

    @staticmethod
    def _to_digits(idx, base, num_levels):
        """Convert integer idx → digits in [0..base-1] of length L."""
        digits = []
        tmp = idx
        for _ in range(num_levels):
            digits.append(tmp % base)
            tmp = tmp // base

        digits = torch.stack(digits, dim=-1)     # finest→coarsest
        digits = torch.flip(digits, dims=[-1])   # coarsest→finest
        return digits

    @staticmethod
    def encode_digits(ix, iy, iz, ih, base=3):
        """(ix,iy,iz,ih) ∈ {0,1,2} → token ∈ [0..80]."""
        return (((ix * base + iy) * base + iz) * base + ih).long()

    @staticmethod
    def decode_token(token, base=3):
        """token ∈ [0..80] → (ix,iy,iz,ih)."""
        ih = token % base
        token =token// base
        iz = token % base
        token = token//base
        iy = token % base
        token =token// base
        ix = token
        return ix.long(), iy.long(), iz.long(), ih.long()

    @staticmethod
    def _bins_to_continuous(idx, v_min, v_max, n_bins):
        """bin center reconstruction."""
        width = (v_max - v_min) / float(n_bins)
        return v_min + (idx.to(torch.float32) + 0.5) * width

    # -------------------- Encode --------------------

    def forward(self, pos, heading):
        """
        pos: (...,3)
        heading: (...)

        returns:
            tokens: (..., num_levels), each ∈ [0..80]
        """
        orig_shape = pos.shape[:-1]
        pos_flat = pos.reshape(-1, 3)
        x = pos_flat[:, 0]
        y = pos_flat[:, 1]
        z = pos_flat[:, 2]
        h = heading.reshape(-1)

        # wrap heading
        h = ((h - self.h_min) % (self.h_max - self.h_min)) + self.h_min

        # finest-level bin index for each dim
        ix_fine = self._continuous_to_bins(x, self.x_min, self.x_max, self.n_total)
        iy_fine = self._continuous_to_bins(y, self.y_min, self.y_max, self.n_total)
        iz_fine = self._continuous_to_bins(z, self.z_min, self.z_max, self.n_total)
        ih_fine = self._continuous_to_bins(h, self.h_min, self.h_max, self.n_total)

        # digits per level
        ix_digits = self._to_digits(ix_fine, self.base, self.num_levels)
        iy_digits = self._to_digits(iy_fine, self.base, self.num_levels)
        iz_digits = self._to_digits(iz_fine, self.base, self.num_levels)
        ih_digits = self._to_digits(ih_fine, self.base, self.num_levels)

        # token per level
        tokens_flat = self.encode_digits(
            ix_digits, iy_digits, iz_digits, ih_digits, self.base
        )  # [N, L]

        tokens = tokens_flat.view(*orig_shape, self.num_levels)
        return tokens

    # -------------------- Decode --------------------

    @torch.no_grad()
    def decode_tokens_to_state(self, tokens):
        """
        tokens: (..., num_levels), each ∈ [0..80]

        returns:
            pos: (...,3)
            heading: (...)
        """
        prefix_shape = tokens.shape[:-1]
        L = tokens.shape[-1]
        assert L == self.num_levels

        tokens_flat = tokens.reshape(-1, L)   # [N,L]
        N = tokens_flat.size(0)

        # decode all digits
        ix_list = []
        iy_list = []
        iz_list = []
        ih_list = []

        for l in range(L):
            ix_l, iy_l, iz_l, ih_l = self.decode_token(tokens_flat[:, l], self.base)
            ix_list.append(ix_l.unsqueeze(-1))
            iy_list.append(iy_l.unsqueeze(-1))
            iz_list.append(iz_l.unsqueeze(-1))
            ih_list.append(ih_l.unsqueeze(-1))

        ix_digits = torch.cat(ix_list, dim=-1)  # [N,L]
        iy_digits = torch.cat(iy_list, dim=-1)
        iz_digits = torch.cat(iz_list, dim=-1)
        ih_digits = torch.cat(ih_list, dim=-1)

        # reconstruct finest-level index
        ix_fine = (ix_digits * self.pow_b).sum(dim=-1)
        iy_fine = (iy_digits * self.pow_b).sum(dim=-1)
        iz_fine = (iz_digits * self.pow_b).sum(dim=-1)
        ih_fine = (ih_digits * self.pow_b).sum(dim=-1)

        # continuous center
        x = self._bins_to_continuous(ix_fine, self.x_min, self.x_max, self.n_total)
        y = self._bins_to_continuous(iy_fine, self.y_min, self.y_max, self.n_total)
        z = self._bins_to_continuous(iz_fine, self.z_min, self.z_max, self.n_total)
        h = self._bins_to_continuous(ih_fine, self.h_min, self.h_max, self.n_total)

        pos = torch.stack([x, y, z], dim=-1).reshape(*prefix_shape, 3)
        heading = h.reshape(*prefix_shape)
        return pos, heading


# ---------------- Example Usage ----------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = HierarchicalStateTokenizer().to(device)

    pos = torch.tensor(
        [
            [0.0, 0.0, 10.0],
            [-50.0, 80.0, 30.0],
        ],
        device=device,
    )
    heading = torch.tensor([0.1, -2.0], device=device)

    tokens = tokenizer(pos, heading)
    print("Tokens:", tokens)  # shape [2,4], each ∈ [0..80]

    pos_rec, heading_rec = tokenizer.decode_tokens_to_state(tokens)
    print("Reconstructed:", pos_rec, heading_rec)
