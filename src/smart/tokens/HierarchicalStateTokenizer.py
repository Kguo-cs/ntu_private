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

    - finest resolution bins:
        x,y,z,h each use 3^4 = 81 bins

    API
    ----
    forward(pos, heading) -> tokens
        pos: (..., 3)
        heading: (...)
        returns tokens: (..., 4), each in [0..80]

    decode_tokens_to_state(tokens, prefix_levels=None)
        tokens: (..., 4)
        prefix_levels:
            None or 4 → use all 4 tokens (finest)
            1         → use only first token (coarsest)
            2         → use first two tokens
            3         → use first three tokens
        returns:
            pos: (..., 3)
            heading: (...)

    sequential_decode_tokens(tokens)
        tokens: (..., 4)
        returns:
            pos_seq: (..., 4, 3)
            heading_seq: (..., 4)
        where level k uses first (k+1) tokens.
    """

    def __init__(
        self,
        x_range=(-75.0, 50.0),
        y_range=(-20.0, 90.0),
        z_range=(0.0, 60.0),
        h_range=(-math.pi, math.pi),
        num_levels=4,
        base=4,
    ):
        super().__init__()

        self.x_min, self.x_max = float(x_range[0]), float(x_range[1])
        self.y_min, self.y_max = float(y_range[0]), float(y_range[1])
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.h_min, self.h_max = float(h_range[0]), float(h_range[1])

        self.num_levels = num_levels
        self.base = base

        # total bins at finest level for each dimension
        self.n_total = base ** num_levels  # = 3^4 = 81

        # powers for reconstructing fine index at L = num_levels:
        # idx_fine = sum_{l=0}^{L-1} digits[l] * base^(L-1-l)
        pow_b = torch.tensor(
            [base ** (num_levels - 1 - l) for l in range(num_levels)],
            dtype=torch.long,
        )
        self.register_buffer("pow_b", pow_b, persistent=False)

        # token range: 3^4 = 81
        self.token_cardinality = base ** 4  # 81

    # ---------------- low-level helpers ----------------

    @staticmethod
    def _continuous_to_bins(x, x_min, x_max, n_bins):
        """
        Map continuous x in [x_min, x_max] to integer bins [0, n_bins-1].
        Vectorized, device-agnostic.
        """
        eps = 1e-6
        x_norm = (x - x_min) / (x_max - x_min + eps)
        x_norm = x_norm.clamp(0.0, 1.0 - eps)
        x_idx = (x_norm * n_bins).floor().long()
        return x_idx

    @staticmethod
    def _to_digits(idx, base, num_levels):
        """
        Convert integer idx → base-'base' digits of length num_levels.

        Returns digits with shape [N, num_levels] where:
            digits[:, 0] = coarsest (most significant)
            digits[:, -1] = finest (least significant)
        """
        digits = []
        tmp = idx
        for _ in range(num_levels):
            digits.append(tmp % base)
            tmp = tmp // base
        digits = torch.stack(digits, dim=-1)     # finest → coarsest
        digits = torch.flip(digits, dims=[-1])   # coarsest → finest
        return digits

    @staticmethod
    def encode_digits(ix, iy, iz, ih, base=3):
        """
        Encode per-level digits for one level:
            ix, iy, iz, ih ∈ {0,1,2}
        → token ∈ [0, base^4 - 1] = [0, 80]
        """
        return (((ix * base + iy) * base + iz) * base + ih).long()

    @staticmethod
    def decode_token(token, base=3):
        """
        token ∈ [0, base^4 - 1] → (ix, iy, iz, ih), each in [0, base-1].
        Vectorized for any shape of token.
        """
        ih = token % base
        token = token // base
        iz = token % base
        token = token // base
        iy = token % base
        token = token // base
        ix = token
        return ix.long(), iy.long(), iz.long(), ih.long()

    @staticmethod
    def _bins_to_continuous(idx, v_min, v_max, n_bins):
        """
        Map bin index [0..n_bins-1] to bin center in [v_min, v_max].
        """
        width = (v_max - v_min) / float(n_bins)
        idx_f = idx.to(torch.float32)
        return v_min + (idx_f + 0.5) * width

    # --------- core helper: decode 1 axis with first K digits ---------

    def _decode_axis_prefix(self, digits, v_min, v_max, prefix_levels):
        """
        digits: [N, L] (coarse→fine), each in [0..base-1]
        prefix_levels: K ∈ {1..L}
        Returns centers: [N], using only first K digits.
        """
        device = digits.device
        N, L = digits.shape
        assert 1 <= prefix_levels <= L

        K = prefix_levels
        digits_k = digits[:, :K]  # [N, K]

        # bins under this coarser grid
        n_bins_used = self.base ** K

        # powers for K-digit representation (coarse→fine)
        powers_k = torch.tensor(
            [self.base ** (K - 1 - j) for j in range(K)],
            device=device,
            dtype=torch.long,
        )  # [K]

        idx = (digits_k * powers_k).sum(dim=-1)  # [N]
        centers = self._bins_to_continuous(idx, v_min, v_max, n_bins_used)
        return centers

    # ---------------- main encode (forward) ----------------

    def forward(self, pos, heading):
        """
        Encode (x, y, z, heading) → per-level tokens.

        Args:
            pos:     (..., 3)  (x, y, z)
            heading: (...)     in radians

        Returns:
            tokens:  (..., num_levels), each in [0..80]
        """
        orig_shape = pos.shape[:-1]  # prefix ...

        pos_flat = pos.reshape(-1, 3)
        x = pos_flat[:, 0]
        y = pos_flat[:, 1]
        z = pos_flat[:, 2]
        h = heading.reshape(-1)

        # wrap heading into [h_min, h_max]
        h = ((h - self.h_min) % (self.h_max - self.h_min)) + self.h_min

        # finest-level indices
        ix_fine = self._continuous_to_bins(x, self.x_min, self.x_max, self.n_total)
        iy_fine = self._continuous_to_bins(y, self.y_min, self.y_max, self.n_total)
        iz_fine = self._continuous_to_bins(z, self.z_min, self.z_max, self.n_total)
        ih_fine = self._continuous_to_bins(h, self.h_min, self.h_max, self.n_total)

        # digits per level (coarse→fine)
        ix_digits = self._to_digits(ix_fine, self.base, self.num_levels)  # [N,L]
        iy_digits = self._to_digits(iy_fine, self.base, self.num_levels)
        iz_digits = self._to_digits(iz_fine, self.base, self.num_levels)
        ih_digits = self._to_digits(ih_fine, self.base, self.num_levels)

        # encode at each level into one token
        tokens_flat = self.encode_digits(
            ix_digits, iy_digits, iz_digits, ih_digits, self.base
        )  # [N,L]

        tokens = tokens_flat.view(*orig_shape, self.num_levels)
        return tokens

    # ---------------- decode: state using first K tokens ----------------

    @torch.no_grad()
    def decode_tokens_to_state(self, tokens, prefix_levels=None):
        """
        Decode per-level tokens → state (x,y,z,h) using only the first K tokens.

        Args:
            tokens: (..., num_levels)
            prefix_levels:
                None or num_levels → uses all tokens (finest, default)
                1                 → uses only tokens[..., :1]
                2                 → uses tokens[..., :2]
                3                 → uses tokens[..., :3]

        Returns:
            pos:     (..., 3)
            heading: (...)
        """
        prefix_shape = tokens.shape[:-1]
        L = tokens.shape[-1]
        assert L == self.num_levels

        if prefix_levels is None:
            prefix_levels = L
        assert 1 <= prefix_levels <= L
        K = prefix_levels

        tokens_flat = tokens.reshape(-1, L)  # [N,L]
        N = tokens_flat.shape[0]

        # decode tokens → digits per axis
        ix_list, iy_list, iz_list, ih_list = [], [], [], []
        for l in range(L):
            ix_l, iy_l, iz_l, ih_l = self.decode_token(tokens_flat[:, l], self.base)
            ix_list.append(ix_l.unsqueeze(-1))
            iy_list.append(iy_l.unsqueeze(-1))
            iz_list.append(iz_l.unsqueeze(-1))
            ih_list.append(ih_l.unsqueeze(-1))

        ix_digits_all = torch.cat(ix_list, dim=-1)  # [N,L]
        iy_digits_all = torch.cat(iy_list, dim=-1)
        iz_digits_all = torch.cat(iz_list, dim=-1)
        ih_digits_all = torch.cat(ih_list, dim=-1)

        # decode each axis using first K digits
        x = self._decode_axis_prefix(ix_digits_all, self.x_min, self.x_max, K)
        y = self._decode_axis_prefix(iy_digits_all, self.y_min, self.y_max, K)
        z = self._decode_axis_prefix(iz_digits_all, self.z_min, self.z_max, K)
        h = self._decode_axis_prefix(ih_digits_all, self.h_min, self.h_max, K)

        # tensor([[1, 2, 1, 0],
        #         [0, 1, 2, 1]], device='cuda:0')

        pos = torch.stack([x, y, z], dim=-1).reshape(*prefix_shape, 3)
        heading = h.reshape(*prefix_shape)
        return pos, heading

    # ------------- sequential decode: per-level states -------------

    @torch.no_grad()
    def sequential_decode_tokens(self, tokens):
        """
        Sequentially decode per-level tokens → per-level states.

        Args:
            tokens: (..., num_levels), each in [0..80]

        Returns:
            pos_seq:     (..., num_levels, 3)
            heading_seq: (..., num_levels)

            For each sample n:
                level 0 uses first token (K=1)
                level 1 uses first two tokens (K=2)
                level 2 uses first three tokens (K=3)
                level 3 uses all four tokens (K=4)
        """
        prefix_shape = tokens.shape[:-1]
        L = tokens.shape[-1]
        assert L == self.num_levels

        tokens_flat = tokens.reshape(-1, L)  # [N,L]
        N = tokens_flat.shape[0]

        # decode all tokens to digits per axis
        ix_list, iy_list, iz_list, ih_list = [], [], [], []
        for l in range(L):
            ix_l, iy_l, iz_l, ih_l = self.decode_token(tokens_flat[:, l], self.base)
            ix_list.append(ix_l.unsqueeze(-1))
            iy_list.append(iy_l.unsqueeze(-1))
            iz_list.append(iz_l.unsqueeze(-1))
            ih_list.append(ih_l.unsqueeze(-1))

        ix_digits_all = torch.cat(ix_list, dim=-1)  # [N,L]
        iy_digits_all = torch.cat(iy_list, dim=-1)
        iz_digits_all = torch.cat(iz_list, dim=-1)
        ih_digits_all = torch.cat(ih_list, dim=-1)

        # decode for K = 1..L
        pos_seq_list = []
        heading_seq_list = []
        for K in range(1, L + 1):
            xK = self._decode_axis_prefix(ix_digits_all, self.x_min, self.x_max, K)  # [N]
            yK = self._decode_axis_prefix(iy_digits_all, self.y_min, self.y_max, K)
            zK = self._decode_axis_prefix(iz_digits_all, self.z_min, self.z_max, K)
            hK = self._decode_axis_prefix(ih_digits_all, self.h_min, self.h_max, K)

            posK = torch.stack([xK, yK, zK], dim=-1)  # [N,3]
            pos_seq_list.append(posK.unsqueeze(1))    # [N,1,3]
            heading_seq_list.append(hK.unsqueeze(1))  # [N,1]

        pos_seq = torch.cat(pos_seq_list, dim=1)         # [N,L,3]
        heading_seq = torch.cat(heading_seq_list, dim=1) # [N,L]

        pos_seq = pos_seq.view(*prefix_shape, L, 3)
        heading_seq = heading_seq.view(*prefix_shape, L)

        return pos_seq, heading_seq


# ---------------- Example usage ----------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = HierarchicalStateTokenizer().to(device)

    pos = torch.tensor(
        [
            [0.0, 0.0, 10.0],
            [-50.0, 80.0, 30.0],
        ],
        device=device,
    )
    heading = torch.tensor([0.1, -2.0], device=device)

    # Encode
    tokens = tok(pos, heading)          # [2,4]
    print("tokens:", tokens)

    # Decode using all 4 tokens (finest)
    pos_fine, h_fine = tok.decode_tokens_to_state(tokens)
    print("fine pos:", pos_fine)
    print("fine heading:", h_fine)

    # Decode using only first token (coarsest)
    pos_coarse, h_coarse = tok.decode_tokens_to_state(tokens[:,:1])
    print("coarse pos:", pos_coarse)
    print("coarse heading:", h_coarse)

    # Decode using first two tokens
    pos_mid, h_mid = tok.decode_tokens_to_state(tokens, prefix_levels=2)
    print("mid pos:", pos_mid)
    print("mid heading:", h_mid)

    # Sequential decode (all levels at once)
    pos_seq, h_seq = tok.sequential_decode_tokens(tokens)
    print("pos_seq shape:", pos_seq.shape)       # [2,4,3]
    print("heading_seq shape:", h_seq.shape)     # [2,4]

    print(pos_seq)
    print(h_seq)
