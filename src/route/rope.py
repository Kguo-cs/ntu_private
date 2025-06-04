import torch
import math

def build_rope_block_diag_matrix(pos, dim):
    """
    Create a block-diagonal position embedding matrix for RoPE.

    Args:
        seq_len (int): Length of the sequence (number of positions).
        dim (int): Dimension of the embedding. Must be even.

    Returns:
        torch.Tensor: A tensor of shape (seq_len, dim, dim), where each matrix[i]
                      is a block-diagonal matrix composed of 2x2 rotation matrices.
    """
    assert dim % 2 == 0, "Embedding dimension must be even."

    # Generate frequencies: typically log-scale as in the Transformer paper
    half_dim = dim // 2
    theta = 10000 ** (-torch.arange(0, half_dim, 2).float() / half_dim)  # (dim//4)


    x=pos[...,0]
    y=pos[...,1]
    #print(x,y)
    # Compute angles: outer product of position indices and frequency scales
    angles1 = x[:, None] * theta[None, :]  # (seq_len, dim//4)
    angles2 = y[:, None] * theta[None, :]  # (seq_len, dim//4)

    angles=torch.cat([angles1, angles2],dim=-1)

    # Get cos and sin values
    cos = angles.cos()#.repeat_interleave(2, dim=1)  # (seq_len, dim//2)
    sin = angles.sin()#.repeat_interleave(2, dim=1)  # (seq_len, dim//2)

    # Construct block rotation matrices
    rot_mats = []
    for i in range(pos.shape[0]):
        blocks = []
        for j in range(0, dim, 2):
            c = cos[i, j // 2]
            s = sin[i, j // 2]
            block = torch.tensor([[c, -s],
                                  [s,  c]])
            blocks.append(block)
        rot_matrix = torch.block_diag(*blocks)  # (dim, dim)
        rot_mats.append(rot_matrix)

    return torch.stack(rot_mats)  # (seq_len, dim, dim)


seq_len=10
start=1

pos = torch.randn((2,2))#torch.arange(start,seq_len+start, dtype=torch.float32)  # (seq_len,)

x=build_rope_block_diag_matrix(pos, 4)
print(x)

k=x[0].T@x[1]

pos[...,0]+=1
pos[...,1]+=2

x=build_rope_block_diag_matrix(pos, 4)
print(x)

k1=x[0].T@x[1]
print(torch.allclose(k,k1))