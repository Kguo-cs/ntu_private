"""Initialize the existing local heading offset to identity; no layout change."""
import hashlib
import torch
from torch import nn


def non_heading_hash(model):
    final = model.to_out_m_delta.mlp[-1]
    names = {id(final.weight): 'weight', id(final.bias): 'bias'}
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        selected = value.detach()
        if id(value) in names:
            selected = torch.cat((selected[:2], selected[4:]), 0)
        digest.update(str((name, selected.shape, selected.dtype)).encode())
        digest.update(selected.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def initialize_heading_identity(model):
    if not model.x_pred or model.heading_velocity:
        raise ValueError('Identity offset initialization requires the vector x0 heading head')
    final = model.to_out_m_delta.mlp[-1]
    if not isinstance(final, nn.Linear) or final.bias is None or final.out_features < 4:
        raise ValueError('Expected the native state head with local heading columns2/3')
    before = non_heading_hash(model)
    final.weight[2:4].zero_()
    final.bias[2:4].copy_(final.bias.new_tensor([1., 0.]))
    after = non_heading_hash(model)
    assert before == after
    return dict(non_heading_parameters_sha256=after, changed_rows=[2, 3],
                local_heading_vector=[1., 0.], architecture_and_inference_unchanged=True)
