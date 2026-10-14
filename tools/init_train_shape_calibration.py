"""Experimental TRAIN-fitted monotone physical-size postprocessing.

This is not a training objective or an evaluator modification. A fixed mapping
is applied to real generated vehicle boxes, so collision metrics see those boxes.
"""
from __future__ import annotations
import types
import torch


def vehicle_mask(shape, kind, ego):
    if shape.ndim != 2 or shape.shape[1] != 2 or kind.shape != shape.shape[:1] or ego.shape != kind.shape:
        raise ValueError('Expected physical [N,2] sizes and aligned type/ego vectors')
    return (kind == 0) & ~ego.bool() & torch.isfinite(shape).all(-1) & (shape > 0).all(-1)


def fit_transport(generated, generated_type, generated_ego, target, target_type, target_ego):
    """257 predeclared log-quantile knots; no evaluation data or tuning argument."""
    source_mask = vehicle_mask(generated, generated_type, generated_ego)
    target_mask = vehicle_mask(target, target_type, target_ego)
    source = generated[source_mask].detach().cpu().double().log()
    destination = target[target_mask].detach().cpu().double().log()
    if min(len(source), len(destination)) < 256:
        raise ValueError('Need at least256 valid non-ego TRAIN vehicles in each distribution')
    q = torch.linspace(.001, .999, 257, dtype=torch.float64)
    curves = []
    for index in range(2):
        x = torch.quantile(source[:, index], q)
        y = torch.quantile(destination[:, index], q)
        # Equal source knots cannot represent distinct target quantiles with a
        # deterministic map. Average their targets; never add artificial jitter.
        unique, inverse, counts = torch.unique_consecutive(x, return_inverse=True, return_counts=True)
        means = torch.zeros_like(unique).scatter_add_(0, inverse, y) / counts
        if len(unique) < 2:
            raise ValueError('Collapsed generated sizes cannot be repaired by a deterministic quantile map')
        curves.append(dict(x=unique.tolist(), y=means.tolist()))
    return dict(schema=1, fields=['length', 'width'], transform='log_quantile_piecewise_linear',
                quantile_range=[.001, .999], quantile_knots=257, curves=curves,
                tail_slope_range=[0., 4.], source_count=len(source), target_count=len(destination),
                selection='positive finite generated/GT type0 non-ego vehicles, separately',
                generated_excluded=int(len(generated)-source_mask.sum()),
                target_excluded=int(len(target)-target_mask.sum()),
                target_split='TRAIN', evaluation_data_used_for_fit=False)


def _map_log(values, curve):
    x = torch.as_tensor(curve['x'], dtype=torch.float64, device=values.device)
    y = torch.as_tensor(curve['y'], dtype=torch.float64, device=values.device)
    if len(x) < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Invalid calibration curve')
    if not torch.all(x[1:] > x[:-1]) or not torch.all(y[1:] >= y[:-1]):
        raise ValueError('Calibration must be monotone with strictly increasing source knots')
    z = values.double().log()
    i = torch.searchsorted(x, z.contiguous(), right=True).sub(1).clamp(0, len(x)-2)
    slope = (y[i+1]-y[i])/(x[i+1]-x[i])
    # Extrapolate continuously instead of clipping values at quantile boundaries.
    slope = torch.where((z < x[0]) | (z > x[-1]), slope.clamp(0., 4.), slope)
    origin_x = torch.where(z > x[-1], x[-1], x[i])
    origin_y = torch.where(z > x[-1], y[-1], y[i])
    return (origin_y + (z-origin_x)*slope).exp().to(values.dtype)


def apply_transport(shape, kind, ego, artifact):
    if artifact['schema'] != 1 or artifact['target_split'] != 'TRAIN' or artifact['evaluation_data_used_for_fit']:
        raise ValueError('Only the registered TRAIN-fitted schema is supported')
    mask = vehicle_mask(shape, kind, ego)
    result = shape.clone()
    for i, curve in enumerate(artifact['curves']):
        result[mask, i] = _map_log(shape[mask, i], curve)
    if not torch.isfinite(result[mask]).all() or not (result[mask] > 0).all():
        raise ValueError('Calibrated physical sizes must remain finite and positive')
    return result


def install_transport(model, artifact):
    """Wrap InitDiffusion._infer: all downstream consumers receive changed boxes."""
    if not model.encoder.initial_scene_only or model.encoder.init_decoder_name != 'flow':
        raise ValueError('This experiment supports initial-scene InitDiffusion only')
    decoder = model.encoder.init_decoder
    if hasattr(decoder, '_train_shape_transport'):
        raise ValueError('Size transport is already installed')
    original = decoder._infer
    stats = dict(calls=0, changed_vehicles=0, invalid_generated_vehicles=0)

    def infer(instance, agent, map_feature):
        if instance.training:
            raise RuntimeError('TRAIN-fitted postprocessing is inference-only')
        position, heading, tokens, shape, velocity = original(agent, map_feature)
        transformed = apply_transport(shape, agent['type'], agent['ego_mask'], artifact)
        selected = (agent['type'] == 0) & ~agent['ego_mask'].bool()
        stats['calls'] += 1
        stats['changed_vehicles'] += int((transformed != shape).any(-1).sum())
        stats['invalid_generated_vehicles'] += int((selected & ~vehicle_mask(shape, agent['type'], agent['ego_mask'])).sum())
        return position, heading, tokens, transformed, velocity

    decoder._infer = types.MethodType(infer, decoder)
    decoder._train_shape_transport = artifact
    return stats
