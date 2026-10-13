"""Inference time grids with unchanged model-call count and uniform defaults."""
import math
from collections.abc import Mapping
from numbers import Integral, Real
import torch


def parse_sampling_time_grid(config=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('sampling_time_grid must be a mapping or null')
    options=dict(config or {})
    if set(options)-{'kind','power','terminal_time'}:
        raise ValueError('sampling_time_grid accepts kind, power and terminal_time')
    kind=options.get('kind','power')
    if kind not in ('power','cosine'):
        raise ValueError('sampling_time_grid.kind must be power or cosine')
    power=options.get('power',1.)
    if isinstance(power,bool) or not isinstance(power,Real) or not math.isfinite(power) or power<=0:
        raise ValueError('sampling_time_grid.power must be finite and positive')
    terminal=options.get('terminal_time')
    if terminal is not None and (isinstance(terminal,bool) or not isinstance(terminal,Real)
                                 or not math.isfinite(terminal) or not 0<terminal<1):
        raise ValueError('sampling_time_grid.terminal_time must be in (0,1) or null')
    if kind=='cosine' and power!=1.:
        raise ValueError('cosine grid does not use a power override')
    return dict(kind=kind,power=float(power),terminal_time=None if terminal is None else float(terminal))


def build_sampling_time_grid(steps, *, device, dtype, config=None):
    if isinstance(steps,bool) or not isinstance(steps,Integral) or steps<1:
        raise ValueError('sampling steps must be a positive integer')
    options=parse_sampling_time_grid(config)
    # Keep the original exact operation for default / explicit uniform Euler.
    times=torch.linspace(1.,0.,int(steps)+1,device=device,dtype=dtype)
    if options['kind']=='cosine':
        times=.5*(1.+torch.cos(math.pi*(1.-times)))
    elif options['power']!=1.:
        times=times.pow(options['power'])
    terminal=options['terminal_time']
    if terminal is not None:
        if steps<2:raise ValueError('terminal_time requires at least two sampling steps')
        # Retain the requested interior shape while keeping the last network
        # call at a comparable denoising time; default None reproduces cc grids.
        last=times[-2]
        times=torch.cat(((times[:-1]-last)/(1.-last)*(1.-terminal)+terminal,times.new_zeros(1)))
    if not torch.isfinite(times).all() or not (times[:-1]>times[1:]).all():
        raise ValueError('Time grid is not strictly descending at the latent dtype precision')
    return times
