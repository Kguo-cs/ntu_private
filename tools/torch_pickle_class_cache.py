"""Opt-in public pickle_module proxy for local Torch scene deserialization.

It reuses only Torch's verified ZIP/legacy local UnpicklerWrapper classes.
All actual find_class/persistent_load/storage behavior stays in installed Torch.
No collector, ctypes untracking, tensor conversion, or model code is involved.
"""
from __future__ import annotations

import pickle
import types


class _CachedUnpicklerMeta(type):
    wrappers = {}
    creations = 0
    reuses = 0
    validated_closures = {}

    def __new__(meta, name, bases, namespace, **kwargs):
        if name != 'UnpicklerWrapper':
            return super().__new__(meta, name, bases, namespace, **kwargs)
        if bases != (_ProxyUnpickler,):
            raise RuntimeError('Unexpected Torch UnpicklerWrapper base; class cache refuses unknown loader')
        function = namespace.get('find_class')
        if not isinstance(function, types.FunctionType):
            raise RuntimeError('Torch wrapper has no normal find_class function')
        from torch import serialization
        kind = function.__qualname__.split('.<locals>.', 1)[0]
        expected = {'_load': ('__class__', 'load_module_mapping'), '_legacy_load': ('__class__',)}
        if kind not in expected or function.__qualname__ != kind + '.<locals>.UnpicklerWrapper.find_class':
            raise RuntimeError('Unknown Torch wrapper kind; ZIP and legacy caches are never mixed')
        if function.__globals__ is not serialization.__dict__ or function.__code__.co_freevars != expected[kind]:
            raise RuntimeError('Installed Torch find_class globals/closure changed; refusing unsafe cache')
        closure = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        if kind == '_load' and closure['load_module_mapping'].cell_contents != {'torch.tensor': 'torch._tensor'}:
            raise RuntimeError('Torch ZIP module alias mapping changed; refusing class reuse')
        if closure.get('__class__') is not namespace.get('__classcell__'):
            raise RuntimeError('Torch __class__ closure and classcell disagree')
        key = (kind, function.__code__)
        if key in meta.wrappers:
            cls = meta.wrappers[key]
            # __build_class__ requires the new body cell to refer to the actual
            # returned class, even though that body's function is not retained.
            namespace['__classcell__'].cell_contents = cls
            meta.reuses += 1
            return cls
        cls = super().__new__(meta, name, bases, namespace, **kwargs)
        meta.wrappers[key] = cls
        meta.validated_closures[kind] = list(function.__code__.co_freevars)
        meta.creations += 1
        return cls


class _ProxyUnpickler(pickle.Unpickler, metaclass=_CachedUnpicklerMeta):
    pass


def pickle_module_proxy():
    """Pass as torch.load(..., weights_only=False, pickle_module=proxy)."""
    proxy = types.ModuleType('pickle')
    proxy.__dict__.update(vars(pickle))
    proxy.Unpickler = _ProxyUnpickler
    return proxy


def cache_stats():
    return {'classes_created': _CachedUnpicklerMeta.creations,
            'class_reuses': _CachedUnpicklerMeta.reuses,
            'cached_loader_kinds': sorted(kind for kind, _ in _CachedUnpicklerMeta.wrappers),
            'validated_closures': _CachedUnpicklerMeta.validated_closures}
