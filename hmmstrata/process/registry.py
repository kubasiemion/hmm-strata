from __future__ import annotations
from typing import Callable
from .base import HMM

_REG: dict[str, Callable[..., HMM]] = {}


def register(name: str):
    def deco(fn):
        _REG[name] = fn
        return fn
    return deco


def make(name: str, **params) -> HMM:
    if name not in _REG:
        import_all()
    hmm = _REG[name](**params)
    hmm.params = dict(params)
    hmm.validate()
    return hmm


def names() -> list[str]:
    import_all()
    return sorted(_REG)


def import_all() -> None:
    from . import sync_strata  # noqa: F401  (registers on import)
