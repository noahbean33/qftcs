"""
This package implements the core categorical framework for AQFT.
It defines the categories Loc and Alg, and the functors between them.
"""
from .loc import LocObject, LocMorphism
from .alg import AlgObject, AlgMorphism
from .functor import AQFTFunctor

__all__ = [
    "LocObject",
    "LocMorphism",
    "AlgObject",
    "AlgMorphism",
    "AQFTFunctor",
]
