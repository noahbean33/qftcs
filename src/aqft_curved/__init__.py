"""
AQFT in Curved Spacetime (`aqft_curved`)
=======================================

A Python library for Algebraic Quantum Field Theory (AQFT) in curved spacetimes.
"""

# Import core components into the top-level namespace
from . import spacetime
from . import field
from . import state
from . import algebra
from . import utils

__all__ = [
    "spacetime",
    "field",
    "state",
    "algebra",
    "utils",
]
