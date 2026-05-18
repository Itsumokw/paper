"""Non-invasive HiGMemPlus extensions.

This package keeps the original `baseline/HiGMem` implementation intact and
adds retrieval-time evidence-aware variants for controlled experiments.
"""

from .enhancers import HiGMemPlusEnhancer, METHODS
from .schemas import RawTurn

__all__ = ["HiGMemPlusEnhancer", "RawTurn", "METHODS"]
