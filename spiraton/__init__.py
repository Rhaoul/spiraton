"""
Spiraton - package public API.

Expose:
- SpiratonCell (core, canon stable)
- GatedSpiratonCell (experimental, log-domain + gates + coeffs)
"""
__version__ = "0.1.0"

from .core.cell import SpiratonCell
from .experimental.gated_cell import GatedSpiratonCell

__all__ = ["SpiratonCell", "GatedSpiratonCell"]
