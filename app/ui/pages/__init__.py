from .store_selection import StoreSelection
from .compute import Compute
from .map_plot import MapPlot
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Store selection": StoreSelection,
    "Optimal route": Compute,
    "Route on a map": MapPlot,
}

__all__ = ["PAGE_MAP"]
