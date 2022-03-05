from .store_selection import StoreSelection
from .page2 import Page2
from .page3 import Page3
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Store selection": StoreSelection,
    "Optimal route": Page2,
    "Route on a map": Page3,
}

__all__ = ["PAGE_MAP"]
