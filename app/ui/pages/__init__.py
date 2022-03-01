from .page1 import Page1
from .page2 import Page2
from .page3 import Page3
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Store selection": Page1,
    "Optimal route": Page2,
    "Route on a map": Page3,
}

__all__ = ["PAGE_MAP"]
