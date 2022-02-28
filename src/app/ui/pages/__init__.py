from .page1 import Page1
from .page2 import Page2
from ..utils import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Home": Page1,
    "Route on a map": Page2,
}

__all__ = ["PAGE_MAP"]
