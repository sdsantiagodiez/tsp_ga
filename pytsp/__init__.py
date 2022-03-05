"""Py Travelling Salesman Problem"""

__version__ = "0.4.0"


from .core.computation import (
    Compute,
    ComputeFactory,
    NumbaCompute,
    NumpyCompute,
)
from .core.routing import Routing
from .util.data_generator import DataGenerator
from .util.plot import Mapplot
