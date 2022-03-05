from .compute import Compute
from .numpy_compute import NumpyCompute
from .numba_compute import NumbaCompute


class ComputeFactory(object):
    def __init__(self):
        self.__initialize_computes()

    def __initialize_computes(self):
        self._computes = {}

        self.register_compute(compute_name="numpy", compute=NumpyCompute)
        self.register_compute(compute_name="numba", compute=NumbaCompute)

    def register_compute(self, compute_name: str, compute: Compute):
        self._computes[compute_name] = compute

    def get_compute(self, compute_name: str) -> Compute:
        compute = self._computes.get(compute_name)
        if not compute:
            raise ValueError(compute_name)
        return compute()
