import numpy as np
import sys

sys.path.append(".")  # until structured as package

from util.distances import get_closest_destination, get_route_distance


def get_benchmark_route(distance_matrix: np.ndarray):
    number_of_cities: int = distance_matrix.shape[0]
    all_cities = np.arange(number_of_cities)
    route: np.ndarray = np.full(number_of_cities, -1, dtype=np.int16)
    route[0] = np.random.choice(number_of_cities, 1)[0]

    for i in np.arange(0, number_of_cities - 1, 1, dtype=int):
        destinations_not_in_route: np.ndarray = all_cities[
            ~np.in1d(all_cities, route)
        ]
        route[i + 1] = get_closest_destination(
            distance_matrix, route[i], destinations_not_in_route
        )

    return route, get_route_distance(distance_matrix, route)
