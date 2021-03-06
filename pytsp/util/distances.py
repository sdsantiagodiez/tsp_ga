import numpy as np
from geopy.distance import geodesic
from numba import jit
from numba import prange

_NUMBA_CACHE: bool = True


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_closest_destination(
    distance_matrix: np.ndarray,
    origin: np.int16,
    destination_1: np.int16,
    destination_2: np.int16,
):
    return (
        destination_1
        if distance_matrix[origin, destination_1]
        < distance_matrix[origin, destination_2]
        else destination_2
    )


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_closest_destination_from_available_routes(
    distance_matrix: np.ndarray,
    origin: np.int16,
    available_destinations: np.ndarray,
):
    if not available_destinations.size:
        raise ValueError("Destionation size can't be 0")

    return available_destinations[
        np.argmin(distance_matrix[origin, :][available_destinations])
    ]


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_destinations_not_in_route(number_of_cities: int, route: np.ndarray):
    destionations_not_in_route = np.full(
        np.sum(route == -1), -2, dtype=np.int16
    )
    for i in range(destionations_not_in_route.size):
        for destination in range(number_of_cities):
            if (
                destination not in route
                and destination not in destionations_not_in_route
            ):
                destionations_not_in_route[i] = destination

    return destionations_not_in_route


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_route_distance(distance_matrix: np.ndarray, route: np.ndarray):
    route_size: int = route.size
    one_hot_distances = np.zeros((route_size, route_size), dtype=np.int8)
    for i in range(route_size - 1):
        one_hot_distances[route[i], route[i + 1]] = 1
    one_hot_distances[route[route_size - 1], route[0]] = 1

    return np.nansum(one_hot_distances * distance_matrix)


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_a_fast_route(distance_matrix: np.ndarray):
    number_of_cities: int = distance_matrix.shape[0]
    route: np.ndarray = np.full(number_of_cities, -1, dtype=np.int16)
    route[0] = np.random.choice(number_of_cities, 1)[0]

    for i in np.arange(number_of_cities - 1):
        destinations_not_in_route: np.ndarray = get_destinations_not_in_route(
            number_of_cities, route
        )
        route[i + 1] = get_closest_destination_from_available_routes(
            distance_matrix, route[i], destinations_not_in_route
        )

    return route


@jit(nopython=True, cache=_NUMBA_CACHE)
def get_a_fast_route_and_distance(distance_matrix: np.ndarray):
    benchmark_route = get_a_fast_route(distance_matrix)
    benchmark_route_distance = get_route_distance(
        distance_matrix, benchmark_route
    )

    return benchmark_route, benchmark_route_distance


def get_distance(
    origin_coordinates,
    destination_coordinates,
    distance_type: str = "geodesic",
):
    distance = np.inf
    if distance_type == "geodesic":
        distance = geodesic(origin_coordinates, destination_coordinates).meters

    return distance


@jit(nopython=True, parallel=True, cache=_NUMBA_CACHE)
def get_roundtrip(route: np.ndarray, destionation_coordinates: np.ndarray):
    route_coordinates = np.zeros((route.size + 1, 2), dtype=np.float64)

    for i in prange(route.size):
        route_coordinates[i] = destionation_coordinates[route[i]]
    route_coordinates[route.size] = destionation_coordinates[route[0]]

    return route_coordinates
