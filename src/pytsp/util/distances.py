import numpy as np
from multipledispatch import dispatch


@dispatch(np.ndarray, np.int16, np.ndarray)
def get_closest_destination(
    distance_matrix: np.ndarray,
    origin: np.int16,
    available_destinations: np.ndarray,
):
    closest_distance: np.int64 = np.iinfo(np.int64).max
    closest_city: np.int16

    if not available_destinations.size:
        raise ValueError("Destionation size can't be 0")

    for i in np.arange(available_destinations.size):
        if (
            closest_distance
            > distance_matrix[origin, available_destinations[i]]
        ):
            closest_distance = distance_matrix[
                origin, available_destinations[i]
            ]
            closest_city = available_destinations[i]

    return closest_city


@dispatch(np.ndarray, np.int16, np.int16, np.int16)
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


def get_route_distance(distance_matrix: np.ndarray, route: np.ndarray):
    route_size: int = route.size
    one_hot_distances = np.zeros((route_size, route_size), dtype=np.int8)
    for i in np.arange(route_size - 1).tolist():
        one_hot_distances[route[i], route[i + 1]] = 1
    one_hot_distances[route[route_size - 1], route[0]] = 1

    return np.nansum(one_hot_distances * distance_matrix)


def get_a_fast_route(distance_matrix: np.ndarray):
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

    return route


def get_a_fast_route_and_distance(distance_matrix: np.ndarray):
    benchmark_route = get_a_fast_route(distance_matrix)
    print(benchmark_route)
    benchmark_route_distance = get_route_distance(
        distance_matrix, benchmark_route
    )

    return benchmark_route, benchmark_route_distance
