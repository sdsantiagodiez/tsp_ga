import numpy as np
from tqdm import tqdm

from .compute import Compute
from pytsp.util.distances import (
    get_a_fast_route,
    get_route_distance,
    get_destinations_not_in_route,
    get_closest_destination,
    get_closest_destination_from_available_routes,
)

GENE_DTYPE: type = np.int16
DISTANCES_DTYPE: type = np.int64


class NumpyCompute(Compute):
    def __init__(self) -> None:
        pass

    @staticmethod
    def run_generation(
        distance_matrix: np.ndarray,
        populations: np.ndarray,
        populations_fitness: np.ndarray,
        populations_mutation_rate: np.ndarray,
        population_number: int,
        population_size: int,
        selection_threshold: int,
        disable_tqdm: bool,
    ):
        """Runs a single generation iteration"""
        return _run_generation(
            distance_matrix,
            populations,
            populations_fitness,
            populations_mutation_rate,
            population_number,
            population_size,
            selection_threshold,
            disable_tqdm,
        )

    @staticmethod
    def get_new_populations(
        distance_matrix: np.ndarray,
        population_number: int,
        population_size: int,
        gene_size: int,
        enhanced_individuals: int,
    ):
        """Generates a random population"""
        return _get_new_populations(
            distance_matrix,
            population_number,
            population_size,
            gene_size,
            enhanced_individuals,
        )

    @staticmethod
    def get_populations_mutation_rates(
        population_number: int,
        uniform_population_mutation_rate: bool,
        max_mutation_rate: float,
    ):
        """Generates mutation rates for a population"""
        return _get_populations_mutation_rates(
            population_number,
            uniform_population_mutation_rate,
            max_mutation_rate,
        )

    @staticmethod
    def calculate_fitness(
        distance_matrix: np.ndarray,
        populations: np.ndarray,
        population_number: int,
        population_size: int,
    ):
        """Calculates fitness defined by distance"""
        return _calculate_fitness(
            distance_matrix,
            populations,
            population_number,
            population_size,
        )


def _run_generation(
    distance_matrix: np.ndarray,
    populations: np.ndarray,
    populations_fitness: np.ndarray,
    populations_mutation_rate: np.ndarray,
    population_number: int,
    population_size: int,
    selection_threshold: int,
    disable_tqdm: bool,
):
    for i in tqdm(range(population_number), disable=disable_tqdm):
        fittest_individuals = _selection_on_population(
            populations_fitness[i], selection_threshold
        )
        _crossover(
            distance_matrix,
            populations[i],
            population_size,
            populations_mutation_rate[i],
            fittest_individuals,
        )

    return _calculate_fitness(
        distance_matrix,
        populations,
        population_number,
        population_size,
    )


def _get_new_populations(
    distance_matrix: np.ndarray,
    population_number: int,
    population_size: int,
    gene_size: int,
    enhanced_individuals: int,
):
    populations = np.full(
        (population_number, population_size, gene_size),
        -1,
        dtype=GENE_DTYPE,
    )
    for i in range(population_number):
        populations[i] = _get_new_population(
            distance_matrix, population_size, gene_size, enhanced_individuals
        )

    return populations


def _get_new_population(
    distance_matrix: np.ndarray,
    population_size: int,
    gene_size: int,
    enhanced_individuals: int,
):
    population = np.full((population_size, gene_size), -1, dtype=GENE_DTYPE)
    for i in range(population_size):
        population[i] = np.random.choice(gene_size, gene_size, replace=False)

    random_individuals = np.random.choice(
        population_size, enhanced_individuals, replace=False
    )
    for i in range(enhanced_individuals):
        population[random_individuals[i]] = get_a_fast_route(distance_matrix)

    return population


def _get_populations_mutation_rates(
    population_number: int,
    uniform_population_mutation_rate: bool,
    max_mutation_rate: float,
):
    """
    When 'uniform_population_mutation_rate' is True, same random mutation
    rate is applied to all populations.

    When 'uniform_population_mutation_rate' is False, each population
    receives a random rate.

    In both cases mutation rates are bound between 0 and max_mutation_rate
    """

    population_mutation_rates = np.full(population_number, 0, dtype=np.float32)
    if uniform_population_mutation_rate:
        population_mutation_rates = np.repeat(
            np.random.uniform(0, max_mutation_rate, (1)),
            population_number,
        )
    else:
        population_mutation_rates = np.random.uniform(
            0, max_mutation_rate, (population_number)
        )

    # allow adaptive mutation rates
    return population_mutation_rates


def _selection_on_population(
    population_fitness: np.ndarray, max_selection_threshold: int
):
    """
    Selection strategy used is based on fitness to allow maximum
    parallelization. Roulette and tournament could be considered,
    although it's expected to not perform as fast

    """
    return np.argsort(population_fitness)[:max_selection_threshold]


def _calculate_population_fitness(
    distance_matrix: np.ndarray, population: np.ndarray, population_size: int
):
    population_fitness = np.full(population_size, 0, dtype=DISTANCES_DTYPE)
    for i in range(population_size):
        population_fitness[i] = get_route_distance(
            distance_matrix, population[i]
        )

    return population_fitness


def _calculate_fitness(
    distance_matrix: np.ndarray,
    populations: np.ndarray,
    population_number: int,
    population_size: int,
):
    fitness = np.zeros(
        (population_number, population_size),
        dtype=DISTANCES_DTYPE,
    )
    for i in range(population_number):
        fitness[i] = _calculate_population_fitness(
            distance_matrix, populations[i], population_size
        )

    return fitness


def _crossover(
    distance_matrix: np.ndarray,
    popoulation: np.ndarray,
    population_size: int,
    mutation_rate: np.float32,
    fittest_invidivuals: np.ndarray,
):
    for i in range(population_size):
        if i not in fittest_invidivuals:
            parent_1, parent_2 = _get_crossover_parents(
                popoulation, fittest_invidivuals
            )
            popoulation[i] = _crossover_parents(
                distance_matrix, parent_1, parent_2
            )
            _mutate(popoulation[i], mutation_rate)


def _get_crossover_parents(
    popoulation: np.ndarray, fittest_invidivuals: np.ndarray
):
    parents = np.random.choice(fittest_invidivuals, 2, replace=False)
    return popoulation[parents[0]], popoulation[parents[1]]


def _crossover_parents(
    distance_matrix: np.ndarray, parent_1: np.ndarray, parent_2: np.ndarray
):
    gene_size = parent_1.size
    starting_cty = np.random.choice(gene_size, 1)[0]
    child = np.full(gene_size, -1, dtype=GENE_DTYPE)
    parent_1_starting_idx = np.where(parent_1 == starting_cty)[0][0]
    parent_2_starting_idx = np.where(parent_2 == starting_cty)[0][0]
    parent_1_reordered = np.concatenate(
        (
            parent_1[parent_1_starting_idx:],
            parent_1[:parent_1_starting_idx],
        ),
        axis=0,
    )
    parent_2_reordered = np.concatenate(
        (
            parent_2[parent_2_starting_idx:],
            parent_2[:parent_2_starting_idx],
        ),
        axis=0,
    )

    for i in range(gene_size):
        if (
            parent_1_reordered[i] not in child
            and parent_2_reordered[i] not in child
        ):
            child[i] = get_closest_destination(
                distance_matrix,
                child[i - 1],
                parent_1_reordered[i],
                parent_2_reordered[i],
            )

        elif parent_1_reordered[i] in child and parent_2_reordered[i] in child:
            not_in_child = get_destinations_not_in_route(gene_size, child)
            child[i] = get_closest_destination_from_available_routes(
                distance_matrix, child[i - 1], not_in_child
            )

        elif parent_1_reordered[i] in child:
            child[i] = parent_2_reordered[i]
        else:
            child[i] = parent_1_reordered[i]

    return child


def _mutate(individual: np.ndarray, mutation_rate: np.float32):
    invidivual_mutation_rate = np.random.uniform(0, 1, 1)[0]
    if invidivual_mutation_rate <= mutation_rate:
        genes_to_mutate = np.random.choice(individual.size, 2, replace=False)

        individual[genes_to_mutate[0]], individual[genes_to_mutate[1]] = (
            individual[genes_to_mutate[1]],
            individual[genes_to_mutate[0]],
        )
