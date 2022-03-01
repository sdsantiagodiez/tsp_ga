import numpy as np
from tqdm import tqdm
from numba import jit
from numba import prange

from pytsp.util.distances import (
    get_a_fast_route,
    get_route_distance,
    get_destinations_not_in_route,
    get_closest_destination,
    get_closest_destination_from_available_routes,
)

GENE_DTYPE: type = np.int16
DISTANCES_DTYPE: type = np.int64


class TSP(object):
    __MIN_POPULATION_SIZE: int = 5
    __MIN_POPULATION_NUMBER: int = 1
    __MIN_GENERATION_NUMBER: int = 5
    __MIN_SELECTION_THRESHOLD: int = 2
    __MAX_SELECTION_THRESHOLD: int
    __MAX_ENHANCED_INDIVIDUALS: int

    def __init__(
        self,
        distances: np.ndarray,
        generation_number: int = 10,
        population_number: int = 5,
        population_size: int = 10,
        max_mutation_rate: float = 0.35,
        uniform_population_mutation_rate: bool = False,
        selection_threshold: int = 0,
        enhanced_individuals: int = 0,
    ):
        self.__initialize_distances(distances)
        self.generation_number = generation_number
        self.population_number = population_number
        self.population_size = population_size
        self.max_mutation_rate = max_mutation_rate
        self.uniform_population_mutation_rate = uniform_population_mutation_rate
        self.__MAX_SELECTION_THRESHOLD = self.population_size // 2
        self.__MAX_ENHANCED_INDIVIDUALS = self.population_size
        self.selection_threshold = (
            self.__MAX_SELECTION_THRESHOLD
            if not selection_threshold
            else selection_threshold
        )
        self.enhanced_individuals = enhanced_individuals

    @property
    def generation_number(self):
        return self._generation_number

    @generation_number.setter
    def generation_number(self, value: int):
        if value < self.__MIN_GENERATION_NUMBER:
            value_error_message = "Generation number can't be less than"
            raise ValueError(
                f"{value_error_message} {self.__MIN_GENERATION_NUMBER}"
            )
        self._generation_number = value

    @property
    def population_number(self):
        return self._population_number

    @population_number.setter
    def population_number(self, value: int):
        if value < self.__MIN_POPULATION_NUMBER:
            value_error_message = "Population number can't be less than"
            raise ValueError(
                f"{value_error_message} {self.__MIN_POPULATION_NUMBER}"
            )
        self._population_number = value

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value: int):
        if value < self.__MIN_POPULATION_SIZE:
            value_error_message = "Population size can't be less than"
            raise ValueError(
                f"{value_error_message} {self.__MIN_POPULATION_SIZE}"
            )
        self._population_size = value

    @property
    def max_mutation_rate(self):
        return self._max_mutation_rate

    @max_mutation_rate.setter
    def max_mutation_rate(self, value: float):
        if not (0 <= value <= 1):
            raise ValueError(
                "Max mutation rate should be a float between 0 and 1"
            )
        self._max_mutation_rate = value

    @property
    def selection_threshold(self):
        return self._selection_threshold

    @selection_threshold.setter
    def selection_threshold(self, value: float):
        if not (
            self.__MIN_SELECTION_THRESHOLD
            <= value
            <= self.__MAX_SELECTION_THRESHOLD
        ):
            value_error_message = "Selection threshold can't be between "
            raise ValueError(
                f"{value_error_message} {self.__MIN_SELECTION_THRESHOLD} \n"
                f"and {self.__MAX_SELECTION_THRESHOLD}"
            )
        self._selection_threshold = value

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, value: np.ndarray):
        value_shape = value.shape
        if len(value_shape) != 2 or value_shape[0] != value_shape[1]:
            raise ValueError(
                "2d with (N,N) shape matrix expected was not satisfied in \n"
                " the distance matrix"
            )

        self._distances = value

    @property
    def enhanced_individuals(self):
        return self._enhanced_individuals

    @enhanced_individuals.setter
    def enhanced_individuals(self, value: float):
        if not (0 <= value <= self.__MAX_ENHANCED_INDIVIDUALS):
            value_error_message = "Enhanced individuals should be between "
            raise ValueError(
                f"{value_error_message} 0 and {self.__MAX_SELECTION_THRESHOLD}"
            )
        self._enhanced_individuals = value

    def __initialize_distances(self, distances: np.ndarray):
        self.distances = distances
        self.gene_size = self.distances.shape[0]

    def run(self, verbose: bool = True):
        disable_tqdm = not verbose

        self.populations = _get_new_populations(
            self.distances,
            self.population_number,
            self.population_size,
            self.gene_size,
            self.enhanced_individuals,
        )

        self.populations_mutation_rate = _get_populations_mutation_rates(
            self.population_number,
            self.uniform_population_mutation_rate,
            self.max_mutation_rate,
        )

        fitness = _calculate_fitness(
            self.distances,
            self.populations,
            self.population_number,
            self.population_size,
        )
        print(f"Baseline fitness: {np.min(fitness):,} meters")
        for generation in range(self.generation_number):
            fitness = _run_generation(
                self.distances,
                self.populations,
                fitness,
                self.populations_mutation_rate,
                self.population_number,
                self.population_size,
                self.selection_threshold,
                disable_tqdm,
            )
            print(f"Gen {generation+1} fitness: {np.min(fitness):,} meters")


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


@jit(nopython=True, parallel=True)
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
    for i in prange(population_number):
        populations[i] = _get_new_population(
            distance_matrix, population_size, gene_size, enhanced_individuals
        )

    return populations


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
def _selection_on_population(
    population_fitness: np.ndarray, max_selection_threshold: int
):
    """
    Selection strategy used is based on fitness to allow maximum
    parallelization. Roulette and tournament could be considered,
    although it's expected to not perform as fast

    """
    return np.argsort(population_fitness)[:max_selection_threshold]


@jit(nopython=True)
def _calculate_population_fitness(
    distance_matrix: np.ndarray, population: np.ndarray, population_size: int
):
    population_fitness = np.full(population_size, 0, dtype=DISTANCES_DTYPE)
    for i in range(population_size):
        population_fitness[i] = get_route_distance(
            distance_matrix, population[i]
        )

    return population_fitness


@jit(nopython=True, parallel=True)
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
    for i in prange(population_number):
        fitness[i] = _calculate_population_fitness(
            distance_matrix, populations[i], population_size
        )

    return fitness


@jit(nopython=True, parallel=True)
def _crossover(
    distance_matrix: np.ndarray,
    popoulation: np.ndarray,
    population_size: int,
    mutation_rate: np.float32,
    fittest_invidivuals: np.ndarray,
):
    for i in prange(population_size):
        if i not in fittest_invidivuals:
            parent_1, parent_2 = _get_crossover_parents(
                popoulation, fittest_invidivuals
            )
            popoulation[i] = _crossover_parents(
                distance_matrix, parent_1, parent_2
            )
            _mutate(popoulation[i], mutation_rate)


@jit(nopython=True)
def _get_crossover_parents(
    popoulation: np.ndarray, fittest_invidivuals: np.ndarray
):
    parents = np.random.choice(fittest_invidivuals, 2, replace=False)
    return popoulation[parents[0]], popoulation[parents[1]]


@jit(nopython=True)
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


@jit(nopython=True)
def _mutate(individual: np.ndarray, mutation_rate: np.float32):
    invidivual_mutation_rate = np.random.uniform(0, 1, 1)[0]
    if invidivual_mutation_rate <= mutation_rate:
        genes_to_mutate = np.random.choice(individual.size, 2, replace=False)

        individual[genes_to_mutate[0]], individual[genes_to_mutate[1]] = (
            individual[genes_to_mutate[1]],
            individual[genes_to_mutate[0]],
        )
