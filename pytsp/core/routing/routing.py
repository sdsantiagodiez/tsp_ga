import numpy as np
from pytsp import ComputeFactory

GENE_DTYPE: type = np.int16
DISTANCES_DTYPE: type = np.int64


class Routing(object):
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
        compute: str = "numba",
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

        self.compute = ComputeFactory().get_compute(compute)

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

    def initialize_population(self):
        self.populations = self.compute.get_new_populations(
            self.distances,
            self.population_number,
            self.population_size,
            self.gene_size,
            self.enhanced_individuals,
        )

        self.populations_mutation_rate = (
            self.compute.get_populations_mutation_rates(
                self.population_number,
                self.uniform_population_mutation_rate,
                self.max_mutation_rate,
            )
        )

        self.fitness = self.compute.calculate_fitness(
            self.distances,
            self.populations,
            self.population_number,
            self.population_size,
        )

    def __get_fittest_path(self):
        return self.populations[0][0]  # to be replaced

    def run_generation(self, disable_tqdm: bool = False):
        self.fitness = self.compute.run_generation(
            self.distances,
            self.populations,
            self.fitness,
            self.populations_mutation_rate,
            self.population_number,
            self.population_size,
            self.selection_threshold,
            disable_tqdm,
        )
        self.fittest = np.min(self.fitness)
        self.fittest_path = self.__get_fittest_path()

    def run(self, verbose: bool = True):
        disable_tqdm = not verbose

        self.initialize_population()

        print(f"Baseline fitness: {np.min(self.fitness):,} meters")
        for generation in range(self.generation_number):
            self.run_generation(disable_tqdm)
            print(
                f"Gen {generation+1} fitness: {np.min(self.fitness):,} meters"
            )
