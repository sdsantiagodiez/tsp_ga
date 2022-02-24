import sys
import numpy as np
from DataGenerator import DataGenerator

sys.path.append(".")  # until structured as package


class TSP(object):
    __MIN_POPULATION_SIZE: int = 5
    __MIN_POPULATION_NUMBER: int = 1
    __MIN_GENERATION_NUMBER: int = 5
    __MIN_SELECTION_THRESHOLD: int = 2
    __MAX_SELECTION_THRESHOLD: int = 2

    def __init__(
        self,
        generation_number: int = 10,
        population_number: int = 5,
        population_size: int = 10,
        num_cities: int = 10,
        max_mutation_rate: float = 0.35,
        uniform_population_mutation_rate: bool = False,
        selection_threshold: int = 0,
        seed: int = 42,
    ):
        self.generation_number = generation_number
        self.population_number = population_number
        self.population_size = population_size
        self.city_data = DataGenerator(num_cities, seed)
        self.gene_size = self.city_data.num_cities
        self.max_mutation_rate = max_mutation_rate
        self.uniform_population_mutation_rate = uniform_population_mutation_rate
        self.__MAX_SELECTION_THRESHOLD = self.population_size // 2
        self.selection_threshold = (
            self.__MAX_SELECTION_THRESHOLD
            if not selection_threshold
            else selection_threshold
        )

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

    def __get_new_population(self):
        population = np.full(
            (self.population_size, self.gene_size), -1, dtype=np.int8
        )
        for i in np.arange(self.population_size).tolist():
            population[i] = np.random.choice(
                self.gene_size, self.gene_size, replace=False
            )

        return population

    def __get_new_populations(self):
        populations = np.full(
            (self.population_number, self.population_size, self.gene_size),
            -1,
            dtype=np.int8,
        )
        for i in np.arange(self.population_number).tolist():
            populations[i] = self.__get_new_population()

        return populations

    def __get_populations_mutation_rates(self):
        """
        When 'uniform_population_mutation_rate' is True, same random mutation
        rate is applied to all populations.

        When 'uniform_population_mutation_rate' is False, each population
        receives a random rate.

        In both cases mutation rates are bound between 0 and max_mutation_rate
        """
        population_mutation_rates = np.full(
            (self.population_number), 0, dtype=np.float16
        )
        if self.uniform_population_mutation_rate:
            population_mutation_rates = np.repeat(
                np.random.uniform(0, self.max_mutation_rate, (1)),
                self.population_number,
            )
        else:
            population_mutation_rates = np.random.uniform(
                0, self.max_mutation_rate, (self.population_number)
            )

        # allow adaptive mutation rates
        return population_mutation_rates

    def __initialize_populations(self):
        self.populations = self.__get_new_populations()
        self.populations_mutation_rate = self.__get_populations_mutation_rates()

    def __calculate_individual_fitness(self, gene: np.ndarray):
        one_hot_distances = np.zeros(
            (self.gene_size, self.gene_size), dtype=np.int8
        )
        for i in np.arange(self.gene_size - 1).tolist():
            one_hot_distances[gene[i], gene[i + 1]] = 1
        one_hot_distances[gene[self.gene_size - 1], gene[0]] = 1

        return np.nansum(one_hot_distances * self.city_data.distances)

    def __calculate_population_fitness(self, population: np.ndarray):
        population_fitness = np.full(self.population_size, 0, dtype=np.float32)
        for i in np.arange(self.population_size).tolist():
            population_fitness[i] = self.__calculate_individual_fitness(
                population[i]
            )

        return population_fitness

    def __calculate_fitness(self):
        fitness = np.zeros(
            (self.population_number, self.population_size), dtype=np.float32
        )
        for i in np.arange(self.population_number).tolist():
            fitness[i] = self.__calculate_population_fitness(
                self.populations[i]
            )

        return fitness

    def __selection_on_population(
        self, population: np.ndarray, population_fitness: np.ndarray
    ):
        """
        Selection strategy used is based on fitness to allow maximum
        parallelization. Roulette and tournament could be considered,
        although it's expected to not perform as fast

        """
        population_fitness_sorted = np.argsort(population_fitness)
        # fittest_individuals = population[
        #    population_fitness_sorted[: self.__MAX_SELECTION_THRESHOLD]
        # ]

        return population_fitness_sorted[: self.__MAX_SELECTION_THRESHOLD]

    def __get_crossover_parents(
        self, popoulation: np.ndarray, fittest_invidivuals: np.ndarray
    ):
        parents = np.random.choice(fittest_invidivuals, 2, replace=False)
        return popoulation[parents[0]], popoulation[parents[1]]

    def __crossover(
        self,
        popoulation: np.ndarray,
        mutation_rate: np.float16,
        fittest_invidivuals: np.ndarray,
    ):
        for i in np.arange(self.population_size).tolist():
            if i not in fittest_invidivuals:
                parent_1, parent_2 = self.__get_crossover_parents(
                    popoulation, fittest_invidivuals
                )
                popoulation[i] = self.__crossover_parents(parent_1, parent_2)
                self.__mutate(popoulation[i], mutation_rate)

    def __crossover_parents(self, parent_1: np.ndarray, parent_2: np.ndarray):
        starting_cty = np.random.choice(self.gene_size, 1)[0]
        child = np.full(self.gene_size, -1, dtype=np.int8)
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

        for i in np.arange(self.gene_size).tolist():
            if parent_1_reordered[i] == parent_2_reordered[i]:
                child[i] = parent_1_reordered[i]
            else:
                child[i] = self.__get_shortest_path(
                    child[i - 1], parent_1_reordered[i], parent_2_reordered[i]
                )
        return child

    def __get_shortest_path(
        self, origin: np.int8, destination_1: np.int8, destination_2: np.int8
    ):
        """
        By default the shortests path will be used as selection for
        which destination to choose. This could be change to other
        strategies
        """
        return (
            destination_1
            if self.city_data.distances[origin, destination_1]
            < self.city_data.distances[origin, destination_2]
            else destination_2
        )

    def __mutate(self, individual: np.ndarray, mutation_rate: np.float16):
        invidivual_mutation_rate = np.random.uniform(0, 1, 1)[0]
        if invidivual_mutation_rate <= mutation_rate:
            genes_to_mutate = np.random.choice(self.gene_size, 2, replace=False)
            individual[[genes_to_mutate[0], genes_to_mutate[1]]] = individual[
                [genes_to_mutate[1], genes_to_mutate[0]]
            ]

    def run(self):
        self.__initialize_populations()

        for _ in np.arange(self.generation_number).tolist():
            fitness = self.__calculate_fitness()
            print(fitness)
            for i in np.arange(self.population_number).tolist():
                population = self.populations[i]
                population_fitness = fitness[i]
                mutation_rate = self.populations_mutation_rate[i]
                fittest_individuals = self.__selection_on_population(
                    population, population_fitness
                )
                self.__crossover(population, mutation_rate, fittest_individuals)


def main():
    tsp_base = TSP()
    tsp_base.run()


if __name__ == "__main__":
    main()
