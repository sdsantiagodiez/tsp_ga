import sys
import numpy as np
from random import sample

from DataGenerator import DataGenerator

sys.path.append(".")  # until structured as package


class TSP(object):
    MIN_POPULATION_SIZE: int = 5
    MIN_POPULATION_NUMBER: int = 1
    MIN_GENERATION_NUMBER: int = 5

    def __init__(
        self,
        generation_number: int = 10,
        population_number: int = 5,
        population_size: int = 10,
        num_cities: int = 10,
        seed: int = 42,
    ):
        self.generation_number = generation_number
        self.population_number = population_number
        self.population_size = population_size
        self.city_data = DataGenerator(num_cities, seed)
        self.gene_size = self.city_data.num_cities

        self.__initialize_populations()

    @property
    def generation_number(self):
        return self._generation_number

    @generation_number.setter
    def generation_number(self, value: int):
        if value < self.MIN_GENERATION_NUMBER:
            value_error_message = "Population number can't be less than"
            raise ValueError(
                f"{value_error_message} {self.MIN_GENERATION_NUMBER}"
            )
        self._generation_number = value

    @property
    def population_number(self):
        return self._population_number

    @population_number.setter
    def population_number(self, value: int):
        if value < self.MIN_POPULATION_NUMBER:
            value_error_message = "Population number can't be less than"
            raise ValueError(
                f"{value_error_message} {self.MIN_POPULATION_NUMBER}"
            )
        self._population_number = value

    @property
    def population_size(self):
        return self._population_size

    @population_size.setter
    def population_size(self, value: int):
        if value < self.MIN_POPULATION_SIZE:
            raise ValueError(
                f"Population size can't be less than {self.MIN_POPULATION_SIZE}"
            )
        self._population_size = value

    def __get_new_population(self):
        population = np.full(
            (self.population_size, self.gene_size), -1, dtype=np.int8
        )
        for i in range(self.population_size):
            population[i] = sample(range(self.gene_size), self.gene_size)

        return population

    def __get_new_populations(self):
        populations = np.full(
            (self.population_number, self.population_size, self.gene_size),
            -1,
            dtype=np.int8,
        )
        for i in range(self.population_number):
            populations[i] = self.__get_new_population()

        return populations

    def __initialize_populations(self):
        self.populations = self.__get_new_populations()

    def __select_parents(self):
        pass

    def __crossover(self):
        pass

    def __calculate_fitness(self, gene, distances: np.ndarray):
        one_hot_distances = np.zeros(
            (self.gene_size, self.gene_size), dtype=np.float16
        )
        for i in range(self.gene_size - 1):
            one_hot_distances[gene[i], gene[i + 1]] = 1
        one_hot_distances[gene[self.gene_size - 1], gene[0]] = 1

        return 1 / np.nansum(one_hot_distances * distances)

    def __mutate(self, gene):
        pass

    def run(self):
        distance = self.__calculate_fitness(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], self.city_data.distances
        )
        print(f"test distance is: {distance}")
        # raise NotImplemented("run not implemented")


def main():
    tsp_base = TSP(
        generation_number=1, population_number=1, num_cities=10, seed=42
    )
    tsp_base.run()


if __name__ == "__main__":
    main()
