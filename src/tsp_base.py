import sys
import numpy as np
from DataGenerator import DataGenerator

sys.path.append(".")  # until structured as package


class TSP(object):
    def __init__(
        self,
        generations: int,
        populations: int,
        num_cities: int = 10,
        seed: int = 42,
    ):
        self.generations = generations
        self.populations = populations
        self.city_data = DataGenerator(num_cities, seed)
        self.gene_length = self.city_data.num_cities

    @property
    def generations(self):
        return self._generations

    @generations.setter
    def generations(self, value: int):
        if value < 1:
            raise ValueError("Generations can't be less than 1")
        self._generations = value

    @property
    def populations(self):
        return self._populations

    @populations.setter
    def populations(self, value: int):
        if value < 1:
            raise ValueError("Populations can't be less than 1")
        self._populations = value

    def __calculate_fitness(self, gene, distances: np.ndarray):
        one_hot_distances = np.zeros((self.gene_length, self.gene_length))
        for i in range(self.gene_length - 1):
            one_hot_distances[gene[i], gene[i + 1]] = 1
        one_hot_distances[gene[self.gene_length - 1], gene[0]] = 1

        return np.nansum(one_hot_distances * distances)

    def run(self):
        distance = self.__calculate_fitness(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], self.city_data.distances
        )
        print(f"test distance is: {distance}")
        # raise NotImplemented("run not implemented")


def main():
    tsp_base = TSP(generations=1, populations=1, num_cities=10, seed=42)
    tsp_base.run()


if __name__ == "__main__":
    main()
