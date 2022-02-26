import numpy as np

GENE_DTYPE: type = np.int16
DISTANCES_DTYPE: type = np.int64


class TSP(object):
    __MIN_POPULATION_SIZE: int = 5
    __MIN_POPULATION_NUMBER: int = 1
    __MIN_GENERATION_NUMBER: int = 5
    __MIN_SELECTION_THRESHOLD: int = 2
    __MAX_SELECTION_THRESHOLD: int

    def __init__(
        self,
        distances: np.ndarray,
        generation_number: int = 10,
        population_number: int = 5,
        population_size: int = 10,
        max_mutation_rate: float = 0.35,
        uniform_population_mutation_rate: bool = False,
        selection_threshold: int = 0,
    ):
        self.__initialize_distances(distances)
        self.generation_number = generation_number
        self.population_number = population_number
        self.population_size = population_size
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

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, value: np.ndarray):
        value_shape = value.shape
        if len(value_shape) != 2:
            raise ValueError(
                "2d with (N,N) shape matrix expected was not satisfied in \n"
                " the distance matrix"
            )
        elif value_shape[0] != value_shape[1]:
            raise ValueError(
                "2d with (N,N) shape matrix expected was not satisfied in \n"
                " the distance matrix"
            )

        self._distances = value

    def __initialize_distances(self, distances: np.ndarray):
        self.distances = distances
        self.gene_size = self.distances.shape[0]

    def __get_new_population(self):
        population = np.full(
            (self.population_size, self.gene_size), -1, dtype=GENE_DTYPE
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
            dtype=GENE_DTYPE,
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

        return np.nansum(one_hot_distances * self.distances)

    def __calculate_population_fitness(self, population: np.ndarray):
        population_fitness = np.full(
            self.population_size, 0, dtype=DISTANCES_DTYPE
        )
        for i in np.arange(self.population_size).tolist():
            population_fitness[i] = self.__calculate_individual_fitness(
                population[i]
            )

        return population_fitness

    def __calculate_fitness(self):
        fitness = np.zeros(
            (self.population_number, self.population_size),
            dtype=DISTANCES_DTYPE,
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
        child = np.full(self.gene_size, -1, dtype=GENE_DTYPE)
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
            if (
                parent_1_reordered[i] not in child
                and parent_2_reordered[i] not in child
            ):
                child[i] = self.__get_shortest_path(
                    child[i - 1],
                    parent_1_reordered[i],
                    parent_2_reordered[i],
                )
            elif (
                parent_1_reordered[i] in child
                and parent_2_reordered[i] in child
            ):
                all_cities = np.arange(self.gene_size)
                not_in_child = all_cities[~np.in1d(all_cities, child)]
                child[i] = self.__get_shortest_path_from_array(
                    child[i - 1], not_in_child
                )

            elif parent_1_reordered[i] in child:
                child[i] = parent_2_reordered[i]
            else:
                child[i] = parent_1_reordered[i]

        return child

    def __get_shortest_path_from_array(
        self, origin: GENE_DTYPE, destionations: np.ndarray
    ):
        shortests_path: DISTANCES_DTYPE = np.iinfo(DISTANCES_DTYPE).max
        for i in np.arange(destionations.size):
            if shortests_path > self.distances[origin, destionations[i]]:
                shortests_path = destionations[i]

        return shortests_path

    def __get_shortest_path(
        self,
        origin: GENE_DTYPE,
        destination_1: GENE_DTYPE,
        destination_2: GENE_DTYPE,
    ):
        """
        By default the shortests path will be used as selection for
        which destination to choose. This could be change to other
        strategies
        """
        return (
            destination_1
            if self.distances[origin, destination_1]
            < self.distances[origin, destination_2]
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

        for generation in np.arange(self.generation_number).tolist():
            fitness = self.__calculate_fitness()
            print(f"Shortest path gen {generation}: {np.min(fitness):,} meters")
            for i in np.arange(self.population_number).tolist():
                population = self.populations[i]
                population_fitness = fitness[i]
                mutation_rate = self.populations_mutation_rate[i]
                fittest_individuals = self.__selection_on_population(
                    population, population_fitness
                )
                self.__crossover(population, mutation_rate, fittest_individuals)
