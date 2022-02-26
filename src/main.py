import sys
from DataGenerator import DataGenerator
from tsp_base import TSP

sys.path.append(".")  # until structured as package


def main():
    num_cities = 10
    seed = 42
    city_data = DataGenerator(num_cities, seed)
    print(city_data.selected_cities)
    print(city_data.distances)
    tsp_base = TSP(
        city_data.distances,
        generation_number=10,
        population_number=20,
        population_size=20,
        max_mutation_rate=0.35,
        uniform_population_mutation_rate=False,
        selection_threshold=0,
    )
    tsp_base.run()


if __name__ == "__main__":
    main()
