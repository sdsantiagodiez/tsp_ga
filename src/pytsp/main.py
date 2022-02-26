import sys
import argparse

sys.path.append(".")  # until structured as package

from util.data_generator import DataGenerator  # noqa: E402
from genetic_algorithm.base import TSP  # noqa: E402

__DEFAULT_VALUES = {
    "num-cities": 10,
    "seed-cities": 42,
    "generation-number": 10,
    "population-number": 20,
    "population-size": 20,
    "max-mutation-rate": 0.35,
    "uniform-population-mutation-rate": False,
    "selection-threshold": 0,
}


def main(
    num_cities: int,
    seed_cities: int,
    generation_number: int,
    population_number: int,
    population_size: int,
    max_mutation_rate: float,
    uniform_population_mutation_rate: bool,
    selection_threshold: int,
):
    print("Generating cities...")
    city_data = DataGenerator(num_cities=num_cities, seed=seed_cities)

    print("Running algorithm")
    tsp_base = TSP(
        city_data.distances,
        generation_number=generation_number,
        population_number=population_number,
        population_size=population_size,
        max_mutation_rate=max_mutation_rate,
        uniform_population_mutation_rate=uniform_population_mutation_rate,
        selection_threshold=selection_threshold,
    )
    tsp_base.run()


def get_args():
    parser = argparse.ArgumentParser(
        description="Run genetic algorithm on Travelling Salesman problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-cities",
        help="Number of cities to randomly generate",
        type=int,
        default=__DEFAULT_VALUES["num-cities"],
    )
    parser.add_argument(
        "--seed-cities",
        help="Seed used to randomly select and reproduce selection of city",
        type=int,
        default=__DEFAULT_VALUES["seed-cities"],
    )
    parser.add_argument(
        "--generation-number",
        help="Number of generations/iterations to run genetic algorithm",
        type=int,
        default=__DEFAULT_VALUES["generation-number"],
    )
    parser.add_argument(
        "--population-number",
        help="Number of populations to generate",
        type=int,
        default=__DEFAULT_VALUES["population-number"],
    )
    parser.add_argument(
        "--population-size",
        help="Number of routes to generate on each population",
        type=int,
        default=__DEFAULT_VALUES["population-size"],
    )
    parser.add_argument(
        "--max-mutation-rate",
        help="Max mutation rate for all populations",
        type=float,
        default=__DEFAULT_VALUES["max-mutation-rate"],
    )
    parser.add_argument(
        "--uniform-population-mutation-rate",
        help="Set same mutation rate for all populations",
        type=bool,
        default=__DEFAULT_VALUES["uniform-population-mutation-rate"],
    )
    parser.add_argument(
        "--selection-threshold",
        help="Number of individuals to select as fittest on each generation",
        type=int,
        default=__DEFAULT_VALUES["selection-threshold"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(
        num_cities=args.num_cities,
        seed_cities=args.seed_cities,
        generation_number=args.generation_number,
        population_number=args.population_number,
        population_size=args.population_size,
        max_mutation_rate=args.max_mutation_rate,
        uniform_population_mutation_rate=args.uniform_population_mutation_rate,
        selection_threshold=args.selection_threshold,
    )
