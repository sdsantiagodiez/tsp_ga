from abc import ABC
from abc import abstractmethod
from numpy import ndarray


class Compute(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod
    @abstractmethod
    def run_generation(
        distance_matrix: ndarray,
        populations: ndarray,
        populations_fitness: ndarray,
        populations_mutation_rate: ndarray,
        population_number: int,
        population_size: int,
        selection_threshold: int,
        disable_tqdm: bool,
    ):
        """Runs a single generation iteration"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_new_populations(
        distance_matrix: ndarray,
        population_number: int,
        population_size: int,
        gene_size: int,
        enhanced_individuals: int,
    ):
        """Generates a random population"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_populations_mutation_rates(
        population_number: int,
        uniform_population_mutation_rate: bool,
        max_mutation_rate: float,
    ):
        """Generates mutation rates for a population"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def calculate_fitness(
        distance_matrix: ndarray,
        populations: ndarray,
        population_number: int,
        population_size: int,
    ):
        """Calculates fitness defined by distance"""
        raise NotImplementedError
