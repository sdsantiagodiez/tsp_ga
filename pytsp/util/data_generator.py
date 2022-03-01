import pandas as pd
import numpy as np
from importlib.resources import path
from tqdm import tqdm

from pytsp.util.distances import get_distance

DEFAULT_CITIES_DATA_PATH: str = path("pytsp.data", "starbucks_us_locations.csv")
DEFAULT_SEED: int = 42
DEFAULT_NUM_CITIES: int = 10
DEFAULT_ALLOW_REPEATING_CITIES: bool = False
DEFAULT_VERBOSE: bool = True


class DataGenerator(object):
    MIN_NUM_CITIES: int = 5

    def __init__(
        self,
        num_cities: int = DEFAULT_NUM_CITIES,
        seed: int = DEFAULT_SEED,
        allow_repeating_cities: bool = DEFAULT_ALLOW_REPEATING_CITIES,
        cities_data_path: str = DEFAULT_CITIES_DATA_PATH,
        verbose: bool = DEFAULT_VERBOSE,
    ):
        self.all_cities = self.__get_all_cities(cities_data_path)
        self.__set_num_cities(num_cities)
        self.generate_new_cities_selection(
            num_cities, seed, allow_repeating_cities, verbose
        )

    @property
    def num_cities(self):
        return self._num_cities

    def __set_num_cities(self, value: int):
        if value is None:
            raise ValueError("Number of cities can't be None")
        if value < self.MIN_NUM_CITIES:
            raise ValueError(
                f"Number of cities can't be less than {self.MIN_NUM_CITIES}"
            )
        self._num_cities = value

    @property
    def selected_cities(self):
        return self._selected_cities

    @property
    def distances(self):
        return self._distances

    def __get_all_cities(
        self, cities_data_path: str = DEFAULT_CITIES_DATA_PATH
    ):
        columns = ["longitude", "latitude", "id", "address"]
        cities = pd.read_csv(cities_data_path, names=columns, header=None)
        cities.dropna(inplace=True)
        cities[["state", "city"]] = cities.id.str.split("-", expand=True)[
            [1, 2]
        ]
        cities["city"] = cities["city"].str[:-7]
        cities["city"] = cities["city"].str.replace(r"\[.*", "", regex=True)
        cities["state_city"] = cities["state"] + "-" + cities["city"]
        cities["coordinates"] = list(
            zip(cities["latitude"], cities["longitude"])
        )

        return cities

    def __get_selected_cities(
        self,
        seed: int = DEFAULT_SEED,
        allow_repeating_cities: bool = DEFAULT_ALLOW_REPEATING_CITIES,
    ):
        exists_more_distinct_cities_than_selected = (
            len(self.all_cities["state_city"].unique()) > self.num_cities
        )
        if (
            not exists_more_distinct_cities_than_selected
            and not allow_repeating_cities
        ):
            raise ValueError("Available distinct cities are less than selected")

        city_columns = self.all_cities.columns.tolist()
        selected_cities = pd.DataFrame(columns=city_columns)
        random_state = seed
        for _ in range(self.num_cities):
            random_city = self.all_cities.sample(random_state=random_state)
            if not allow_repeating_cities:
                while (
                    random_city["state_city"]
                    .isin(selected_cities["state_city"])
                    .any()
                ):
                    random_state += 1
                    random_city = self.all_cities.sample(
                        random_state=random_state
                    )
            random_state += 1
            selected_cities = pd.concat([selected_cities, random_city])

        return selected_cities.reset_index(drop=True)

    def __generate_distances(
        self, distance_type: str = "geodesic", verbose: bool = DEFAULT_VERBOSE
    ):
        disable_tqdm = not verbose
        cities_distance = np.full(
            (self.num_cities, self.num_cities), np.inf, dtype=np.int64
        )

        for origin_index, origin_coordinates in tqdm(
            self.selected_cities["coordinates"].to_dict().items(),
            disable=disable_tqdm,
        ):
            for destination_index, destination_coordinates in (
                self.selected_cities["coordinates"].to_dict().items()
            ):
                if origin_index != destination_index:
                    cities_distance[
                        origin_index, destination_index
                    ] = get_distance(
                        origin_coordinates,
                        destination_coordinates,
                        distance_type,
                    )

        return cities_distance

    def generate_new_cities_selection(
        self,
        num_cities: int = None,
        seed: int = DEFAULT_SEED,
        allow_repeating_cities: bool = DEFAULT_ALLOW_REPEATING_CITIES,
        verbose: bool = DEFAULT_VERBOSE,
    ) -> None:
        if num_cities is not None:
            self.__set_num_cities(num_cities)
        self._selected_cities = self.__get_selected_cities(
            seed, allow_repeating_cities
        )
        self._distances = self.__generate_distances(verbose=verbose)
