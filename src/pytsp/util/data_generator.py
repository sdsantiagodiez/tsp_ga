import pandas as pd
import numpy as np
from geopy.distance import geodesic

CITIES_DATA_PATH: str = "../../data/starbucks_us_locations.csv"


class DataGenerator(object):
    MIN_NUM_CITIES: int = 5

    def __init__(
        self,
        num_cities: int = 10,
        seed: int = 42,
        allow_repeating_cities: bool = False,
        cities_data_path: str = CITIES_DATA_PATH,
    ):
        self.all_cities = self.__get_all_cities(cities_data_path)
        self.__set_num_cities(num_cities)
        self.generate_new_cities_selection(
            num_cities, seed, allow_repeating_cities
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

    def __get_all_cities(self, cities_data_path: str = CITIES_DATA_PATH):
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
        self, seed: int = 42, allow_repeating_cities: bool = False
    ):
        city_columns = self.all_cities.columns.tolist()
        selected_cities = pd.DataFrame(columns=city_columns)
        random_state = seed
        for city in range(self.num_cities):
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

    def __generate_distances(self, distance_type: str = "geodesic"):
        cities_distance = np.full(
            (self.num_cities, self.num_cities), np.inf, dtype=np.int64
        )

        for origin_index, origin in self.selected_cities.iterrows():
            for (
                destination_index,
                destination,
            ) in self.selected_cities.iterrows():
                if origin_index != destination_index:
                    cities_distance[
                        origin_index, destination_index
                    ] = self.__get_distance(
                        origin["coordinates"],
                        destination["coordinates"],
                        distance_type=distance_type,
                    )

        return cities_distance

    def __get_distance(
        self,
        origin_coordinates,
        destination_coordinates,
        distance_type: str = "geodesic",
    ):
        distance = np.inf
        if distance_type == "geodesic":
            distance = geodesic(
                origin_coordinates, destination_coordinates
            ).meters

        return distance

    def generate_new_cities_selection(
        self,
        num_cities: int = None,
        seed: int = 42,
        allow_repeating_cities: bool = False,
    ) -> None:
        if num_cities is not None:
            self.__set_num_cities(num_cities)
        self._selected_cities = self.__get_selected_cities(
            seed, allow_repeating_cities
        )
        self._distances = self.__generate_distances()
