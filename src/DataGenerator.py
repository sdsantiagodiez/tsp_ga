import pandas as pd
import numpy as np
from geopy.distance import geodesic


class DataGenerator(object):
    MIN_NUM_CITIES: int = 5
    CITIES_DATA_PATH: str = "../data/starbucks_us_locations.csv"
    DISTANCE_ROUNDING: int = 4

    def __init__(self, num_cities: int = 10):
        self.all_cities = self.__get_all_cities()
        self.num_cities = num_cities
        self.selected_cities = self.__get_selected_cities()
        self.distances = self.__generate_distances()

    @property
    def num_cities(self):
        return self._num_cities

    @num_cities.setter
    def num_cities(self, value: int):
        if value < self.MIN_NUM_CITIES:
            raise ValueError("Number of cities can't be less than 5")
        self._num_cities = value

    def __get_all_cities(self):
        columns = ["longitude", "latitude", "id", "address"]
        cities = pd.read_csv(self.CITIES_DATA_PATH, names=columns, header=None)
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

    def __get_distance(city_1, city_2):
        return geodesic(city_1["coordinates"], city_2["coordinates"]).kilometers

    def __get_selected_cities(self):
        city_columns = self.all_cities.columns.tolist()
        selected_cities = pd.DataFrame(columns=city_columns)
        for city in range(self.num_cities):
            random_city = self.all_cities.sample()
            while (
                random_city["state_city"]
                .isin(selected_cities["state_city"])
                .any()
            ):
                random_city = self.all_cities.sample()
            selected_cities = pd.concat([selected_cities, random_city])

        return selected_cities.reset_index(drop=True)

    def __generate_distances(self):
        cities_distance = np.full((self.num_cities, self.num_cities), np.inf)

        for origin_index, origin in self.selected_cities.iterrows():
            for (
                destination_index,
                destination,
            ) in self.selected_cities.iterrows():
                if origin_index != destination_index:
                    cities_distance[origin_index, destination_index] = round(
                        geodesic(
                            origin["coordinates"], destination["coordinates"]
                        ).kilometers,
                        self.DISTANCE_ROUNDING,
                    )

        return cities_distance

    def get_distances(self, new_generation: bool = False):
        if new_generation:
            self.distances = self.__generate_distances()

        return self.distances


def main():
    dg = DataGenerator()
    print(f"DataGenerator class with {dg.num_cities} cities.")


if __name__ == "__main__":
    main()
