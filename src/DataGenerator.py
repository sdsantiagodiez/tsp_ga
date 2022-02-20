import pandas as pd


class DataGenerator(object):
    MIN_NUM_CITIES: int = 5
    CITIES_DATA_PATH: str = "../data/starbucks_us_locations.csv"

    def __init__(self, num_cities: int = 10):
        self.num_cities = num_cities
        self.all_cities = self.__get_all_cities()
        self.selected_cities = self.__get_selected_cities()
        self.distances = self.__generate_random_distances()

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
        cities = pd.read_csv(
            "../data/starbucks_us_locations.csv", names=columns, header=None
        )
        cities.dropna(inplace=True)
        cities[["state", "city"]] = cities.id.str.split("-", expand=True)[
            [1, 2]
        ]
        cities["city"] = cities["city"].str[:-7]
        cities["city"] = cities["city"].str.replace(r"\[.*", "", regex=True)
        cities["state_city"] = cities["state"] + "-" + cities["city"]

        return cities

    def __get_selected_cities(self):

        return None

    def __generate_random_distances(self):

        for city in range(self.num_cities):
            pass

    def get_distances(self, new_generation: bool = False):
        if new_generation:
            self.distances = self.__generate_random_distances()

        return self.distances


def main():
    dg = DataGenerator()
    print(f"DataGenerator class with {dg.num_cities} cities.")


if __name__ == "__main__":
    main()
