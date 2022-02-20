class DataGenerator(object):
    def __init__(self, num_cities=10):
        self.num_cities = num_cities

    @property
    def num_cities(self):
        return self._num_cities

    @num_cities.setter
    def num_cities(self, value):
        if value < 5:
            raise ValueError("Number of cities can't be less than 5")
        self._num_cities = value


def main():
    dg = DataGenerator()
    print(f"DataGenerator class with {dg.num_cities} cities.")


if __name__ == "__main__":
    main()
