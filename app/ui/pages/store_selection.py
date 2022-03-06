import streamlit as st
from ..utils import Page
from pytsp import DataGenerator


class StoreSelection(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        self.__build_static_content()
        self.__build_inputs()
        self.__build_outputs()

    def __build_static_content(self):
        st.title("Coffe Road Trip")
        st.caption(
            "Play around with the parameters on the left to randomly \
                select Starbuck's stores across the US"
        )

    def __build_inputs(self):
        number_of_stores = st.sidebar.slider(
            "Select number of Starbucks stores to visit",
            value=self.state.client_config["num_cities"],
            min_value=5,
            max_value=1000,
            step=1,
        )
        self.state.client_config["num_cities"] = number_of_stores

        seed_cities = st.sidebar.slider(
            "Select seed for random city selection",
            value=self.state.client_config["seed_cities"],
            min_value=5,
            max_value=1000,
            step=1,
        )
        self.state.client_config["seed_cities"] = seed_cities

        allow_repeating_cities = st.sidebar.checkbox(
            "Allow selecting multiple stores in the same citiy",
            value=self.state.client_config["allow_repeating_cities"],
        )
        self.state.client_config["population_number"] = allow_repeating_cities

        self.__add_generate_button()

    def __add_generate_button(self):
        if st.sidebar.button("Generate stores"):
            self.__run_button()

    def __run_button(self):
        with st.spinner("Randomly selecting stores to visit..."):
            city_data = DataGenerator(
                num_cities=self.state.client_config["num_cities"],
                seed=self.state.client_config["seed_cities"],
                allow_repeating_cities=self.state.client_config[
                    "allow_repeating_cities"
                ],
            )
            self.state.client_config[
                "selected_cities"
            ] = city_data.selected_cities
            self.state.client_config[
                "selected_cities_coordinates"
            ] = city_data.get_selected_cities_coordinates()
            self.state.client_config["distance_matrix"] = city_data.distances

    def __build_outputs(self):
        if self.state.client_config["selected_cities"] is not None:
            st.success("Done!")
            st.dataframe(self.state.client_config["selected_cities"])
