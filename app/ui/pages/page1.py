import streamlit as st
from ..utils import Page
from pytsp import DataGenerator


class Page1(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        self.__build_static_content()
        self.build_inputs()

    def __build_static_content(self):
        st.title("Coffe Road Trip")

    def build_inputs(self):
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
            "Allow multiple stores in the same citiy",
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
            selected_starbucks_stores = city_data.selected_cities

            self.state.client_config[
                "selected_cities"
            ] = selected_starbucks_stores
            self.state.client_config["distance_matrix"] = city_data.distances
        st.success("Done!")
        st.dataframe(self.state.client_config["selected_cities"])
