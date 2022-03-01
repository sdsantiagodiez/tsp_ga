import streamlit as st
import numpy as np
from ..utils import Page
from pytsp.main import run


class Page1(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        self.__build_static_content()
        self.build_inputs()

    def __build_static_content(self):
        st.title("The Coffee Road")

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

        population_number = st.sidebar.slider(
            "Select number of individuals per population",
            value=self.state.client_config["population_number"],
            min_value=5,
            max_value=1000,
            step=1,
        )
        self.state.client_config["population_number"] = population_number

        self.__add_calculate_button()

    def __add_calculate_button(self):
        if st.sidebar.button("Calculate"):
            self.__run_button()
        else:
            pass

    def __run_button(self):
        with st.spinner("Calculating distances..."):
            city_data = run(seed_cities=self.state.client_config["seed_cities"])
        st.success("Done!")
        selected_starbucks_stores = city_data.selected_cities
        self.state.client_config["selected_cities"] = selected_starbucks_stores
        self.state.client_config["route"] = np.array(
            [k for k in selected_starbucks_stores["coordinates"]]
            + [selected_starbucks_stores["coordinates"][0]]
        )
        st.dataframe(self.state.client_config["selected_cities"])
