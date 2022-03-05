import streamlit as st
from ..utils import Page
from stqdm import stqdm
import numpy as np

from pytsp import Routing


class Page2(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        self.__build_static_content()
        self.__build_inputs()
        self.__build_outputs()

    def __build_static_content(self):
        st.title("Compute the best route")
        st.caption(
            "Play around with the values on the left in order to tell \
            the genetic algorithm working in the background how it should \
            calculate the optimal route for your caffeinic road trip ;)"
        )

    def __build_inputs(self):

        self.__add_generate_button()

    def __add_generate_button(self):
        if st.sidebar.button("Compute best route"):
            self.__run_button()

    def __run_button(self):
        with st.spinner("Warming up genetic algorithm..."):
            distances = self.state.client_config["distance_matrix"]
            routing = Routing(distances=distances)
            routing.initialize_population()

        with st.spinner("Visiting stores and calculating distances..."):
            text_area_shortest_distance = st.empty()
            text_area_computed_routes = st.empty()
            routes_evaluated = 0
            for generation in stqdm(
                range(routing.generation_number), desc="Computing distances..."
            ):
                routing.run_generation(disable_tqdm=True)
                routes_evaluated += (
                    routing.population_number * routing.population_size
                )

                text_area_shortest_distance.text(
                    f"Generation {generation} "
                    f"shortest distance: {routing.fittest/1000:,} km"
                )

                text_area_computed_routes.text(
                    f"Routes evaluated: " f"{routes_evaluated:,}"
                )

            self.state.client_config["route"] = routing.fittest_individual
            self.state.client_config[
                "route_coordinates"
            ] = self.__get_route_coordinates()
            self.state.client_config["route_distance"] = routing.fittest

    def __get_route_coordinates(self):
        route = self.state.client_config["route"]
        coordinates = self.state.client_config["selected_cities_coordinates"]
        route_coordinates = np.zeros((len(route) + 1, 2))
        for i in range(len(route)):
            route_coordinates[i] = coordinates[route[i]]
        route_coordinates[len(route)] = coordinates[route[0]]

        return route_coordinates

    def __build_outputs(self):
        if self.state.client_config["route"] is not None:
            route_distance = self.state.client_config["route_distance"] / 1000
            st.success(
                f"Fastest calculated route would have a distance of "
                f"{route_distance:,} km"
            )
