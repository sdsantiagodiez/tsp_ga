import streamlit as st
from ..utils import Page
from stqdm import stqdm

from pytsp import Routing
from pytsp.util.distances import get_roundtrip


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
        generation_number = st.sidebar.slider(
            "Number of generations to compute",
            value=self.state.client_config["generation_number"],
            min_value=5,
            max_value=100,
            step=1,
        )
        self.state.client_config["generation_number"] = generation_number
        """
        population_number = st.sidebar.slider(
            "Number of populations",
            value=self.state.client_config["population_number"],
            min_value=5,
            max_value=1000,
            step=5,
        )
        self.state.client_config["population_number"] = population_number
        """
        population_size = st.sidebar.slider(
            "Individuals per population",
            value=self.state.client_config["population_size"],
            min_value=5,
            max_value=1000,
            step=5,
        )
        self.state.client_config["population_size"] = population_size

        max_mutation_rate = st.sidebar.slider(
            "Max mutation rate",
            value=self.state.client_config["max_mutation_rate"],
            min_value=0.0,
            max_value=1.0,
            step=0.01,
        )
        self.state.client_config["max_mutation_rate"] = max_mutation_rate

        uniform_population_mutation_rate = st.sidebar.checkbox(
            "Uniform population mutation rate",
            value=self.state.client_config["uniform_population_mutation_rate"],
        )
        self.state.client_config[
            "uniform_population_mutation_rate"
        ] = uniform_population_mutation_rate

        # selection_threshold = self.state.client_config["selection_threshold"],
        selection_threshold = st.sidebar.slider(
            "Selection threshold",
            value=self.state.client_config["selection_threshold"],
            min_value=0,
            max_value=self.state.client_config["population_size"] // 2,
            step=1,
        )
        self.state.client_config["selection_threshold"] = selection_threshold

        enhanced_individuals = st.sidebar.slider(
            "Number of enhanced individuals",
            value=self.state.client_config["enhanced_individuals"],
            min_value=0,
            max_value=self.state.client_config["population_size"],
            step=1,
        )
        self.state.client_config["enhanced_individuals"] = enhanced_individuals

        compute = st.sidebar.selectbox(
            "Compute mode",
            options=["numba", "numpy"]
            # add default
        )
        self.state.client_config["compute"] = compute

        self.__add_generate_button()

    def __add_generate_button(self):
        if st.sidebar.button("Compute best route"):
            self.__run_button()

    def __run_button(self):
        with st.spinner("Warming up genetic algorithm..."):
            routing = Routing(
                distances=self.state.client_config["distance_matrix"],
                generation_number=self.state.client_config["generation_number"],
                population_number=20,
                population_size=self.state.client_config["population_size"],
                max_mutation_rate=self.state.client_config["max_mutation_rate"],
                uniform_population_mutation_rate=self.state.client_config[
                    "uniform_population_mutation_rate"
                ],
                selection_threshold=self.state.client_config[
                    "selection_threshold"
                ],
                enhanced_individuals=self.state.client_config[
                    "enhanced_individuals"
                ],
                compute=self.state.client_config["compute"],
            )
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

            self.state.client_config["route_coordinates"] = get_roundtrip(
                routing.fittest_individual,
                self.state.client_config["selected_cities_coordinates"],
            )
            self.state.client_config["route_distance"] = routing.fittest

    def __build_outputs(self):
        if self.state.client_config["route_distance"] is not None:
            route_distance = self.state.client_config["route_distance"] / 1000
            st.success(
                f"Fastest calculated route would have a distance of "
                f"{route_distance:,} km"
            )
