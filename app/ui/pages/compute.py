import streamlit as st
from ..utils import Page
from stqdm import stqdm

from pytsp import Routing
from pytsp.util.distances import get_roundtrip


class Compute(Page):
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
            "Generations to compute",
            value=self.state.client_config["generation_number"],
            min_value=5,
            max_value=5000,
            step=5,
            help="Number of iterations to run the genetic algorithm",
        )
        self.state.client_config["generation_number"] = generation_number

        population_number = st.sidebar.slider(
            "Populations to create",
            value=self.state.client_config["population_number"],
            min_value=5,
            max_value=1000,
            step=1,
            help="Number of populations to generate and compute on each \
                iteration",
        )
        self.state.client_config["population_number"] = population_number

        population_size = st.sidebar.slider(
            "Individuals per population",
            value=self.state.client_config["population_size"],
            min_value=5,
            max_value=1000,
            step=5,
            help="Number of individuals (routes) to compute on each population",
        )
        self.state.client_config["population_size"] = population_size

        max_mutation_rate = st.sidebar.slider(
            "Max mutation rate",
            value=self.state.client_config["max_mutation_rate"],
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Maximum mutation rate value to be assigned randomly \
                to each population",
        )
        self.state.client_config["max_mutation_rate"] = max_mutation_rate

        uniform_population_mutation_rate = st.sidebar.checkbox(
            "Uniform population mutation rate",
            value=self.state.client_config["uniform_population_mutation_rate"],
            help="When selected, the same mutation rate is randomly selected \
                for all populations. If not selected, each population would \
                be assigned a different mutation rate",
        )
        self.state.client_config[
            "uniform_population_mutation_rate"
        ] = uniform_population_mutation_rate

        selection_threshold = st.sidebar.slider(
            "Selection threshold",
            value=self.state.client_config["selection_threshold"],
            min_value=0,
            max_value=self.state.client_config["population_size"] // 2,
            step=1,
            help="Number of individuals (routes) to be selected as 'fittests' \
                from each population in order to survive into the next \
                generation and use their genes for new individuals",
        )
        self.state.client_config["selection_threshold"] = selection_threshold

        enhanced_individuals = st.sidebar.slider(
            "Number of enhanced individuals",
            value=self.state.client_config["enhanced_individuals"],
            min_value=0,
            max_value=self.state.client_config["population_size"],
            step=1,
            help="Number of individuals to not be generated completly \
                randomly. While non-enhanced individuals are generated \
                randomly selecting cities from the list of available \
                destinations, enhanced individuals select a random starting \
                point and subsequent destinations are selected based on \
                nearest distance",
        )
        self.state.client_config["enhanced_individuals"] = enhanced_individuals

        compute_options = ["numba", "numpy"]
        compute = st.sidebar.selectbox(
            "Compute mode",
            options=compute_options,
            index=compute_options.index(self.state.client_config["compute"]),
            help="Numpy: pure numpy implementation serving as a baseline \
                of what could be achieved with 'base' python methods.\n\
                Numba: still pythonic, but leveraging the numba library \
                in order to compile optimized machine code",
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
                population_number=self.state.client_config["population_number"],
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
