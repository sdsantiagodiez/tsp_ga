import streamlit as st
from streamlit_folium import folium_static
from ..utils import Page
from pytsp.util.plot import Mapplot


class Page3(Page):
    def __init__(self, state):
        self.state = state

    def write(self):
        self.__build_static_content()
        self.__build_inputs()
        self.__build_outputs()

    def __build_static_content(self):
        st.title("Coffee Road directions")

    def __build_inputs(self):
        width = st.sidebar.slider(
            "Map width",
            value=self.state.client_config["width"],
            min_value=240,
            max_value=2000,
            step=1,
            key=1,
        )
        self.state.client_config["width"] = width

        height = st.sidebar.slider(
            "Map height",
            value=self.state.client_config["height"],
            min_value=240,
            max_value=2000,
            step=1,
        )
        self.state.client_config["height"] = height

        line_color = st.sidebar.color_picker(
            "Connecting lines color",
            value=self.state.client_config["line_color"],
        )
        self.state.client_config["line_color"] = line_color

        line_weight = st.sidebar.slider(
            "Select line weight",
            value=self.state.client_config["line_weight"],
            min_value=1,
            max_value=10,
            step=1,
        )
        self.state.client_config["line_weight"] = line_weight

        directional_arrows = st.sidebar.slider(
            "Amount of directional arrows between two coordinate points",
            value=self.state.client_config["directional_arrows"],
            min_value=1,
            max_value=10,
            step=1,
        )
        self.state.client_config["directional_arrows"] = directional_arrows

        directional_arrows_color = st.sidebar.color_picker(
            "Directional arrows color",
            value=self.state.client_config["directional_arrows_color"],
        )
        self.state.client_config[
            "directional_arrows_color"
        ] = directional_arrows_color

        directional_arrows_radius = st.sidebar.slider(
            "Select directional arrow radius",
            value=self.state.client_config["directional_arrows_radius"],
            min_value=1,
            max_value=10,
            step=1,
        )
        self.state.client_config[
            "directional_arrows_radius"
        ] = directional_arrows_radius

    def __get_map_to_plot(self):
        return Mapplot.plot_map(
            coordinates=self.state.client_config["route_coordinates"],
            width=self.state.client_config["width"],
            height=self.state.client_config["height"],
            line_color=self.state.client_config["line_color"],
            line_weight=self.state.client_config["line_weight"],
            line_name=self.state.client_config["line_name"],
            directional_arrows=self.state.client_config["directional_arrows"],
            directional_arrows_color=self.state.client_config[
                "directional_arrows_color"
            ],
            directional_arrows_radius=self.state.client_config[
                "directional_arrows_radius"
            ],
        )

    def __build_outputs(self):
        if self.state.client_config["route_coordinates"] is not None:
            with st.spinner("Plotting map..."):
                map_to_plot = self.__get_map_to_plot()
                folium_static(
                    fig=map_to_plot,
                    width=self.state.client_config["width"],
                    height=self.state.client_config["height"],
                )
