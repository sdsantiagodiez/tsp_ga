from branca.element import Figure
from folium import Map
from folium import Marker
from folium import FeatureGroup
from folium.vector_layers import PolyLine
import numpy as np


class Mapplot(object):
    @staticmethod
    def plot_map(
        coordinates: np.ndarray,
        width: int = 400,
        height: int = 400,
        line_color: str = "blue",
        line_weight: int = 2,
    ):
        initial_location = coordinates[0]

        figure = Figure(width=width, height=height)
        map = Map(
            width=width,
            height=height,
            location=initial_location,
        )

        # map.fit_bounds(_get_bounds(coordinates))

        for i in range(len(coordinates)):

            Marker(tuple(coordinates[i])).add_to(map)

        feature_group_name = "benchmark path"
        feature = FeatureGroup(feature_group_name)
        # folium.RegularPolygonMarker(location=initial_location,
        #  fill_color='blue',
        # number_of_sides=3, radius=10, rotation=35).add_to(map) #arrow
        PolyLine(coordinates, color=line_color, weight=line_weight).add_to(
            feature
        )

        feature.add_to(map)

        figure.add_child(map)
        return figure

    @staticmethod
    def _get_bounds(coordinates: list):
        nw = coordinates.max()
        se = coordinates.min()

        return [se, nw]
