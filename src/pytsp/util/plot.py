from branca.element import Figure
from folium import Map
from folium import Marker
from folium import FeatureGroup
from folium.vector_layers import PolyLine
from folium import RegularPolygonMarker
import numpy as np
import math


class Mapplot(object):
    @staticmethod
    def plot_map(
        coordinates: np.ndarray,
        width: int = 400,
        height: int = 400,
        line_color: str = "blue",
        line_weight: int = 2,
    ):
        initial_location = tuple(coordinates[0])

        figure = Figure(width=width, height=height)
        map = Map(
            width=width,
            height=height,
            location=initial_location,
        )

        for i in range(len(coordinates)):

            Marker(tuple(coordinates[i])).add_to(map)

        feature_group_name = "benchmark path"
        feature = FeatureGroup(feature_group_name)
        rotation = Mapplot.__get_rotation_angle(coordinates[0], coordinates[1])
        RegularPolygonMarker(
            location=initial_location,
            fill_color="blue",
            number_of_sides=3,
            radius=10,
            rotation=rotation,
        ).add_to(map)
        PolyLine(coordinates, color=line_color, weight=line_weight).add_to(
            feature
        )

        feature.add_to(map)
        figure.add_child(map)

        map.fit_bounds(Mapplot._get_bounds(coordinates))
        return figure

    @staticmethod
    def __get_rotation_angle(coordinates_origin, coordinates_destination):
        dy = coordinates_destination[0] - coordinates_origin[0]
        dx = math.cos(math.pi / 180 * coordinates_origin[0]) * (
            coordinates_destination[1] - coordinates_origin[1]
        )
        ang = (math.atan2(dy, dx) / math.pi) * -180

        return round(ang, 2)

    @staticmethod
    def _get_bounds(coordinates: np.ndarray):
        southests = coordinates.transpose()[0].min()
        northests = coordinates.transpose()[0].max()
        westests = coordinates.transpose()[1].min()
        eastest = coordinates.transpose()[1].max()

        sw = [southests, westests]
        ne = [northests, eastest]
        return [sw, ne]
