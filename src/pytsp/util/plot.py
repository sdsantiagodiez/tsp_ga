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
        rotation = Mapplot._get_degrees_approximation(
            coordinates[0], coordinates[1]
        )
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
    def _get_degrees_approximation(coordinates_origin, coordinates_destination):
        lat1 = math.radians(coordinates_origin[0])
        lat2 = math.radians(coordinates_destination[0])

        diffLong = math.radians(
            coordinates_destination[1] - coordinates_origin[1]
        )

        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
        )

        initial_bearing = math.atan2(x, y)

        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    @staticmethod
    def _get_bounds(coordinates: np.ndarray):
        southests = coordinates.transpose()[0].min()
        northests = coordinates.transpose()[0].max()
        westests = coordinates.transpose()[1].min()
        eastest = coordinates.transpose()[1].max()

        sw = [southests, westests]
        ne = [northests, eastest]
        return [sw, ne]
