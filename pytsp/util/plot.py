from branca.element import Figure
from folium import Map
from folium import Marker
from folium import FeatureGroup
from folium.vector_layers import PolyLine
from folium import RegularPolygonMarker
import numpy as np
import math


class Mapplot(object):
    __DEFAULTS = {
        "width": 400,
        "height": 400,
        "line_color": "red",
        "line_weight": 2,
        "line_name": "",
        "directional_arrows": 4,
        "directional_arrows_color": "blue",
        "directional_arrows_radius": 5,
    }

    @staticmethod
    def plot_map(
        coordinates: np.ndarray,
        width: int = __DEFAULTS["width"],
        height: int = __DEFAULTS["height"],
        line_color: str = __DEFAULTS["line_color"],
        line_weight: int = __DEFAULTS["line_weight"],
        line_name: str = __DEFAULTS["line_name"],
        directional_arrows: int = __DEFAULTS["directional_arrows"],
        directional_arrows_color: str = __DEFAULTS["directional_arrows_color"],
        directional_arrows_radius: int = __DEFAULTS[
            "directional_arrows_radius"
        ],
    ):
        initial_location = tuple(coordinates[0])

        map = Map(
            width=width,
            height=height,
            location=initial_location,
        )

        Mapplot._add_coordinates(map=map, coordinates=coordinates)

        Mapplot._add_route_lines(
            map=map,
            coordinates=coordinates,
            feature_group_name=line_name,
            line_color=line_color,
            line_weight=line_weight,
        )

        Mapplot._add_directional_arrows(
            map=map,
            coordinates=coordinates,
            number_between_each_location=directional_arrows,
            color=directional_arrows_color,
            radius=directional_arrows_radius,
        )

        map.fit_bounds(Mapplot._get_bounds(coordinates))

        return Mapplot._get_figure(map=map, width=width, height=height)

    @staticmethod
    def _add_coordinates(map: Map, coordinates: np.ndarray):
        for i in range(len(coordinates)):
            Marker(tuple(coordinates[i])).add_to(map)

    @staticmethod
    def _add_route_lines(
        map: Map,
        coordinates: np.ndarray,
        feature_group_name: str = __DEFAULTS["line_name"],
        line_color: str = __DEFAULTS["line_color"],
        line_weight: int = __DEFAULTS["line_weight"],
    ):
        feature = FeatureGroup(feature_group_name)
        PolyLine(coordinates, color=line_color, weight=line_weight).add_to(
            feature
        )

        feature.add_to(map)

    @staticmethod
    def _add_directional_arrows(
        map: Map,
        coordinates: np.ndarray,
        number_between_each_location: int = __DEFAULTS["directional_arrows"],
        color: str = __DEFAULTS["directional_arrows_color"],
        radius: int = __DEFAULTS["directional_arrows_radius"],
    ):
        for i in range(len(coordinates) - 1):
            origin = coordinates[i]
            destination = coordinates[i + 1]
            rotation = Mapplot.__get_rotation_angle(origin, destination)
            for i in range(1, number_between_each_location + 1):
                mid_point = Mapplot._get_mid_point(
                    origin, destination, i / (number_between_each_location + 1)
                )
                Mapplot._add_arrow(
                    map=map,
                    location=mid_point,
                    color=color,
                    radius=radius,
                    rotation=rotation,
                )

    @staticmethod
    def _add_arrow(
        map: Map,
        location: tuple,
        color: str = __DEFAULTS["directional_arrows_color"],
        radius: int = __DEFAULTS["directional_arrows_radius"],
        rotation: float = 0,
    ):
        number_of_sides = 3
        RegularPolygonMarker(
            location=location,
            color=color,
            number_of_sides=number_of_sides,
            radius=radius,
            rotation=rotation,
        ).add_to(map)

    @staticmethod
    def __get_rotation_angle(
        coordinates_origin: np.ndarray, coordinates_destination: np.ndarray
    ):
        dy = coordinates_destination[0] - coordinates_origin[0]
        dx = math.cos(math.pi / 180 * coordinates_origin[0]) * (
            coordinates_destination[1] - coordinates_origin[1]
        )
        ang = (math.atan2(dy, dx) / math.pi) * -180

        return round(ang, 2)

    @staticmethod
    def _get_mid_point(
        coordinates_origin: np.ndarray,
        coordinates_destination: np.ndarray,
        mid_point_proportion: float = 0.5,
    ):
        distance_between_points = Mapplot._get_distance(
            coordinates_origin, coordinates_destination
        )
        mid_point_distance = distance_between_points * mid_point_proportion

        mid_point = tuple(coordinates_origin)
        if distance_between_points > mid_point_distance:
            ratio = (
                distance_between_points - mid_point_distance
            ) / distance_between_points
            x = coordinates_destination[0] - ratio * (
                coordinates_destination[0] - coordinates_origin[0]
            )
            y = coordinates_destination[1] - ratio * (
                coordinates_destination[1] - coordinates_origin[1]
            )
            mid_point = tuple([x, y])

        return tuple(mid_point)

    @staticmethod
    def _get_distance(
        coordinates_origin: np.ndarray, coordinates_destination: np.ndarray
    ):
        x = coordinates_destination[0] - coordinates_origin[0]
        y = coordinates_destination[1] - coordinates_origin[1]

        return math.sqrt(pow(x, 2) + pow(y, 2))

    @staticmethod
    def _get_bounds(coordinates: np.ndarray):
        southests = coordinates.transpose()[0].min()
        northests = coordinates.transpose()[0].max()
        westests = coordinates.transpose()[1].min()
        eastest = coordinates.transpose()[1].max()

        sw = [southests, westests]
        ne = [northests, eastest]
        return [sw, ne]

    @staticmethod
    def _get_figure(
        map: Map,
        width: int = __DEFAULTS["width"],
        height: int = __DEFAULTS["height"],
    ):
        figure = Figure(width=width, height=height)
        figure.add_child(map)

        return figure
