"""
station.py

This module contains the Station class.

Classes:
- Station: A class to represent a station.

Author: Aaron Berkhoff
Date: 2025-01-30
"""

import numpy as np

from python_propagate.platforms import Platform
from python_propagate.scenario import Scenario


class Station(Platform):
    """
    A class to represent a station.

    Inherits from Platform.

    Attributes
    ----------
    sensor : str
        The sensor of the station.
    name : str
        The name of the station.
    minimum_elevation_angle : float
        The minimum elevation angle of the station.
    """

    def __init__(
        self,
        latlong: tuple,
        scenario: Scenario,
        sensor: str = "none",
        name: str = "none",
        altitude: float = 0.0,
        minimum_elevation_angle: float = 0.0,
        identity: int = 0,
    ):
        """
        Constructs all the necessary attributes for the Station object.

        Parameters
        ----------
        latitude : float
            The latitude of the station in degrees.
        longitude : float
            The longitude of the station in degrees.
        scenario : Scenario
            The scenario to which the station is attached.
        sensor : str
            The sensor of the station.
        name : str
            The name of the station.
        altitude : float, optional
            The altitude of the station (default is 0.0).
        minimum_elevation_angle : float, optional
            The minimum elevation angle of the station (default is 0.0).
        """
        super().__init__(latlong, scenario, altitude)
        self._sensor = sensor
        self._name = name
        self._minimum_elevation_angle = minimum_elevation_angle
        self._identity = identity

    @property
    def sensor(self):
        """Gets the sensor of the station."""
        return self._sensor

    @property
    def name(self):
        """Gets the name of the station."""
        return self._name

    @property
    def minimum_elevation_angle(self):
        """Gets the minimum elevation angle of the station."""
        return self._minimum_elevation_angle

    def calculate_range_and_range_rate_from_target(self, state):
        """Calculates the range and range rate from the station to the target agent."""
        diff_x = self.state.x_ecef - state.x_ecef
        diff_y = self.state.y_ecef - state.y_ecef
        diff_z = self.state.z_ecef - state.z_ecef

        rho = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        diff_vx = self.state.vx_ecef - state.vx_ecef
        diff_vy = self.state.vy_ecef - state.vy_ecef
        diff_vz = self.state.vz_ecef - state.vz_ecef

        rho_dot = (diff_x * diff_vx + diff_y * diff_vy + diff_z * diff_vz) / rho

        return rho, rho_dot

    def calculate_azimuth_and_elevation(self, state):
        """Calculates the azimuth and elevation angles from the station to the target"""
        diff_x = -(self.state.x_ecef - state.x_ecef)
        diff_y = -(self.state.y_ecef - state.y_ecef)
        diff_z = -(self.state.z_ecef - state.z_ecef)

        enu_matrix = np.array(
            [
                [-np.sin(self.state.longitude), np.cos(self.state.longitude), 0],
                [
                    -np.sin(self.state.latitude) * np.cos(self.state.longitude),
                    -np.sin(self.state.latitude) * np.sin(self.state.longitude),
                    np.cos(self.state.latitude),
                ],
                [
                    np.cos(self.state.latitude) * np.cos(self.state.longitude),
                    np.cos(self.state.latitude) * np.sin(self.state.longitude),
                    np.sin(self.state.latitude),
                ],
            ]
        )

        enu = enu_matrix @ np.array([diff_x, diff_y, diff_z])

        e, n, u = enu

        azimuth = np.arctan2(e, n) % (2 * np.pi)
        elevation = np.arcsin(u / np.sqrt(e**2 + n**2 + u**2))

        return azimuth, elevation

    def calculate_ra_and_dec(self, state):
        """Calculates the right ascension and declination angles from the station to the target"""
        dec = np.arcsin(state.z_eci / state.radius)
        ra = np.arctan2(state.y_eci, state.x_eci)

        return ra, dec
