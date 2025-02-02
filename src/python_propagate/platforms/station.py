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
        lat_long_alt: tuple,
        sensor: str = "none",
        name: str = "none",
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
        super().__init__(lat_long_alt)
        self._sensor = sensor
        self._name = name
        self._minimum_elevation_angle = minimum_elevation_angle
        self._identity = identity

    def __repr__(self):
        """
        Returns a string representation of the Station object.

        Returns
        -------
        str
            A string representation of the Station object.
        """
        return (f"Station(lat_long_alt={self.lat_long_alt}, scenario={self.scenario}, sensor={self.sensor!r}, "
                f"name={self.name!r}, altitude={self.altitude}, minimum_elevation_angle={self.minimum_elevation_angle}, "
                f"identity={self._identity})")

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
        diff_x = self.state.position_ecef[0] - state.position_ecef[0]
        diff_y = self.state.position_ecef[1] - state.position_ecef[1]
        diff_z = self.state.position_ecef[2] - state.position_ecef[2]

        rho = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        diff_vx = self.state.velocity_ecef[0] - state.velocity_ecef[0]
        diff_vy = self.state.velocity_ecef[1] - state.velocity_ecef[1]
        diff_vz = self.state.velocity_ecef[2] - state.velocity_ecef[2]

        rho_dot = (diff_x * diff_vx + diff_y * diff_vy + diff_z * diff_vz) / rho

        return rho, rho_dot

    def calculate_azimuth_and_elevation(self, state):
        """Calculates the azimuth and elevation angles from the station to the target"""
        diff_x = -(self.state.position_ecef[0] - state.position_ecef[0])
        diff_y = -(self.state.position_ecef[1] - state.position_ecef[1])
        diff_z = -(self.state.position_ecef[2] - state.position_ecef[2])

        enu_matrix = np.array(
            [
                [-np.sin(self.state.latlong[1]), np.cos(self.state.latlong[1]), 0],
                [
                    -np.sin(self.state.latlong[0]) * np.cos(self.state.latlong[1]),
                    -np.sin(self.state.latlong[0]) * np.sin(self.state.latlong[1]),
                    np.cos(self.state.latlong[0]),
                ],
                [
                    np.cos(self.state.latlong[0]) * np.cos(self.state.latlong[1]),
                    np.cos(self.state.latlong[0]) * np.sin(self.state.latlong[1]),
                    np.sin(self.state.latlong[0]),
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
        dec = np.arcsin(state.position_eci[2] / np.linalg.norm(state.position_eci))
        ra = np.arctan2(state.position_eci[1], state.position_eci[2])

        return ra, dec
    
