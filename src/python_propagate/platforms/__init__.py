"""
platform.py

This module contains the Platform class.

Classes:
- Platform: A class to represent a platform.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

import spiceypy as spice
import numpy as np

from python_propagate.utilities.units import RAD2DEG, DEG2RAD
from python_propagate.agents.state import State
from python_propagate.scenario import Scenario


class Platform:
    """
    A class to represent a platform.

    Attributes
    ----------
    latitude : float
        The latitude of the platform.
    longitude : float
        The longitude of the platform.
    state : dict
        The state of the platform containing position and velocity.
    """

    def __init__(self, lat_long_alt: tuple):
        """
        Constructs all the necessary attributes for the Platform object.

        Parameters
        ----------
        latitude : float
            The latitude of the platform in degrees
        longitude : float
            The longitude of the platform in degrees
        scenario : Scenario
            The scenario to which the platform is attached.
        """
        self._latitude = lat_long_alt[0] * DEG2RAD
        self._longitude = lat_long_alt[1] * DEG2RAD
        self._altitude = lat_long_alt[2]
        self.scenario = None
        self.state = None

    @property
    def latitude(self):
        """Gets the latitude of the platform."""
        return self._latitude

    @property
    def longitude(self):
        """Gets the longitude of the platform."""
        return self._longitude

    @property
    def altitude(self):
        """Gets the altitude of the platform."""
        return self._altitude
    
    def add_scenario(self,scenario: Scenario):
        """Adds and links the station to a scenario."""        
        self.scenario = scenario

    def calculate_state_ecef(self):
        """Calculates the initial state (position and velocity in the ECEF frame) using spiceypy."""

        curvature = self._scenario.central_body.radius / np.sqrt(
            1 - self._scenario.central_body.eccentricity * np.sin(self.latitude)
        )

        factor = curvature + self._altitude

        x_ecef = factor * np.cos(self._latitude) * np.cos(self._longitude)
        y_ecef = factor * np.cos(self._latitude) * np.sin(self._longitude)
        z_ecef = (
            (1 - self._scenario.central_body.eccentricity) * curvature + self._altitude
        ) * np.sin(self._latitude)

        state = State(
            position=np.array([x_ecef, y_ecef, z_ecef]),
            velocity=np.array([0, 0, 0]),
            frame="ECEF",
        )

        return state
