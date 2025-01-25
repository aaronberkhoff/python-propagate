import spiceypy as spice
from python_propagate.Agents.state import State
from python_propagate.Scenario import Scenario
import numpy as np
from python_propagate.Utilities.units import RAD2DEG, DEG2RAD

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

    def __init__(self, latitude: float, longitude: float,scenario:Scenario, altitude: float = 0.0):
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
        self._latitude = latitude * DEG2RAD
        self._longitude = longitude * DEG2RAD
        self._altitude = altitude
        self._scenario = scenario
        self.state = self.calculate_state_ECEF()

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

    @property
    def time(self):
        """Gets the current time from the scenario."""
        return self._scenario.start_time


    def calculate_state_ECEF(self):
        """Calculates the initial state (position and velocity in the ECEF frame) using spiceypy."""

        curvature = self._scenario.central_body.radius / np.sqrt(1 - self._scenario.central_body.eccentricity * np.sin(self.latitude))
        factor = (curvature + self._altitude)

        x_ecef = factor * np.cos(self._latitude) * np.cos(self._longitude)
        y_ecef = factor * np.cos(self._latitude) * np.sin(self._longitude)
        z_ecef = ((1 - self._scenario.central_body.eccentricity) * curvature + self._altitude) * np.sin(self._latitude)

        state = State(position=np.array([x_ecef,y_ecef,z_ecef]),
                      velocity=np.array([0,0,0]),frame='ECEF')
        
        return state

# Ensure the file is not empty
pass