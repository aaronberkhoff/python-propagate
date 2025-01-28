from python_propagate.Platforms import Platform
from python_propagate.Scenario import Scenario
from python_propagate.Agents import Agent
import numpy as np


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

    def __init__(self, latitude: float, longitude: float, 
                 scenario: Scenario, 
                 sensor: str = None, 
                 name: str = None, 
                 altitude: float = 0.0, 
                 minimum_elevation_angle: float = 0.0,
                 id:int = None):
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
        super().__init__(latitude, longitude, scenario, altitude)
        self._sensor = sensor
        self._name = name
        self._minimum_elevation_angle = minimum_elevation_angle
        self._id = id

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
        diff_x = self.state.x_ECEF - state.x_ECEF
        diff_y = self.state.y_ECEF - state.y_ECEF
        diff_z = self.state.z_ECEF - state.z_ECEF

        range = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        diff_vx = self.state.vx_ECEF - state.vx_ECEF
        diff_vy = self.state.vy_ECEF - state.vy_ECEF
        diff_vz = self.state.vz_ECEF - state.vz_ECEF

        range_rate = (diff_x * diff_vx + diff_y * diff_vy + diff_z * diff_vz) / range

        return range, range_rate
    
    def calculate_azimuth_and_elevation(self, state):
        """Calculates the azimuth and elevation angles from the station to the target """
        diff_x = - (self.state.x_ECEF - state.x_ECEF)
        diff_y = - (self.state.y_ECEF - state.y_ECEF)
        diff_z = - (self.state.z_ECEF - state.z_ECEF)

        # range = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        # enu_matrix = np.array([
        # [-np.sin(self.state.longitude), np.cos(self.state.longitude), 0],
        # [-np.cos(self.state.longitude)*np.sin(self.state.latitude), -np.sin(self.state.longitude)*np.sin(self.state.latitude), np.cos(self.state.latitude)],
        # [np.cos(self.state.longitude)*np.cos(self.state.latitude), np.sin(self.state.longitude)*np.cos(self.state.latitude), np.sin(self.state.latitude)]
        # ])
        enu_matrix = np.array([
        [-np.sin(self.state.longitude), np.cos(self.state.longitude), 0],
        [-np.sin(self.state.latitude)*np.cos(self.state.longitude), -np.sin(self.state.latitude)*np.sin(self.state.longitude), np.cos(self.state.latitude)],
        [np.cos(self.state.latitude)*np.cos(self.state.longitude), np.cos(self.state.latitude)*np.sin(self.state.longitude), np.sin(self.state.latitude)]
        ])

        enu = enu_matrix @ np.array([diff_x, diff_y, diff_z])

        e, n, u = enu
        range = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
        
        azimuth = np.arctan2(e, n) % (2*np.pi)
        elevation = np.arcsin(u / np.sqrt(e**2 + n**2 + u**2)) 

        # range = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

        # elevation = np.arcsin(diff_z / range) 
        # azimuth = np.arctan2(diff_y, diff_x) 


        return azimuth, elevation
    
    def calculate_ra_and_dec(self,state):

        dec = np.arcsin(state.z_ECI/ state.radius)
        ra = np.arctan2(state.y_ECI, state.z_ECI)

        return ra,dec



