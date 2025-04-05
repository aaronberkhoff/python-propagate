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

from python_propagate.utilities.constants import AU, C1, PHI 
from python_propagate.utilities.calculations import calc_shadow   

from python_propagate.agents import Agent

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
        color="red",
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
        self._color = color

    def __repr__(self):
        """
        Returns a string representation of the Station object.

        Returns
        -------
        str
            A string representation of the Station object.
        """
        return (
            f"Station(lat_long_alt={self.lat_long_alt}, scenario={self.scenario}, sensor={self.sensor!r}, "
            f"name={self.name!r}, altitude={self.altitude}, minimum_elevation_angle={self.minimum_elevation_angle}, "
            f"identity={self._identity}, color={self.color})"
        )

    @property
    def sensor(self):
        """Gets the sensor of the station."""
        return self._sensor

    @property
    def name(self):
        """Gets the name of the station."""
        return self._name

    @property
    def identity(self):
        """Gets the identity of the station."""
        return self._identity

    @property
    def color(self):
        """Gets the color of the station."""
        return self._color

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
    
    def calculate_light_flux(self, state, agent: Agent):
        """
        Calculate the light flux received at the station from the target agent.

        Parameters
        ----------
        state : State
            The state of the target agent.
        
        Returns
        -------
        float
            The light flux received at the station in W/m^2.
        """

        object_dict = {obj.name.lower(): obj for obj in agent.scenario.celestial_bodies}
        state_sun =  object_dict.get("sun").get_state(state.time)

        # Check for shadowing: if in shadow, return zero acceleration.
        shadow_bool, shadow_value = calc_shadow(
            state_agent=state,
            state_sun=state_sun,
            reference_body_radius=agent.scenario.central_body.radius
        )

        if shadow_bool:
            # If in shadow, return zero flux
            return 0.0, 0.0
        
        #check if day time
        self.state.time = state.time # ensure the time is set for the state of the station, this is required for the sun state to be correct
        shadow_bool, shadow_value = calc_shadow(
            state_agent=self.state,
            state_sun=state_sun,
            reference_body_radius=agent.scenario.central_body.radius
        )

        if not shadow_bool:
            # If not in shadow, return zero flux
            return 0.0, 0.0



        agent.bus.set_orientation(state)

        # Get the face normals and areas from the bus shape.
        normals = agent.bus.shape.face_normals
        areas = agent.bus.shape.area_faces

        # Compute the cosine of the angle between each face normal and the sunlight direction.
        # The sun direction is taken as the unit vector from the spacecraft toward the sun.
        sun_direction = state_sun.position / np.linalg.norm(state_sun.position)
        cos_theta = np.dot(normals, sun_direction)

        # Only consider faces exposed to the sun (cos_theta > 0)
        exposed = cos_theta > 0
        cos_theta = cos_theta[exposed]
        normals = normals[exposed]
        # Also filter the corresponding areas.
        areas_exposed = areas[exposed]

        # Retrieve the face properties for the exposed faces.
        # Note: self.agent.bus.shape.face_properties is a dict indexed by triangle index.
        # We convert the values to a NumPy array and then select the exposed indices.
        all_cs = np.array([value['Cs'] for key, value in agent.bus.shape.face_properties.items()])
        all_cd = np.array([value['Cd'] for key, value in agent.bus.shape.face_properties.items()])
        cs_data = all_cs[exposed]
        cd_data = all_cd[exposed]

        # Compute reflectivity model coefficients
        mus = 0.5 * cs_data         # Albedo coefficient for SRP
        nu = (1.0 / 3.0) * cd_data    
        btheta = 2 * nu * cos_theta + 4 * mus * cos_theta**2

        # Compute the vector from the spacecraft to the sun and its magnitude
        agent_to_sun = state_sun.position - state.position
        r_sun = np.linalg.norm(agent_to_sun)
        # agent_to_sun_unit = agent_to_sun / r_sun

        # Compute the distance in AU (note: r_sun is in meters)
        distance_au = (r_sun / AU)**2

        flux_from_satellite = PHI / distance_au * (
            btheta  + (1 - mus) * (cos_theta**2)
        ) * areas_exposed

        # === 1. Define the direction from the agent (satellite) to the ground station ===
        agent_to_station = self.state.position - state.position  # Vector from satellite to ground station
        r_station = np.linalg.norm(agent_to_station)
        agent_to_station_unit = agent_to_station / r_station

        # === 2. Project the reflected flux toward the station ===
        # Dot product of face normal and observer direction
        cos_phi = np.dot(normals, agent_to_station_unit)

        # Only faces that can "see" the ground station (cos_phi > 0)
        visible = cos_phi > 0
        cos_phi = cos_phi[visible]

        # Select only the visible faces' areas and flux
        areas_visible = areas_exposed[visible]
        flux_visible = flux_from_satellite[visible]

        
        # === 3. Assuming Lambertian reflection, flux emitted toward the observer scales with cos_phi ===
        # Sum the contributions from all visible faces
        # Lambertian: divide by pi to account for isotropic scattering
        flux_received = np.sum(
            (flux_visible * cos_phi) * areas_visible
        ) / (np.pi * r_station**2)

        v_band_magnitude = 3.6e-9 # Approximate V-band magnitude for the Sun in W/m^2, used for apparent magnitude calculation
        apparent_magnitude = -2.5 * np.log10(flux_received / v_band_magnitude)
        # === 4. Return total flux in W/m^2 at ground station ===
        return flux_received, apparent_magnitude

        
