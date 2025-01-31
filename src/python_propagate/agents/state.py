"""
state.py

This module contains the State class.

Classes:
- State: A class to represent the state of an agent.
- Station: A class to represent a station.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

import numpy as np
import spiceypy as spice

from python_propagate.utilities.transforms import cart2classical


TARGET = "EARTH"
ECI = "J2000"
ECEF = "ITRF93"


class State:
    """
    A class to represent the state of an agent.

    Attributes
    ----------
    position : array-like, optional
        The position vector of the agent (default is None).
    velocity : array-like, optional
        The velocity vector of the agent (default is None).
    acceleration : array-like, optional
        The acceleration vector of the agent (default is None).
    stm : array-like, optional
        The state transition matrix (default is None).
    stm_dot : array-like, optional
        The time derivative of the state transition matrix (default is None).
    time : datetime, optional
        The time of the state (default is None).
    dimension : int, optional
        The dimension of the state vector (default is 6).
    frame : str, optional
        The reference frame of the state ('inertial' or 'ECEF', default is 'inertial').

    Methods
    -------
    __init__(self, position=None, velocity=None, acceleration=None, stm=None, stm_dot=None, time=None, dimension=6, frame="inertial"):
        Initializes the State with the given parameters.
    position_eci(self):
        Returns the position in the ECI frame.
    position_ecef(self):
        Returns the position in the ECEF frame.
    velocity_eci(self):
        Returns the velocity in the ECI frame.
    velocity_ecef(self):
        Returns the velocity in the ECEF frame.
    radius(self):
        Returns the radius of the position vector.
    x_eci(self):
        Returns the x-component of the position vector in the ECI frame.
    y_eci(self):
        Returns the y-component of the position vector in the ECI frame.
    z_eci(self):
        Returns the z-component of the position vector in the ECI frame.
    x_ecef(self):
        Returns the x-component of the position vector in the ECEF frame.
    y_ecef(self):
        Returns the y-component of the position vector in the ECEF frame.
    z_ecef(self):
        Returns the z-component of the position vector in the ECEF frame.
    vx_eci(self):
        Returns the x-component of the velocity vector in the ECI frame.
    vy_eci(self):
        Returns the y-component of the velocity vector in the ECI frame.
    vz_eci(self):
        Returns the z-component of the velocity vector in the ECI frame.
    vx_ecef(self):
        Returns the x-component of the velocity vector in the ECEF frame.
    vy_ecef(self):
        Returns the y-component of the velocity vector in the ECEF frame.
    vz_ecef(self):
        Returns the z-component of the velocity vector in the ECEF frame.
    latitude(self):
        Returns the latitude of the position vector in the ECEF frame.
    longitude(self):
        Returns the longitude of the position vector in the ECEF frame.
    compile(self):
        Compiles the state vector.
    to_keplerian(self, mu):
        Converts the state vector to Keplerian elements.
    dot(self):
        Returns the time derivative of the state vector.
    """

    def __init__(
        self,
        position=None,
        velocity=None,
        acceleration=None,
        stm=None,
        stm_dot=None,
        time=None,
        dimension=6,
        frame="inertial",
    ):
        """
        Constructs all the necessary attributes for the State object.

        Parameters
        ----------
        position : array-like, optional
            The position vector of the agent (default is None).
        velocity : array-like, optional
            The velocity vector of the agent (default is None).
        acceleration : array-like, optional
            The acceleration vector of the agent (default is None).
        stm : array-like, optional
            The state transition matrix (default is None).
        stm_dot : array-like, optional
            The time derivative of the state transition matrix (default is None).
        time : datetime, optional
            The time of the state (default is None).
        dimension : int, optional
            The dimension of the state vector (default is 6).
        frame : str, optional
            The reference frame of the state ('inertial' or 'ECEF', default is 'inertial').
        """
        self.position = position  # Assume kilometers by default
        self.velocity = velocity  # Assume kilometers per second
        self.acceleration = acceleration
        self.frame = frame
        self.dimension = dimension
        self.stm = stm
        self.time = time
        self.stm_dot = stm_dot

    @property
    def position_eci(self):
        """
        Returns the position in the ECI (Earth-Centered Inertial) frame.

        Returns
        -------
        array-like
            The position vector in the ECI frame.
        """
        if self.frame == "inertial":
            position = self.position
        elif self.frame == "ECEF":
            et = spice.str2et(self.time.strftime("%Y-%m-%dT%H:%M:%S"))
            rotation_matrix = spice.pxform("ECEF", "ECI", et)
            position = rotation_matrix @ self.position
        else:
            raise ValueError(
                f"Frame <{self.frame}> is spelled wrong or is not supported"
            )

        return position

    @property
    def position_ecef(self):
        """
        Returns the position in the ECEF (Earth-Centered Earth-Fixed) frame.

        Returns
        -------
        array-like
            The position vector in the ECEF frame.
        """
        if self.frame == "ECEF":
            position = self.position
        elif self.frame == "inertial":
            et = spice.str2et(self.time.strftime("%Y-%m-%dT%H:%M:%S"))
            rotation_matrix = spice.pxform("ECI", "ECEF", et)
            position = rotation_matrix @ self.position
        else:
            raise ValueError(
                f"Frame <{self.frame}> is spelled wrong or is not supported"
            )

        return position

    @property
    def velocity_eci(self):
        """
        Returns the velocity in the ECI (Earth-Centered Inertial) frame.

        Returns
        -------
        array-like
            The velocity vector in the ECI frame.
        """
        if self.frame == "inertial":
            velocity = self.velocity
        elif self.frame == "ECEF":
            et = spice.str2et(self.time.strftime("%Y-%m-%dT%H:%M:%S"))
            rotation_matrix = spice.pxform("ECEF", "ECI", et)
            velocity = rotation_matrix @ self.velocity
        else:
            raise ValueError(
                f"Frame <{self.frame}> is spelled wrong or is not supported"
            )

        return velocity

    @property
    def velocity_ecef(self):
        """
        Returns the velocity in the ECEF (Earth-Centered Earth-Fixed) frame.

        Returns
        -------
        array-like
            The velocity vector in the ECEF frame.
        """
        if self.frame == "ECEF":
            velocity = self.velocity
        elif self.frame == "inertial":
            et = spice.str2et(self.time.strftime("%Y-%m-%dT%H:%M:%S"))
            rotation_matrix = spice.pxform("ECI", "ECEF", et)
            velocity = rotation_matrix @ self.velocity
        else:
            raise ValueError(
                f"Frame <{self.frame}> is spelled wrong or is not supported"
            )

        return velocity

    @property
    def radius(self):
        """
        Returns the radius of the position vector.

        Returns
        -------
        float
            The radius of the position vector.
        """
        return np.sqrt(
            self.position[0] ** 2 + self.position[1] ** 2 + self.position[2] ** 2
        )

    @property
    def x_eci(self):
        """
        Returns the x-component of the position vector in the ECI frame.

        Returns
        -------
        float
            The x-component of the position vector in the ECI frame.
        """
        return self.position_eci[0]

    @property
    def y_eci(self):
        """
        Returns the y-component of the position vector in the ECI frame.

        Returns
        -------
        float
            The y-component of the position vector in the ECI frame.
        """
        return self.position_eci[1]

    @property
    def z_eci(self):
        """
        Returns the z-component of the position vector in the ECI frame.

        Returns
        -------
        float
            The z-component of the position vector in the ECI frame.
        """
        return self.position_eci[2]

    @property
    def x_ecef(self):
        """
        Returns the x-component of the position vector in the ECEF frame.

        Returns
        -------
        float
            The x-component of the position vector in the ECEF frame.
        """
        return self.position_ecef[0]

    @property
    def y_ecef(self):
        """
        Returns the y-component of the position vector in the ECEF frame.

        Returns
        -------
        float
            The y-component of the position vector in the ECEF frame.
        """
        return self.position_ecef[1]

    @property
    def z_ecef(self):
        """
        Returns the z-component of the position vector in the ECEF frame.

        Returns
        -------
        float
            The z-component of the position vector in the ECEF frame.
        """
        return self.position_ecef[2]

    @property
    def vx_eci(self):
        """
        Returns the x-component of the velocity vector in the ECI frame.

        Returns
        -------
        float
            The x-component of the velocity vector in the ECI frame.
        """
        return self.velocity_eci[0]

    @property
    def vy_eci(self):
        """
        Returns the y-component of the velocity vector in the ECI frame.

        Returns
        -------
        float
            The y-component of the velocity vector in the ECI frame.
        """
        return self.velocity_eci[1]

    @property
    def vz_eci(self):
        """
        Returns the z-component of the velocity vector in the ECI frame.

        Returns
        -------
        float
            The z-component of the velocity vector in the ECI frame.
        """
        return self.velocity_eci[2]

    @property
    def vx_ecef(self):
        """
        Returns the x-component of the velocity vector in the ECEF frame.

        Returns
        -------
        float
            The x-component of the velocity vector in the ECEF frame.
        """
        return self.velocity_ecef[0]

    @property
    def vy_ecef(self):
        """
        Returns the y-component of the velocity vector in the ECEF frame.

        Returns
        -------
        float
            The y-component of the velocity vector in the ECEF frame.
        """
        return self.velocity_ecef[1]

    @property
    def vz_ecef(self):
        """
        Returns the z-component of the velocity vector in the ECEF frame.

        Returns
        -------
        float
            The z-component of the velocity vector in the ECEF frame.
        """
        return self.velocity_ecef[2]

    @property
    def latitude(self):
        """
        Returns the latitude of the position vector in the ECEF frame.

        Returns
        -------
        float
            The latitude of the position vector in the ECEF frame.
        """
        return np.arctan2(self.z_ecef, np.sqrt(self.x_ecef**2 + self.y_ecef**2))

    @property
    def longitude(self):
        """
        Returns the longitude of the position vector in the ECEF frame.

        Returns
        -------
        float
            The longitude of the position vector in the ECEF frame.
        """
        return np.arctan2(self.y_ecef, self.x_ecef)

    def compile(self):
        """
        Compiles the state vector.

        Returns
        -------
        array-like
            The compiled state vector.
        """
        if self.stm is not None:
            state_dot = np.hstack((self.position, self.velocity, self.stm.flatten()))
        else:
            state_dot = np.hstack((self.position, self.velocity))

        return state_dot

    def to_keplerian(self, mu):
        """
        Converts the state vector to Keplerian elements.

        Parameters
        ----------
        mu : float
            The gravitational parameter of the central body.

        Returns
        -------
        tuple
            The Keplerian elements (sma, ecc, inc, raan, arg, nu).
        """
        sma, ecc, inc, raan, arg, nu = cart2classical(self.compile(), mu)
        return sma, ecc, inc, raan, arg, nu

    def dot(self):
        """
        Returns the time derivative of the state vector.

        Returns
        -------
        array-like
            The time derivative of the state vector.
        """
        if self.stm is not None:
            state_dot = np.hstack(
                (self.velocity, self.acceleration, self.stm_dot.flatten())
            )
        else:
            state_dot = np.hstack((self.velocity, self.acceleration))

        return state_dot

    def extract_position(self):
        """
        Extracts the position components.

        Returns
        -------
        tuple
            The x, y, and z components of the position vector.
        """
        return self.position[0], self.position[1], self.position[2]

    def extract_velocity(self):
        """
        Extracts the velocity components.

        Returns
        -------
        tuple
            The x, y, and z components of the velocity vector.
        """
        return self.velocity[0], self.velocity[1], self.velocity[2]

    def update_acceleration_from_state(self, state):
        """
        Updates acceleration from another state.

        Parameters
        ----------
        state : State
            The state from which to update the acceleration.
        """
        if state.acceleration is not None:
            self.acceleration = self.acceleration + state.acceleration

        if state.stm_dot is not None:
            self.stm_dot = state.stm_dot
