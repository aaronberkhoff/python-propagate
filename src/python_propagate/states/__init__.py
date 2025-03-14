"""
state.py

This module contains the State class.

Classes:
- State: A class to represent the state of an agent.
- Station: A class to represent a station.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

from collections import namedtuple

import numpy as np
import spiceypy as spice

from python_propagate.utilities.transforms import cart2classical, classical2cart


# TODO: Remove hard coded TARGET
TARGET = "EARTH"
ECI = "J2000"
ECEF = "ITRF93"
MU = 398600.4415

OrbitalElements = namedtuple(
    "OrbitalElements", ["sma", "ecc", "inc", "arg", "raan", "nu"]
)


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
    latlong(self):
        Returns the latitude and longitude of the position vector.
    compile(self):
        Compiles the state vector.
    to_keplerian(self, mu):
        Converts the state vector to Keplerian elements.
    to_cartesian(self, mu):
        Converts the state vector to Cartesian coordinates.
    dot(self):
        Returns the time derivative of the state vector.
    """

    def __init__(self, frame = "inertial",**kwargs):
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
        orbital_elements : tuple, optional
            The orbital elements of the agent (default is None).
        """
        self.stm = None
        self.frame = frame

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __add__(self, other):
        if not isinstance(other, State):
            return NotImplemented

        new_state = State(**vars(self))  # Copy attributes to new instance

        for key in vars(self).keys() | vars(other).keys():  # Union of keys from both
            self_val = getattr(self, key, None)
            other_val = getattr(other, key, None)

            if self_val is None and other_val is None:
                setattr(new_state, key, None)  # Keep None if both are None
            elif self_val is None:
                setattr(new_state, key, other_val)  # Take other if self is None
            elif other_val is None:
                setattr(new_state, key, self_val)  # Take self if other is None
            else:
                setattr(new_state, key, self_val + other_val)  # Normal addition

        return new_state

    def __iadd__(self, other):
        result = self + other  # Use __add__ logic
        for key, value in vars(result).items():
            setattr(self, key, value)
        return self

    def __matmul__(self, other):
        if not isinstance(other, State):
            return NotImplemented  # Ensures correct behavior with unsupported types

        # Check if both states have valid accelerations
        # if self.acceleration is None or other.acceleration is None:
        #     raise ValueError("Both states must have non-None acceleration attributes.")

        # Update the current object's acceleration by adding the other state's acceleration
        if hasattr(other, "acceleration"):
            self.acceleration += other.acceleration  # In-place update

        if hasattr(other, "velocity"):
            self.velocity += other.velocity  # In-place update

        if hasattr(other, "stm_dot"):
            self.stm_dot += other.stm_dot

        if hasattr(other, "entropy_dot"):
            self.entropy_dot += other.entropy_dot

        # Return the updated object itself
        return self

    def __repr__(self):
        """
        Returns a string representation of the State object.

        Returns
        -------
        str
            A string representation of the State object.
        """
        return f"State({vars(self)})"

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
            rotation_matrix = spice.pxform(ECI, ECEF, et)
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
            rotation_matrix = spice.pxform(ECI, ECEF, et)
            velocity = rotation_matrix @ self.velocity
        else:
            raise ValueError(
                f"Frame <{self.frame}> is spelled wrong or is not supported"
            )

        return velocity

    @property
    def latlong(self) -> tuple:
        """
        Returns the latitude of the position vector in the ECEF frame.

        Returns
        -------
        tuple
            The latitude and longitude of the position vector in the ECEF frame.
        """
        lat = np.arctan2(
            self.position_ecef[2],
            np.sqrt(self.position_ecef[0] ** 2 + self.position_ecef[1] ** 2),
        )
        long = np.arctan2(self.position_ecef[1], self.position_ecef[0])

        return (lat, long)

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

    def to_cartesian(self, mu):
        """
        Converts the state vector to Cartesian coordinates.

        Returns
        -------
        tuple
            The position and velocity vectors in Cartesian coordinates.
            TODO: There is a quick fix for nu and M below
        """
        state = classical2cart(
            **self.orbital_elements[0:5], nu=self.orbital_elements[-1], mu=mu
        )
        return state

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
