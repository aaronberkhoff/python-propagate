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
from collections import namedtuple

from python_propagate.utilities.transforms import cart2classical, classical2cart


#TODO: Remove hard coded TARGET
TARGET = "EARTH"
ECI = "J2000"
ECEF = "ITRF93"
MU = 398600.4415

OrbitalElements = namedtuple(
    "OrbitalElements", ["sma","ecc","inc","arg","raan","nu"]
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
        orbital_elements=None,
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
        orbital_elements : tuple, optional
            The orbital elements of the agent (default is None).
        """
        self.position = position  # Assume kilometers by default
        self.velocity = velocity  # Assume kilometers per second
        self.acceleration = acceleration
        self.frame = frame
        self.dimension = dimension
        self.stm = stm
        self.time = time
        self.stm_dot = stm_dot

        if orbital_elements is not None:
            self.orbital_elements = orbital_elements
            #TODO Figure out how to handle mu
            state = classical2cart(sma=orbital_elements[0], 
                                   ecc=orbital_elements[1],
                                   inc=orbital_elements[2],
                                   arg=orbital_elements[3],
                                   raan=orbital_elements[4],
                                   nu=orbital_elements[5],
                                   mu=MU)
            self.position = state[0:3]
            self.velocity = state[3:6]
           
    def __repr__(self):

        """
        Returns a string representation of the State object.

        Returns
        -------
        str
            A string representation of the State object.
        """
        return (f"State(position={self.position}, velocity={self.velocity}, acceleration={self.acceleration}, "
                f"stm={self.stm}, stm_dot={self.stm_dot}, time={self.time}, dimension={self.dimension}, "
                f"frame={self.frame}, orbital_elements={getattr(self, 'orbital_elements', None)})")
        

            
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
        lat  = np.arctan2(self.position_ecef[2], np.sqrt(self.position_ecef[0]**2 + self.position_ecef[1]**2))
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
    
    def to_cartesian(self,mu):
        """
        Converts the state vector to Cartesian coordinates.

        Returns
        -------
        tuple
            The position and velocity vectors in Cartesian coordinates.
            TODO: There is a quick fix for nu and M below
        """
        state = classical2cart(**self.orbital_elements[0:5],nu=self.orbital_elements[-1], mu=mu)
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

