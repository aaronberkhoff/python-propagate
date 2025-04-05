"""
spacecraft.py

This module contains the Spacecraft class.

Classes:
- Spacecraft: A class to represent a spacecraft.
Author: Aaron Berkhoff

Date: 2025-01-30
"""
import numpy as np
import trimesh

from python_propagate.agents import Agent
from python_propagate.states import State
from python_propagate.utilities.transforms import inertial_to_ric
from scipy.spatial.transform import Rotation as R




class Spacecraft(Agent):
    """
    A class to represent a spacecraft.

    Inherits from the Agent class.

    Attributes
    ----------
    state : object
        The initial state of the spacecraft.
    start_time : datetime
        The start time of the simulation.
    duration : float
        The duration of the simulation.
    dt : float
        The time step for the simulation.
    coefficient_of_drag : float, optional
        The coefficient of drag of the spacecraft (default is None).
    mass : float, optional
        The mass of the spacecraft (default is None).
    area : float, optional
        The cross-sectional area of the spacecraft (default is None).

    Methods
    -------
    __init__(self, state, start_time, duration, dt, coefficient_of_drag=None, mass=None, area=None):
        Initializes the Spacecraft with the given parameters.
    """

    def __init__(
        self,
        state,
        start_time,
        duration,
        dt,
        coefficient_of_drag=None,
        mass=None,
        area=None,
        name=None,
        scenario=None,
        dynamics = [],
        manuevers = [],
        bus = None # Allow passing a bus object to the spacecraft, default is None
    ):
        """
        Constructs all the necessary attributes for the Spacecraft object.

        Parameters
        ----------
        state : object
            The initial state of the spacecraft.
        start_time : datetime, str
            The start time of the simulation.
        duration : float
            The duration of the simulation.
        dt : float
            The time step for the simulation.
        coefficient_of_drag : float, optional
            The coefficient of drag of the spacecraft (default is None).
        mass : float, optional
            The mass of the spacecraft (default is None).
        area : float, optional
            The cross-sectional area of the spacecraft (default is None).
        """

        super().__init__(
            state,
            start_time,
            duration,
            dt,
            coefficient_of_drag,
            mass,
            area=area,
            name=name,
            scenario=scenario,
            dynamics=dynamics,
            manuevers=manuevers,
            bus = bus
        )

    def __repr__(self):
        """
        Returns a string representation of the Spacecraft object.

        Returns
        -------
        str
            A string representation of the Spacecraft object.
        """
        return (
            f"Spacecraft(state={self.state}, start_time={self.start_time}, duration={self.duration}, "
            f"dt={self.dt}, coefficient_of_drag={self.coefficient_of_drag}, mass={self.mass}, area={self.area}, name={self.name})"
        )

class Bus():


    def __init__(self, agent: Agent, name=None, shape = 'box', extents=[1,1,1], orientation='nadir'):

        self.name = name
        self.orientation = orientation # Default orientation of the bus, can be 'nadir' or other orientations as needed.

        if shape == 'box':
            self.shape = trimesh.creation.box(extents=extents)  # Create a box with the specified extents.
        else:
            raise NotImplementedError(
                f"Shape '{shape}' is not implemented for the spacecraft bus."
            )
    
    def visualize_shape(self):
        """
        Visualizes the shape of the spacecraft bus using trimesh.
        """
        if hasattr(self, 'shape'):
            self.shape.show()
        else:
            raise AttributeError("Shape not defined for the spacecraft bus.")
        
    def set_orientation(self,state: State, transform=None) ->  None:
      
        """
        Sets the orientation of the bus based on the state of the spacecraft.

        Parameters:
            state (State): The state of the spacecraft.
            transform (None, optional): A transformation matrix if needed.

        Returns:
            None
        """
        rv = state.compile()

        if self.orientation == 'nadir':
            """
            For nadir orientation, align the bus with the velocity vector of the spacecraft.
            """
            rotation_matrix = self._nadir_transform(state)
            self.shape.apply_transform(rotation_matrix)

        else: 
            """
            For other orientations, you can implement specific logic based on the desired orientation.
            """
            raise NotImplementedError(
                f"Orientation '{self.orientation}' is not implemented for the spacecraft bus."
            )
        
    def _nadir_transform(self, state: State) -> np.ndarray:
        """
        Computes the nadir transform for the spacecraft bus based on the state.

        Parameters:
            state (State): The state of the spacecraft.

        Returns:
            np.ndarray: The transformation matrix for nadir orientation.
        """
        nadir = -state.position / np.linalg.norm(state.position)  # Get the nadir vector (opposite of position vector)

        x_body = np.array([1, 0, 0])

        #Compute rotation to align x_body to nadir
        v = np.cross(x_body, nadir)
        c = np.dot(x_body, nadir)

        if np.allclose(v, 0) and c > 0:
            # Already aligned
            r_align = np.eye(3)
        elif np.allclose(v, 0) and c < 0:
            # Opposite direction (180° rotation)
            r_align = R.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix()
        else:
            s = np.linalg.norm(v)
            kmat = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            r_align = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        # Convert to 4x4 transform and apply
        transform = np.eye(4)
        transform[:3, :3] = r_align

        return transform








