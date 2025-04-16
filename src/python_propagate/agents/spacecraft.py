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
import json
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
        dynamics=[],
        manuevers=[],
        bus=None,  # Allow passing a bus object to the spacecraft, default is None
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
            bus=bus,
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


class Bus:
    _cached_face_properties = {}

    def __init__(
        self,
        name=None,
        shape="box",
        extents=[1, 1, 1],
        orientation="nadir",
        properties_file="data/bus/default_properties.json",
    ):

        extents = (
            np.array(extents, dtype=float) / 1000
        )  # Convert extents to kilometers for consistency with typical spacecraft dimensions in astrodynamics.

        self.name = name
        self.orientation = orientation
        self._base_orientation = orientation
        self.extents = extents  # The extents of the bus in the x, y, z dimensions

        # Load face properties from the JSON file (use cached version if available)
        if properties_file not in Bus._cached_face_properties:
            with open(properties_file, "r") as file:
                Bus._cached_face_properties[properties_file] = json.load(file)
        self.face_properties = Bus._cached_face_properties[properties_file]

        if shape == "box":
            self.shape = trimesh.creation.box(
                extents=extents, face_attributes=self.face_properties
            )
            self.face_properties = self._order_face_properties_box()
        else:
            raise NotImplementedError(
                f"Shape '{shape}' is not implemented for the spacecraft bus."
            )

    def __repr__(self):
        """
        Returns a string representation of the Bus object.

        Returns
        -------
        str
            A string representation of the Bus object.
        """
        return f"Bus(name={self.name}, orientation={self.orientation}, extents={self.extents})"

    @property
    def base_orientation(self):
        return self._base_orientation

    def visualize_shape(self):
        """
        Visualizes the shape of the spacecraft bus using trimesh.
        """
        if hasattr(self, "shape"):
            self.shape.show()
        else:
            raise AttributeError("Shape not defined for the spacecraft bus.")

    def set_orientation(self, state: "State", transform=None):
        """
        Sets the orientation of the bus based on the state of the spacecraft.
        """
        if "orientation" in state.metadata:
            self.orientation = state.metadata["orientation"]

        if self.orientation == "nadir":
            rotation_matrix = self._nadir_transform(state)
            self.shape.apply_transform(rotation_matrix)
        elif self.orientation == "free":
            pass  # keep the current bus orientation (free to rotate)
        else:
            raise NotImplementedError(
                f"Orientation '{self.orientation}' is not implemented for the spacecraft bus."
            )

    def _nadir_transform(self, state: "State") -> np.ndarray:
        """
        Computes the nadir transform for the spacecraft bus based on the state.
        """
        nadir = -state.position / np.linalg.norm(
            state.position
        )  # Get the nadir vector (opposite of position vector)
        x_body = np.array([1, 0, 0])
        v = np.cross(x_body, nadir)
        c = np.dot(x_body, nadir)

        if np.allclose(v, 0) and c > 0:
            r_align = np.eye(3)
        elif np.allclose(v, 0) and c < 0:
            r_align = -np.eye(3)  # Opposite direction (180° rotation)
        else:
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            r_align = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))

        transform = np.eye(4)
        transform[:3, :3] = r_align
        return transform

    def _order_face_properties_box(self):
        """
        Order the face properties based on the face index to ensure consistent ordering for visualization and other purposes.
        """
        normal_to_property = {
            (1, 0, 0): "xp_face",
            (-1, 0, 0): "xm_face",
            (0, 1, 0): "yp_face",
            (0, -1, 0): "ym_face",
            (0, 0, 1): "zp_face",
            (0, 0, -1): "zm_face",
        }

        properties_ordered = {}
        for i, normal in enumerate(self.shape.face_normals):
            normal_tuple = tuple(
                np.round(normal).astype(int)
            )  # Convert normal to a tuple of integers
            if normal_tuple in normal_to_property:
                properties_ordered[i] = self.face_properties[
                    normal_to_property[normal_tuple]
                ]
            else:
                raise ValueError(
                    f"Normal vector {normal} does not match any predefined face properties."
                )

        return properties_ordered
