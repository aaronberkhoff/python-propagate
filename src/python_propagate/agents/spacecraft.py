"""
spacecraft.py

This module contains the Spacecraft class.

Classes:
- Spacecraft: A class to represent a spacecraft.
Author: Aaron Berkhoff

Date: 2025-01-30
"""

from python_propagate.agents import Agent


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
    coefficent_of_drag : float, optional
        The coefficient of drag of the spacecraft (default is None).
    mass : float, optional
        The mass of the spacecraft (default is None).
    area : float, optional
        The cross-sectional area of the spacecraft (default is None).

    Methods
    -------
    __init__(self, state, start_time, duration, dt, coefficent_of_drag=None, mass=None, area=None):
        Initializes the Spacecraft with the given parameters.
    """

    def __init__(
        self,
        state,
        start_time,
        duration,
        dt,
        coefficent_of_drag=None,
        mass=None,
        area=None,
        name=None,
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
        coefficent_of_drag : float, optional
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
            coefficent_of_drag,
            mass,
            area=area,
            name=name,
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
            f"dt={self.dt}, coefficent_of_drag={self.coefficent_of_drag}, mass={self.mass}, area={self.area}, name={self.name})"
        )
