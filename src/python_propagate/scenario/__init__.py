"""
scenario.py

This module contains the Scenario class.

Classes:
- Scenario: A class to represent a simulation scenario.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

from datetime import datetime, timedelta

from python_propagate.environment.planets import Planet
from python_propagate.utilities.load_spice import load_spice


class Scenario:
    """
    A class to represent a simulation scenario.

    Attributes
    ----------
    central_body : Planet
        The central body of the scenario.
    start_time : datetime
        The start time of the simulation.
    duration : timedelta
        The duration of the simulation.
    dt : timedelta
        The time step of the simulation.
    agents : list
        A list of agents in the scenario.
    stations : list
        A list of stations in the scenario.

    Methods
    -------
    __init__(self, central_body: Planet, start_time: datetime, duration: timedelta, dt: timedelta):
        Initializes the Scenario with the given parameters.
    central_body(self):
        Returns the central body of the scenario.
    start_time(self):
        Returns the start time of the simulation.
    duration(self):
        Returns the duration of the simulation.
    """

    def __init__(
        self,
        central_body: Planet,
        start_time: datetime,
        duration: timedelta,
        dt: timedelta,
        agents :list = [],
        stations:list = [],
    ):
        """
        Initializes the Scenario with the given parameters.

        Parameters
        ----------
        central_body : Planet
            The central body of the scenario.
        start_time : datetime
            The start time of the simulation.
        duration : timedelta
            The duration of the simulation.
        dt : timedelta
            The time step of the simulation.
        agents : tuple
            A tuple of agents in the scenario.
        stations : tuple, optional
            A tuple of stations in the scenario (default is empty tuple).
        """
        self._central_body = central_body
        self._start_time = start_time
        self._duration = duration
        self._dt = dt
        self.agents = agents
        self.stations = stations

        load_spice()

    @property
    def central_body(self):
        """
        Returns the central body of the scenario.

        Returns
        -------
        Planet
            The central body of the scenario.
        """
        return self._central_body

    @property
    def start_time(self):
        """
        Returns the start time of the simulation.

        Returns
        -------
        datetime
            The start time of the simulation.
        """
        return self._start_time

    @property
    def duration(self):
        """
        Returns the duration of the simulation.

        Returns
        -------
        timedelta
            The duration of the simulation.
        """
        return self._duration

    @property
    def dt(self):
        """
        Returns the time step of the simulation.

        Returns
        -------
        timedelta
            The time step of the simulation.
        """
        return self._dt

    def add_dynamics(self, dynamics: tuple):
        """Adds dynamics to the agents in the scenario."""
        for agent in self.agents:
            agent.add_dynamics(dynamics)

    def add_agents(self, agents: tuple):
        """Adds agents to the agents in the scenario."""
        for agent in agents:
            self.agents.append(agent)

    def add_stations(self, stations: tuple):
        """Adds stations to the stations in the scenario."""
        for station in stations:
            self.stations.append(station)

    def run(self):
        """Runs the simulation."""
        for agent in self.agents:
            agent.propagate()
