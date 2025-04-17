"""
scenario.py

This module contains the Scenario class.

Classes:
- Scenario: A class to represent a simulation scenario.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

from datetime import datetime, timedelta
from typing import Iterable

from python_propagate.environment.planets import Planet
from python_propagate.utilities.load_spice import load_spice
from python_propagate.utilities.string_format import DATESTR


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
        agents=[],
        stations=[],
        use_spice=False,
        name="Scenario",
        celestial_bodies: Iterable[Planet] = []
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
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, DATESTR)

        if isinstance(duration, dict):
            duration = timedelta(**duration)

        if isinstance(dt, dict):
            dt = timedelta(**dt)

        if celestial_bodies:
            load_spice()

        self._central_body = central_body
        self._celestial_bodies = celestial_bodies
        self._start_time = start_time
        self._duration = duration
        self._dt = dt

        if agents:
            self.agents = [agent.set_scenario(self) for agent in agents]
        else:
            self.agents = agents

        if stations:
            self.stations = [station.set_scenario(self) for station in stations]
        else:
            self.stations = stations




        if use_spice:
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
    def celestial_bodies(self):
        """
        Returns the celestial bodies of the scenario.

        Returns
        -------
        Planet
            The celestial bodies of the scenario.
        """
        return self._celestial_bodies

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
