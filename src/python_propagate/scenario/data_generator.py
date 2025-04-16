"""
data_generator.py

This module contains the DataGenerator class.

Classes:
- DataGenerator: A class to represent a simulation DataGenerator.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

from pathlib import Path
import pandas as pd
import numpy as np
from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from python_propagate.scenario import Scenario
from python_propagate.utilities.units import RAD2DEG, ARC2DEG

np.random.seed(100)


class DataGenerator(Scenario):
    """
    A class to generate observational data from agents in a scenario.

    Attributes:
        central_body: The celestial body around which the scenario is defined.
        start_time: The start time of the scenario.
        duration: The duration of the scenario.
        dt: The time step for the scenario.
        data_types (tuple): Types of data to generate (default: ("right_ascension", "declination")).
        agents: The agents (e.g., satellites) involved in the scenario.
        stations: The ground stations collecting data.
        plots: Plotting options for the scenario.
        output_directory (str): Directory to save the output files (default: "examples/results").
        name (str): The scenario name (default: "None").
    """

    def __init__(
        self,
        central_body,
        start_time,
        duration,
        dt,
        data_types: tuple = ("right_ascension", "declination"),
        agents=...,
        stations=...,
        plots=None,
        output_directory: str = "examples/results",
        name: str = "None",
        output_type="csv",
    ):
        """
        Initializes the DataGenerator instance.

        Args:
            central_body: The celestial body for the scenario.
            start_time: The start time of the scenario.
            duration: The duration of the scenario.
            dt: The time step for the scenario.
            data_types (tuple, optional): Types of data to generate. Defaults to ("right_ascension", "declination").
            agents: The agents (e.g., satellites) in the scenario.
            stations: The ground stations observing the agents.
            plots (optional): Plotting configurations. Defaults to None.
            output_directory (str, optional): Directory to save results. Defaults to "examples/results".
            name (str, optional): The scenario name. Defaults to "None".

        """

        self._data_types = data_types
        self._plots = plots
        self._name = name

        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

        if isinstance(output_type, str):
            self._output_type[output_type]
        if isinstance(output_type, Iterable):
            self._output_type = output_type
        super().__init__(
            central_body, start_time, duration, dt, agents, stations, use_spice=True
        )

    @property
    def data_types(self):
        """Returns the data types to be generated."""
        return self._data_types

    @property
    def output_directory(self):
        """Returns the output directory path."""
        return self._output_directory

    @property
    def plots(self):
        """Returns the plot configuration."""
        return self._plots

    @property
    def name(self):
        """Returns the scenario name."""
        return self._name

    @property
    def output_type(self):
        """Returns the scenario output_type."""
        return self._output_type

    def run(self):
        """
        Runs the DataGenerator simulation, collecting observational data from agents and saving it to HDF5, Excel, and CSV formats.
        """
        print("Generating Data...")
        self.generate_data()

        # Now plot the orbit
        if "orbit" in self.plots:
            print(f"Plotting Orbit...\n")
            self.plot_orbit()
        if "ground_track" in self.plots:
            print("Plotting Ground track...")
            plot_ground_track(
                self.agents, self.stations, self.output_directory, name=self.name
            )
        plot_orbital_elements(self.agents, self, self.output_directory, name=self.name)
