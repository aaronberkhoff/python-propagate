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
        super().__init__(central_body, start_time, duration, dt, agents, stations)

        self._data_types = data_types
        self._plots = plots
        self._name = name

        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(parents=True, exist_ok=True)

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

    def run(self):
        """
        Runs the DataGenerator simulation, collecting observational data from agents and saving it to HDF5, Excel, and CSV formats.
        """
        print("Generating Data...")
        self.generate_data()

        #Now plot the orbit
        if 'orbit' in self.plots:
            print(f"Plotting Orbit...\n")
            self.plot_orbit()
        if 'ground_track' in self.plots:
            print("Plotting Ground track...")
            plot_ground_track(self.agents,
                              self.stations,
                              self.output_directory,
                              name=self.name)

    def generate_data(self):

        data_all = []  # List to store data for all agents

        for agent in self.agents:
            agent.propagate()  # Update agent state
            data_agent = []  # Store data for this agent

            for i, state in enumerate(agent.state_data):
                for station in self.stations:
                    # Calculate azimuth and elevation (convert to degrees)
                    az, el = station.calculate_azimuth_and_elevation(state=state)
                    az *= RAD2DEG
                    el *= RAD2DEG

                    #TODO: Hard coded noise to data
                    az += np.random.normal(0,5 * ARC2DEG)
                    el += np.random.normal(0,5 * ARC2DEG)

                    # Calculate right ascension and declination (convert to degrees)
                    ra, dec = station.calculate_ra_and_dec(state=state)
                    ra *= RAD2DEG
                    dec *= RAD2DEG

                    ra += np.random.normal(0,5 * ARC2DEG)
                    dec += np.random.normal(0,5 * ARC2DEG)

                    # Calculate range and range rate
                    rho, rhodot = station.calculate_range_and_range_rate_from_target(
                        state=state
                    )
                    rho += np.random.normal(0,10e-3)
                    rhodot += np.random.normal(0,10e-6)

                    # Store data if elevation is above the station's minimum threshold
                    if el > station.minimum_elevation_angle:
                        data_agent.append(
                            {
                                "agent": agent.name,
                                "index": i,
                                "time": state.time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "RA_DEG": ra,
                                "DEC_DEG": dec,
                                "AZ_DEG": az,
                                "EL_DEG": el,
                                "Range_KM": rho,
                                "Range_Rate_KMS": rhodot,
                                "station": station.name,
                                "station_id": station.identity,
                            }
                        )

            data_all.extend(data_agent)

        # Convert data to a pandas DataFrame
        df = pd.DataFrame(data_all)

        # Define output file paths
        output_path_h5 = self.output_directory / f"{self.name}.h5"
        output_path_xlsx = self.output_directory / f"{self.name}.xlsx"
        output_path_csv = self.output_directory / f"{self.name}.csv"

        # Save data to various formats
        df.to_hdf(output_path_h5, key="df", mode="w")
        df.to_excel(output_path_xlsx, index=False)
        df.to_csv(output_path_csv, index=False)

        print(
            f"Files written:\n  HDF5: {output_path_h5}\n  Excel: {output_path_xlsx}\n  CSV: {output_path_csv}"
        )
            

    def plot_orbit(self):
        """
        Plots an isometric view of the orbit, showing four different projections:
        - axxy: XY plane
        - axxz: XZ plane
        - axyz: YZ plane
        - ax3d: 3D orbit view

        The central body is represented by a sphere at the origin with a color gradient in the 3D view.
        Its radius is determined by self.central_body.radius.
        """
        
        # Create a new figure with a grid of subplots
        fig = plt.figure(figsize=(14, 14))
        axxy = fig.add_subplot(221)
        axxz = fig.add_subplot(223)
        axyz = fig.add_subplot(224)
        ax3d = fig.add_subplot(222, projection='3d')

        # Determine sphere properties for the central body
        sphere_radius = self.central_body.radius
        central_color = 'blue'  # Base color for 2D plots
        sphere_alpha = 0.3      # Transparency for the sphere

        # Draw the sphere on the 2D plots as a circle centered at (0, 0)
        sphere_circle = Circle((0, 0), sphere_radius, color=central_color, alpha=sphere_alpha)
        axxy.add_patch(sphere_circle)
        sphere_circle_xz = Circle((0, 0), sphere_radius, color=central_color, alpha=sphere_alpha)
        axxz.add_patch(sphere_circle_xz)
        sphere_circle_yz = Circle((0, 0), sphere_radius, color=central_color, alpha=sphere_alpha)
        axyz.add_patch(sphere_circle_yz)

        # Draw the sphere on the 3D plot using a parametric surface with a color gradient.
        # Generate the mesh for the sphere.
        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x_sphere = sphere_radius * np.cos(u) * np.sin(v)
        y_sphere = sphere_radius * np.sin(u) * np.sin(v)
        z_sphere = sphere_radius * np.cos(v)
        
        # Use a colormap (e.g., 'viridis') to create a color gradient across the surface.
        # The colormap will map the z_sphere values to colors.
        surface = ax3d.plot_surface(x_sphere, y_sphere, z_sphere,
                                    cmap='viridis',  # Choose your favorite colormap here
                                    alpha=sphere_alpha,
                                    rstride=1, cstride=1,  # Adjust these parameters for resolution
                                    linewidth=0, antialiased=True)

        # # Optionally, add a color bar for reference
        # fig.colorbar(surface, ax=ax3d, shrink=0.5, aspect=10)

        # Loop through all agents to plot their orbits and key points (start and end)
        for agent in self.agents:
            # Plot start and end markers along with the trajectory on the XY plane
            axxy.plot(agent.state_data[0].position[0], agent.state_data[0].position[1],
                    'g*', label='start', fillstyle='none')
            axxy.plot(agent.state_data[-1].position[0], agent.state_data[-1].position[1],
                    'rs', label='end', fillstyle='none')
            axxy.plot([state.position[0] for state in agent.state_data],
                    [state.position[1] for state in agent.state_data],
                    label=agent.name, linewidth=1.0)
            axxy.set_xlabel('X [KM]')
            axxy.set_ylabel('Y [KM]')

            # Plot on the XZ plane
            axxz.plot(agent.state_data[0].position[0], agent.state_data[0].position[2],
                    'g*', label='start', fillstyle='none')
            axxz.plot(agent.state_data[-1].position[0], agent.state_data[-1].position[2],
                    'rs', label='end', fillstyle='none')
            axxz.plot([state.position[0] for state in agent.state_data],
                    [state.position[2] for state in agent.state_data],
                    label=agent.name, linewidth=1.0)
            axxz.set_xlabel('X [KM]')
            axxz.set_ylabel('Z [KM]')

            # Plot on the YZ plane
            axyz.plot(agent.state_data[0].position[1], agent.state_data[0].position[2],
                    'g*', label='start', fillstyle='none')
            axyz.plot(agent.state_data[-1].position[1], agent.state_data[-1].position[2],
                    'rs', label='end', fillstyle='none')
            axyz.plot([state.position[1] for state in agent.state_data],
                    [state.position[2] for state in agent.state_data],
                    label=agent.name, linewidth=1.0)
            axyz.set_xlabel('Y [KM]')
            axyz.set_ylabel('Z [KM]')

            # Plot on the 3D view
            ax3d.plot([state.position[0] for state in agent.state_data],
                    [state.position[1] for state in agent.state_data],
                    [state.position[2] for state in agent.state_data],
                    label=agent.name, linewidth=1.0)
            ax3d.plot([agent.state_data[0].position[0]],
                    [agent.state_data[0].position[1]],
                    [agent.state_data[0].position[2]],
                    'g*', label='start', fillstyle='none')
            ax3d.plot([agent.state_data[-1].position[0]],
                    [agent.state_data[-1].position[1]],
                    [agent.state_data[-1].position[2]],
                    'rs', label='end', fillstyle='none')
            ax3d.set_xlabel('X [KM]')
            ax3d.set_ylabel('Y [KM]')
            ax3d.set_zlabel('Z [KM]')

        # Add legend and grid to the 2D subplots
        axxy.legend()
        axxy.grid(True)
        axxz.grid(True)
        axyz.grid(True)

        axxy.set_aspect('equal', adjustable='datalim')
        axxz.set_aspect('equal', adjustable='datalim')
        axyz.set_aspect('equal', adjustable='datalim')
        ax3d.set_aspect('equal', adjustable='datalim')

        plt.tight_layout()

        # Save the figure to the designated output directory
        saveas = self.output_directory / f"{self.name}_orbit_plot.png"
        print(f"Files saved:\n {saveas}: ")
        plt.savefig(saveas, dpi=100)
        plt.close(fig)


def plot_ground_track(agents, stations, output_directory, name="ground_track_visibility"):
    """
    Plots the entire ground track of each agent and overlays points (in the station's color)
    where the agent's elevation exceeds the station's minimum elevation angle.
    
    Args:
        agents (list): List of agent objects. Each must have a `state_data` attribute where each
                       element (state) has attributes `latitude` and `longitude` in radians.
        stations (list): List of station objects. Each station must have:
                          - a method `calculate_azimuth_and_elevation(state)` that returns (az, el)
                          - an attribute `minimum_elevation_angle` (in degrees)
                          - optionally, a `color` attribute to be used for plotting.
        output_directory (str or Path): Directory where the resulting plot will be saved.
        name (str, optional): Base name for the output file. Defaults to "ground_track_visibility".
    """
    
    # Create a figure with a PlateCarree projection (suitable for geographic data)
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Add map features: coastlines and borders
    ax.coastlines(resolution="110m", color="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")
    
    # Optionally add gridlines with labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")
    gl.xlabel_style = {"size": 10, "color": "gray"}
    gl.ylabel_style = {"size": 10, "color": "gray"}
    ax.set_global()
    
    # Plot the entire ground track for each agent (in a neutral color)
    for agent in agents:
        # Compute full track: convert longitudes and latitudes from radians to degrees
        track_lons = [state.latlong[1] * RAD2DEG for state in agent.state_data]
        track_lats = [state.latlong[0] * RAD2DEG for state in agent.state_data]
        ax.plot(track_lons, track_lats, color="k", marker = 'x',linestyle = 'none',
                transform=ccrs.PlateCarree(), label=f"{agent.name} Track")
        
    # Now, for each station, find and plot the points where the agent is visible.
    # (That is, where the elevation > station.minimum_elevation_angle.)
    for station in stations:
        # Set a default color if the station does not have one
        station_color = getattr(station, "color", "magenta")
        
        # Loop through agents and accumulate the visible points
        for agent in agents:
            visible_lons = []
            visible_lats = []
            
            for state in agent.state_data:
                # Calculate azimuth and elevation at the given state from this station.
                az, el = station.calculate_azimuth_and_elevation(state=state)
                # Convert the elevation to degrees for comparison
                if (el * RAD2DEG) > station.minimum_elevation_angle:
                    visible_lons.append(state.latlong[1] * RAD2DEG)
                    visible_lats.append(state.latlong[0] * RAD2DEG)
            
            # If there are any visible points for this agent at this station, plot them.
            if visible_lons:
                ax.plot(visible_lons, visible_lats, marker="o", linestyle="None",
                        color=station_color, markersize=6,
                        transform=ccrs.PlateCarree(),
                        label=f"{station.name} Visibility")
    
    # Plot station locations with a distinctive marker.
    for station in stations:
        st_lon = station.longitude * RAD2DEG
        st_lat = station.latitude * RAD2DEG
        station_color = getattr(station, "color", "magenta")
        ax.plot(st_lon, st_lat, marker="*", color=station_color, markersize=15,
                transform=ccrs.PlateCarree(), label=station.name)
    
    # Set map limits (optional: full globe)
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    
    # Create a legend and remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    from collections import OrderedDict
    unique = OrderedDict()
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), loc="lower left", fontsize="small", ncol=2)
    
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    plt.title("Ground Track with Station Visibility")
    
    # Save the plot to file
    output_path = output_directory / f"{name}_ground_track.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Ground track with visibility plot saved as {output_path}")
