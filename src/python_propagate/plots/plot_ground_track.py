import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import matplotlib.pyplot as plt
from python_propagate.utilities.units import RAD2DEG


def plot_ground_track(
    agents, stations, output_directory, name="ground_track_visibility", legend=True
):
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
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # Add map features: coastlines and borders
    ax.coastlines(resolution="110m", color="black", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="gray")

    # Optionally add gridlines with labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--"
    )
    gl.xlabel_style = {"size": 10, "color": "gray"}
    gl.ylabel_style = {"size": 10, "color": "gray"}
    ax.set_global()

    # Plot the entire ground track for each agent (in a neutral color)
    for agent in agents:
        # Compute full track: convert longitudes and latitudes from radians to degrees
        track_lons = [state.latlong[1] * RAD2DEG for state in agent.state_data]
        track_lats = [state.latlong[0] * RAD2DEG for state in agent.state_data]
        ax.plot(
            track_lons,
            track_lats,
            color="k",
            marker="x",
            linestyle="none",
            transform=projection,
            label=f"{agent.name} Track",
        )

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
                ax.plot(
                    visible_lons,
                    visible_lats,
                    marker="o",
                    linestyle="None",
                    color=station_color,
                    markersize=6,
                    transform=projection,
                    label=f"{station.name} Visibility",
                )

    # Plot station locations with a distinctive marker.
    for station in stations:
        st_lon = station.longitude * RAD2DEG
        st_lat = station.latitude * RAD2DEG
        station_color = getattr(station, "color", "magenta")
        ax.plot(
            st_lon,
            st_lat,
            marker="*",
            color=station_color,
            markersize=15,
            transform=projection,
            label=station.name,
        )

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
    if legend:
        ax.legend(
            unique.values(), unique.keys(), loc="lower left", fontsize="small", ncol=2
        )

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    plt.title("Ground Track with Station Visibility")

    # Save the plot to file
    output_path = output_directory / f"{name}_ground_track.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    # plt.close(fig)
    print(f"Ground track with visibility plot saved as {output_path}")
