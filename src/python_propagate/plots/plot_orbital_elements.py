import numpy as np
import matplotlib.pyplot as plt
from python_propagate.utilities.units import RAD2DEG
from pathlib import Path

def plot_orbital_elements(agents, scenario, output_directory, name="orbital_elements",legend = True, save = True):
    """
    Plots the orbital elements (SMA, Eccentricity, Inclination, RAAN, Argument of Periapsis, True Anomaly)
    over time for each agent.

    Args:
        agents (list): List of agent objects. Each must have a `state_data` attribute containing state objects.
                       Each state object must have:
                         - a method `to_keplerian(mu)` returning (a, e, i, RAAN, arg_periapsis, true_anomaly)
                         - a `time` attribute (in seconds, or another consistent time unit)
        scenario: An object that contains the central body's gravitational parameter, accessible as
                  `scenario.central_body.mu`.
        output_directory (str or Path): Directory where the resulting plot will be saved.
        name (str, optional): Base name for the output file. Defaults to "orbital_elements".
    """
    # Create subplots for the 6 orbital elements.
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()  # for easier indexing: 0:SMA, 1:ecc, 2:inc, 3:RAAN, 4:Arg Periapsis, 5:true anomaly

    # Loop over agents
    for agent in agents:
        # Extract times and orbital elements for this agent
        times = np.array([state.time for state in agent.state_data])
        # Calculate orbital elements: [a, e, i, RAAN, arg_periapsis, true_anomaly]
        orbital_element_data = [state.to_keplerian(scenario.central_body.mu) for state in agent.state_data]

        sma_data =  np.array([oe.sma for oe in orbital_element_data])
        ecc_data =  np.array([oe.ecc for oe in orbital_element_data])
        inc_data =  np.array([oe.inc for oe in orbital_element_data])
        arg_data =  np.array([oe.arg for oe in orbital_element_data])
        raan_data = np.array([oe.raan for oe in orbital_element_data])
        nu_data =   np.array([oe.nu for oe in orbital_element_data])
        # Unpack each element (converting angles to degrees)
        # sma = elements[:, 0]                  # Semi-major axis (assumed to be in km)
        # ecc = elements[:, 1]                  # Eccentricity
        # inc = np.degrees(inc_data)      # Inclination
        # raan = np.degrees(elements[:, 3])     # Right Ascension of Ascending Node
        # arg_periapsis = np.degrees(elements[:, 4])  # Argument of Periapsis
        # true_anomaly = np.degrees(elements[:, 5])   # True Anomaly

        inc_data  *= RAD2DEG
        raan_data *= RAD2DEG     # Right Ascension of Ascending Node
        arg_data  *= RAD2DEG  # Argument of Periapsis
        nu_data   *= RAD2DEG   # True Anomaly

        # Plot each element over time
        axes[0].plot(times, sma_data, label=f"{agent.name}")
        axes[1].plot(times, ecc_data, label=f"{agent.name}")
        axes[2].plot(times, inc_data, label=f"{agent.name}")
        axes[3].plot(times, raan_data, label=f"{agent.name}")
        axes[4].plot(times, arg_data, label=f"{agent.name}")
        axes[5].plot(times, nu_data, label=f"{agent.name}")

    # Set labels and grid for each subplot
    element_titles = [
        "Semi-Major Axis [km]",
        "Eccentricity",
        "Inclination [deg]",
        "RAAN [deg]",
        "Argument of Periapsis [deg]",
        "True Anomaly [deg]"
    ]
    for ax, title in zip(axes, element_titles):
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(title)
        ax.grid(True)
        if legend:
            ax.legend(fontsize="small")
    
    if save:
        plt.tight_layout()
        if not isinstance(output_directory, Path):
            output_directory = Path(output_directory)
        if not output_directory.exists():
            print(f"Creating output directory: {output_directory}")
            output_directory.mkdir(parents=True, exist_ok=True)
        # Save the figure to the designated output directory
        output_path = output_directory / f"{name}_orbital_elements.png"
        plt.savefig(output_path, dpi=150)
        # plt.close(fig)
        print(f"Orbital elements plot saved as {output_path}")

    return fig, axes

