from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
from python_propagate.utilities.units import RAD2DEG
from python_propagate.agents import Agent
from python_propagate.utilities.constants import AU, PHI

from python_propagate.agents import Agent

import os



def plot_light_curve(
    agents: Iterable,  # or Iterable[Agent] if Agent is defined
    output_directory: str,
    name: str = "light_curves",
    legend: bool = True
):
    """
    Plot light curves for multiple agents on the same figure.
    """
    os.makedirs(output_directory, exist_ok=True)

    norm = PHI / (AU**2  / 1000**2)
    # Create a single figure OUTSIDE the loop
    fig, (ax_flux, ax_mag) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for idx, agent in enumerate(agents):
        times = np.array([state.time for state in agent.state_data])
        flux = np.array([state.metadata['flux_w_m2'] for state in agent.state_data]) / norm
        mags = np.array([state.metadata['apparent_magnitude'] for state in agent.state_data])

        # Use different colors for each agent automatically
        color = f"C{idx % 10}"  # matplotlib default cycle has 10 colors

        # Plot each agent
        ax_flux.plot(times, flux, label=f"{agent.name}", color=color)
        ax_mag.plot(times, mags, label=f"{agent.name}", color=color)

    # Beautify flux axis
    ax_flux.set_ylabel("Flux [Percent of Solar Flux]") 
    ax_flux.grid(True, ls=":")
    if legend:
        ax_flux.legend(loc="upper right")

    # Beautify magnitude axis
    ax_mag.set_ylabel("Apparent Magnitude")
    ax_mag.set_xlabel("Time")
    ax_mag.invert_yaxis()  # brighter = up
    ax_mag.grid(True, ls=":")
    if legend:
        ax_mag.legend(loc="upper right")

    fig.tight_layout()

    # Save once after all agents are plotted
    outpath = os.path.join(output_directory, f"{name}_light_curve.png")
    fig.savefig(outpath, dpi=150)
    print(f"Light curve plot saved as {outpath}")
    # plt.close(fig)
