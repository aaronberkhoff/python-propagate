import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from python_propagate.filters.data_handler import DataHandler
from python_propagate.utilities.units import RAD2DEG


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_residuals(
    true_data,
    pred_data,
    dt,
    output_directory: str,
    agent_name ,
    plot_type: str = "position",
    ylim=None,
    show=False,
    
):
    """
    Plots either state residuals or measurement residuals with ±3σ bounds.

    Parameters:
        data_handler (DataHandler): Object containing residuals and covariance history.
        path (str): File path to save the generated plot.
        plot_type (str): 'state' for state residuals, 'measurement' for measurement residuals.
    """
    rel_res = np.array([np.abs(tru.position - pred.position) / (tru.position) for tru, pred in zip(true_data,pred_data) ])
    abs_res = np.array([np.abs(tru.position - pred.position) for tru, pred in zip(true_data,pred_data) ])
    
    times = np.arange(0,rel_res.shape[0])* dt
    labels = ["X [KM]", "Y [KM]", "Z [KM]"]
    # title = "Position Residuals"
    index = 0


    fig, axs = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    # fig.suptitle(title, fontsize=16)

    for i, (ax, label) in enumerate(zip(axs, labels)):
        plot_single_residual(
            ax, rel_res[:, i + index], times, label, ylim=ylim
        )

    axs[-1].set_xlabel("Time [HR]")
    fig.tight_layout()
    
    if show:
        plt.show()

    # Save the figure to the designated output directory
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)
    if not output_directory.exists():
        print(f"Creating output directory: {output_directory}")
        output_directory.mkdir(parents=True, exist_ok=True)

    name = agent_name + '_rel_error'
    saveas = output_directory / f"{name}_positions.png"
    print(f"Files saved:\n {saveas}: ")
    plt.savefig(saveas, dpi=100)

    #Absolute error

    fig, axs = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    # fig.suptitle(title, fontsize=16)

    for i, (ax, label) in enumerate(zip(axs, labels)):
        plot_single_residual(
            ax, abs_res[:, i + index], times, label, ylim=ylim
        )

    axs[-1].set_xlabel("Time [HR]")
    fig.tight_layout()
    if show:
        plt.show()
    name = agent_name + '_abs_error'
    saveas = output_directory / f"{name}_positions.png"
    print(f"Files saved:\n {saveas}: ")
    plt.savefig(saveas, dpi=100)


def plot_single_residual(ax, residuals,  times, label, ylim=None):
    """
    Plots a single residual with ±3σ bounds.

    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        residuals (np.ndarray): Residual values for the state/measurement.
        variances (np.ndarray): Variance values for the state/measurement.
        label (str): Label for the variable.
    """
    # sigma = 3 * np.sqrt(variances)

    ax.plot(
        times / 3600, residuals, marker="x", linestyle="none", label=f"{label} Residual"
    )
    # ax.plot(times / 3600, sigma, color="red", label=r"$\pm3\sigma$")
    # ax.plot(times / 3600, -sigma, color="red")

    ax.set_ylabel(label)
    ax.set_ylim(ylim)
    ax.grid(True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=10)

    rms = np.sqrt(np.mean(np.square(residuals)))
    ax.set_title(f"{label} - RMS: {rms:.4e}")

    # Dynamic y-limit setting
    # max_abs_val = max(np.max(np.abs(residuals)), np.max(sigma))
    # ax.set_ylim([-max_abs_val*1.1, max_abs_val*1.1])


# def plot_uncertainty(agents,scenario, output_directory,name):

