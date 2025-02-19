import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from python_propagate.filters.data_handler import DataHandler
from python_propagate.utilities.units import RAD2DEG


import numpy as np
import matplotlib.pyplot as plt

def plot_residuals(data_handler: DataHandler, path: str, plot_type: str = 'position', ylim = None, show = False):
    """
    Plots either state residuals or measurement residuals with ±3σ bounds.
    
    Parameters:
        data_handler (DataHandler): Object containing residuals and covariance history.
        path (str): File path to save the generated plot.
        plot_type (str): 'state' for state residuals, 'measurement' for measurement residuals.
    """
    if plot_type == 'position':
        cov = np.array(data_handler.covariance_estimate_history)
        res = np.array([res.ravel() for res in data_handler.state_residuals])
        times = data_handler.state_times
        labels = ["X [KM]", "Y [KM]", "Z [KM]"]
        title = "Position Residuals"
        index = 0
    elif plot_type == 'velocity':
        cov = np.array(data_handler.covariance_estimate_history)
        res = np.array([res.ravel() for res in data_handler.state_residuals])
        times = data_handler.state_times
        labels = ["Vx [KM/S]", "Vx [KM/S]", "Vz [KM/S]"]
        title = "Velocity Residuals"
        index = 3
    elif plot_type == 'measurement':
        cov = np.array(data_handler.measurement_covariances) 
        res = np.array([res.ravel() for res in data_handler.residuals]) 
        times = data_handler.measurement_times
        labels = ["RA [DEG]", "DEC [DEG]"]
        title = "Measurement Residuals"
        index = 0
    else:
        raise ValueError(f"Invalid plot_type <{plot_type}>. Check spelling or support plot types")

    fig, axs = plt.subplots(len(labels), 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=16)

    for i, (ax, label) in enumerate(zip(axs, labels)):
        plot_single_residual(ax, res[:, i+index], cov[:, i+index, i+index], times, label,ylim=ylim)

    axs[-1].set_xlabel("Time [HR]")
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()

def plot_single_residual(ax, residuals, variances, times, label, ylim = None):
    """
    Plots a single residual with ±3σ bounds.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        residuals (np.ndarray): Residual values for the state/measurement.
        variances (np.ndarray): Variance values for the state/measurement.
        label (str): Label for the variable.
    """
    sigma = 3 * np.sqrt(variances)
    
    ax.plot(times,residuals, marker="x", linestyle="none", label=f"{label} Residual")
    ax.plot(times,sigma, color="red", label=r"$\pm3\sigma$")
    ax.plot(times,-sigma, color="red")
    
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

