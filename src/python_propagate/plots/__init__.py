import numpy as np
import matplotlib.pyplot as plt

class Plot:

    def __init__(self):
        pass


def remove_duplicate_legend(ax=None):
    """
    Removes duplicate labels from a Matplotlib legend.
    
    Parameters:
        ax (matplotlib.axes.Axes, optional): The axes object to clean up.
                                             If None, uses the current axes.
    """
    if ax is None:
        ax = plt.gca()  # Get current axes if not provided
    
    # Get all handles and labels
    handles, labels = ax.get_legend_handles_labels()

    # Remove duplicates while maintaining order
    unique = dict(zip(labels, handles))

    # Set new legend with unique labels
    ax.legend(unique.values(), unique.keys())