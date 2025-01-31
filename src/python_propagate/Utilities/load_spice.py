"""
load_spice.py

This module contains functions for loading and unloading SPICE kernels.

functions:
- load_spice: Load the necessary SPICE kernels for the simulation.
- unload_spice: Unload the SPICE kernels from the simulation.

Author: Aaron Berkhoff
Date: 2025-01-30
"""

import spiceypy as spice

# TODO Have the user indicate which spice files to download


def load_spice():
    """Load the necessary SPICE kernels for the simulation."""
    spice.furnsh("data/naif0011.tls")
    spice.furnsh("data/earth_000101_250421_250124.bpc")
    # spice.furnsh('data/pck00010.tpc')
    # spice.furnsh('data/de430.bsp')


def unload_spice():
    """Unload the SPICE kernels from the simulation."""
    spice.unload("data/naif0011.tls")
    spice.unload("data/earth_000101_250421_250124.bpc")
    # spice.unload('data/pck00010.tpc')
    # spice.unload('data/de430.bsp')
