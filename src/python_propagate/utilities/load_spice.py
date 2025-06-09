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
from pathlib import Path

# Get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# TODO Have the user indicate which spice files to download

def load_spice():
    """Load the necessary SPICE kernels for the simulation."""
    spice.furnsh(str(DATA_DIR / "naif0011.tls"))
    spice.furnsh(str(DATA_DIR / "earth_000101_250421_250124.bpc"))
    # spice.furnsh(str(DATA_DIR / "de430.bsp"))
    spice.furnsh(str(DATA_DIR / "de442.bsp"))
    # spice.furnsh(DATA_DIR / "pck00010.tpc")

def unload_spice():
    """Unload the SPICE kernels from the simulation."""
    spice.unload((DATA_DIR / "naif0011.tls"))
    spice.unload((DATA_DIR / "earth_000101_250421_250124.bpc"))
    # spice.unload((DATA_DIR / "de430.bsp"))
    spice.unload(str(DATA_DIR / "de442.bsp"))
    # spice.unload(DATA_DIR / "pck00010.tpc")
