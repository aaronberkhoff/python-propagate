"""
units.py

This module contains functions for converting between units.

Functions:
- rad2deg: Converts radians to degrees.
- deg2rad: Converts degrees to radians.

Constants:
- RAD2DEG: Conversion factor from radians to degrees.
- DEG2RAD: Conversion factor from degrees to radians.
- DEG2ARC: Conversion factor from degrees to arcseconds.
- ARC2DEG: Conversion factor from arcseconds to degrees.
- RAD2ARC: Conversion factor from radians to arcseconds.
- ARC2RAD: Conversion factor from arcseconds to radians.

Author: Aaron Berkhoff
Date: 2025-01-30
"""

import numpy as np

RAD2DEG = 180 / np.pi
DEG2RAD = np.pi / 180
DEG2ARC = 3600
ARC2DEG = 1 / 3600
RAD2ARC = RAD2DEG * DEG2ARC
ARC2RAD = ARC2DEG * DEG2RAD


def rad2deg(radians: float):
    """Converts radians to degrees."""
    return radians * RAD2DEG


def deg2rad(radians: float):
    """Converts degrees to radians."""
    return radians * DEG2RAD
