"""
Useful constants for the python_propagate module.
"""

SPEEDOFLIGHT = 299792458  # Speed of light in vacuum in meters per second (m/s)
PHI = 1367.0  # Solar constant (W/m^2), average solar flux at 1 AU from the Sun
AU = 149597870.7  # Astronomical Unit in kilometers (km), average distance from Earth to Sun
C1 = (
    PHI / SPEEDOFLIGHT
)  # Constant used in SRP calculations, derived from solar constant and speed of light
