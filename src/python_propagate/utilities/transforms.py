"""
transforms.py:

This module contains functions for converting between reference frames and orbital elements.

Functions:
- classical2cart: Converts classical orbital elements to cartesian state vector.
- cart2classical: Converts cartesian state vector to classical orbital elements.
- mean2true: Converts mean anomaly to true anomaly.
- true2mean: Converts true anomaly to mean anomaly.

Author: Aaron Berkhoff
Date: 2025-01-30
"""

import numpy as np
from scipy.optimize import newton


def classical2cart(sma, ecc, inc, arg, raan, mu, nu=None, mean_anomaly=None):
    """Converts classical orbital elements to cartesian state vector."""

    if nu is None and mean_anomaly is None:
        raise ValueError("A true anomaly (nu) or mean anomaly (M) must be specified")

    if nu is None:
        nu, _ = mean2true(mean_anomaly, ecc)

    p = sma * (1 - ecc**2)
    r = p / (1 + ecc * np.cos(nu))

    r_perifocal = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    v_perifocal = np.array(
        [-np.sqrt(mu / p) * np.sin(nu), np.sqrt(mu / p) * (ecc + np.cos(nu)), 0]
    )

    transform = np.array(
        [
            [
                np.cos(raan) * np.cos(arg) - np.sin(raan) * np.sin(arg) * np.cos(inc),
                -np.cos(raan) * np.sin(arg) - np.sin(raan) * np.cos(arg) * np.cos(inc),
                np.sin(raan) * np.sin(inc),
            ],
            [
                np.sin(raan) * np.cos(arg) + np.cos(raan) * np.sin(arg) * np.cos(inc),
                -np.sin(raan) * np.sin(arg) + np.cos(raan) * np.cos(arg) * np.cos(inc),
                -np.cos(raan) * np.sin(inc),
            ],
            [np.sin(arg) * np.sin(inc), np.cos(arg) * np.sin(inc), np.cos(inc)],
        ]
    )

    xyz = transform @ r_perifocal

    vxyz = transform @ v_perifocal

    return np.hstack((xyz, vxyz))


def cart2classical(state, mu, nu_bool=True):
    """Converts cartesian state vector to classical orbital elements."""

    # x, y, z, vx, vy, vz = state

    r = np.linalg.norm(state[0:3])

    v = np.linalg.norm(state[3:6])

    z_hat = [0, 0, 1]

    h = np.cross(state[0:3], state[3:6])

    e_vector = (
        1
        / mu
        * (
            (v**2 - mu / r) * np.array(state[0:3])
            - np.dot(state[0:3], state[3:6]) * np.array(state[3:6])
        )
    )

    ecc = np.linalg.norm(e_vector)

    p = np.linalg.norm(h) ** 2 / mu

    sma = p / (1 - ecc**2)

    inc = np.arccos(h[2] / np.linalg.norm(h))

    # node line direction
    ahat = (np.cross(z_hat, h)) / (np.linalg.norm(np.cross(z_hat, h)))

    # find raan
    # xhat=[1, 0, 0]
    raan = np.arccos(np.dot([1, 0, 0], ahat))

    # Quad check
    if ahat[1] < 0:
        raan = 2 * np.pi - raan

    # find w
    arg = np.arccos((np.dot(ahat, e_vector)) / np.linalg.norm(e_vector))

    # quad check
    if e_vector[2] < 0:
        arg = 2 * np.pi - arg

    nu = np.arccos(np.dot(e_vector, state[0:3]) / (ecc * r))

    if np.dot(state[0:3], state[3:6]) < 0:
        nu = 2 * np.pi - nu

    if nu_bool:
        anomaly = nu

    else:
        eccentric_amomaly = np.arctan2(
            np.sqrt(1 - ecc**2) * np.sin(nu), ecc + np.cos(nu)
        )
        mean_anomaly = eccentric_amomaly - ecc * np.sin(eccentric_amomaly)
        anomaly = mean_anomaly

    return sma, ecc, inc, arg, raan, anomaly


def mean2true(mean, eccentricity):
    """Converts mean anomaly to true anomaly."""
    eccentric_anomaly = newton(lambda E: E - eccentricity * np.sin(E) - mean, x0=mean)
    true_anomaly = 2 * np.arctan(
        (
            np.sqrt((1 + eccentricity) / (1 - eccentricity))
            * np.tan(eccentric_anomaly / 2)
        )
    )

    true_anomaly = true_anomaly % (2 * np.pi)
    return true_anomaly, eccentric_anomaly


def true2mean(true_anomaly, eccentricity):
    """Converts true anomaly to mean anomaly."""
    eccentric_amomaly = 2 * np.arctan(
        np.sqrt((1 - eccentricity))
        / np.sqrt(1 + eccentricity)
        * np.tan(true_anomaly / 2)
    )
    mean_anomaly = eccentric_amomaly - eccentricity * np.sin(eccentric_amomaly)
    return mean_anomaly
