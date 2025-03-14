import numpy as np
from scipy.linalg import cho_solve, cho_factor


def mahalanobis_distance(x, y, covariance):
    diff = x - y

    # Use Cholesky decomposition for efficient solving
    c_factor = cho_factor(covariance, lower=True)
    mahal_dist_sq = diff.T @ cho_solve(c_factor, diff)

    return np.sqrt(mahal_dist_sq)[0, 0]


def calc_ellipse(mean, covariance, sigma=3, resolution=100, rotate = True, translate = True):

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(covariance)
    # Sort in descending order
    theta = np.linspace(0, 2 * np.pi, resolution)
    ellipse_points = np.array(
        [
            sigma * np.sqrt(eigenvals[0]) * np.cos(theta),
            sigma * np.sqrt(eigenvals[1]) * np.sin(theta),
        ]
    )

    if rotate:
        rotated_ellipse = eigenvecs @ ellipse_points
    else:
        rotated_ellipse = ellipse_points


    # The ellipse width and height: 2*nsigma*sqrt(eigenvalue)
    if translate:
        translated_ellipse = rotated_ellipse + mean
    else:
        translated_ellipse = rotated_ellipse

    return translated_ellipse, eigenvecs

def orbital_period(sma, mu):
    """
    Calculates the orbital period of an object.

    Parameters:
    sma (float): Semi-major axis in meters
    mu (float): Gravitational parameter (m^3/s^2)

    Returns:
    float: Orbital period in seconds
    """
    return 2 * np.pi * np.sqrt(sma**3 / mu)