import numpy as np
from scipy.linalg import cho_solve, cho_factor


def mahalanobis_distance(x, y, covariance):
    diff = x - y

    # Use Cholesky decomposition for efficient solving
    c_factor = cho_factor(covariance, lower=True)
    mahal_dist_sq = diff.T @ cho_solve(c_factor, diff)

    return np.sqrt(mahal_dist_sq)[0, 0]


def calc_ellipse(mean, covariance, sigma=3, resolution=100):

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

    rotated_ellipse = eigenvecs @ ellipse_points

    # The ellipse width and height: 2*nsigma*sqrt(eigenvalue)
    translated_ellipse = rotated_ellipse + mean

    return translated_ellipse
