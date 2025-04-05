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


def calc_shadow(state_agent, state_sun, reference_body_radius):

    rsat = np.linalg.norm(state_agent.position_eci)  # Satellite position vector (in km)
    rsun = np.linalg.norm(state_sun.position_eci)  # Sun position vector (in km)

    rsathat = state_agent.position_eci / rsat
    rsunhat = state_sun.position_eci / rsun

    # Angular radii (in radians) for umbra and penumbra
    alpha_umb = 0.264121687
    alpha_pen = 0.269007205

    cos_angle = np.dot(rsunhat, rsathat)

    shadow_bool = False
    shadow_value = 1.0

    if cos_angle < 0:
        sin_angle = np.sqrt(1.0 - cos_angle**2)

        # Projection of satellite in direction of light
        sat_horz = rsat * cos_angle
        sat_vert = rsat * sin_angle

        x = reference_body_radius / np.sin(alpha_pen)
        pen_vert = np.tan(alpha_pen) * (x + sat_horz)

        if sat_vert <= pen_vert:
            shadow_value = 0.5  # Penumbra

            y = reference_body_radius / np.sin(alpha_umb)
            umb_vert = np.tan(alpha_umb) * (y - sat_horz)

            if sat_vert <= umb_vert:
                shadow_value = 0.0  # Umbra
                shadow_bool = True

    return shadow_bool, shadow_value
