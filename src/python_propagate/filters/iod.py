import numpy as np
import itertools
from scipy.stats import multivariate_normal


def admissible_region(
    range_data,
    range_rate_data,
    measurements,
    station_state,
    sma_min,
    sma_max,
    ecc_max,
    mu=398600.4415,
):

    # define constants
    energy_min = -mu / (2 * sma_min)
    energy_max = -mu / (2 * sma_max)
    # determine the radius of the measurement
    ra = measurements[0]
    dec = measurements[1]

    ra_rate = measurements[2]
    dec_rate = measurements[3]

    station_position = station_state.position
    station_velocity = station_state.velocity

    admissible_region = []

    for rho in range_data:

        for rhodot in range_rate_data:
            # sma constraint
            up = np.array(
                [np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)]
            )
            ualpha = np.array([-np.sin(ra) * np.cos(dec), np.cos(ra) * np.cos(dec), 0])
            udelta = np.array(
                [-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)]
            )

            w0 = np.linalg.norm(station_position) ** 2
            w1 = 2 * (np.dot(station_velocity, up))
            w2 = ra_rate**2 * np.cos(dec) ** 2 + dec_rate**2
            w3 = 2 * ra_rate * np.dot(station_velocity, ualpha) + 2 * dec_rate * np.dot(
                station_velocity, udelta
            )
            w4 = np.linalg.norm(station_velocity) ** 2
            w5 = 2 * np.dot(station_position, up)

            radius2 = rho**2 + w5 * rho + w0
            velo2 = rhodot**2 + w1 * rhodot + w2 * rho**2 + w3 * rho + w4

            energy = velo2 / 2 - mu / np.sqrt(radius2)

            ##Functions
            fof_rho = (
                w2 * rho**2 + w3 * rho + w4 - 2 * mu / np.sqrt(rho**2 + w5 * rho + w0)
            )

            sma_test_min = rhodot**2 + w1 * rhodot + fof_rho - 2 * energy_min
            sma_test_max = rhodot**2 + w1 * rhodot + fof_rho - 2 * energy_max

            # eccentricity constraint
            h1 = np.cross(station_position, up)
            h2 = np.cross(up, ra_rate * ualpha + dec_rate * udelta)
            h3 = np.cross(up, station_velocity) + np.cross(
                station_position, ra_rate * ualpha + dec_rate * udelta
            )
            h4 = np.cross(station_position, station_velocity)

            angular_momentum = h1 * rhodot + h2 * rho**2 + h3 * rho + h4

            ecc = np.sqrt(
                1 + 2 * energy * np.linalg.norm(angular_momentum) ** 2 / mu**2
            )

            ##define scalars
            c0 = np.linalg.norm(h1) ** 2
            c1 = 2 * np.dot(h1, h2)
            c2 = 2 * np.dot(h1, h3)

            c3 = 2 * np.dot(h1, h4)
            c4 = np.linalg.norm(h2) ** 2
            c5 = 2 * np.dot(h2, h3)

            c6 = 2 * np.dot(h2, h4) + np.linalg.norm(h3) ** 2
            c7 = 2 * np.dot(h3, h4)
            c8 = np.linalg.norm(h4) ** 2

            # functions
            pof_rho = c1 * rho**2 + c2 * rho + c3
            uof_rho = c4 * rho**4 + c5 * rho**3 + c6 * rho**2 + c7 * rho + c8

            # final constants
            a4 = c0
            a3 = pof_rho + c0 * w1
            a2 = uof_rho + c0 * fof_rho + pof_rho * w1
            a1 = fof_rho * pof_rho + w1 * uof_rho
            a0 = fof_rho * uof_rho + mu**2 * (1 - ecc_max**2)

            ecc_test = (
                a4 * rhodot**4 + a3 * rhodot**3 + a2 * rhodot**2 + a1 * rhodot + a0
            )

            if sma_test_min >= 0 and sma_test_max <= 0:

                if ecc_test <= 0:

                    # print(f"Energy: {energy} rho: {rho} rhodot: {rhodot}")

                    admissible_region.append((rho, rhodot))

    return np.array(admissible_region)


def admissible_probabilities(measurement, covariances, admissible_region, weights):

    probabilities = []

    samples = len(weights)

    range_data = np.linspace(35300, 38500, samples)
    range_rate_data = np.linspace(-0.40, 0.40, samples)

    covariance = np.eye(len(covariances)) * np.asarray(covariances)
    # cnt = 0
    for rho, rhodot in itertools.product(range_data, range_rate_data):
        prob = 0
        for wi, mean in zip(weights, admissible_region):

            prob += wi * multivariate_normal.pdf(
                np.concatenate((measurement, rho, rhodot)), covariance
            )

        probabilities.append((rho, rhodot, prob))
        # cnt += 1
        # print(cnt)
