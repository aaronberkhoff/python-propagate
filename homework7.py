from collections import namedtuple
from scipy.io import loadmat
from scipy.stats import chi2
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import scipy as sp
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


from python_propagate.constructors.yaml_constructors import load_yaml

from python_propagate.sensors.optical import Optical
from python_propagate.utilities.units import ARC2RAD, RAD2DEG
from python_propagate.filters.jpda import JPDA

from python_propagate.agents import Agent
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.filters.iod import admissible_region, admissible_probabilities
from python_propagate.utilities.calculations import calc_ellipse
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.states import State

from python_propagate.plots.residual_plot import plot_residuals
from python_propagate.filters.gaussian_mixture import GaussianMixture

from python_propagate.utilities.transforms import reconstruct_sigma_points

import pickle

Parameters = namedtuple(
    "Parameters", ("covariances", "means", "observations", "times", "truths", "data2")
)


def plot_zl(data_handlers, path):

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharey=True, sharex=True)

    for i, (ax, data_handler) in enumerate(zip(axs, data_handlers)):

        missed = len(
            [miss for miss in data_handler.metadata[f"num_zl{i}"] if miss == 0]
        )
        ax.plot(data_handler.measurement_times, data_handler.metadata[f"num_zl{i}"])

        ax.set_title(f"Object {i} - Missed Detections: {missed}")
        ax.set_ylabel("Number of Components")
        ax.grid(True)

    axs[-1].set_xlabel("Time [HR]")

    fig.savefig(path)


def load_parameters():

    datap1 = loadmat("homework7/data/hw07_data.mat")
    datap2 = np.loadtxt("homework7/data/hw07_prob2_data.dat")

    covariances = np.array([datap1["P_apriori"], datap1["P_apriori"]])
    means = np.squeeze(np.array([datap1["X_apriori1"], datap1["X_apriori2"]]))
    obs = datap1["observations"].transpose(2, 1, 0)
    # ob2 = datap2['observations'].transpose(2,1,0)
    times = datap1["times"]
    truths = np.array([datap1["truStates1"], datap1["truStates2"]]).transpose(1, 0, 2)
    # weights = datap1["weights"].T
    # measurements = data[:, 1:3]
    # truths = data[:, 3:]
    # times = data[:, 0]

    return Parameters(
        covariances=covariances,
        means=means,
        observations=obs,
        data2=datap2,
        times=times,
        truths=truths,
    )


def load_scenario():

    data = load_yaml("homework7/config.yaml")
    delta = (10 * ARC2RAD) ** 2
    sensor = Optical(noise_mean=[0, 0], noise_covariance=delta * np.eye(2))
    spacecraft = data["agents"][0]

    return spacecraft, sensor


def ra_dec_to_cartesian(observation, station_state):

    ra, dec, ra_rate, dec_rate, rho, rhodot = observation.compile().tolist()
    station_position = station_state.position
    station_velocity = station_state.velocity

    up = np.array([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
    ualpha = np.array([-np.sin(ra) * np.cos(dec), np.cos(ra) * np.cos(dec), 0])
    udelta = np.array(
        [-np.cos(ra) * np.sin(dec), -np.sin(ra) * np.sin(dec), np.cos(dec)]
    )

    rho_vec = rho * up
    rho_dot_vec = rhodot * up + rho * ra_rate * ualpha + rho * dec_rate * udelta

    position = station_position + rho_vec
    velocity = station_velocity + rho_dot_vec

    return np.concatenate([position, velocity])


def calculate_range_and_range_rate_from_target(station_state, state):
    """Calculates the range and range rate from the station to the target agent."""
    diff_x = station_state.position[0] - state[0]
    diff_y = station_state.position[1] - state[1]
    diff_z = station_state.position[2] - state[2]

    rho = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

    diff_vx = station_state.velocity[0] - state[3]
    diff_vy = station_state.velocity[1] - state[4]
    diff_vz = station_state.velocity[2] - state[5]

    rho_dot = (diff_x * diff_vx + diff_y * diff_vy + diff_z * diff_vz) / rho

    return rho, rho_dot


def problem1(params, spacecraft, sensor):

    jpda = JPDA(
        agent=spacecraft,
        means=params.means,
        covariances=params.covariances,
        sensor=sensor,
        process_noise_mean=[0, 0, 0],
        process_noise_covariance=1e-16,
        probability_of_detection=0.9,
        probability_of_gating=0.99,
        clutter_poisson_mean=1.5,
    )

    # jpda.run(observations=params.observations, times=params.times)

    # data_handler_list = [data for data in jpda.data_handler]

    # # # # Save the instance to a file
    # for i,data_handler in enumerate(data_handler_list):
    #     with open(f'homework7/data_handler{i}.pkl', 'wb') as file:
    #         pickle.dump(data_handler, file, protocol=pickle.HIGHEST_PROTOCOL)

    data_handler_list = []

    for i, data_handler in enumerate(jpda.data_handler):
        with open(f"homework7/data_handler{i}.pkl", "rb") as file:
            data_handler_list.append(pickle.load(file))

    for i, data_handler in enumerate(data_handler_list):
        data_handler.truths = params.truths[:, i, :]
        plot_residuals(
            data_handler=data_handler,
            path=f"homework7/latex/figures/problem1_residual_meas_plot_agent{i}.png",
            plot_type="measurement",
            show=False,
        )

        plot_residuals(
            data_handler=data_handler,
            path=f"homework7/latex/figures/problem1_residual_pos_plot_agent{i}.png",
            plot_type="position",
            show=False,
        )

        plot_residuals(
            data_handler=data_handler,
            path=f"homework7/latex/figures/problem1_residual_vel_plot_agent{i}.png",
            plot_type="velocity",
            show=False,
        )

        # plot_residuals(
        #     data_handler=data_handler,
        #     path=f"homework7/latex/figures/problem1_residual_merged_plot_agent{i}.png",
        #     plot_type="merged",
        #     show=True,
        # )

    plot_zl(
        data_handlers=data_handler_list,
        path="homework7/latex/figures/problem1_missed.png",
    )

    plt.show()

    pass


def problem2(params, spacecraft, sensor):

    data2 = params.data2
    # data_gmm = loadmat("homework3/data/hw03_gmm.mat")

    measurements = data2[:, 1:3]
    observation = np.array([0.87406, -0.07313, 7.29274e-5, 6.81829e-7])
    times = data2[:, 0]

    # weights = data_gmm["weights"].T
    # covariances = data_gmm["covars"]
    # means = data_gmm["means"]

    sigma_rho = 75.0**2
    sigma_rhodot = (0.05) ** 2

    truth = data2[:, 3:]

    sensor_state = State(
        position=np.array([5655.719, 1925.262, 2234.277]),
        velocity=np.array([-0.1404001, 0.4121821, 0.0002260]),
    )

    range_data = np.arange(35300, 38500 + 100, 100)
    range_rate_data = np.arange(-0.4, 0.4 + 0.02, 0.02)

    sma_min = 36000
    sma_max = 47000
    ecc_max = 0.1

    admissible = admissible_region(
        range_data,
        range_rate_data,
        observation,
        sensor_state,
        sma_min=sma_min,
        sma_max=sma_max,
        ecc_max=ecc_max,
    )

    weights = (np.ones(len(admissible)) / len(admissible)).reshape(-1, 1)
    means = []
    covariances = []
    sigma_ra, sigma_dec = (10 * ARC2RAD) ** 2, (10 * ARC2RAD) ** 2
    sigma_radot, sigma_decdot = 1e-16, 1e-16

    covariance = np.diag(
        [sigma_ra, sigma_dec, sigma_radot, sigma_decdot, sigma_rho, sigma_rhodot]
    )
    cnt = 1
    # for ranges in admissible:

    #     mean = np.concatenate([observation,ranges]).reshape(-1,1)
    #     sigma_point = SigmaPoints(mean=mean, covariance=covariance)

    #     state_sigma_point = np.vstack([ra_dec_to_cartesian(obs,station_state=sensor_state) for obs in sigma_point]).T

    #     state_mean, state_covariance = reconstruct_sigma_points(state_sigma_point,sigma_point.weights)

    #     sigma_point = SigmaPoints(mean=state_mean,covariance=state_covariance)

    #     sigma_point.propagate(agent = spacecraft, duration = 6000, parallel=False)

    #     state_mean, state_covariance = sigma_point.reconstruct_state()

    #     print(f'State: {cnt}')
    #     cnt += 1

    #     try:
    #        np.linalg.cholesky(state_covariance)
    #     except np.linalg.LinAlgError as e:
    #         raise e

    #     means.append(state_mean)
    #     covariances.append(state_covariance)

    # means = np.squeeze(np.array(means))
    # covariances = np.array(covariances)

    # np.save('homework7/data/means.npy', means)
    # np.save('homework7/data/covariances.npy', covariances)

    means = np.load("homework7/data/means.npy")
    covariances = np.load("homework7/data/covariances.npy")

    gaussian_mixture = GaussianMixture(
        means=means,
        covariances=covariances,
        weights=weights,
        sensor=sensor,
        process_noise_mean=[0, 0, 0],
        process_noise_covariance=1e-16,
        filter=UnscentedKalman,
        agent=spacecraft,
    )

    gaussian_mixture.data_handler.truths = truth

    filter_state, _ = gaussian_mixture.run(measurements=measurements,times=times)

    # np.save('homework7/data/final_mean.npy',filter_state.mean)
    # np.save('homework7/data/final_covariance.npy',filter_state.covariance)

    mean = np.load("homework7/data/final_mean.npy")
    cov = np.load("homework7/data/final_covariance.npy")

    # final_sigma = SigmaPoints(mean=filter_state.mean,covariance=filter_state.covariance)
    final_sigma = SigmaPoints(mean=mean, covariance=cov)

    final_sigma.propagate(agent=spacecraft, duration=-times[-1], parallel=True)

    meas_sigma = np.vstack(
        [
            calculate_range_and_range_rate_from_target(sensor_state, state)
            for state in final_sigma.state.T
        ]
    ).T

    meas_mean, meas_cov = reconstruct_sigma_points(
        meas_sigma, weights=final_sigma.weights
    )

    ellip = calc_ellipse(mean=meas_mean, covariance=meas_cov)

    ellip[0] = ellip[0] / spacecraft.scenario.central_body.radius

    plot_residuals(
            data_handler=gaussian_mixture.data_handler,
            path=f"homework7/latex/figures/problem2_residual_meas_plot_agent.png",
            plot_type="measurement",
            show=False,
        )

    plot_residuals(
        data_handler=gaussian_mixture.data_handler,
        path=f"homework7/latex/figures/problem2_residual_pos_plot_agent.png",
        plot_type="position",
        show=False,
    )

    plot_residuals(
        data_handler=gaussian_mixture.data_handler,
        path=f"homework7/latex/figures/problem2_residual_vel_plot_agent.png",
        plot_type="velocity",
        show=False,
    )

    # plot_residuals(
    #     data_handler=data_handler,
    #     path=f"homework7/latex/figures/problem1_residual_merged_plot_agent{i}.png",
    #     plot_type="merged",
    #     show=True,
    # )

    # plt.show()

    # # Create a grid of points
    X, Y = np.meshgrid(range_data / 6378.1363, range_rate_data)
    X_energy, Y_energy = np.meshgrid(admissible[:, 0], admissible[:, 1])

    # Compute the convex hull
    hull = ConvexHull(admissible)

    # Extract the boundary points
    boundary_points = admissible[hull.vertices]
    boundary_points = np.vstack([boundary_points, boundary_points[0]])

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the grid of points
    ax.scatter(
        X, Y, color="black", marker="o", facecolors="none", s=10
    )  # s=10 sets the size of the points
    ax.scatter(
        admissible[:, 0] / 6378.1363,
        admissible[:, 1],
        color="red",
        marker="o",
        s=40,
    )  # s=10 sets the size of the points
    ax.plot(
        boundary_points[:, 0] / 6378.1363,
        boundary_points[:, 1],
        color="blue",
        marker="none",
    )
    # plt.fill(admissible[:,0], admissible_energy[:,1], color='cyan', alpha=0.3)
    ax.fill(ellip[0], ellip[1], label=r"3$\sigma$", color="blue", alpha=0.5)
    ax.plot(meas_mean[0], meas_mean[1], label=r"Mean", color="blue", marker="x")
    ax.set_xlim([5.54, 6.0])
    ax.set_ylim([-0.41, 0.41])
    ax.set_xlabel("Range [Earth Radii]")
    ax.set_ylabel("Range Rate [km/s]")
    ax.grid(True)  # Add a grid for better visualization
    plt.savefig("homework7/latex/figures/admissible_region.png")

    plt.show()

    pass


if __name__ == "__main__":
    params = load_parameters()
    jones_sat1, sensor = load_scenario()
    # problem1(params=params, spacecraft=jones_sat1, sensor=sensor)
    problem2(params=params, spacecraft=jones_sat1, sensor=sensor)
