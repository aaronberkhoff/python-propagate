from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from collections import OrderedDict

import time


from python_propagate.states import State
from python_propagate.states.sigma_points import SigmaPoints

from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.scenario import Scenario
from python_propagate.environment.planets import Earth
from python_propagate.utilities.transforms import (
    inertial_to_ric,
    unscented_transform,
    reconstruct_sigma_points,
    unscented_weights,
)
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.states.unscented_state import FilterState
from python_propagate.utilities.units import ARC2RAD

from python_propagate.sensors.optical import Optical
from python_propagate.dynamics import Dynamic

np.random.seed(100)


def load_scenario():

    # PARAMTERS
    start_time = datetime.strptime("2024-03-03T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(hours=24)
    dt = timedelta(seconds=30)
    mass = 1350
    area = 3.6
    cd = 2.0

    # SCENARIO
    earth = Earth()
    scenario = Scenario(
        central_body=earth,
        start_time=start_time,
        duration=duration,
        dt=dt,
        use_spice=False,
    )

    # SPACEOBJECT
    jones_sat = Spacecraft(
        state=None,
        start_time=start_time,
        duration=duration,
        dt=scenario.dt,
        coefficient_of_drag=cd,
        mass=mass,
        area=area,
    )

    jones_sat.set_scenario(scenario=scenario)
    dynamics = ("kepler", "J2", "J3", "drag")
    jones_sat.add_dynamics(dynamics=dynamics)

    return jones_sat




def test_ukf():
    tic = time.time()
    state_bar = np.array([-2011.990, -382.065, 6316.376, 5.419783, -5.945319, 1.37398])

    indentity = np.eye(3)
    zeros = np.zeros((3, 3))

    covariance_bar = np.block([[indentity, zeros], [zeros, indentity * 1e-6]])

    jones_sat = load_scenario()

    jones_sat.state = State(position=state_bar[0:3], velocity=state_bar[3:6])

    # noise
    delta = (10 * ARC2RAD)**2
    optical = Optical(noise_mean=[0,0], noise_covariance=delta*np.eye(2))

    # process noise
    process_noise_mean = [0,0,0]
    process_noise_covariance = 1e-16

    # sensors
    # optical = Optical(noise_mean=noise_mean, noise_covariance=noise_covariance)

    # data
    n = -1
    data = np.loadtxt("/Users/vanhaslett/Documents/MATLAB/hw02_data.dat")
    time_hist = data[:, 0][0:]
    measurments = data[:, 1:3][0:]
    truths = data[:, 3:][0:]


    unscented_kalman = UnscentedKalman(
        spacecraft=jones_sat,
        mean=jones_sat.state.compile(),
        covariance=covariance_bar,
        sensor=optical,
        process_noise_mean=process_noise_mean,
        process_noise_covariance=process_noise_covariance,
    )


    state_estimate, covariance_estimate = unscented_kalman.run(measurements=measurments,time=time_hist)

    residual_state_hist = unscented_kalman.residuals_data
    estimated_covariance_hist = unscented_kalman.covariance_estimate_hist

    # print(state_estimate)
    # print(covariance_estimate)
    # unscented_kalman.plot_res(path="homework2/results/residuals_plot.png")
    # unscented_kalman.plot_state_res(
    #     path="homework2/results/state_residuals_plot.png", truths=truths
    # )
    toc = time.time()
    print("Runtime: ", toc - tic, "seconds.")

    pass


if __name__ == "__main__":
 
    test_ukf()