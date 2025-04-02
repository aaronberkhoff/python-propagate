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
# from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.states.unscented_state import FilterState
from python_propagate.utilities.units import ARC2RAD
from python_propagate.forge.genes import Gene, Genes
from python_propagate.filters.hierarchical_mixture_of_experts import HierarchicalMixtureExperts

from python_propagate.sensors.optical import Optical
from python_propagate.dynamics import Dynamic
from collections import namedtuple

np.random.seed(100)


def load_scenario():

    # PARAMTERS
    start_time = datetime.strptime("2024-03-03T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(minutes=10)
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




def test_hme():
    tic = time.time()
    state_bar = np.array([-2011.990, -382.065, 6316.376, 5.419783, -5.945319, 1.37398])


    indentity = np.eye(3)
    zeros = np.zeros((3, 3))

    covariance_bar = np.block([[indentity, zeros], [zeros, indentity * 1e-6]])

    jones_sat = load_scenario()

    jones_sat.state = State(position=state_bar[0:3], velocity=state_bar[3:6])

    # sensor noise
    delta = (10 * ARC2RAD)**2
    noise_mean=np.zeros(2)
    noise_covariance=delta*np.eye(2)
    optical = Optical(noise_mean, noise_covariance)
   

    # process noise
    process_noise_mean = np.zeros(3)
    process_noise_covariance = 1e-16

    # data
    n = -1
    data = np.loadtxt("/Users/vanhaslett/Documents/MATLAB/hw02_data.dat")

    data_lenth = 16 # len(data)
    time_hist = data[:, 0][0:data_lenth]
    measurments = data[:, 1:3][0:data_lenth]
    truths = data[:, 3:][0:data_lenth]

    # Create gene agents
    gene = Gene(agent_base=jones_sat, attribute="area", samples=3, area_min=1, area_max=50)

    hierarchical_mixture = HierarchicalMixtureExperts(
        gene=gene,
        mean=state_bar,
        covariance=covariance_bar,
        sensor=optical,
        noise_mean=noise_mean,
        noise_covariance=noise_covariance,
        process_noise_mean=process_noise_mean,
        process_noise_covariance=process_noise_covariance,
    )


    expert_choice, ai_list = hierarchical_mixture.run(measurments, time_hist)

    # Plot each column of ai_list against the time array
    plt.figure(figsize=(10, 6))
    for i in range(gene.samples):
        line_width = 4 - (i / 2)  # Decrease line width as i increases
        plt.plot(time_hist, ai_list[:,i], label=f"Expert {i+1}", linewidth=line_width)

    # Add labels and title
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bank Choice Certainty')
    plt.title('Plot of Epxerts against Time')
    plt.legend()
    plt.grid(True)

    # Save plot
    # plt.show()
    plt.savefig('/Users/vanhaslett/Development/python-propagate/tests/results/hme_JAH_test_plot.png')



    toc = time.time()
    print("Runtime: ", toc - tic, "seconds.")

    pass


if __name__ == "__main__":

    test_hme()