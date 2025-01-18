import pytest
import numpy as np
from numpy.testing import assert_allclose

from python_propagate.Scenario import Scenario
from python_propagate.Environment.Planets import Planet, initialize_planet
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Agents import State
from datetime import datetime, timedelta
from python_propagate.Dynamics import J2, J3, keplerian
from scipy.io import loadmat



def test_accel_are_equal_kepler():

    earth = initialize_planet('Earth')

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)
    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    initial_state = State(position=position,velocity=velocity)

    

    # ax_j2, ay_j2, az_j2 = J2.J2_motion(initial_state(),time=None,scenario=scenario)
    ax_keplerian, ay_keplerian, az_keplerian = keplerian.keplerian_motion(initial_state(),time=None,scenario=scenario)

    ax = ax_keplerian
    ay = ay_keplerian
    az = az_keplerian

    # temp = 
    actual_accel = np.array([ax,ay,az])[:,np.newaxis]
    
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_accel = data['accel_TwoBody'][3:6]
    assert_allclose(actual_accel, expected_accel,rtol = 1e-7)






def test_end_states_are_equal_kepler():
    earth = initialize_planet('Earth')

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    initial_state = State(position=position,velocity=velocity)

    jah_sat = Spacecraft(initial_state,start_time=start_time,duration=duration, dt = scenario.dt)
    jah_sat.set_scenario(scenario=scenario)
    jah_sat.add_dynamics((keplerian.keplerian_motion,))

    # scenario.add_agent(jah_sat)
    # scenario.add_dynamics((J2.J2_motion,J3.J3_motion,keplerian.keplerian_motion))

    jah_sat.propagate()

    actual_end = jah_sat.state()[np.newaxis,:]
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_end = data['endState_TwoBody']
    assert_allclose(actual_end, expected_end,rtol = 1e-7)


if __name__ == "__main__":

    test_accel_are_equal_kepler()
    test_end_states_are_equal_kepler()



