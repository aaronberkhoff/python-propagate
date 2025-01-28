import pytest
import numpy as np
from numpy.testing import assert_allclose

from python_propagate.Scenario import Scenario
from python_propagate.Environment.Planets import Earth
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Agents import State
from datetime import datetime, timedelta
from python_propagate.Dynamics.keplerian import Keplerian
from python_propagate.Dynamics.J2 import J2
from python_propagate.Dynamics.J3 import J3
from python_propagate.Dynamics.drag import Drag
from scipy.io import loadmat



def test_accel_are_equal_drag():

    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 / (1000 ** 2)
    initial_state = State(position=position,velocity=velocity)

    jah_sat = Spacecraft(initial_state,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)

    drag = Drag(scenario=scenario,agent=jah_sat)

    result = drag(initial_state,None)

    actual_accel = result.acceleration[:,np.newaxis]
    
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_accel = data['accel_All'][3:6] - data['accel_TwoBody_J2_J3'][3:6]
    diff = (actual_accel - expected_accel)
    assert_allclose(actual_accel, expected_accel,rtol = 0.0, atol = 1e-18)



def test_end_states_are_equal_drag():
    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]
    initial_state = State(position=position,velocity=velocity)

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 * 1e-6

    jah_sat = Spacecraft(initial_state,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    jah_sat.set_scenario(scenario=scenario)
    dynamics = ('kepler','J2','J3','drag',)
    jah_sat.add_dynamics(dynamics=dynamics)

    # scenario.add_agent(jah_sat)
    # scenario.add_dynamics((J2.J2_motion,J3.J3_motion,keplerian.keplerian_motion))

    jah_sat.propagate()

    actual_end = jah_sat.state.compile()[np.newaxis,:]
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_end = data['endState_All']
    assert_allclose(actual_end, expected_end,rtol = 1e-2)



if __name__ == "__main__":

    test_accel_are_equal_drag()
    test_end_states_are_equal_drag()