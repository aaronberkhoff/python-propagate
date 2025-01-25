import pytest
import numpy as np
from numpy.testing import assert_allclose

from python_propagate.Scenario import Scenario
from python_propagate.Environment.Planets import Earth
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Agents import State
from datetime import datetime, timedelta
from python_propagate.Dynamics.stm import STM
from scipy.io import loadmat

def test_stm_acceleration():

    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    initial_state = State(position=position,velocity=velocity,stm=np.eye(6))

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 / 1000**2

    jah_sat = Spacecraft(initial_state,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)

    stm = STM(scenario=scenario,agent=jah_sat)

    result = stm(initial_state,None)

    actual_accel = result.stm_dot.flatten()[:,np.newaxis]
    
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_accel = data['accel_All'][6:]
    diff = (actual_accel - expected_accel)
    # diff = (actual_accel - expected_accel).reshape((6,6))

    assert_allclose(actual_accel, expected_accel,rtol = 0, atol = 1e-9)

def test_stm_final():
    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)
    position = np.array([1340.745, -6663.403, -132.528])
    velocity = np.array([5.457807, 1.368701, -5.614317])

    state1 = State(position=np.array(position),velocity=np.array(velocity),stm = np.eye(6))
    state2 = State(position=np.array(position)* (1+10e-7),velocity=np.array(velocity)* (1+10e-7),stm = np.eye(6))

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 * 1e-6

    jah_sat1 = Spacecraft(state1,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    jah_sat2 = Spacecraft(state2,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    jah_sat1.set_scenario(scenario=scenario)
    jah_sat2.set_scenario(scenario=scenario)

    dynamics = ('kepler','J2','J3','drag','stm')
    jah_sat1.add_dynamics(dynamics=dynamics)
    jah_sat2.add_dynamics(dynamics=dynamics)
    

    #First propagate
    jah_sat1.propagate()
    xf = jah_sat1.state
    stmf = xf.stm

    delta_x0 = np.hstack((state1.position * 10e-7, state1.velocity * 10e-7))
    delta_xf = stmf @ delta_x0


    #second propagate
    jah_sat2.propagate()

    xf_test = jah_sat2.state

    #test
    delta_xf2_pos  = (xf.position - xf_test.position)
    delta_xf2_velo = (xf.velocity - xf_test.velocity)

    delta_xf2 = np.hstack((delta_xf2_pos,delta_xf2_velo))

    diff_xf = delta_xf2 - delta_xf

    assert delta_xf2 == pytest.approx(delta_xf, rel=1e-6)  # Relative tolerance


    pass




if __name__ == "__main__":

    test_stm_acceleration()
    test_stm_final()




