from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose

from python_propagate.scenario import Scenario
from python_propagate.environment.planets import Earth
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State
from datetime import datetime, timedelta
from python_propagate.dynamics.keplerian import Keplerian


def test_accel_are_equal_kepler():
    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(
        central_body=earth, start_time=start_time, duration=duration, dt=dt
    )
    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    initial_state = State(position=position, velocity=velocity)

    # ax_j2, ay_j2, az_j2 = J2.J2_motion(initial_state(),time=None,scenario=scenario)
    kepler = Keplerian(scenario=scenario)

    result = kepler(initial_state, None)

    actual_accel = result.acceleration[:, np.newaxis]

    data = loadmat("tests\data\Dynamics_ComparisonResults.mat")
    expected_accel = data["accel_TwoBody"][3:6]
    diff = actual_accel - expected_accel
    assert_allclose(actual_accel, expected_accel, rtol=0, atol=1e-16)


def test_end_states_are_equal_kepler():
    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(
        central_body=earth, start_time=start_time, duration=duration, dt=dt
    )

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]

    initial_state = State(position=position, velocity=velocity)

    jah_sat = Spacecraft(
        initial_state, start_time=start_time, duration=duration, dt=scenario.dt
    )
    jah_sat.set_scenario(scenario=scenario)
    dynamics = ("kepler",)
    jah_sat.add_dynamics(dynamics=dynamics)

    # scenario.add_agent(jah_sat)
    # scenario.add_dynamics((J2.J2_motion,J3.J3_motion,keplerian.keplerian_motion))

    jah_sat.propagate()

    actual_end = jah_sat.state.compile()[np.newaxis, :]
    data = loadmat("tests\data\Dynamics_ComparisonResults.mat")
    expected_end = data["endState_TwoBody"]
    assert_allclose(actual_end, expected_end, rtol=1e-7)


if __name__ == "__main__":

    test_accel_are_equal_kepler()
    test_end_states_are_equal_kepler()
