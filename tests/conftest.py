import pytest
from python_propagate.Scenario import Scenario
from python_propagate.Environment.Planets import Earth
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Agents import State
from datetime import datetime, timedelta
from python_propagate.Dynamics import J2, J3, keplerian
from scipy.io import loadmat

@pytest.fixture
def test_scenario():
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
    
    return {"sat":jah_sat, "scenario":scenario, "initial_state":initial_state}
