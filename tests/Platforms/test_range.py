import pytest
from datetime import datetime
import numpy as np
from python_propagate.Platforms import Platform
from python_propagate.Platforms.station import Station
from python_propagate.Scenario import Scenario
from datetime import datetime, timedelta
from python_propagate.Environment.Planets import Earth
from python_propagate.Utilities.units import RAD2DEG, DEG2RAD
from python_propagate.Agents.state import State
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Utilities.load_spice import load_spice
import spiceypy as spice

TARGET = 'EARTH'
ECI = 'J2000'
ECEF = 'ITRF93'

@pytest.fixture
def scenario():
    # Create a mock scenario object
    earth = Earth(flattening=0.0)

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)
    return Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

@pytest.fixture
def agent():
    # Create a mock scenario object
    earth = Earth(flattening=0.0)

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)
    scen = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = np.array([1340.745, -6663.403, -132.528])
    velocity = np.array([5.457807, 1.368701, -5.614317])
    initial_state = State(position=position,velocity=velocity,time=start_time)

    jah_sat = Spacecraft(initial_state,start_time=start_time,duration=duration, dt = scen.dt)
    jah_sat.set_scenario(scenario=scen)
    agent = jah_sat

    return agent

def test_frame_conversion(agent):

    # load_spice()

    position_ECI = agent.state.position_ECI

    position_ECEF = agent.state.position_ECEF 

    et = spice.str2et(agent.start_time.strftime('%Y-%m-%dT%H:%M:%S'))
    rotation_matix = spice.pxform(ECEF,ECI,et)

    position_ECI2 = rotation_matix @ position_ECEF

    velocity_ECI = agent.state.velocity_ECI

    velocity_ECEF = agent.state.velocity_ECEF 

    et = spice.str2et(agent.start_time.strftime('%Y-%m-%dT%H:%M:%S'))
    rotation_matix = spice.pxform(ECEF,ECI,et)

    velocity_ECI2 = rotation_matix @ velocity_ECEF

    np.testing.assert_array_almost_equal(position_ECI, position_ECI2, decimal=12)
    np.testing.assert_array_almost_equal(velocity_ECI, velocity_ECI2, decimal=12)

    

def test_data_from_target(scenario,agent):
    latitude = 34.05  # Example latitude
    longitude = -118.25  # Example longitude
    altitude = 0.0  # Example altitude
    assert isinstance(scenario,Scenario)
    assert isinstance(agent,Spacecraft)
    platform = Station(latitude, longitude, scenario, altitude)

    range, range_rate = platform.calculate_range_and_range_rate_from_target(state=agent.state)
    az, el = platform.calculate_azimuth_and_elevation(state=agent.state)


    pass



if __name__ == "__main__":

    earth = Earth(flattening=0)

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)
    scen = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = np.array([1340.745, -6663.403, -132.528])
    velocity = np.array([5.457807, 1.368701, -5.614317])
    initial_state = State(position=position,velocity=velocity,time=start_time)

    jah_sat = Spacecraft(initial_state,start_time=start_time,duration=duration, dt = scen.dt)
    jah_sat.set_scenario(scenario=scen)

    test_frame_conversion(scenario=scen,agent=jah_sat)
    test_data_from_target(scenario=scen,agent=jah_sat)