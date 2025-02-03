import pytest
from datetime import datetime, timedelta
from python_propagate.platforms import Platform
from python_propagate.scenario import Scenario

from python_propagate.environment.planets import Earth
from python_propagate.utilities.units import RAD2DEG, DEG2RAD


@pytest.fixture
def scenario():
    # Create a mock scenario object
    earth = Earth(flattening_bool=False)

    start_time = datetime.strptime("2025-01-15T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)
    return Scenario(central_body=earth, start_time=start_time, duration=duration, dt=dt)


def test_platform_initialization(scenario):
    latitude = 34.05  # Example latitude
    longitude = -118.25  # Example longitude
    altitude = 0.0  # Example altitude

    platform = Platform((latitude, longitude,altitude))
    platform.set_scenario(scenario=scenario)

    assert platform.latitude == latitude * DEG2RAD
    assert platform.longitude == longitude * DEG2RAD
    assert platform.altitude == altitude


def test_latlon_convesion(scenario):
    latitude = 34.05  # Example latitude
    longitude = -118.25  # Example longitude
    altitude = 0.0  # Example altitude

    platform = Platform((latitude, longitude,altitude))
    platform.set_scenario(scenario=scenario)

    state = platform.state

    lat = state.latlong[0] * RAD2DEG
    lon = state.latlong[1] * RAD2DEG

    assert lat == latitude
    assert lon == longitude


if __name__ == "__main__":

    earth = Earth(flattening_bool=False)

    start_time = datetime.strptime("2025-01-15T12:30:00", "%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)
    scen = Scenario(central_body=earth, start_time=start_time, duration=duration, dt=dt)

    test_platform_initialization(scenario=scen)
    test_latlon_convesion(scenario=scen)
