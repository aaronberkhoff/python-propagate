from datetime import datetime, timedelta

from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State

from python_propagate.connectors.spacetrack import SpaceTrackConnector




def test_spacetrack_connector() -> None:
    """
    Test the SpaceTrack connector by initializing it and fetching TLE data for a known satellite.
    This test ensures that the connector is working correctly and can retrieve TLE data from SpaceTrack.
    """

    connector = SpaceTrackConnector()

    tle = connector.fetch_latest_tle(25544)  # ISS NORAD ID

    print(tle)

def test_spacetrack_tle_history() -> None:
    """
    Test the TLE history retrieval from SpaceTrack for a known satellite.
    This test ensures that the connector can fetch historical TLE data correctly.
    """

    connector = SpaceTrackConnector()

    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 10)

    tle_history = connector.get_tle_history(25544, start_time, end_time)  # ISS NORAD ID

    for timestamp, tle in tle_history:
        print(f"{timestamp}: {tle}")


if __name__ == "__main__":

    #hSqYJsqFJ_8zE.X
    # test_spacetrack_connector()
    test_spacetrack_tle_history()
    
    

    