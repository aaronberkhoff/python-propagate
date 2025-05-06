import numpy as np
from python_propagate.connectors.astria import AstriaConnector






def test_get_single_tle():

    astria = AstriaConnector()

    norad_id = 25544 #ISS
    time = "2019-10-22"

    state = astria.get_cart_state(norad_id,time)
    oe = astria.get_oe_and_epoch(norad_id,time)
    state = astria.get_full_state(norad_id=norad_id,time=time)


    state = astria.get_full_state(norad_id=norad_id,time=time)

    position_eci = state.position_eci
    velocity_eci = state.velocity_eci

    position_ecef = state.position_ecef
    velocity_ecef = state.velocity_ecef

    latitude, longitude = state.latlong

    altitude = np.linalg.norm(position_eci)

    pass

def test_get_multiple_tle():

    astria = AstriaConnector()

    norad_id = 25544 #ISS
    time = "2019-10-22"

    state = astria.get_cart_state(norad_id,time)
    oe = astria.get_oe_and_epoch(norad_id,time)
    state = astria.get_full_state(norad_id=norad_id,time=time)


    state = astria.get_full_state(norad_id=norad_id,time=time)

    position_eci = state.position_eci
    velocity_eci = state.velocity_eci

    position_ecef = state.position_ecef
    velocity_ecef = state.velocity_ecef

    latitude, longitude = state.latlong

    altitude = np.linalg.norm(position_eci)

    pass

if __name__ == "__main__":

    test_get_single_tle()

    