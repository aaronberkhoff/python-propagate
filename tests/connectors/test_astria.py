from python_propagate.connectors.astria import AstriaConnector





def test_get_single_tle():

    astria = AstriaConnector()

    norad_id = 25544 #ISS
    time = "2018-09-21"

    # state = astria.get_cart_state(norad_id,time)
    oe = astria.get_oe_and_epoch(norad_id,time)

    pass

if __name__ == "__main__":

    test_get_single_tle()

    