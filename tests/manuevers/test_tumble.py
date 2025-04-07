

from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.agents.spacecraft import Spacecraft, Bus
from python_propagate.agents import State
from datetime import datetime, timedelta
from python_propagate.dynamics.srp import SRP
from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.states.proper_orbital_elements import ProperElements
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements

from python_propagate.plots.plot_orbit import plot_orbit
from python_propagate.plots.plot_light_curve import plot_light_curve



def test_free_rotate_unit(config) -> None:
    """
    Test the srp acceleration calculation for a spacecraft body.
    """
    # Load the configuration fil

    spacecraft = config['agents'][0]

    spacecraft.ensure_cart_state()

    tumble = spacecraft.manuevers[0]  # Get the tumble maneuver configuration

    result = tumble(spacecraft.state,0)
    assert spacecraft.bus.orientation == spacecraft.bus.base_orientation

    result = tumble(spacecraft.state,tumble.execution_time)
    assert spacecraft.bus.orientation == 'free'


    np.testing.assert_equal(np.zeros(3),result.acceleration)

    




   

    pass

def test_tumble_with_visual(config):
    """
    Test the srp acceleration calculation with visualization.
    """
    spacecraft = config['agents'][0]

    spacecraft.propagate()

    output_directory = "tests/results"
    name = "tumble_test"  # Name for the output files
    plot_orbit([spacecraft],spacecraft.scenario,output_directory,name,legend = True)
    plot_orbital_elements([spacecraft],spacecraft.scenario,output_directory,name,legend = True)

    

    plt.show()


if __name__ == "__main__":

    config = load_yaml('tests/configs/test_manuevers.yaml')

    # test_free_rotate_unit(config=config)
    test_tumble_with_visual(config=config)