

from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.agents.spacecraft import Spacecraft, Bus
from python_propagate.agents import State
from datetime import datetime, timedelta
from python_propagate.dynamics.drag import Drag
from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.states.proper_orbital_elements import ProperElements
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements

from python_propagate.plots.plot_orbit import plot_orbit



def test_body_drag_accel(config) -> None:
    """
    Test the drag acceleration calculation for a spacecraft body.
    """
    # Load the configuration fil

    spacecraft = config['agents'][1]

    side = np.sqrt(spacecraft.area)  

    bus = Bus(name="bus",extents=[side,side,side])  # Create a bus instance for the spacecraft

    spacecraft.bus = bus  # Assign the bus to the spacecraft instance

    spacecraft.ensure_cart_state()  # Ensure the spacecraft has a Cartesian state for drag calculation
    complex_drag = Drag(scenario=spacecraft.scenario, agent=spacecraft, complex_drag=True)

    drag = Drag(scenario=spacecraft.scenario, agent=spacecraft, complex_drag=False)

    result_complex = complex_drag(spacecraft.state, None)

    result = drag(spacecraft.state, None)

    diff = result_complex.acceleration - result.acceleration

    assert_allclose(result_complex.acceleration, result.acceleration, rtol=1e-7, atol=1e-7)

    pass

def test_drag_with_visual(config):
    """
    Test the drag acceleration calculation with visualization.
    """
    spacecraft1 = config['agents'][1]
    spacecraft0 = config['agents'][2]
    
    side = np.sqrt(spacecraft1.area)  

    bus = Bus(name="bus",extents=[side,side,side])  # Create a bus instance for the spacecraft

    spacecraft1.bus = bus  # Assign the bus to the spacecraft instance

    spacecraft1.propagate()
    spacecraft0.propagate()

    output_directory = "tests/results"
    name = "complex_drag_test"  # Name for the output files
    plot_orbit([spacecraft0,spacecraft1],spacecraft0.scenario,output_directory,name,legend = True)
    plot_orbital_elements([spacecraft0,spacecraft1],spacecraft0.scenario,output_directory,name,legend = True)
    

    # plt.show()


if __name__ == "__main__":

    config = load_yaml('tests/configs/test_config.yaml')

    test_body_drag_accel(config=config)
    test_drag_with_visual(config=config)