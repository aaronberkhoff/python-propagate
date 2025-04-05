

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

def test_station_flux(config) -> None:

    """
    Test the srp acceleration calculation for a spacecraft body.
    """
    # Load the configuration file

    spacecraft = config['agents'][3]
    
    side = np.sqrt(spacecraft.area)  

    bus = Bus(name="bus",agent=spacecraft,extents=[side,side,side])  # Create a bus instance for the spacecraft

    spacecraft.bus = bus  # Assign the bus to the spacecraft instance

    spacecraft.propagate()

    station = spacecraft.scenario.stations[0]
    flux = []
    for state in spacecraft.state_data:

        flux.append(station.calculate_light_flux(state, agent = spacecraft))

    times = np.array([state.time for state in spacecraft.state_data])
    flux = np.array(flux)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(times, flux[:,1], label='Light Flux', color='blue')

    ax.set_title('Light Flux over Time')
    ax.set_xlabel('Time Step (arbitrary units)')  # X-axis label, can be replaced with actual time if needed
    ax.set_ylabel('Light Flux (W/m^2)')  # Y-axis label for light flux



    plt.savefig("tests/results/station_flux_test_lightcurve.png", dpi=300)  # Save the figure to a file
    
    plt.show()

    
    

    pass

    # Note: Not asserting anything here as this is just a test run for flux calculation.

def test_flux_with_visual(config):
    """
    Test the srp acceleration calculation with visualization.
    """
    spacecraft1 = config['agents'][0]
    spacecraft0 = config['agents'][0]
    
    side = np.sqrt(spacecraft1.area)  

    bus = Bus(name="bus",agent=spacecraft1,extents=[side,side,side])  # Create a bus instance for the spacecraft

    spacecraft1.bus = bus  # Assign the bus to the spacecraft instance

    spacecraft1.propagate()
    spacecraft0.propagate()

    output_directory = "tests/results"
    name = "station_flux_test"  # Name for the output files
    plot_orbit([spacecraft0,spacecraft1],spacecraft0.scenario,output_directory,name,legend = True)
    plot_orbital_elements([spacecraft0,spacecraft1],spacecraft0.scenario,output_directory,name,legend = True)
    

    plt.show()


if __name__ == "__main__":

    config = load_yaml('tests/configs/test_config.yaml')

    test_station_flux(config=config)  # Test the station flux calculation
    test_flux_with_visual(config=config)