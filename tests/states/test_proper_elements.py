from scipy.io import loadmat
import numpy as np
from numpy.testing import assert_allclose
import matplotlib.pyplot as plt

from python_propagate.scenario import Scenario
from python_propagate.environment.planets import Earth
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State
from datetime import datetime, timedelta
from python_propagate.dynamics.keplerian import Keplerian
from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.states.proper_orbital_elements import ProperElements
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements

from python_propagate.utilities.units import RAD2DEG

def plot_mean_elements(data, agent, fig, axes):
        
        times = np.array([state.time for state in agent.state_data])

        sma_data =  data[0]
        ecc_data =  data[1]
        inc_data =  data[2] * RAD2DEG  
        arg_data =  data[3] * RAD2DEG
        raan_data = data[4] * RAD2DEG
        nu_data =   data[5] * RAD2DEG

        axes[0].plot(times, sma_data, label=f"Mean Elements")
        axes[1].plot(times, ecc_data, label=f"Mean Elements")
        axes[2].plot(times, inc_data, label=f"Mean Elements")
        axes[3].plot(times, raan_data, label=f"Mean Elements")
        axes[4].plot(times, arg_data, label=f"Mean Elements")
        axes[5].plot(times, nu_data, label=f"Mean Elements")


def plot_proper_elements(data, agent, fig, axes):
        
        times = np.array([state.time for state in agent.state_data])

        sma_data =  data[0]
        ecc_data =  data[1]
        inc_data =  data[2] * RAD2DEG  
        arg_data =  data[3] * RAD2DEG
        raan_data = data[4] * RAD2DEG
        nu_data =   data[5] * RAD2DEG

        axes[0].plot(times, sma_data, label=f"Proper Elements")
        axes[1].plot(times, ecc_data, label=f"Proper Elements")
        axes[2].plot(times, inc_data, label=f"Proper Elements")
        axes[3].plot(times, raan_data, label=f"Proper Elements")
        axes[4].plot(times, arg_data, label=f"Proper Elements")
        axes[5].plot(times, nu_data, label=f"Proper Elements")



def test_proper_elements(config):

    agent = config['agents'][0]

    agent.propagate()

    #get the orbital elements
    elements = [state.to_keplerian(agent.scenario.central_body.mu) for state in agent.state_data]

    proper_elements = ProperElements(orbital_element_data=elements, dt = agent.dt, duration=agent.duration)

    fig, axes = plot_orbital_elements(agents=[agent],scenario=agent.scenario,output_directory=None,save=False)

    plot_mean_elements(proper_elements.mean_orbital_elements_data,agent,fig,axes)
    plot_proper_elements(proper_elements.proper_orbital_elements_data,agent,fig,axes)
    for ax in axes:
         ax.legend()
    plt.tight_layout()
    plt.show()

    pass


if __name__ == "__main__":

    config = load_yaml('tests/configs/test_config.yaml')

    test_proper_elements(config=config)

