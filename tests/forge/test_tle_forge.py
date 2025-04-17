from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State

from python_propagate.forge.astrometric_forge import TLEForge
from python_propagate.constructors.yaml_constructors import load_yaml

from python_propagate.utilities.load_spice import load_spice




def test_data_generate() -> None:


    config = load_yaml(yaml_file='tests/configs/test_tle_forge.yaml')

    forge = config["forge"]
    parellel = 5
    load_spice()
    forge.run(parellel)
    plt.show()
 





if __name__ == "__main__":

    
    """
    Entry point for the test script.
    """
    test_data_generate()
    
    

    