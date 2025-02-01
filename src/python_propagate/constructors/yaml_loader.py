import yaml

from python_propagate.constructors.yaml_constructors import *


def load_yaml(yaml_file):

    with open(yaml_file, 'r') as file:
        # raw = file.read()
        return yaml.load(file,Loader=yaml.FullLoader)
    
yaml.add_constructor('!Spacecraft',spacecraft_constructor)
yaml.add_constructor('!Station',station_constructor)
yaml.add_constructor('!Scenario',scenario_constructor)
yaml.add_constructor('!State',state_constructor)
yaml.add_constructor('!OrbitalElements',orbital_elements_constructor)