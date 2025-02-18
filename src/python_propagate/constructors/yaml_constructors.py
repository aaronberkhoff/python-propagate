"""
yaml_constructor.py 

Description:


"""

import yaml

from python_propagate.platforms.station import Station
from python_propagate.environment.planets import Earth
from python_propagate.platforms.station import Station
from python_propagate.scenario import Scenario
from python_propagate.scenario.data_generator import DataGenerator
from python_propagate.scenario.data_generator import Scenario
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.states import State, OrbitalElements

class ClassConstructor:

    def __init__(self,object):

        self.object = object
        pass
    
    def constructor(self,loader,node):

        values = loader.construct_mapping(node, deep = True)

        return self.object(**values)


def load_yaml(yaml_file):
    
    yaml.add_constructor("!Spacecraft", ClassConstructor(Spacecraft).constructor)
    yaml.add_constructor("!Station", ClassConstructor(Station).constructor)
    yaml.add_constructor("!Scenario", ClassConstructor(Scenario).constructor)
    yaml.add_constructor("!DataGenerator", ClassConstructor(DataGenerator).constructor)
    yaml.add_constructor("!State", ClassConstructor(State).constructor)
    yaml.add_constructor("!OrbitalElements", ClassConstructor(OrbitalElements).constructor)
    yaml.add_constructor("!Earth", ClassConstructor(Earth).constructor)


    with open(yaml_file, "r") as file:
        # raw = file.read()
        return yaml.load(file, Loader=yaml.FullLoader)


