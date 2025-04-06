"""
yaml_constructor.py

Description:


"""

import yaml

from python_propagate.platforms.station import Station
from python_propagate.environment.earth import Earth
from python_propagate.environment.moon import Moon
from python_propagate.environment.sun import Sun
from python_propagate.platforms.station import Station
from python_propagate.scenario import Scenario
from python_propagate.scenario.data_generator import DataGenerator
from python_propagate.scenario.data_generator import Scenario
from python_propagate.agents.spacecraft import Spacecraft, Bus
from python_propagate.states import State, OrbitalElements

from python_propagate.dynamics.manuevers import ImpulseManuever, ThrustManuever, StationKeepLoss
from python_propagate.forge.photometric_forge import PhotoForge
from python_propagate.forge.astrometric_forge import AstroForge
from python_propagate.forge.tle_forge import TLEForge
from python_propagate.forge.genes import Genes, Gene
# from python_propagate.forge.photometric_forge import PhotoForge

class ClassConstructor:

    def __init__(self, object):

        self.object = object
        pass

    def constructor(self, loader, node):

        values = loader.construct_mapping(node, deep=True)

        return self.object(**values)


def load_yaml(yaml_file):

    yaml.add_constructor("!Spacecraft", ClassConstructor(Spacecraft).constructor)
    yaml.add_constructor("!Bus", ClassConstructor(Bus).constructor)

    yaml.add_constructor("!Station", ClassConstructor(Station).constructor)
    yaml.add_constructor("!Scenario", ClassConstructor(Scenario).constructor)

    yaml.add_constructor("!DataGenerator", ClassConstructor(DataGenerator).constructor)
    yaml.add_constructor("!State", ClassConstructor(State).constructor)
    yaml.add_constructor(
        "!OrbitalElements", ClassConstructor(OrbitalElements).constructor
    )
    yaml.add_constructor("!Earth", ClassConstructor(Earth).constructor)
    yaml.add_constructor("!Moon", ClassConstructor(Moon).constructor)
    yaml.add_constructor("!Sun", ClassConstructor(Sun).constructor)

    yaml.add_constructor("!ImpulseManuever", ClassConstructor(ImpulseManuever).constructor)
    yaml.add_constructor("!ThrustManuever", ClassConstructor(ThrustManuever).constructor)
    yaml.add_constructor("!StationKeepLoss", ClassConstructor(StationKeepLoss).constructor)

    yaml.add_constructor("!AstroForge", ClassConstructor(AstroForge).constructor)
    yaml.add_constructor("!TLEForge", ClassConstructor(TLEForge).constructor)
    yaml.add_constructor("!PhotoForge", ClassConstructor(PhotoForge).constructor)

    yaml.add_constructor("!Gene", ClassConstructor(Gene).constructor)
    yaml.add_constructor("!Genes", ClassConstructor(Genes).constructor)
    

    with open(yaml_file, "r") as file:
        # raw = file.read()
        return yaml.load(file, Loader=yaml.FullLoader)
