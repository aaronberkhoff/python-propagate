from datetime import datetime, timedelta
from collections import namedtuple

from python_propagate.platforms.station import Station
from python_propagate.environment.planets import Earth
from python_propagate.platforms.station import Station
from python_propagate.scenario import Scenario
from python_propagate.scenario.data_generator import DataGenerator
from python_propagate.scenario.data_generator import Scenario
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents.state import State, OrbitalElements
from python_propagate.dynamics import j2, j3, keplerian

from python_propagate.utilities.string_format import DATESTR


def spacecraft_constructor(loader, node):
    """
    Constructor function for the !Spacecraft tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Earth object.

    Returns
    -------
    Spacecraft
        An instance of the Spacecraft class.
    """
    values = loader.construct_mapping(node,deep=True)
    return Spacecraft(**values)

def station_constructor(loader, node):
    """
    Constructor function for the !Station tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Station object.

    Returns
    -------
    Station
        An instance of the Station class.
    """
    values = loader.construct_mapping(node)
    
    return Station(**values)

def scenario_constructor(loader, node):
    """
    Constructor function for the !Scenario tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Station object.

    Returns
    -------
    Station
        An instance of the Station class.
    """
    values = loader.construct_mapping(node,deep = True)

    if values['central_body'].lower() == 'earth':

        central_body = Earth(flattening_bool=values.pop('flattening'))
        values['central_body'] = central_body
    else:
        raise ValueError(f'central_body <{values["central_body"]}> not support')

    return Scenario(**values)

def data_generator_constructor(loader, node):
    """
    Constructor function for the !Scenario tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Station object.

    Returns
    -------
    Station
        An instance of the Station class.
    """
    values = loader.construct_mapping(node,deep = True)

    if values['central_body'].lower() == 'earth':

        central_body = Earth(flattening_bool=values.pop('flattening'))
        values['central_body'] = central_body
    else:
        raise ValueError(f'central_body <{values["central_body"]}> not support')    
    
    return DataGenerator(**values)

def state_constructor(loader, node):
    """
    Constructor function for the !State tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Station object.

    Returns
    -------
    Station
        An instance of the Station class.
    """
    values = loader.construct_mapping(node)
    
    return State(**values)

def orbital_elements_constructor(loader, node):
    """
    Constructor function for the !Scenario tag in the YAML file.

    Parameters
    ----------
    loader : yaml.Loader
        The YAML loader.
    node : yaml.Node
        The YAML node containing the data for the Station object.

    Returns
    -------
    Station
        An instance of the Station class.
    """
    values = loader.construct_mapping(node)
    
    return OrbitalElements(**values)

