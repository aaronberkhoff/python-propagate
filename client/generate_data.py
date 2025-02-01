
from datetime import datetime, timedelta

import click
import yaml

from python_propagate.constructors.yaml_loader import load_yaml


@click.command()
@click.option('--infile', type=str, help= "YAML infile for the scenario to generate data", required = True)
def main(infile:str):
    
    #TODO: add parse scenario from yaml
    #TODO: handle orbital elements
    config = load_yaml(yaml_file=infile)

    scenario = config['scenario']
    agents = config['agents']
    stations = config['stations']
    dynamics = config['dynamics']

    scenario.add_agents(agents)
    scenario.add_stations(stations)

    for agent in agents:
        agent.add_dynamics(dynamics)
        agent.set_scenario(scenario)

    pass


if __name__ == "__main__":
    
    main()
    # Earth(flatening_bool=True)
    pass