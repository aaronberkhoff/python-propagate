from datetime import datetime, timedelta
import click

from python_propagate.constructors.yaml_loader import load_yaml


@click.command()
@click.option(
    "--infile",
    type=str,
    help="YAML infile for the scenario to generate data",
    required=True,
)
# @click.option('--plot_ground', type=bool, help= "Plot the ground track of the scenario", default = False, required = False)
# @click.option('--plot_orbit', type=bool, help= "Plot the isometric view of the scenario", default = False, required = False)
def main(infile: str):

    config = load_yaml(yaml_file=infile)

    data_generator = config["scenario"]
    agents = config["agents"]
    stations = config["stations"]
    dynamics = config["dynamics"]
    # TODO: I do not like that I have to set the scenarion first then the dynamics in that order
    for agent in agents:
        agent.set_scenario(data_generator)
        agent.add_dynamics(dynamics)

    for station in stations:
        station.set_scenario(data_generator)

    data_generator.add_agents(agents)
    data_generator.add_stations(stations)

    data_generator.run()

    pass

    # TODO: handle orbital elements


if __name__ == "__main__":

    main()

    pass
