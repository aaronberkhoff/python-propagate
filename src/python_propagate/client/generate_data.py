"""
generate_data.py

Description:
    Client that generates satellite data

"""

import click

from python_propagate.constructors.yaml_constructors import load_yaml


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
    """
    Description:
        main client funtion
    Arguments:
        infile: string
            configuration file to be used
    """

    config = load_yaml(yaml_file=infile)

    data_generator = config["scenario"]

    # TODO: I do not like that I have to set the scenarion first then the dynamics in that order


    data_generator.run()

    pass

    # TODO: handle orbital elements


if __name__ == "__main__":

    main()

    pass
