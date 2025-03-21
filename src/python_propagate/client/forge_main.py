"""
forge_main.py

Description:
    Client that generates satellite data

"""

import click
import matplotlib.pyplot as plt

from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.utilities.load_spice import load_spice


@click.command()
@click.option(
    "--infile",
    type=str,
    help="YAML infile for the scenario to generate data",
    required=True,
)
@click.option(
    "--parallel",
    type=int,
    help="YAML infile for the scenario to generate data",
    required=False,
)
# @click.option('--plot_ground', type=bool, help= "Plot the ground track of the scenario", default = False, required = False)
# @click.option('--plot_orbit', type=bool, help= "Plot the isometric view of the scenario", default = False, required = False)
def main(infile: str, parallel = 0):
    """
    Description:
        main client funtion
    Arguments:
        infile: string
            configuration file to be used
    """

    config = load_yaml(yaml_file=infile)

    forge = config["forge"]


    load_spice()
    forge.run(parallel=parallel)
    plt.show()

    pass

    # TODO: handle orbital elements


if __name__ == "__main__":

    main()

    pass
