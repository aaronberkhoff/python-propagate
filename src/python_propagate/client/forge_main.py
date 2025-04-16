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
    help="Number of parallel processes to use for the forge run. Default is 0 (no parallelism).",
    required=False,
)
def main(infile: str, parallel=0):
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
    # plt.show()

    pass


if __name__ == "__main__":

    main()

    pass
