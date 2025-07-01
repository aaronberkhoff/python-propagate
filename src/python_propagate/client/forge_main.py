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
@click.option(
    "--case",
    multiple = True,
    help="The cases to run",
    default = ('all',)
)
def main(infile: str, case: str,parallel = 0):
    """
    Description:
        main client funtion
    Arguments:
        infile: string
            configuration file to be used
    """

    config = load_yaml(yaml_file=infile)

    forges = config["cases_to_run"]
    cases_to_run = [forge for forge in forges if forge.name in case or 'all' in case]

    if not cases_to_run:
        raise ValueError(f'Cases {case} not found in yaml input file')


    load_spice()
    for forge in cases_to_run:
        click.echo(f'Running case <{forge.name}> with {parallel} cores')
        forge.run(parallel=parallel)

    pass


if __name__ == "__main__":

    main()

    pass
