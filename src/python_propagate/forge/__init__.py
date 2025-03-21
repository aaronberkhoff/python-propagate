from pathlib import Path
from typing import Iterable
from python_propagate.plots.plot_ground_track import plot_ground_track
from python_propagate.plots.plot_orbit import plot_orbit
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements

class Forge:

    def __init__(self, scenario, genes,output_directory, datatypes, output_types, plots = None, name = 'Forge'):

        self.scenario = scenario
        self.database = None
        self.name = name
        self.datatypes = datatypes
        self.output_types = output_types
        self.plots = plots
        self.genes = genes

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        if isinstance(output_types,str):
            self.output_types [output_types]
        if isinstance(output_types,Iterable):
            self.output_types = output_types

    def generate_data(self):

        raise NotImplementedError("Must specify the forge used. This is the Default Forge and does not have a generation function associated with it.")
    
    def generate_data_parallel(self):

        raise NotImplementedError("Must specify the forge used. This is the Default Forge and does not have a generation function associated with it.")
    
    def run(self,parallel):
        """
        Runs the DataGenerator simulation, collecting observational data from agents and saving it to HDF5, Excel, and CSV formats.
        """
        print("Generating Data...")

        if parallel:
            self.generate_data_parallel(cores=parallel)
        else:
            self.generate_data()

        if len(self.genes.agents) > 7:
            legend = False
        else:
            legend = True
        # Now plot the orbit
        if self.plots:
            if "orbit" in self.plots:
                print(f"Plotting Orbit...\n")
                plot_orbit(self.genes.agents,self.scenario,self.output_directory,name=self.name,legend=legend)
            if "ground_track" in self.plots:
                print("Plotting Ground track...")
                plot_ground_track(
                    self.genes.agents, self.scenario.stations, self.output_directory, name=self.name,legend=legend
                )
            if "orbital_elements" in self.plots:
                print("Plotting Orbital Elements...")
                plot_orbital_elements(self.genes.agents, self.scenario, self.output_directory, name=self.name,legend=legend)
    