from pathlib import Path
from typing import Iterable
import pandas as pd
import numpy as np
import h5py

from python_propagate.plots.plot_ground_track import plot_ground_track
from python_propagate.plots.plot_orbit import plot_orbit
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements
from python_propagate.plots.plot_light_curve import plot_light_curve

from python_propagate.forge.genes import Genes
from python_propagate.agents import Agent




class Forge:

    def __init__(self, scenario, agents,output_directory, data_types, output_types, add_noise = False, plots = None, name = 'Forge'):

        self.scenario = scenario
        self.database = None
        self.name = name
        self.data_types = data_types
        self.output_types = output_types
        self.plots = plots

        if isinstance(agents,Genes): #handles agents defined as genes
            self.agents = [agent for agent in agents.agents]
        # elif isinstance(agents,Iterable[Genes]): #handles iterable of genes
        #     self.agents = [item.agents for sublist in agents for item in sublist]
        elif isinstance(agents,list): # handles a case where some agents are defines as list and/or genes
            self.agents = []
            for sublist in agents:
                if isinstance(sublist,Genes):
                    self.agents.extend([agent for agent in sublist.agents])
                elif isinstance(sublist,list):
                    self.agents.extend(sublist)
                elif isinstance(sublist,Agent):
                    self.agents.append(sublist)
        else:
            raise ValueError('unexpected behavior')
            

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

        self.add_noise = add_noise

        if isinstance(output_types,str):
            self.output_types [output_types]
        if isinstance(output_types,Iterable):
            self.output_types = output_types
            

    def process_agent(self, agent, scenario, datatypes, add_noise = False, propagate=True):

        raise NotImplementedError("The method 'process_agent' should be implemented in the subclass or outside this class.")

    def save_data_to_files(self, data_all):
            """
            Saves the simulation data to CSV, HDF5, and Excel, with unique datasets for each agent.
            """
            # Convert list of all agent data into a DataFrame
            df_all = pd.DataFrame(data_all)

            # Define output file paths
            if 'csv' in self.output_types:
                output_path_csv = self.output_directory / f"{self.name}.csv"
                df_all.to_csv(output_path_csv, index=False)
                print(f"CSV File written:\n CSV: {output_path_csv}")

            if 'h5' in self.output_types:
                output_path_h5 = self.output_directory / f"{self.name}.h5"
                with h5py.File(output_path_h5, "w") as h5file:
                    for agent_name, df_agent in df_all.groupby("agent"):
                        agent_group = h5file.create_group(agent_name)  # Create group for each agent
                        
                        for column in df_agent.columns:
                            if column != "agent":
                                try:   # Skip agent name since it's used as a key
                                    agent_group.create_dataset(column, data=df_agent[column].values)
                                except (AttributeError,TypeError): 
                                    stacked_array = np.stack(df_agent[column].values)
                                    agent_group.create_dataset(column, data=stacked_array)


                print(f"HDF5 File written:\n h5: {output_path_h5}")

            if 'xlsx' in self.output_types:
                output_path_xlsx = self.output_directory / f"{self.name}.xlsx"
                with pd.ExcelWriter(output_path_xlsx) as writer:
                    for agent_name, df_agent in df_all.groupby("agent"):
                        df_agent.to_excel(writer, sheet_name=agent_name, index=False)
                print(f"XLSX File written:\n xlsx: {output_path_xlsx}")
        



    def generate_data(self):
        
        data_all = []  # List to store data for all agents

        for agent in self.agents:
            data_agent, agent = self.process_agent(agent,self.scenario,self.data_types,add_noise=self.add_noise)
            data_all.extend(data_agent)

        self.save_data_to_files(data_all)
        return data_all
        
    def generate_data_parallel(self,cores):

        import concurrent.futures

        # In your main method, replace the serial loop with a parallel one:
        data_all = []
        updated_agents = []

        # Choose the appropriate executor: ProcessPoolExecutor for CPU-bound tasks,
        # or ThreadPoolExecutor for I/O-bound tasks.
        with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
            # Submit a task for each agent.
            # Note: if you need to deepcopy agent_base or similar inside each task,
            # you can do that within process_agent.
            futures = [
                executor.submit(self.process_agent, agent, self.scenario, self.data_types,add_noise=self.add_noise)
                for agent in self.agents
            ]
            # As each future completes, extend the data_all list.
            for future in concurrent.futures.as_completed(futures):
                agent_data, updated_agent = future.result()
                updated_agents.append(updated_agent)  # Store the updated agent
                data_all.extend(agent_data)  # Store collected data

        # Convert collected data into a pandas DataFrame.
        self.save_data_to_files(data_all=data_all)
        self.agents = updated_agents

    
    def run(self,parallel=0):
        """
        Runs the DataGenerator simulation, collecting observational data from agents and saving it to HDF5, Excel, and CSV formats.
        """

        if parallel:
            self.generate_data_parallel(cores=parallel)
        else:
            self.generate_data()

        if len(self.agents) > 7:
            legend = False
        else:
            legend = True
        # Now plot the orbit
        if self.plots:
            if "orbit" in self.plots:
                plot_orbit(self.agents,self.scenario,self.output_directory,name=self.name,legend=legend)
            if "ground_track" in self.plots:
                plot_ground_track(
                    self.agents, self.scenario.stations, self.output_directory, name=self.name,legend=legend
                )
            if "orbital_elements" in self.plots:
                plot_orbital_elements(self.agents, self.scenario, self.output_directory, name=self.name,legend=legend)

            if "light_curve" in self.plots:
                plot_light_curve(self.agents, self.output_directory, name=self.name, legend=legend)
    