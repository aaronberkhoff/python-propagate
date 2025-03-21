import pandas as pd
import numpy as np

from python_propagate.utilities.units import RAD2DEG, ARC2DEG
from python_propagate.forge import Forge
from python_propagate.utilities.load_spice import load_spice

def process_agent(agent, scenario, datatypes):
        load_spice()
        agent.propagate()  # Update agent state
        data_agent = []
        
        for i, state in enumerate(agent.state_data):
            for station in scenario.stations:
                az, el = station.calculate_azimuth_and_elevation(state=state)
                az *= RAD2DEG
                el *= RAD2DEG

                az += np.random.normal(0, 5 * ARC2DEG)
                el += np.random.normal(0, 5 * ARC2DEG)

                ra, dec = station.calculate_ra_and_dec(state=state)
                ra *= RAD2DEG
                dec *= RAD2DEG

                ra += np.random.normal(0, 5 * ARC2DEG)
                dec += np.random.normal(0, 5 * ARC2DEG)

                rho, rhodot = station.calculate_range_and_range_rate_from_target(state=state)
                rho += np.random.normal(0, 1e-3)
                rhodot += np.random.normal(0, 1e-6)

                if el > station.minimum_elevation_angle:
                    data_entry = {
                        "agent": agent.name,
                        "index": i,
                        "time": state.time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "station": station.name,
                        "station_id": station.identity,
                        "RA_DEG": ra,
                        "DEC_DEG": dec,
                        "AZ_DEG": az,
                        "EL_DEG": el,
                        "RANGE_KM": rho,
                        "RANGE_RATE_KMS": rhodot,
                        "X_INERTIAL_KM": state.position[0],
                        "Y_INERTIAL_KM": state.position[1],
                        "Z_INERTIAL_KM": state.position[2],
                        "VX_INERTIAL_KMS": state.velocity[0],
                        "VY_INERTIAL_KMS": state.velocity[1],
                        "VZ_INERTIAL_KMS": state.velocity[2],
                    }

                    for data_type in datatypes:
                        if data_type in data_entry:
                            pass

                    data_agent.append(data_entry)
        return data_agent, agent

class AstroForge(Forge):

    def __init__(self, scenario,genes, output_directory, data_types, output_types, plots=None, name='Forge'):
        super().__init__(scenario,genes, output_directory, data_types, output_types, plots, name)

    def save_data_to_files(self,data_all):

        # Convert data to a pandas DataFrame
        df = pd.DataFrame(data_all)

        # Define output file paths
        if 'csv' in self.output_types:
            output_path_csv = self.output_directory / f"{self.name}.csv"
            df.to_csv(output_path_csv, index=False)
            print(
            f"CSV File written:\n CSV: {output_path_csv}"
            )

        if 'h5' in self.output_types:
            output_path_h5 = self.output_directory / f"{self.name}.h5"
            df.to_hdf(output_path_h5, key="df", mode="w")
            print(
            f"h5 File written:\n h5: {output_path_h5}"
            )

        if 'xlsx' in self.output_types:
            output_path_xlsx = self.output_directory / f"{self.name}.xlsx"
            df.to_excel(output_path_xlsx, index=False)
            print(
            f"XLSX File written:\n xlsx: {output_path_xlsx}"
            )
    



    def generate_data(self):
        
        data_all = []  # List to store data for all agents

        for agent in self.genes.agents:
            data_agent = process_agent(agent,self.scenario,self.datatypes)
            data_all.extend(data_agent)

        return data_all
        
    def generate_doata_parallel(self,cores):

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
                executor.submit(process_agent, agent, self.scenario, self.datatypes)
                for agent in self.genes.agents
            ]
            # As each future completes, extend the data_all list.
            for future in concurrent.futures.as_completed(futures):
                agent_data, updated_agent = future.result()
                updated_agents.append(updated_agent)  # Store the updated agent
                data_all.extend(agent_data)  # Store collected data

        # Convert collected data into a pandas DataFrame.
        self.save_data_to_files(data_all=data_all)
        self.genes.agents = updated_agents


