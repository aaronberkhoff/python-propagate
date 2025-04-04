import pandas as pd
import numpy as np
import h5py
from collections import namedtuple
import sqlite3
from datetime import datetime, timedelta
from copy import deepcopy

from sgp4.api import Satrec

from python_propagate.utilities.units import RAD2DEG, ARC2DEG, DEG2RAD
from python_propagate.forge import Forge
from python_propagate.utilities.load_spice import load_spice
from python_propagate.states.proper_orbital_elements import ProperElements

from python_propagate.agents import Agent
from python_propagate.utilities.string_format import DATESTR
from python_propagate.states import OrbitalElements, State  

from python_propagate.utilities.transforms import mean2true


def process_agent(agent, scenario, datatypes, add_noise = False, propagate=True):
        load_spice()
        if propagate:
            agent.propagate()  # Update agent state

        elements = [state.to_keplerian(agent.scenario.central_body.mu) for state in agent.state_data]

        proper_elements = ProperElements(orbital_element_data=elements, dt = agent.dt, duration=agent.duration)

        mean_oe = np.array(proper_elements.mean_orbital_elements_data).T
        proper_oe = np.array(proper_elements.proper_orbital_elements_data).T

        data_agent = []
        
        for i, (state,oe,moe,poe) in enumerate(zip(agent.state_data,elements,mean_oe,proper_oe)):
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
                cnt = 0

                if add_noise:

                    az_noise = np.random.normal(0, 5 * ARC2DEG)
                    el_noise = np.random.normal(0, 5 * ARC2DEG)

                    ra_noise = np.random.normal(0, 5 * ARC2DEG)
                    dec_noise = np.random.normal(0, 5 * ARC2DEG)

                    rho_noise = np.random.normal(0, 1e-3)
                    rhodot_noise = np.random.normal(0, 1e-6)

                    sma_noise = np.random.normal(0,10)
                    ecc_noise = np.random.normal(0,.001)
                    inc_noise = np.random.normal(0,2)
                    arg_noise = np.random.normal(0,2)
                    raan_noise = np.random.normal(0,2)
                    nu_noise = np.random.normal(0,2)

                else: 

                    az_noise = 0
                    el_noise = 0

                    ra_noise = 0
                    dec_noise = 0

                    rho_noise = 0
                    rhodot_noise = 0

                    sma_noise = 0
                    ecc_noise = 0
                    inc_noise = 0
                    arg_noise = 0
                    raan_noise = 0
                    nu_noise = 0

                if el > station.minimum_elevation_angle:
                    data_entry = {
                        "agent": agent.name,
                        "index": i,
                        "epoch_time": state.time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "time_sec": i * agent.dt.total_seconds(),
                        "station": station.name,
                        "station_id": station.identity,

                        "RA_DEG": (ra + ra_noise) % 360,
                        "DEC_DEG": max(-90, min(90,dec + dec_noise)),
                        "AZ_DEG": az + az_noise,
                        "EL_DEG": el + el_noise,
                        "RANGE_KM": rho + rho_noise,
                        "RANGE_RATE_KMS": rhodot + rhodot_noise,

                        "X_INERTIAL_KM": state.position[0],
                        "Y_INERTIAL_KM": state.position[1],
                        "Z_INERTIAL_KM": state.position[2],
                        "VX_INERTIAL_KMS": state.velocity[0],
                        "VY_INERTIAL_KMS": state.velocity[1],
                        "VZ_INERTIAL_KMS": state.velocity[2],

                        "LAT_DEG": state.latlong[0] * RAD2DEG,
                        "LON_DEG": state.latlong[1] * RAD2DEG,
                        "ALT_KM": np.linalg.norm(state.position) - agent.scenario.central_body.radius,

                        "SMA_KM"   : oe.sma + sma_noise,
                        "ECC_KM"   : oe.ecc + ecc_noise,
                        "INC_DEG"  : (oe.inc * RAD2DEG + inc_noise) % 180,
                        "ARG_DEG"  : (oe.arg  * RAD2DEG + arg_noise) % 360,
                        "RAAN_DEG" : (oe.raan  * RAD2DEG + raan_noise) % 360,
                        "NU_DEG"   : (oe.nu  * RAD2DEG + nu_noise) % 360,

                        "MEAN_SMA_KM"   : moe[0] + sma_noise,
                        "MEAN_ECC_KM"   : moe[1] + ecc_noise,
                        "MEAN_INC_DEG"  : (moe[2] * RAD2DEG + inc_noise) % 180,
                        "MEAN_ARG_DEG"  : (moe[3]  * RAD2DEG + arg_noise) % 360,
                        "MEAN_RAAN_DEG" : (moe[4]   * RAD2DEG + raan_noise) % 360,
                        "MEAN_NU_DEG"   : (moe[5] * RAD2DEG + nu_noise) % 360,

                        "PROP_SMA_KM"   : poe[0] + sma_noise,
                        "PROP_ECC_KM"   : poe[1] + ecc_noise,
                        "PROP_INC_DEG"  : (poe[2] * RAD2DEG + inc_noise) % 180,
                        "PROP_ARG_DEG"  : (poe[3]  * RAD2DEG + arg_noise) % 360,
                        "PROP_RAAN_DEG" : (poe[4]   * RAD2DEG + raan_noise) % 360,
                        "PROP_NU_DEG"   : (poe[5] * RAD2DEG + nu_noise) % 360,
                    }

                    filtered_data_entry = {key: value for key, value in data_entry.items() if key in datatypes or key in {"agent", "index", "time_sec","epoch_time", "station", "station_id"}}
                    # for data_type in datatypes:
                    #     if data_type in data_entry:
                    #         pass

                    data_agent.append(filtered_data_entry)
                    cnt += 1

        return data_agent, agent

class AstroForge(Forge):

    def __init__(self, scenario,genes, output_directory, data_types, output_types,add_noise = False, plots=None, name='Forge'):
        super().__init__(scenario,genes, output_directory, data_types, output_types, add_noise,plots, name)

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
                            if column != "agent":  # Skip agent name since it's used as a key
                                agent_group.create_dataset(column, data=df_agent[column].values)

                print(f"HDF5 File written:\n h5: {output_path_h5}")

            if 'xlsx' in self.output_types:
                output_path_xlsx = self.output_directory / f"{self.name}.xlsx"
                with pd.ExcelWriter(output_path_xlsx) as writer:
                    for agent_name, df_agent in df_all.groupby("agent"):
                        df_agent.to_excel(writer, sheet_name=agent_name, index=False)
                print(f"XLSX File written:\n xlsx: {output_path_xlsx}")
        



    def generate_data(self):
        
        data_all = []  # List to store data for all agents

        for agent in self.genes.agents:
            data_agent, agent = process_agent(agent,self.scenario,self.datatypes,add_noise=self.add_noise)
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
                executor.submit(process_agent, agent, self.scenario, self.datatypes,add_noise=self.add_noise)
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


class TLEForge(AstroForge):
    """
    TLEForge is a specialized forge for generating TLE data, inheriting from AstroForge.
    """

    def __init__(self, scenario, database, norad_ids, agent_base: Agent, output_directory, data_types, output_types, add_noise=False, plots=None, name='TLEForge'):

        Genes = namedtuple('Genes', ['agents'])  # Create a namedtuple for genes, as TLEForge needs agents to be passed in.
        self.agent_base = agent_base  # Store the agent base for TLEForge, used to extract agents from the database.
        genes = Genes(agents=self._get_agents_from_database(database=database,norad_ids=norad_ids,scenario=scenario))  # Extract agents from the database.

        super().__init__(scenario, genes, output_directory, data_types, output_types, add_noise=add_noise, plots=plots, name=name)


    def _get_agents_from_database(self,database,norad_ids,scenario):
        """
        Helper function to extract agents from the database for TLEForge, if needed.
        """
        # Placeholder for extracting agents from a database, if applicable.
        # This would typically involve querying a database and returning the agents.
        conn = sqlite3.connect(database)
        cursor = conn.cursor()

        # Query to get the list of all table names in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")

        tables = cursor.fetchall()

        # Execute a query to get data from the TLE table, sorted by NORAD ID

        agents = []

        for norad_id in norad_ids:
            cursor.execute(f"SELECT * FROM {tables[0][0]} WHERE norad_cat_id = {norad_id}")  # Adjust 'tle_data' and 'norad_id' to your table/column names

            # Fetch all the sorted rows of the query result
            rows = cursor.fetchall()

            agent = deepcopy(self.agent_base)
            

            for i, row in enumerate(rows):

                if scenario.duration > (datetime.strptime(row[11], DATESTR) - datetime.strptime(rows[0][11], DATESTR)):

                    sat = Satrec.twoline2rv(row[39], row[40])

                    jd, fr = sat.jdsatepoch, sat.jdsatepochF
                    _, radius, velo = sat.sgp4(jd, fr)

                    state = State(position=np.array(radius),  # Position in km
                                  velocity=np.array(velo),  # Velocity in km/s
                                  time=datetime.strptime(row[11], DATESTR))
                    
                    agent.name = f"Agent_{norad_id}"  # Set a unique name for the agent based on NORAD ID
                    agent.state = state
                    # agent.ensure_cart_state()  # Ensure the agent has state data for propagation
                    # agent.state.time = datetime.strptime(row[3], DATESTR)  # Set the initial time for the agent from the database row
                    

                    if i == len(rows) - 1:
                        agent.start_time = datetime.strptime(rows[0][11], DATESTR)  
                    else:
                        agent.start_time = datetime.strptime(row[11], DATESTR)  
                        duration = datetime.strptime(rows[i+1][11], DATESTR) - agent.start_time  # Assuming row[1] is the epoch time in the database, adjust as needed

                        if duration.total_seconds() == 0:
                            pass
                        else:
                            agent.duration = duration  # Set the duration for the agent based on the TLE data
                            agent.propagate()  # Propagate the agent for the duration of the TLE data

                        times = [state.time for state in agent.state_data]  # Get the times of the propagated states

                    pass 
                else: 
                    break

            agents.append(agent)  # Append the agent to the list of agents
            print(f"Extracted agent for NORAD ID {norad_id}: {agent.name} with {len(agent.state_data)} states.")  # Debugging output
            pass 

        return agents
    

    def generate_data(self):
        
        data_all = []  # List to store data for all agents

        for agent in self.genes.agents:
            data_agent, agent = process_agent(agent,self.scenario,self.datatypes,add_noise=self.add_noise,propagate=False)
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
                executor.submit(process_agent, agent, self.scenario, self.datatypes,add_noise=self.add_noise,propagate=False)
                for agent in self.genes.agents
            ]
            # As each future completes, extend the data_all list.
            for future in concurrent.futures.as_completed(futures):
                agent_data, updated_agent = future.result()
                updated_agents.append(updated_agent)  # Store the updated agent
                data_all.extend(agent_data)  # Store collected data

        # Convert collected data into a pandas DataFrame.
        self.save_data_to_files(data_all=data_all)


