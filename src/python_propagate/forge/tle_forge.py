
import numpy as np
from collections import namedtuple
import sqlite3
from datetime import datetime, timedelta
from copy import deepcopy

from sgp4.api import Satrec

from python_propagate.agents import Agent
from python_propagate.utilities.string_format import DATESTR

from python_propagate.forge.astrometric_forge import AstroForge
from python_propagate.forge.photometric_forge import PhotoForge
from python_propagate.forge import Forge  
from python_propagate.states import State  


class TLEForge(Forge):
    """
    TLEForge is a specialized forge for generating TLE data, inheriting from AstroForge or PhotoForge.
    """

    def __init__(self, scenario, database, norad_ids, agent_base: Agent, output_directory, data_types, output_types, add_noise=False, plots=None, name='TLEForge', forge: str = 'AstroForge'):

        Genes = namedtuple('Genes', ['agents'])  # Create a namedtuple for genes, as TLEForge needs agents to be passed in.
        self.agent_base = agent_base  # Store the agent base for TLEForge, used to extract agents from the database.
        genes = Genes(agents=self._get_agents_from_database(database=database, norad_ids=norad_ids, scenario=scenario))  # Extract agents from the database.
        
        # Dynamically choose the appropriate parent class
        if forge == 'AstroForge':
            self.parent_class = AstroForge
        elif forge == 'PhotoForge':
            self.parent_class = PhotoForge
        else:
            raise ValueError(f"Invalid forge type '{forge}' specified. Use 'AstroForge' or 'PhotoForge'.")

        # Call the parent constructor dynamically
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
    
    def process_agent(self, agent, scenario, datatypes, add_noise=False, propagate=False):
        return self.parent_class.process_agent(self,agent, scenario, datatypes, add_noise, propagate)
    
    def generate_data_parallel(self, cores):

        print("WARNING: Parallel is not supported for TLE forge right now. Reverting to serial mode.")
        return super().generate_data()


