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


class PhotoForge(Forge):
    """
    A class to represent a photometric forge for generating orbits from photometric data.

    Inherits from the Forge class.

    Attributes
    ----------
    None

    Methods
    -------
    
    """

    def __init__(self, scenario, agents, output_directory, data_types, output_types, add_noise=False, plots=None, name='PhotoForge'):
        super().__init__(scenario, agents, output_directory, data_types, output_types, add_noise, plots, name)

    def get_orientation_history_from_manuever(self,agent,time_data): #TODO This needs to be changed like crazy
        orientation_data = []
        
        for state,time in zip(agent.state_data, time_data):
            for man in agent.manuevers:

                man(state,time)

            orientation_data.append(agent.bus.orientation)

        return orientation_data

    def process_agent(self, agent, scenario, datatypes, add_noise = False, propagate=True):
        load_spice() # This is required so that this function works in parallel mode, otherwise spice will not be loaded in the worker process.
        if propagate:
            agent.propagate()  # Update agent state

        data_agent = []
        state_data = []
        orientation_data = self.get_orientation_history_from_manuever(agent,[i * agent.dt.total_seconds() for i, _ in enumerate(agent.state_data)])
        for i, (state, orientation) in enumerate(zip(agent.state_data,orientation_data)):
            
            for station in scenario.stations:
                
                az, el = station.calculate_azimuth_and_elevation(state=state)
                az *= RAD2DEG
                el *= RAD2DEG


                ra, dec = station.calculate_ra_and_dec(state=state)
                ra *= RAD2DEG
                dec *= RAD2DEG

                rho, rhodot = station.calculate_range_and_range_rate_from_target(state=state)
                cnt = 0

                flux_received, apparent_magnitude, is_visible = station.calculate_light_flux(state=state,agent=agent)

                state.metadata['flux_w_m2'] = flux_received # Preserve any existing metadata in the state object, if present.
                state.metadata['apparent_magnitude'] = apparent_magnitude # Store the apparent magnitude in the state metadata for reference.
                state.metadata['orientation'] = orientation

                state_data.append(deepcopy(state)) #TODO: SOmething is wrong here. I should Not have to do a deep copy
                
                if add_noise:

                    az_noise = np.random.normal(0, 5 * ARC2DEG)
                    el_noise = np.random.normal(0, 5 * ARC2DEG)

                    ra_noise = np.random.normal(0, 5 * ARC2DEG)
                    dec_noise = np.random.normal(0, 5 * ARC2DEG)

                    rho_noise = np.random.normal(0, 1e-3)
                    rhodot_noise = np.random.normal(0, 1e-6)


                else: 

                    az_noise = 0
                    el_noise = 0

                    ra_noise = 0
                    dec_noise = 0

                    rho_noise = 0
                    rhodot_noise = 0


                if el > station.minimum_elevation_angle:
                    # Only record data if the elevation is above the minimum angle and the target is visible
                    data_entry = {
                        "agent": agent.name,
                        "index": i,
                        "epoch_time": state.time.strftime(DATESTR),
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

                        "FLUX_W_M2": flux_received if is_visible else 0.0,
                        "FLUX_APPARENT_MAG": apparent_magnitude if is_visible else 0.0,   # Placeholder for apparent magnitude, can be calculated from flux if needed.


                    }

                    

                    filtered_data_entry = {key: value for key, value in data_entry.items() if key in datatypes or key in {"agent", "index", "time_sec","epoch_time", "station", "station_id"}}
                    # for data_type in datatypes:
                    #     if data_type in data_entry:
                    #         pass

                    data_agent.append(filtered_data_entry)
                    cnt += 1
        agent.state_data = state_data
        return data_agent, agent
