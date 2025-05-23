import os 
import requests
import json
import pandas as pd
import warnings
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import AuthError, ServiceUnavailable
import yaml
import getpass
import numpy as np
from typing import Iterable, Literal

from python_propagate.utilities.load_spice import load_spice
from python_propagate.states import State
from python_propagate.agents import Agent
from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.utilities.string_format import DATESTR
from python_propagate.utilities.units import RAD2DEG

from datetime import datetime, timedelta

from pathlib import Path





# URL = "http://astria.tacc.utexas.edu/AstriaGraph/"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/cesium/Assets/IAU2006_XYS/IAU2006_XYS_0.json"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/SP_ephemeris/08/8822.oem.gz"
URI = "bolt+s://astria003.pods.astria.tapis.io:443"
CREDENTIALS_FILE = "credentials.yaml"
# CYPHER_QUERY = "MATCH (n) RETURN count(n) as num"
DATABASE = 'astria'
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROP_YAML = BASE_DIR / "data" / "defaults" / "astria_propagator.yaml"



class AstriaConnector:

    
    
    _full_query = lambda norad_id, start, end : f"""MATCH (p:SpaceObject)-[:has_orbit]->(Orbit)
                                                    WHERE p.NoradId = "{norad_id}"
                                                    AND Orbit.Epoch > datetime("{start.strftime(DATESTR)}")
                                                    AND Orbit.Epoch < datetime("{end.strftime(DATESTR)}")
                                                    RETURN Orbit;"""
    

    def __init__(self, propagator:Agent = None, prop_index = 0):
        self.username, self.password = self._get_credentials()
        self.client = GraphDatabase.driver(URI, auth=basic_auth(self.username, self.password))

        if not propagator:
            config = load_yaml(yaml_file=PROP_YAML)
            try: 
                self.propagator = config['propagators'][prop_index]
            except ValueError:
                print(f'Input file <{PROP_YAML}> must have a propagators field defined or must be a valid index. Check the yaml file.')


    def _get_credentials(self):
        """Retrieves stored credentials from a YAML file or prompts the user."""
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "r") as f:
                creds = yaml.safe_load(f) or {}
                astria_creds = creds.get("astria", {})
                username = astria_creds.get("username")
                password = astria_creds.get("password")

                if username and password and self._validate_login(username, password):
                    print("✅ Loaded valid credentials from YAML file.")
                    return username, password
                else:
                    print("⚠️ Stored credentials are invalid or missing.")
        
        return self._prompt_for_credentials()

    def _prompt_for_credentials(self):
        """Prompts user for credentials and stores them if valid."""
        while True:
            username = input("Enter your Astria username: ")
            password = getpass.getpass("Enter your Astria password: ")

            if self._validate_login(username, password):
                self._store_credentials(username, password)
                print("✅ Credentials validated and stored in YAML file.")
                return username, password
            else:
                print("❌ Login failed. Please try again.")

    def _validate_login(self, username, password):
        """Checks if the provided credentials are valid."""
        try:
            client = GraphDatabase.driver(URI, auth=(username, password))
            with client as session:
                result = session.verify_connectivity()  # Test query
                print(f"Authentication Passed")
            return True
        except AuthError as e:
            print("❌ Authentication error:", e)
            return False
        except ServiceUnavailable as e:
            print("Could not connect to Neo4j:", e)
        finally:
            if 'driver' in locals():
                client.close()


    def _store_credentials(self, username, password):
        """Stores credentials securely in a YAML file."""
        creds = {"astria": {"username": username, "password": password}}
        with open(CREDENTIALS_FILE, "w") as f:
            yaml.safe_dump(creds, f)

        # Restrict file permissions to owner only
        os.chmod(CREDENTIALS_FILE, 0o600)
       
    def run_query(self, query,**kwargs):

        with self.client.session(database=DATABASE) as session:
            result = session.run(query(**kwargs))
            return [record.data() for record in result] 


    def get_state_data(self, norad_ids: Iterable[int], start_time: datetime, end_time: datetime) -> dict:

        data = {}

        for norad_id in norad_ids:

            results_raw = self.run_query(
                AstriaConnector._full_query,
                norad_id=norad_id,
                start=start_time,
                end=end_time
            )

            if not results_raw:
                warnings.warn(
                    f'No data for Norad_id: <{norad_id}> for date range {start_time} to {end_time}',
                    UserWarning
                )
            else:
                state_data = [
                    State(
                        position=np.array(result['Orbit']['Cart'][:3]) / 1000,
                        velocity=np.array(result['Orbit']['Cart'][3:]) / 1000,
                        time=result['Orbit']['Epoch'].to_native().replace(tzinfo=None)
                    )
                    for result in results_raw
                ]

                state_data_sorted = sorted(state_data, key=lambda s: s.time)

                data[norad_id] = state_data_sorted

        return data

    
    def get_propagated_data(self, norad_ids:Iterable[int], start_time:str, end_time:str, dt = timedelta(seconds=30), csv_path = None): 


        results = []  # List to collect all records

        data = self.get_state_data(norad_ids=norad_ids, start_time=start_time,end_time=end_time)

        for norad_id in norad_ids:
            # current_time = start_time
            # unique_dates, states = self._unique_dates(norad_id=norad_id,start_time=start_time,end_time=end_time)
            state_data = data[norad_id]

            for state_km1,state_k in zip(state_data[0:-1],state_data[1:]) :
                #fill in the gaps
                # time_to_prop = state_k.time - state_km1.time
                self.propagator.state = state_km1
                self.propagator.dt = dt
                self.propagator.duration = state_k.time - state_km1.time
                self.propagator.start_time = state_km1.time
                self.propagator.propagate()

                # if state.time
            for state in self.propagator.state_data:
                results.append({
                    'norad_id': norad_id,
                    'time_sec': (state.time - state_data[0].time).total_seconds(),
                    'epoch': state.time,
                    'X_INERTIAL_KM': state.position_eci[0],
                    'Y_INERTIAL_KM': state.position_eci[1],
                    'Z_INERTIAL_KM': state.position_eci[2],

                    'VX_INERTIAL_KMS': state.velocity_eci[0],
                    'VY_INERTIAL_KMS': state.velocity_eci[1],
                    'VZ_INERTIAL_KMS': state.velocity_eci[2],

                    'X_BODY_FIXED_KM': state.position_ecef[0],
                    'Y_BODY_FIXED_KM': state.position_ecef[1],
                    'Z_BODY_FIXED_KM': state.position_ecef[2],

                    'VX_BODY_FIXED_KMS': state.velocity_ecef[0],
                    'VY_BODY_FIXED_KMS': state.velocity_ecef[1],
                    'VZ_BODY_FIXED_KMS': state.velocity_ecef[2],

                    'LAT_DEG': state.latlong[0] *  RAD2DEG,
                    'LON_DEG': state.latlong[1] * RAD2DEG,
                    'ALT_KM': np.linalg.norm(state.position_eci) - 6378.1363 # Radius of Earth
                })
            self.propagator.state_data = []

        # Convert to DataFrame
        df = pd.DataFrame(results)

        if csv_path:
            print(f"Saved csv to {csv_path}")
            df.to_csv(path_or_buf=csv_path)

        return df
