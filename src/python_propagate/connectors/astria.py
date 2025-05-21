import os 
import requests
import json
import pandas as pd
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
from python_propagate.utilities.units import RAD2DEG, ARC2DEG, DEG2RAD

from datetime import datetime, timedelta

from pathlib import Path





# URL = "http://astria.tacc.utexas.edu/AstriaGraph/"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/cesium/Assets/IAU2006_XYS/IAU2006_XYS_0.json"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/SP_ephemeris/08/8822.oem.gz"
URI = "bolt+s://astria001.pods.astria.tapis.io:443"
CREDENTIALS_FILE = "credentials.yaml"
CYPHER_QUERY = "MATCH (n) RETURN count(n) as num"
DATABASE = 'astria'
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PROP_YAML = BASE_DIR / "data" / "defaults" / "astria_propagator.yaml"



class AstriaConnector:

    
    _cart_query = lambda time, norad_id : f"""MATCH (p:`SpaceObject`:`{time}`)-[:has_orbit]->(Orbit)
                                              WHERE p.NoradId = "{norad_id}"
                                              RETURN Orbit.Cart as Cart"""
    
    _full_query = lambda time, norad_id : f"""MATCH (p:`SpaceObject`:`{time}`)-[:has_orbit]->(Orbit)
                                              WHERE p.NoradId = "{norad_id}"
                                              RETURN Orbit as orbit"""
    
    _oe_query = lambda time, norad_id : f"""MATCH (p:`SpaceObject`:`{time}`)-[:has_orbit]->(Orbit)
                                              WHERE p.NoradId = "{norad_id}"
                                              RETURN Orbit.SMA as sma,
                                              Orbit.Ecc as ecc,
                                              Orbit.ArgP as arg,
                                              Orbit.RAAN as raan,
                                              Orbit.MeanAnom as ma,
                                              Orbit.Epoch as epoch"""

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


        

    def get_cart_state(self, norad_id, time):

        if isinstance(norad_id,int):
            norad_id = str(norad_id)

        result = np.array(self.run_query(AstriaConnector._cart_query, norad_id = norad_id, time = time))[0]

        return result
    
    def get_oe_and_epoch(self,norad_id,time):

        if isinstance(norad_id,int):
            norad_id = str(norad_id)

        result = self.run_query(AstriaConnector._oe_query, norad_id = norad_id, time = time)[0]

        return result
    
    
    def get_full_state(self,norad_id,time) -> State:

        load_spice()

        if isinstance(norad_id,int):
            norad_id = str(norad_id)
        try:
            result = self.run_query(AstriaConnector._full_query, norad_id = norad_id, time = time)[0]
            cartesian = np.array(result['orbit']['Cart']) / 1000
            epoch = result['orbit']['Epoch'].to_native().replace(tzinfo=None)
            state = State(position = cartesian[:3] ,velocity = cartesian[3:], time = epoch)
            return state
        except IndexError:
            Warning(f"Data for object <{norad_id}> at time <{time}> not found in database. Returning None for state")
            cartesian = [None for i in range(6)]
            epoch = datetime.strptime(time,"%Y-%m-%d")
            state = State(position = cartesian[:3] ,velocity = cartesian[3:], time = epoch)
            return State

        

        
    
    def get_data(self, norad_ids:Iterable[int], start_time:str, end_time:str, csv_path = None): 


        data = {}
        start_time = datetime.strptime(start_time,"%Y-%m-%d")
        end_time = datetime.strptime(end_time,"%Y-%m-%d")
        dt = timedelta(days=1)

        results = []  # List to collect all records

        for norad_id in norad_ids:
            current_time = start_time
            while current_time <= end_time:
                time_str = current_time.strftime("%Y-%m-%d")
                state = self.get_full_state(norad_id=norad_id, time=time_str)

                results.append({
                    'norad_id': norad_id,
                    'query_time': time_str,
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

                current_time += timedelta(hours=12)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Drop duplicate epochs *per satellite*
        df = df.drop_duplicates(subset=['norad_id', 'epoch'])

        # Optional: reset index
        df.reset_index(drop=True, inplace=True)

        if csv_path:
            print(f"Saved csv to {csv_path}")
            df.to_csv(path_or_buf=csv_path)

        return df
    
    def _unique_dates(self, norad_id:int, start_time:datetime, end_time:datetime):

        #TODO need to add start and end dates to the lists

        seen = set()
        unique_ordered = []
        states = []

        current_time = start_time

        while current_time < end_time:
            time_str = current_time.strftime("%Y-%m-%d")
            state = self.get_full_state(norad_id=norad_id, time=time_str)
            
            if state.time not in seen:
                if state.time > start_time:
                    unique_ordered.append(state.time)
                    seen.add(state.time)
                    states.append(state)

            current_time += timedelta(hours=1)

    

        return unique_ordered, states


    
    def get_propagated_data(self, norad_ids:Iterable[int], start_time:str, end_time:str, dt = timedelta(seconds=30), csv_path = None): 


        data = {}
        start_time = datetime.strptime(start_time,"%Y-%m-%d")
        end_time = datetime.strptime(end_time,"%Y-%m-%d")
        self.propagator.dt = dt

        # times_to_evaluate = timedelta_list(start=start_time, end=end_time, step=dt)

        results = []  # List to collect all records

        for norad_id in norad_ids:
            # current_time = start_time
            unique_dates, states = self._unique_dates(norad_id=norad_id,start_time=start_time,end_time=end_time)
            for tkm1,tk,xkm1 in zip(unique_dates[0:-1],unique_dates[1:],states[0:-1]) :
                #fill in the gaps
                time_to_prop = tk - tkm1
                self.propagator.state = xkm1
                self.propagator.start_time = tkm1
                self.propagator.propagate(duration = time_to_prop.total_seconds())
                # self.propagator.state_data.append

                # if state.time
                for state in self.propagator.state_data:
                    results.append({
                        'norad_id': norad_id,
                        'time_sec': (state.time - unique_dates[0]).total_seconds(),
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

        # Drop duplicate epochs *per satellite*
        df = df.drop_duplicates(subset=['norad_id', 'epoch'])

        # Optional: reset index
        df.reset_index(drop=True, inplace=True)

        if csv_path:
            print(f"Saved csv to {csv_path}")
            df.to_csv(path_or_buf=csv_path)

        return df



def timedelta_list(start: datetime, end: datetime, step: timedelta):
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += step
    return times








        

        



       