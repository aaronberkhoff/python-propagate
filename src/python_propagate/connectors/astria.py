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

from datetime import datetime, timedelta





# URL = "http://astria.tacc.utexas.edu/AstriaGraph/"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/cesium/Assets/IAU2006_XYS/IAU2006_XYS_0.json"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/SP_ephemeris/08/8822.oem.gz"
URI = "bolt+s://astria001.pods.astria.tapis.io:443"
CREDENTIALS_FILE = "credentials.yaml"
CYPHER_QUERY = "MATCH (n) RETURN count(n) as num"
DATABASE = 'astria'



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

    def __init__(self):
        self.username, self.password = self._get_credentials()
        self.client = GraphDatabase.driver(URI, auth=basic_auth(self.username, self.password))

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
            epoch = result['orbit']['Epoch'].to_native()
            state = State(position = cartesian[:3] ,velocity = cartesian[3:], time = epoch)
            return state
        except IndexError:
            Warning(f"Data for object <{norad_id}> at time <{time}> not found in database. Returning None for state")
            cartesian = [None for i in range(6)]
            epoch = datetime.strptime(time,"%Y-%m-%d")
            state = State(position = cartesian[:3] ,velocity = cartesian[3:], time = epoch)

        

        
    
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

                    'latitude': state.latlong[0],
                    'longitude': state.latlong[1],
                    'altitude': np.linalg.norm(state.position_eci) - 6378.1363 # Radius of Earth
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










        

        



       