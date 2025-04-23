import os 
import requests
import json
from neo4j import GraphDatabase, basic_auth
from neo4j.exceptions import AuthError, ServiceUnavailable
import yaml
import getpass
import numpy as np

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



        

        



       