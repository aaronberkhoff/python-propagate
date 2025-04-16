import os 
import requests
import json

# URL = "http://astria.tacc.utexas.edu/AstriaGraph/"
URL = "http://astria.tacc.utexas.edu/AstriaGraph/cesium/Assets/IAU2006_XYS/IAU2006_XYS_0.json"
# URL = "http://astria.tacc.utexas.edu/AstriaGraph/SP_ephemeris/08/8822.oem.gz"

class AstriaConnector:


    def __init__(self):

        with open("python-propagate/src/python_propagate/connectors/auth_keys.json", "r") as f:
            token_data = json.load(f)

        # Step 2: Extract the access token (assuming only one top-level key)
        access_token = list(token_data.keys())[0]  # '51BAC0C9F0A871A83001F5F599E24EFF'

        # Step 3: Use it in the request
        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        response = requests.get(URL, headers=headers)

        data = response.json()

        pass