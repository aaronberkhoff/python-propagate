import yaml
import os
import getpass
from datetime import timedelta
import time


from spacetrack import SpaceTrackClient, AuthenticationError
import spacetrack.operators as op


CREDENTIALS_FILE = "credentials.yaml"


class SpaceTrackConnector:
    def __init__(self):
        self.username, self.password = self._get_credentials()
        self.client = SpaceTrackClient(self.username, self.password)

    def _get_credentials(self):
        """Retrieves stored credentials from a YAML file or prompts the user."""
        if os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "r") as f:
                creds = yaml.safe_load(f) or {}
                space_track_creds = creds.get("space_track", {})
                username = space_track_creds.get("username")
                password = space_track_creds.get("password")

                if username and password and self._validate_login(username, password):
                    print("✅ Loaded valid credentials from YAML file.")
                    return username, password
                else:
                    print("⚠️ Stored credentials are invalid or missing.")

        return self._prompt_for_credentials()

    def _prompt_for_credentials(self):
        """Prompts user for credentials and stores them if valid."""
        while True:
            username = input("Enter your Space-Track username: ")
            password = getpass.getpass("Enter your Space-Track password: ")

            if self._validate_login(username, password):
                self._store_credentials(username, password)
                print("✅ Credentials validated and stored in YAML file.")
                return username, password
            else:
                print("❌ Login failed. Please try again.")

    def _validate_login(self, username, password):
        """Checks if the provided credentials are valid."""
        try:
            client = SpaceTrackClient(username, password)
            client.authenticate()  # Test query
            return True
        except AuthenticationError as e:
            print("❌ Authentication error:", e)
            return False

    def _store_credentials(self, username, password):
        """Stores credentials securely in a YAML file."""
        creds = {"username": username, "password": password}
        with open(CREDENTIALS_FILE, "w") as f:
            yaml.safe_dump(creds, f)

        # Restrict file permissions to owner only
        os.chmod(CREDENTIALS_FILE, 0o600)

    def fetch_latest_tle(self, norad_id):
        """Fetch the latest TLE for a given NORAD ID."""
        try:
            return self.client.tle_latest(
                norad_cat_id=norad_id, ordinal=1, format="tle"
            )
        except Exception as e:
            print(f"Error fetching TLE: {e}")
            return None

    def get_tle_history(self, norad_id, start_time, end_time):
        """
        Retrieves the TLE history for a given NORAD ID between the start and end UTC times.

        Parameters:
        - norad_id: The NORAD catalog ID of the satellite.
        - start_time: The start time as a datetime object (in UTC).
        - end_time: The end time as a datetime object (in UTC).

        Returns:
        - A list of tuples with (timestamp, TLE).
        """
        # Convert the start_time and end_time to UTC strings that SpaceTrack can understand
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

        tle_history = []
        current_time = start_time

        drange = op.inclusive_range(start_time, end_time)

        # Loop through each day in the date range to fetch TLEs
        try:
            # Query SpaceTrack for the TLE closest to the current time
            lines = self.client.gp_history(
                iter_lines=True, creation_date=drange, orderby="TLE_LINE1", format="tle"
            )
            if tle_data:
                for tle in tle_data:
                    timestamp = tle.get(
                        "epoch"
                    )  # Or a specific timestamp field depending on the API response
                    tle_history.append((timestamp, tle.get("tle")))

            # Move to the next day
            current_time += timedelta(days=1)

            # Optional: sleep to avoid hitting API limits too quickly
            time.sleep(1)

        except Exception as e:
            print(
                f"Error fetching TLE for {norad_id} between {start_time} and {end_time}: {e}"
            )

        return tle_history

    def clear_credentials(self):
        """Deletes stored credentials."""
        if os.path.exists(CREDENTIALS_FILE):
            os.remove(CREDENTIALS_FILE)
            print("✅ Credentials cleared from YAML file.")
