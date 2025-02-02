
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd


from python_propagate.scenario import Scenario
from python_propagate.utilities.units import RAD2DEG, DEG2RAD

class DataGenerator(Scenario):

    def __init__(self, central_body, 
                start_time,
                duration, 
                dt,
                data_types: tuple = ('right_ascension','declination'),
                agents = ...,
                stations = ...,
                plots = None,
                output_directory: str = 'exmples/results',
                name: str = 'None'
                ):
        super().__init__(central_body, start_time, duration, dt, agents, stations)

        self._data_types = data_types
        self._plots = plots
        self._name = name
        
        self._output_directory = Path(output_directory)
        self._output_directory.mkdir(exist_ok=True)

    @property
    def data_types(self):
        return self._data_types
    
    @property
    def output_directory(self):
        return self._output_directory
    
    @property
    def plots(self):
        return self._plots
    
    @property
    def name(self):
        return self._name


    def run(self):
        # List to accumulate data for all agents
        data_all = []

        for agent in self.agents:
            agent.propagate()  # Update or propagate the agent state
            data_agent = []  # List for this agent's measurement data
            
            for i, state in enumerate(agent.state_data):
                for station in self.stations:
                    # Calculate azimuth and elevation (convert to degrees)
                    az, el = station.calculate_azimuth_and_elevation(state=state)
                    az *= RAD2DEG
                    el *= RAD2DEG

                    # Calculate right ascension and declination (convert to degrees)
                    ra, dec = station.calculate_ra_and_dec(state=state)
                    ra *= RAD2DEG
                    dec *= RAD2DEG

                    # Calculate range and range rate
                    rho, rhodot = station.calculate_range_and_range_rate_from_target(state=state)

                    # Only include data if the elevation is above the station's minimum angle
                    if el > station.minimum_elevation_angle:
                        data_agent.append({
                            "agent": agent.name,  # Include the agent (satellite) name
                            "index": i,
                            "time": state.time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "RA_DEG": ra,
                            "DEC_DEG": dec,
                            "AZ_DEG": az,
                            "EL_DEG": el,
                            "Range_KM": rho,
                            "Range_Rate_KMS": rhodot,
                            "station": station.name,
                            "station_id": station._identity,
                        })
            
            # Append this agent's data to the master list
            data_all.extend(data_agent)

        # Create a DataFrame from the flattened list of measurement dictionaries.
        df = pd.DataFrame(data_all)

        # Prepare file names using self.name
        file_name_h5 = self.name + ".h5"
        file_name_xlsx = self.name + ".xlsx"
        file_name_csv = self.name + ".csv"
        
        # Create full paths using the output directory (assumed to be a Path object)
        output_path_h5 = self.output_directory.joinpath(file_name_h5)
        output_path_xlsx = self.output_directory.joinpath(file_name_xlsx)
        output_path_csv = self.output_directory.joinpath(file_name_csv)

        # Write the DataFrame to an HDF5 file.
        # mode='w' ensures the file is overwritten if it exists.
        df.to_hdf(output_path_h5, key='df', mode='w')

        # Write the DataFrame to an Excel file.
        # Setting index=False avoids writing the DataFrame index as a separate column.
        df.to_excel(output_path_xlsx, index=False)

        # (Optional) Write to a CSV file.
        df.to_csv(output_path_csv, index=False)

        print(f"Files written:\n  HDF5: {output_path_h5}\n  Excel: {output_path_xlsx}\n  CSV: {output_path_csv}")





