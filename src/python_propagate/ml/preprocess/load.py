import h5py as h5
import numpy as np



def load_data_from_h5(file):

    data_to_extract = [#'index',
                       'time_sec',
                       'X_INERTIAL_KM',
                       'Y_INERTIAL_KM',
                       'Z_INERTIAL_KM',
                       'VX_INERTIAL_KMS',
                       'VY_INERTIAL_KMS',
                       'VZ_INERTIAL_KMS',
                       ]

    data_dict = {}
        
    # Access a specific dataset by name (replace 'dataset_name' with the actual name)
    with h5.File(file, "r") as h5file:
        for agent_name in h5file.keys():  # Loop through each agent
            agent_data = []  # Store valid attributes per agent
            
            for attribute in data_to_extract:  # Loop through attributes
                data = np.array(h5file[agent_name][attribute])  # Convert to NumPy array
                
                # Only store numeric attributes
                if np.issubdtype(data.dtype, np.number):
                    agent_data.append(data)

            # Convert agent data to a single NumPy array
            if agent_data:
                # Stack attributes as feature vectors and add to data_list
                agent_data_array = np.column_stack(agent_data)
                data_dict[agent_name] = agent_data_array
        
        # Now 'data' is a NumPy array containing the dataset's contents
        # print(data)
        return data_dict


    


