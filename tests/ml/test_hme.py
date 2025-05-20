import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from python_propagate.constructors.yaml_constructors import load_yaml
from python_propagate.mems.hme import HME
import pandas as pd


def load_data_from_h5(file):

    data_to_extract = ['epoch_time',
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
                # if np.issubdtype(data.dtype, np.number):
                agent_data.append(data)

            # Convert agent data to a single NumPy array
            if agent_data:
                # Stack attributes as feature vectors and add to data_list
                agent_data_array = np.column_stack(agent_data)
                data_dict[agent_name] = agent_data_array
        
        # Now 'data' is a NumPy array containing the dataset's contents
        # print(data)
        return pd.DataFrame(data_dict)
    


def load_data(agent_name = 'GEO1'):
    file = 'examples/hme/results/test1/case1.h5'
    # data = load_data_from_h5(file = 'tests/ml/results/geo_test1/MLForge.h5')
    # data = load_data_from_h5(file = 'examples/hme/results/test1/case1.h5')
    data_dict = {}
    with h5.File(file, "r") as f:
    # Inspect keys
        print(list(f.keys()))
        data_dict[agent_name] = {}
        # Assume 'data' is the key containing your dataset
        for att in f[agent_name].keys():
            
            data_dict[agent_name][att] = f[agent_name][att]
        df = pd.DataFrame(data_dict[agent_name])

    return df


def test_hme() -> None:

    #load experts
    # config = load_yaml(yaml_file='examples/hme_example.yaml')
    
    # experts = config['experts']
    # experts = config['experts'].agents

    #load data to test
    data = load_data(agent_name='GEO_Sat2')

    #initialize the hme

    hme = HME(experts_file='examples/hme_example.yaml',device='cpu')

    #run HME
    expert_dict = hme.run(observations_dataframe=data,num_epochs=10000, parallel=0)


    # T, N = weights.shape
    # time = np.arange(len(data[1:]))

    plt.figure(figsize=(12, 6))
    for expert in expert_dict.keys():
        plt.plot(expert_dict[expert]['probabilities'], label=f'Expert: {expert}, MAX: {np.max(expert_dict[expert]["probabilities"])}')

    plt.xlabel('Time step')
    plt.ylabel('Gating weight')
    plt.title('Gating network weights over time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('examples/hme/results/test1/experts_weights.png')
    plt.show()
    






if __name__ == "__main__":

    test_hme()
    
    
