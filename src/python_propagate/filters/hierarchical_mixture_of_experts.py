from python_propagate.forge.genes import Gene, Genes
from python_propagate.scenario import Scenario
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State, Agent
from python_propagate.sensors import Sensor
from datetime import datetime, timedelta
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.filters.gating_network import GatingNetwork
from python_propagate.states.unscented_state import FilterState
import numpy as np
import concurrent.futures
from collections import namedtuple

# Define the Expert named tuple
Expert = namedtuple('Expert', ['residuals_data', 'covariance_estimate_hist', 'measurement_mean_hist', 'measurement_covariance_hist', 'state_estimate_hist', 'state_mean_hist'])

class HierarchicalMixtureExperts:

    """
    A class to represent a hierarchical mixture of experts.

    Attributes
    ----------
    scenario : Scenario
        The scenario of the dynamic.
    agent : Agent
        The agent of the dynamic.
    stm : STM
        The state transition matrix of the dynamic.
    function : function
        The function of the dynamic.
    """


    def __init__(self, gene, mean, covariance, sensor, noise_mean, noise_covariance, process_noise_mean, process_noise_covariance, filter_name=UnscentedKalman):

        # Define agents
        self.agents = [chrom for chrom in gene.chromosomes]

        # Define filter parameters
        self.mean = mean
        self.covariance = covariance
        self.sensor = sensor
        self.process_noise_mean = process_noise_mean
        self.process_noise_covariance =  process_noise_covariance
    
        # Build filter (UKF)
        self.filter = filter_name
            
     


    
    # Build Expert
    def hme_expert(self, spacecraft, time, measurements):

        # Run Filter
        unscented_kalman = UnscentedKalman(
                                spacecraft=spacecraft,
                                mean=np.array(self.mean),
                                covariance=self.covariance,
                                sensor=self.sensor,
                                process_noise_mean=self.process_noise_mean,
                                process_noise_covariance=self.process_noise_covariance,
                            )
        state_estimate, covariance_estimate = unscented_kalman.run(measurements,time)

        # Create the Expert named tuple
        expert = Expert(
            residuals_data=np.array(unscented_kalman.residuals_data),
            covariance_estimate_hist=unscented_kalman.covariance_estimate_hist,
            measurement_mean_hist=unscented_kalman.measurement_mean_hist,
            measurement_covariance_hist=unscented_kalman.measurement_covariance_hist,
            state_estimate_hist=unscented_kalman.state_estimate_hist, 
            state_mean_hist=unscented_kalman.state_mean_hist
        )

        return expert
        
    
        
    def run(self, measurements, time):

        # Use ThreadPoolExecutor to parallelize the execution of hme_expert for each agent
        print("Filtering experts")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(self.hme_expert, self.agents, [measurements] * len(self.agents), [time] * len(self.agents)))
        
        # Run gating network
        # Instantiate the GatingNetwork class
        print("Weighting results")
        gating_network = GatingNetwork(eta=1.75)
        howManyObs = len(time)
        
        # Call the gating_network method with the prepared data
        expert_choice, ai_list = gating_network.run(results, measurements, howManyObs)
        
        return expert_choice, ai_list



