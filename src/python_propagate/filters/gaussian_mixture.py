import numpy as np
from scipy.stats import multivariate_normal

from python_propagate.filters import Filter
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.utilities.calculations import mahalanobis_distance
from python_propagate.utilities.transforms import reconstruct_sigma_points

from python_propagate.states.unscented_state import FilterState
from python_propagate.filters.data_handler import DataHandler
from python_propagate.plots.residual_plot import plot_residuals

class GaussianMixture:

    def __init__(self, means, covariances, weights,
                 agent, sensor,
                 process_noise_mean, process_noise_covariance, filter: Filter,
                 prune_threshold = 1e-5, merge_threshold = 3):
        
        self.weights = weights
        self._sensor  = sensor
        self.data_handler = DataHandler()
        # self._measurements = measurements
        # self._time = time
        self.component_hist = []
        process_noise_mean = np.array(process_noise_mean)[:,np.newaxis]
        process_noise_covariance = np.array(process_noise_covariance) * np.eye(len(process_noise_mean))

        self._process_noise_mean = process_noise_mean
        self._process_noise_covariance = process_noise_covariance
        self._agent = agent
        self._prune_threshold = prune_threshold
        self._merge_threshold = merge_threshold
        # self.filters = []
        self.states = [FilterState(
            state_mean=mean[:,np.newaxis],
            state_covariance=cov,
            noise_mean=sensor.noise_mean,
            noise_covariance=sensor.noise_covariance,
            process_noise_mean=process_noise_mean,
            process_noise_covariance=process_noise_covariance,
        ) for mean,cov in zip(means.T,covariances.transpose(2,0,1))]

        
        self.filter = filter(spacecraft = self.agent,
                                covariance = covariances[:,:,0],
                                mean = means[:,0],
                                sensor = self.sensor,
                                process_noise_mean = self.process_noise_mean,
                                process_noise_covariance = self.process_noise_covariance)



    @property
    def sensor(self):
        return self._sensor
    @property
    def measurements(self):
        return self._measurements
    @property
    def time(self):
        return self._time
    @property
    def process_noise_mean(self):
        return self._process_noise_mean
    @property
    def process_noise_covariance(self):
        return self._process_noise_covariance
    @property
    def agent(self):
        return self._agent
    @property
    def merge_threshold(self):
        return self._merge_threshold
    @property
    def prune_threshold(self):
        return self._prune_threshold

    

    
    def run(self,measurements,times):

        #first guess
        state_estimate, covariance_estimate = self.merge_distribution()

        filter_state = FilterState(
            state_mean=state_estimate,
            state_covariance=covariance_estimate,
            noise_mean=self.sensor.noise_mean,
            noise_covariance=self.sensor.noise_covariance,
            process_noise_mean=self.process_noise_mean,
            process_noise_covariance=self.process_noise_covariance,
        )

        # self.data_handler.add_state(filter_state,time = 0)
        
        #first measurement update
        filter_state = self.measurement_update(measurement=measurements[0])

        #store state data
        self.data_handler.add_state(filter_state,time=0)


        self.component_hist.append(len(self.states))
        self.merge_prune_truncate()
        

        for i, (tk, tkm1, measurement) in enumerate(zip(times[1:], times[0:-1], measurements[1:])):

            filter_state = self.time_update(duration=tk - tkm1)

            #store state data
            # self.data_handler.add_state(filter_state,time=tk)

            filter_state = self.measurement_update(measurement=measurement)

            #store state data
            self.data_handler.add_state(filter_state,time=tk)

            self.component_hist.append(len(self.states))
            self.merge_prune_truncate()

        
    

        self.component_hist.append(len(self.states))

        return filter_state, self.component_hist

    #Figure out how to handle sigma points

    def measurement_update(self,measurement):



        for i,(state) in enumerate(self.states):

            sigma_points = SigmaPoints(state.mean, state.covariance)
            state_estimate,  covariance_estimate, measurement_mean, measurement_covariance = self.filter.measurement_update(measurement,
                                                                                                                        sigma_points, 
                                                                                                                        state.state_mean,
                                                                                                                        state.state_covariance)
            self.states[i].state_mean = state_estimate
            self.states[i].state_covariance = covariance_estimate
            q_measurement = multivariate_normal.pdf(x=measurement,mean=measurement_mean.ravel(),cov=measurement_covariance)
            self.weights[i] *= q_measurement

        self.weights = self.weights / np.sum(self.weights)
        state_estimate, covariance_estimate = self.merge_distribution()

        self.data_handler.add_measurement(measurement_mean=measurement_mean,measurement_covariance=measurement_covariance)
        self.data_handler.add_residual(measurement[:,np.newaxis] - measurement_mean)

        return FilterState(
            state_mean=state_estimate,
            state_covariance=covariance_estimate,
            noise_mean=self.sensor.noise_mean,
            noise_covariance=self.sensor.noise_covariance,
            process_noise_mean=self.process_noise_mean,
            process_noise_covariance=self.process_noise_covariance,
        )
    

    def time_update(self, duration):

        for i,state in enumerate(self.states):

           sigma_points = self.filter.time_update(state_estimate = state.mean,covariance_estimate = state.covariance,duration=duration, parallel = False)
           
           self.states[i].state_mean, self.states[i].state_covariance = reconstruct_sigma_points(sigma_points=sigma_points.state, weights=self.filter.weights)
           print(f'state: {i}')

        state_bar, covariance_bar = self.merge_distribution()

        return FilterState(
            state_mean=state_bar,
            state_covariance=covariance_bar,
            noise_mean=self.sensor.noise_mean,
            noise_covariance=self.sensor.noise_covariance,
            process_noise_mean=self.process_noise_mean,
            process_noise_covariance=self.process_noise_covariance,
        )

    def merge_prune_truncate(self,j_max = 800):

        # 2.1
        subset_i = [(state,weight) for state, weight in zip(self.states, self.weights) if weight >= self.prune_threshold]
        subset_new = []
        
        # subset_l = set()
        while subset_i:
            #2.1 
            subset_l = []
            subset_diff = []
            #2.2 find the max value
            max_state, _ = max(subset_i, key=lambda x: x[1])

            #2.3 fill set L with the values that meet threshold
            for (state, weight) in subset_i:
    
                distance = mahalanobis_distance(max_state.state_mean,state.state_mean,max_state.state_covariance)[0,0]
                if distance <= self.merge_threshold:
                    subset_l.append((state, weight))
                else:
                    subset_diff.append((state,weight))
            
            # 2.4 calculate merged mean, covariance, and weight
            weight_l = np.sum(weight for _,weight in subset_l)
            mean_l = np.sum(weight/ weight_l * state.state_mean for state,weight in subset_l)
            covariance_l = np.sum(weight/ weight_l * (state.state_covariance + np.outer(state.state_mean ,state.state_mean)) for state, weight in subset_l)  - np.outer(mean_l,  mean_l)

            subset_new.append((FilterState(
                state_mean=mean_l,
                state_covariance=covariance_l,
                noise_mean=self.sensor.noise_mean,
                noise_covariance=self.sensor.noise_covariance,
                process_noise_mean=self.process_noise_mean,
                process_noise_covariance=self.process_noise_covariance),weight_l))
            

            #2.5 remove L from I
            subset_i = subset_diff
            # subset_i = subset_i - subset_l

        if len(subset_new) > j_max:
            subset_new.sort(key=lambda x: x[1], reverse=True)
            subset_new = subset_new[:j_max]
            

        #normalize weights
        weight_total = sum(weight for _,weight in subset_new)
        subset_new = [(state, weight / weight_total) for state, weight in subset_new]

        #save components
        self.states, self.weights = zip(*subset_new)
        self.weights = np.array(self.weights)

    def merge_distribution(self):
        means = np.array([state.state_mean.ravel() for state in self.states])#.T
        covs = np.array([state.state_covariance for state in self.states])#.transpose(1,2,0)
        state_bar = np.einsum('ij,ik->kj',self.weights,means)
        p_temp = np.array([cov  + np.outer(x_l[:,np.newaxis] - state_bar,x_l[:,np.newaxis] - state_bar) for x_l, cov in zip(means,covs)])
        covariance_bar = np.einsum('ij,ikl->kl',self.weights,p_temp)

        return state_bar, covariance_bar

    def plot_position_residuals(self,path,ylim = None):

        plot_residuals(self.data_handler, path, plot_type='position',ylim=ylim)

    def plot_velocity_residuals(self,path,ylim = None):

        plot_residuals(self.data_handler, path, plot_type='velocity',ylim=ylim)

    def plot_measurement_residuals(self,path,ylim = None):

        plot_residuals(self.data_handler,path,plot_type='measurement',ylim=ylim)

        




                                  

            
        