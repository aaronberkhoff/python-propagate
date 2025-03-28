import numpy as np
from scipy.stats import multivariate_normal

from python_propagate.filters import Filter
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.utilities.calculations import mahalanobis_distance
from python_propagate.utilities.transforms import reconstruct_sigma_points

from python_propagate.states.unscented_state import FilterState
from python_propagate.filters.data_handler import DataHandler
from python_propagate.plots.residual_plot import plot_residuals
from python_propagate.filters.gaussian_mixture import GaussianMixture


class GaussianMixtureSplit(GaussianMixture):

    """
    Class to represent a Gaussian Mixture Filter with a split function without dynamic splitting based on entropy"""

    def __init__(self, means, covariances, weights, agent, sensor, process_noise_mean, process_noise_covariance, filter, prune_threshold=0.00001, merge_threshold=3, component_split = 3):
        

        if component_split == 3:

            self.component_split = component_split
            self.component_data = {"alpha": np.array([0.2252246249, 0.5495507502, 0.2252246249]),
                                   "m": np.array([-1.0575154615, 0.0, 1.0575154615]),
                                   "sigma": np.array([0.6715662887, 0.6715662887, 0.6715662887])}

        elif component_split == 5:
            self.component_data ={"alpha": [0.0763216491, 0.2474417860, 0.3524731300, 0.2474417860, 0.0763216491],
                                  "m": [-1.6899729111, -0.8009283834, 0, 0.8009283834, 1.6899729111],
                                  "sigma": [0.4422555386, 0.4422555386, 0.4422555386, 0.4422555386, 0.4422555386]}
            
            self.component_split = component_split

        elif component_split == 0:
            self.component_split = 0

        else:
            raise ValueError("Component split must be either 3 or 5 or None")
        

        super().__init__(means, covariances, weights, agent, sensor, process_noise_mean, process_noise_covariance, filter, prune_threshold, merge_threshold)



    def split(self):


        new_states = []
        new_weights = []

        for state,weight in zip(self.states,self.weights):

            cov = state.state_covariance
            mean = state.state_mean
            mean = mean.squeeze()
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            lam = np.diag(eigenvals)


            for i in range(self.component_split):
                alpha = self.component_data["alpha"][i]
                mi = self.component_data["m"][i]
                sigma = self.component_data["sigma"][i]

                lami = lam.copy()  # Use `.copy()` to avoid overwriting the original
                lami[-1, -1] *= sigma**2
                cov = eigenvecs @ lami @ eigenvecs.T
                mean_temp = mean + np.sqrt(eigenvals[-1]) * mi * eigenvecs[:, -1]
                weight_temp = alpha * weight

                new_weights.append(weight_temp)
                new_states.append(FilterState(
                    state_mean=mean_temp[:,np.newaxis],
                    state_covariance=cov,
                    noise_mean=self.sensor.noise_mean,
                    noise_covariance=self.sensor.noise_covariance,
                    process_noise_mean=self.process_noise_mean,
                    process_noise_covariance=self.process_noise_covariance,
                ))
        
        self.states = new_states
        self.weights = new_weights


        pass 

    def time_update(self, duration):

        if self.component_split:
            self.split()

        for i, state in enumerate(self.states):

            sigma_points = self.filter.time_update(
                state_estimate=state.mean,
                covariance_estimate=state.covariance,
                duration=duration,
                parallel=False,
            )

            self.states[i].state_mean, self.states[i].state_covariance = (
                reconstruct_sigma_points(
                    sigma_points=sigma_points.state, weights=self.filter.weights
                )
            )
            print(f"state: {i}")

        state_bar, covariance_bar = self.merge_distribution()

        return FilterState(
            state_mean=state_bar,
            state_covariance=covariance_bar,
            noise_mean=self.sensor.noise_mean,
            noise_covariance=self.sensor.noise_covariance,
            process_noise_mean=self.process_noise_mean,
            process_noise_covariance=self.process_noise_covariance,
        )


        