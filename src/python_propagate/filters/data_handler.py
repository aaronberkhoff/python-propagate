import numpy as np
import pandas as pd

from python_propagate.states.unscented_state import FilterState


class DataHandler:

    def __init__(self, truths=None):

        self.states = []
        self.measurement_means = []
        self.measurement_covariances = []

        self.merged_innovation = []
        self.merged_covariances = []
        self.merged_times = []

        self.state_times = []
        self.measurement_times = []
        self.residuals = []
        self.truths = truths

    def add_state(self, state: FilterState, time: float):

        self.states.append(state)
        self.state_times.append(time)

    def add_measurement(self, measurement_mean, measurement_covariance, time):

        self.measurement_means.append(measurement_mean)
        self.measurement_covariances.append(measurement_covariance)
        self.measurement_times.append(time)

    def add_merged(self, merged_innovation, merged_covariance, time):

        self.merged_innovation.append(merged_innovation)
        self.merged_covariances.append(merged_covariance)
        self.merged_times.append(time)

    def add_residual(self, residual):
        self.residuals.append(residual)

    @property
    def state_history(self):
        return [state.state_mean for state in self.states]

    @property
    def covariance_history(self):
        return [state.state_covariance for state in self.states]

    @property
    def state_estimate_history(self):
        return [state.state_mean for state in self.states]

    @property
    def covariance_estimate_history(self):
        return [state.state_covariance for state in self.states]

    @property
    def state_bar_history(self):
        return np.array([state.state_mean for state in self.states])

    @property
    def covariance_bar_history(self):
        return np.array([state.state_covariance for state in self.states])

    @property
    def state_residuals(self):
        return np.array(
            [
                state.ravel() - truth
                for state, truth in zip(self.state_estimate_history, self.truths)
            ]
        )
