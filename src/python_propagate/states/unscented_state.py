import numpy as np

from python_propagate.states import State


class FilterState(State):

    def __init__(
        self,
        state_mean,
        state_covariance,
        noise_mean,
        noise_covariance,
        process_noise_mean=None,
        process_noise_covariance=None,
    ):

        if process_noise_mean is None:
            process_noise_mean = np.array([])

        if process_noise_covariance is None:
            process_noise_covariance = np.array([])

        self.n_state = state_mean.shape[0]
        self.n_noise = noise_mean.shape[0]
        self.n_process = process_noise_mean.shape[0]

        self.length = (
            state_mean.shape[0] + noise_mean.shape[0] + process_noise_mean.shape[0]
        )
        self.state_mean = state_mean
        self.state_covariance = state_covariance
        self.noise_mean = noise_mean
        self.noise_covariance = noise_covariance
        self.process_noise_mean = process_noise_mean
        self.process_noise_covariance = process_noise_covariance

        super().__init__(position=state_mean[0:3], velocity=state_mean[3:6])

    def __eq__(self, other):
        if not isinstance(other, FilterState):
            return False
        return (self.state_mean, self.state_covariance) == (
            other.state_mean,
            other.state_covariance,
        )

    def __hash__(self):
        return hash(
            (tuple(self.state_mean.ravel()), tuple(self.state_covariance.flatten()))
        )

    @property
    def mean(self):
        return np.vstack([self.state_mean, self.process_noise_mean, self.noise_mean])

    @property
    def covariance(self):
        return np.block(
            [
                [
                    self.state_covariance,
                    np.zeros((self.n_state, self.n_process)),
                    np.zeros((self.n_state, self.n_noise)),
                ],
                [
                    np.zeros((self.n_process, self.n_state)),
                    self.process_noise_covariance,
                    np.zeros((self.n_process, self.n_noise)),
                ],
                [
                    np.zeros((self.n_noise, self.n_state)),
                    np.zeros((self.n_noise, self.n_process)),
                    self.noise_covariance,
                ],
            ]
        )
