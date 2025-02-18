import numpy as np
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.sensors import Sensor


class Filter:

    def __init__(
        self,
        spacecraft: Spacecraft,
        covariance: np.ndarray,
        sensor: Sensor,
        process_noise_mean: np.ndarray,
        process_noise_covariance: np.ndarray,
    ):

        self.spacecraft = spacecraft
        self.covariance = covariance
        self._sensor = sensor
        self._process_noise_mean = process_noise_mean
        self._process_noise_covariance = process_noise_covariance


    @property
    def sensor(self):
        return self._sensor

    @property
    def process_noise_mean(self):
        return self._process_noise_mean

    @property
    def process_noise_covariance(self):
        return self._process_noise_covariance
