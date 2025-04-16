import numpy as np


class Sensor:

    def __init__(self, noise_mean, noise_covariance, measurement_map=None):
        self._noise_mean = np.asarray(noise_mean).reshape(-1, 1)
        self._noise_covariance = np.array(noise_covariance) * np.eye(
            len(noise_covariance)
        )

        if measurement_map is not None:
            self.measurement_map = measurement_map  # Assign the default function

    @property
    def noise_mean(self):
        return self._noise_mean

    @property
    def noise_covariance(self):
        return self._noise_covariance


def measurement_map(self, state):
    raise NotImplementedError("Dynamic function must be provided or overridden.")
