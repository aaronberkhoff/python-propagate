import numpy as np


class Sensor:

    def __init__(self, noise_mean, noise_covariance):
        self._noise_mean = np.asarray(noise_mean)[:, np.newaxis]
        self._noise_covariance = np.asarray(noise_covariance) * np.eye(
            len(noise_covariance)
        )

        # if measurement_map is not None:
        #     self.measurement_map = measurement_map  # Assign the default function

    @property
    def noise_mean(self):
        return self._noise_mean

    @property
    def noise_covariance(self):
        return self._noise_covariance


    def measurement_map(self, state):
        raise NotImplementedError("Measurement must be provided or overridden.")
