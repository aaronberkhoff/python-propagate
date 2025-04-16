import numpy as np

from python_propagate.sensors import Sensor
from python_propagate.states import State


class Optical(Sensor):

    def __init__(self, noise_mean, noise_covariance):

        super().__init__(
            noise_mean, noise_covariance, measurement_map=self.ra_dec_measurement_map
        )


    def ra_dec_measurement_map(self,state: State):

        right_ascension = np.arctan2(state.position[1], state.position[0])
        declination = np.arcsin(state.position[2] / np.linalg.norm(state.position))

        return np.array([right_ascension, declination]).reshape(-1,1)
    




