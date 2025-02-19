from concurrent.futures import ProcessPoolExecutor
from functools import partial
import copy

import numpy as np


from python_propagate.utilities.transforms import unscented_transform, unscented_weights, reconstruct_sigma_points
from python_propagate.states import State
from python_propagate.agents import Agent
from python_propagate.sensors import Sensor


class SigmaPoints:

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):

        sigma_points = unscented_transform(mean, covariance)
        self.sigma_points = sigma_points
        self._shape = sigma_points.shape
        self._weights = unscented_weights(shape=sigma_points.shape)

    def __iter__(self):
        return iter(self.states)

    def __sub__(self, matrix):

        if not isinstance(matrix, np.ndarray):
            return NotImplemented

        return self.sigma_points - matrix

    def __matmul___(self, matrix):

        if not isinstance(matrix, np.ndarray):
            return NotImplemented

        return self.sigma_points @ matrix

    @property
    def state(self):
        return self.sigma_points[0:6]

    @property
    def noise(self):
        return self.sigma_points[9:]

    @property
    def process_noise(self):
        return self.sigma_points[6:9]

    @property
    def matrix(self):
        return self.sigma_points

    @property
    def weights(self):
        return self._weights

    @property
    def shape(self):
        return self._shape

    @property
    def states(self):
        return [
            State(
                position=sigma_point[0:3],
                velocity=sigma_point[3:6],
                noise=sigma_point[9:],
                process_noise=sigma_point[6:9],
            )
            for sigma_point in self.sigma_points.T
        ]

    def propagate(self, agent: Agent, duration: int = 30, parallel: bool = False):

        # sigma_points = self.state.sigma_points()

        if parallel:

            process_with_sat = partial(
                parallel_propagate, **{"agent": agent, "duration": duration}
            )

            with ProcessPoolExecutor() as executor:

                state_final = list(executor.map(process_with_sat, self))

            self.sigma_points[0:6, :] = np.array(
                [state.compile() for state in state_final]
            ).T
        else:
            # Collect the updated states in a list

            for i, sigma_point in enumerate(self):

                agent.state = sigma_point  # Update the current state with a sigma point
                agent.propagate(duration=duration)  # Propagate the state
                self.sigma_points[0:6, i] = agent.state.compile()

    def measurement_map(self,sensor:Sensor):

        measurement_map = (
            np.hstack([sensor.measurement_map(state) for state in self])
            + self.noise
        )

        measurement_mean, measurement_covariance = reconstruct_sigma_points(
            measurement_map, weights=self.weights
        )

        return measurement_mean, measurement_covariance
    
    def reconstruct_state(self):

        return reconstruct_sigma_points(sigma_points= self.state, weights=self.weights)

def parallel_propagate(sigma_point: np.ndarray, agent: Agent, duration: int = 30):

    local_agent = copy.deepcopy(agent)
    local_agent.state = sigma_point
    local_agent.propagate(duration=duration)

    return local_agent.state
