import numpy as np
from scipy.stats import chi2
from tqdm import tqdm

from python_propagate.states.unscented_state import FilterState
from python_propagate.filters.data_handler import DataHandler
from python_propagate.utilities.calculations import mahalanobis_distance
from python_propagate.utilities.transforms import reconstruct_sigma_points
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.filters import Filter
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.utilities.progress_bar import progress

from copy import copy


class PDAFilter:

    def __init__(
        self,
        agent,
        mean,
        covariance,
        sensor,
        process_noise_mean,
        process_noise_covariance,
        filter: Filter = UnscentedKalman,
        probability_of_detection=0.9,
        probability_of_gating=0.99,
        clutter_poisson_mean=2,
    ):

        process_noise_mean = np.array(process_noise_mean)[:, np.newaxis]
        process_noise_covariance = np.array(process_noise_covariance) * np.eye(
            len(process_noise_mean)
        )

        self.filter = filter(
            spacecraft=agent,
            mean=mean,
            covariance=covariance,
            sensor=sensor,
            process_noise_mean=process_noise_mean,
            process_noise_covariance=process_noise_covariance,
        )

        self._probability_of_detection = probability_of_detection
        self._probability_of_gating = probability_of_gating
        self._gating_threshold = chi2.ppf(probability_of_gating, 2)
        self.agent = agent
        self.sensor = sensor

        self.data_handler = DataHandler()

        self.state = FilterState(
            state_mean=mean,
            state_covariance=covariance,
            noise_mean=sensor.noise_mean,
            noise_covariance=sensor.noise_covariance,
            process_noise_mean=process_noise_mean,
            process_noise_covariance=process_noise_covariance,
        )

        self._clutter_poisson_mean = clutter_poisson_mean
        self._probability_factor = (
            clutter_poisson_mean
            * (1 - probability_of_detection * probability_of_gating)
            * 2
            * np.pi ** (self.state.n_noise)
        ) / probability_of_detection

    @property
    def probability_of_detection(self):
        return self._probability_of_detection

    @property
    def probability_of_gating(self):
        return self._probability_of_gating

    @property
    def probability_factor(self):
        return self._probability_factor

    @property
    def gating_threshold(self):
        return self._gating_threshold

    @property
    def clutter_poisson_mean(self):
        return self._clutter_poisson_mean

    @progress
    def run(self, observations: np.ndarray, times: np.ndarray):

        # Merge hypothesis

        # Measurement Update
        # #TODO; Can edit sigmapoints to be a property of the class (FilterState or FIlter)
        sigma_points = SigmaPoints(self.state.mean, self.state.covariance)

        self.measurement_update(
            observation=observations[:, :, 0], sigma_points=sigma_points, time=times[0]
        )

        self.data_handler.add_state(copy(self.state), time=times[0])
        # Time loop
        loop_iterable = list(
            zip(times[1:], times[:-1], observations.transpose(2, 0, 1)[1:])
        )
        for i, (tk, tkm1, observation) in enumerate(
            tqdm(loop_iterable, desc="Running PDA Filter", unit="step")
        ):

            ## time update
            sigma_points = self.time_update(duration=tk - tkm1)

            ## merge hypothesis
            ## measurement update
            self.measurement_update(
                observation=observation, sigma_points=sigma_points, time=tk
            )

            self.data_handler.add_state(copy(self.state), time=tk)

        return self.state

    def measurement_update(self, observation, sigma_points: SigmaPoints, time):

        expected_meas_mean, expected_meas_cov, measurement_sigma = (
            sigma_points.measurement_map(sensor=self.sensor)
        )

        merged_innovation, probabilities, innovations = self.merge_hypothesis(
            observation=observation,
            expected_meas_mean=expected_meas_mean,
            expected_meas_cov=expected_meas_cov,
        )
        merged_innovation_cov = self.calc_merged_covariance(
            sigma_points=sigma_points, probabilities=probabilities
        )

        state_difference = sigma_points.state - self.state.state_mean
        measurement_difference = measurement_sigma - expected_meas_mean

        cross_covariance = np.einsum(
            "ij,j,kj->ik",
            state_difference,
            sigma_points.weights[1],
            measurement_difference,
        )

        kalman_gain = np.linalg.solve(expected_meas_cov.T, cross_covariance.T).T

        p_temp = (
            self.state.state_covariance
            - cross_covariance @ kalman_gain.T
            - kalman_gain @ cross_covariance.T
            + kalman_gain @ expected_meas_cov @ kalman_gain.T
        )

        self.state.state_mean += kalman_gain @ merged_innovation
        self.state.state_covariance = (
            probabilities[0] * self.state.state_covariance
            + (1 - probabilities[0]) * p_temp
            + (
                kalman_gain
                @ (
                    (
                        np.einsum(
                            "i,ji,ik->jk", probabilities[1:], innovations, innovations.T
                        )
                    )
                    - np.outer(merged_innovation, merged_innovation)
                )
                @ kalman_gain.T
            )
        )

        self.data_handler.add_measurement(
            measurement_mean=expected_meas_mean,
            measurement_covariance=expected_meas_cov,
            time=time,
        )
        self.data_handler.add_merged(
            merged_innovation=merged_innovation,
            merged_covariance=merged_innovation_cov,
            time=time,
        )
        self.data_handler.add_residual(merged_innovation)

        pass

    def time_update(self, duration: int, parallel=False):

        sigma_points = SigmaPoints(self.state.mean, self.state.covariance)

        sigma_points.propagate(agent=self.agent, duration=duration, parallel=parallel)

        self.state.state_mean, self.state.state_covariance = (
            sigma_points.reconstruct_state()
        )

        return sigma_points

    def merge_hypothesis(
        self, observation: np.ndarray, expected_meas_mean, expected_meas_cov
    ):

        # sigma_points = SigmaPoints(self.state.mean,self.state.covariance)

        # expected_meas_mean, expected_meas_cov = sigma_points.measurement_map(sensor=self.sensor)

        probabilities = self.calc_probabilities(
            expected_meas_mean=expected_meas_mean,
            expected_meas_cov=expected_meas_cov,
            observation=observation,
        )
        innovations = observation - expected_meas_mean

        # merged_measurement = np.sum([innovation * prob for innovation,prob in zip(innovations.T,probabilities[1:])])
        merged_innovation = np.einsum(
            "ij,ki->kj", probabilities[1:, np.newaxis], innovations
        )

        return (
            merged_innovation,
            probabilities,
            innovations,
        )

    def calc_probabilities(self, observation, expected_meas_mean, expected_meas_cov):

        probability_array = np.zeros(observation.shape[-1] + 1)

        alpha = np.sum(
            [
                calc_alpha(
                    expected_meas_mean,
                    meas[:, np.newaxis],
                    expected_meas_cov,
                    self.gating_threshold,
                )
                for meas in observation.T
            ]
        )
        beta = self.probability_factor * np.sqrt(np.linalg.det(expected_meas_cov))

        probability_array[0] = beta / (beta + alpha)

        for i, meas in enumerate(observation.T):

            probability_array[i + 1] = calc_alpha(
                expected_meas_mean,
                meas[:, np.newaxis],
                expected_meas_cov,
                self.gating_threshold,
            ) / (beta + alpha)

        return probability_array

    def calc_merged_covariance(self, sigma_points: SigmaPoints, probabilities):

        measurement_sigma = np.hstack(
            [self.sensor.measurement_map(state) for state in sigma_points]
        )

        _, measurement_covariance = reconstruct_sigma_points(
            measurement_sigma, weights=sigma_points.weights
        )

        return measurement_covariance + self.sensor.noise_covariance * np.sum(
            probabilities**2
        )


def calc_alpha(expected_meas_mean, measurement, expected_meas_cov, gating_threshold):

    dist2 = (
        mahalanobis_distance(expected_meas_mean, measurement, expected_meas_cov) ** 2
    )

    if dist2 > gating_threshold:
        return 0

    return np.exp(-dist2 / 2)
