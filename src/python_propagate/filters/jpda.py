import numpy as np
from scipy.stats import chi2
from tqdm import tqdm
import itertools
from scipy.optimize import linear_sum_assignment

from python_propagate.states.unscented_state import FilterState
from python_propagate.filters.data_handler import DataHandler
from python_propagate.utilities.calculations import mahalanobis_distance
from python_propagate.utilities.transforms import reconstruct_sigma_points
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.filters import Filter
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.utilities.progress_bar import progress
from python_propagate.utilities.hungarian import (
    murty,
    get_best_assignments,
    MurtyKBestAssigner,
)


from copy import copy


class JPDA:

    def __init__(
        self,
        agent,
        means,
        covariances,
        sensor,
        process_noise_mean,
        process_noise_covariance,
        filter: Filter = UnscentedKalman,
        probability_of_detection=0.9,
        probability_of_gating=0.99,
        clutter_poisson_mean=1.5,
        max_hypothesis=20,
    ):

        process_noise_mean = np.asarray(process_noise_mean).reshape(-1, 1)
        process_noise_covariance = np.array(process_noise_covariance) * np.eye(
            len(process_noise_mean)
        )

        # self.filter = filter(
        #     spacecraft=agent,
        #     mean=mean,
        #     covariance=covariance,
        #     sensor=sensor,
        #     process_noise_mean=process_noise_mean,
        #     process_noise_covariance=process_noise_covariance,
        # )

        self._probability_of_detection = probability_of_detection
        self._probability_of_gating = probability_of_gating
        self._gating_threshold = chi2.ppf(probability_of_gating, 2)
        self.agent = agent
        self.sensor = sensor

        self.data_handler = [DataHandler() for _ in range(len(means))]

        self.states = [
            FilterState(
                state_mean=mean.reshape(-1, 1),
                state_covariance=cov,
                noise_mean=sensor.noise_mean,
                noise_covariance=sensor.noise_covariance,
                process_noise_mean=process_noise_mean,
                process_noise_covariance=process_noise_covariance,
            )
            for mean, cov in zip(means, covariances)
        ]

        self._clutter_poisson_mean = clutter_poisson_mean
        self._probability_factor = (
            clutter_poisson_mean
            * (1 - probability_of_detection * probability_of_gating)
            * 2
            * np.pi ** (self.states[0].n_noise)
        ) / probability_of_detection

        self.max_hypothesis = max_hypothesis

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

    # @progress
    def run(self, observations: np.ndarray, times: np.ndarray):

        # Merge hypothesis

        # Measurement Update
        # #TODO; Can edit sigmapoints to be a property of the class (FilterState or FIlter)

        sigma_points = [
            SigmaPoints(state.mean, state.covariance) for state in self.states
        ]

        self.measurement_update(
            observation=observations[0],
            time=times[0],
            sigma_points=sigma_points,
        )
        # Time loop
        loop_iterable = list(zip(times[1:], times[:-1], observations[1:]))
        for i, (tk, tkm1, observation) in enumerate(
            tqdm(loop_iterable, desc="Running JPDA Filter", unit="step")
        ):

            ## time update
            sigma_points = self.time_update(duration=tk - tkm1, time=tk)

            ## merge hypothesis
            ## measurement update
            self.measurement_update(
                observation=observation, sigma_points=sigma_points, time=tk
            )

        return self.states

    def measurement_update(self, observation, time, sigma_points):

        expected_measurements = [
            sigma_point.measurement_map(sensor=self.sensor)
            for sigma_point in sigma_points
        ]

        # Step 1: Gating:
        valid_observations, prob_detection, prob_missed = self._data_association(
            observation=observation, expected_measurements=expected_measurements
        )

        for i, (meas, sigma_point) in enumerate(
            zip(expected_measurements, sigma_points)
        ):
            state_difference = sigma_point.state - self.states[i].state_mean
            measurement_difference = meas[2] - meas[0]

            cross_covariance = np.einsum(
                "ij,j,kj->ik",
                state_difference,
                sigma_point.weights[1],
                measurement_difference,
            )

            kalman_gain = np.linalg.solve(meas[1].T, cross_covariance.T).T

            self._covariance_update(
                kalman_gain=kalman_gain,
                expected_measurement=meas[0],
                expected_measurement_cov=meas[1],
                valid_observations=valid_observations[i],
                prob_detection=prob_detection[i],
                prob_missed_detection=prob_missed[i],
                index=i,
            )

            self._state_update(
                kalman_gain=kalman_gain,
                expected_measurement=meas[0],
                valid_observations=valid_observations[i],
                prob_detection=prob_detection[i],
                index=i,
            )

            self._save_measurement_data(i, expected_measurement=meas, time=time)
            self._save_state_data(i, time)

        pass

    def time_update(self, duration: int, parallel=False, time=None):

        sigma_points = []

        for i, state in enumerate(self.states):

            sigma_point = SigmaPoints(state.mean, state.covariance)

            sigma_point.propagate(
                agent=self.agent, duration=duration, parallel=parallel
            )

            state.state_mean, state.state_covariance = sigma_point.reconstruct_state()

            sigma_points.append(sigma_point)
            # self._save_state_data(i,time)

        return sigma_points

    def _data_association(self, observation, expected_measurements):

        costs = []
        n_states = len(self.states)
        idx_states = n_states - 1

        # pdfs = np.zeros((len(self.states), len(observation)))

        for meas, obs in itertools.product(expected_measurements, observation):

            costs.append(self._cost(obs, meas[0], meas[1]))

        costs = np.reshape(costs, (n_states, len(observation)))

        costs = -np.hstack([self._cost_missed_detection(n_states), costs])

        assigner = MurtyKBestAssigner(costs, k_best=self.max_hypothesis)
        assignments, costs = assigner.find_k_best()

        _, col_assignments = zip(*assignments)
        col_assignments = np.asarray(col_assignments).T
        probabilities = 1 / np.exp(costs)
        probabilities /= np.sum(probabilities)

        prob_missed_detection = [
            probabilities[id <= idx_states] for id in col_assignments
        ]
        prob_detection = [probabilities[id > (idx_states)] for id in col_assignments]

        valid_observations = [
            observation[id[id > idx_states] - n_states] for id in col_assignments
        ]

        try:
            self.data_handler[0].metadata["num_zl0"].append(len(valid_observations[0]))
            self.data_handler[1].metadata["num_zl1"].append(len(valid_observations[1]))
        except KeyError:
            self.data_handler[0].metadata["num_zl0"] = [len(valid_observations[0])]
            self.data_handler[1].metadata["num_zl1"] = [len(valid_observations[1])]

        return valid_observations, prob_detection, prob_missed_detection

    def _covariance_update(
        self,
        kalman_gain,
        expected_measurement,
        expected_measurement_cov,
        valid_observations,
        prob_detection,
        prob_missed_detection,
        index,
    ):
        p0 = np.sum(prob_missed_detection)

        ztilde = (valid_observations - expected_measurement.T)[:, :, np.newaxis]
        Ztilde = np.einsum("j, jik", prob_detection, ztilde)

        self.states[index].state_covariance = (
            p0 * self.states[index].state_covariance
            + (1 - p0)
            * (
                self.states[index].state_covariance
                - kalman_gain @ expected_measurement_cov @ kalman_gain.T
            )
            + (
                kalman_gain
                @ (
                    np.einsum(
                        "i,ijk,ilm->jm",
                        prob_detection,
                        ztilde,
                        ztilde.transpose(0, 2, 1),
                    )
                    - np.einsum("ji,ik->jk", Ztilde, Ztilde.T)
                )
                @ kalman_gain.T
            )
        )

        pass

    def _state_update(
        self,
        kalman_gain,
        expected_measurement,
        valid_observations,
        prob_detection,
        index,
    ):

        ztilde = valid_observations - expected_measurement.T

        ztilde_all = np.einsum("j,jk", prob_detection, ztilde).reshape(-1, 1)

        self.states[index].state_mean += kalman_gain @ ztilde_all

        self._save_residual_data(index, ztilde_all)

        pass

    # TODO combine updates as to not recalc ztilde

    def _save_measurement_data(self, index, expected_measurement, time):

        self.data_handler[index].add_measurement(
            measurement_mean=expected_measurement[0],
            measurement_covariance=expected_measurement[1],
            time=time,
        )

        pass

    def _save_residual_data(self, index, innovation):

        self.data_handler[index].add_residual(innovation)

        pass

    def _save_state_data(self, index, time):

        self.data_handler[index].add_state(copy(self.states[index]), time)

        pass

    def _pdf(self, distance2, covariance):

        pdf = (
            1 / np.sqrt(np.linalg.det(2 * np.pi * covariance)) * np.exp(-distance2 / 2)
        )

        return pdf

    def _cost(self, observation, measurement_expected, measurement_cov):

        dij2 = (
            mahalanobis_distance(
                observation, measurement_expected.ravel(), measurement_cov
            )
            ** 2
        )

        if dij2 > self.gating_threshold:
            return -np.inf

        pdf = self._pdf(dij2, measurement_cov)

        return np.log(self.probability_of_detection * pdf / self.clutter_poisson_mean)

    def _cost_missed_detection(self, n_agents):

        return np.log(
            np.eye(n_agents)
            * (1 - self.probability_of_detection * self.probability_of_gating)
        )
