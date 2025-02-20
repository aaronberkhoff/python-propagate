import numpy as np
from munkres import Munkres, DISALLOWED

from python_propagate.filters import Filter
from python_propagate.filters.unscented_kalman import UnscentedKalman
from python_propagate.utilities.calculations import mahalanobis_distance
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.states.unscented_state import FilterState
from python_propagate.filters.data_handler import DataHandler
from scipy.stats import chi2
from python_propagate.utilities.progress_bar import progress
from tqdm import tqdm
from copy import copy


class SingleHypothesisTracker:

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
        data_association_algo=Munkres(),
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
        self._data_association_algo = data_association_algo

        self.data_handler = DataHandler()

        self.state = FilterState(
            state_mean=mean,
            state_covariance=covariance,
            noise_mean=sensor.noise_mean,
            noise_covariance=sensor.noise_covariance,
            process_noise_mean=process_noise_mean,
            process_noise_covariance=process_noise_covariance,
        )

    @property
    def probability_of_detection(self):
        return self._probability_of_detection

    @property
    def probability_of_gating(self):
        return self._probability_of_gating

    @property
    def gating_threshold(self):
        return self._gating_threshold

    @property
    def data_association_algo(self):
        return self._data_association_algo

    @progress
    def run(self, observations: np.ndarray, times):
        print("Starting...........")
        missed_detections = np.zeros((observations.shape[-1]))
        # data association
        most_likely_measurement, missed_detections[0] = self.data_association(
            observation=observations[:, :, 0]
        )

        # measurement update
        if most_likely_measurement is not None:
            self.measurement_update(most_likely_measurement, time=times[0])

        self.data_handler.add_state(copy(self.state), time=times[0])

        # time loop
        loop_iterable = list(
            zip(times[1:], times[:-1], observations.transpose(2, 0, 1)[1:])
        )
        for i, (tk, tkm1, observation) in enumerate(
            tqdm(loop_iterable, desc="Running SHT Filter", unit="step")
        ):

            ##propagation
            self.time_update(duration=tk - tkm1)

            ##date association
            most_likely_measurement, missed_detections[i + 1] = self.data_association(
                observation=observation
            )

            ##measurement update
            if most_likely_measurement is not None:
                self.measurement_update(most_likely_measurement, time=tk)

            self.data_handler.add_state(copy(self.state), time=tk)

        return self.state, np.sum(missed_detections)

    def data_association(self, observation: np.ndarray):

        obs_shape = observation.shape[1]

        sigma_points = SigmaPoints(self.state.mean, self.state.covariance)

        measurement_mean, measurement_cov, _ = sigma_points.measurement_map(
            sensor=self.sensor
        )

        hypotheses_array = np.array(
            [
                mahalanobis_distance(
                    measurement_mean, meas[:, np.newaxis], measurement_cov
                )
                ** 2
                for meas in observation.T
            ]
        )

        temp = np.full((obs_shape, obs_shape), np.inf)
        np.fill_diagonal(temp, self.gating_threshold)

        hypotheses_array = np.block([[hypotheses_array], [temp]]).T

        indices = self.data_association_algo.compute(hypotheses_array)

        most_likely_measurement = [i for i, idx in enumerate(indices) if idx[1] == 0]

        if most_likely_measurement:
            return (
                observation[:, indices[most_likely_measurement[0]][0]][:, np.newaxis],
                0,
            )

        return (None, 1)

    def measurement_update(self, measurement: np.ndarray, time):

        sigma_points = SigmaPoints(self.state.mean, self.state.covariance)
        state_estimate, covariance_estimate, measurement_mean, measurement_cov = (
            self.filter.measurement_update(
                measurement,
                sigma_points,
                self.state.state_mean,
                self.state.state_covariance,
            )
        )
        self.state.state_mean = state_estimate
        self.state.state_covariance = covariance_estimate

        self.data_handler.add_measurement(
            measurement_mean=measurement_mean,
            measurement_covariance=measurement_cov,
            time=time,
        )
        self.data_handler.add_residual(measurement - measurement_mean)

        pass

    def time_update(self, duration: float):

        sigma_points = self.filter.time_update(
            state_estimate=self.state.mean,
            covariance_estimate=self.state.covariance,
            duration=duration,
        )
        state_bar, cov_bar = sigma_points.reconstruct_state()

        self.state.state_mean = state_bar
        self.state.state_covariance = cov_bar

        pass
