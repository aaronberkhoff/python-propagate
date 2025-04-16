import numpy as np
import matplotlib.pyplot as plt


from python_propagate.filters import Filter
from python_propagate.utilities.transforms import (
    unscented_transform,
    reconstruct_sigma_points,
    unscented_weights,
)
from python_propagate.dynamics import Dynamic
from python_propagate.states import State
from python_propagate.states.unscented_state import FilterState
from python_propagate.sensors import Sensor
from python_propagate.states.sigma_points import SigmaPoints
from python_propagate.agents import Agent
from pathlib import Path

# from python_propagate.agents.spacecraft import Spacecraft


class UnscentedKalman(Filter):

    def __init__(
        self,
        spacecraft,
        mean,
        covariance,
        sensor: Sensor,
        process_noise_mean,
        process_noise_covariance,
    ):
        process_noise_mean = np.array(process_noise_mean)[:, np.newaxis]
        process_noise_covariance = np.array(process_noise_covariance) * np.eye(
            len(process_noise_mean)
        )

        state = FilterState(
            state_mean=mean[:, np.newaxis],
            state_covariance=covariance,
            noise_mean=sensor.noise_mean,
            noise_covariance=sensor.noise_covariance,
            process_noise_mean=process_noise_mean,
            process_noise_covariance=process_noise_covariance,
        )

        self._weights_mean, self._weights_covariance = unscented_weights(
            shape=(state.length, 2 * state.length + 1)
        )

        self.state = state
        self.spacecraft = spacecraft
        self.residuals_data = [[], []]
        self.residual_cov_data = [[], [], [], []]
        self.state_estimate_hist = []
        self.covariance_estimate_hist = []
        self.time_data = []

        if process_noise_mean is not None:
            self.spacecraft.add_dynamics((NoiseDynamic,))

        super().__init__(
            spacecraft,
            covariance,
            sensor,
            process_noise_mean,
            process_noise_covariance,
        )

    @property
    def weights(self):
        return (self._weights_mean, self._weights_covariance)

    @property
    def weights_mean(self):
        return self._weights_mean

    @property
    def weights_covariance(self):
        return self._weights_covariance

    def run(self, time, measurements):

        # process first measurement:

        sigma_points = SigmaPoints(self.state.mean, self.state.covariance)

        state_estimate, covariance_estimate, _, _ = self.measurement_update(
            measurement=measurements[0][:, np.newaxis],
            sigma_points=sigma_points,
            state_bar=self.state.state_mean,
            covariance_bar=self.state.state_covariance,
        )

        # new initial guess
        self.state.state_mean = state_estimate
        self.state.state_covariance = covariance_estimate

        for i, (tk, tkm1, measurement) in enumerate(
            zip(time[1:], time[0:-1], measurements[1:])
        ):

            sigma_points = self.time_update(
                self.state.mean, self.state.covariance, duration=tk - tkm1
            )

            state_bar, covariance_bar = reconstruct_sigma_points(
                sigma_points=sigma_points.state, weights=self.weights
            )

            state_estimate, covariance_estimate, _, _ = self.measurement_update(
                measurement=measurement[:, np.newaxis],
                sigma_points=sigma_points,
                state_bar=state_bar,
                covariance_bar=covariance_bar,
            )

            self.state.state_mean = state_estimate
            self.state.state_covariance = covariance_estimate

            self.time_data.append(i)

            print(f"Progress: {(i / len(measurements))*100}")
            # self.spacecraft.state.position = state_estimate[0:3]
            # self.spacecraft.state.velocity = state_estimate[3:6]
            # self.spacecraft.state.covariance_estimate = covariance_estimate

        return state_estimate, covariance_estimate

    def time_update(
        self, state_estimate, covariance_estimate, duration: int, parallel=False
    ):

        sigma_points = SigmaPoints(state_estimate, covariance_estimate)

        sigma_points.propagate(
            agent=self.spacecraft, duration=duration, parallel=parallel
        )

        return sigma_points

    def measurement_update(
        self,
        measurement: np.ndarray,
        sigma_points,
        state_bar,
        covariance_bar,
        return_kalman=False,
    ):

        # measurement_sigma = np.vstack([(self.sensor.measurement_map(sigma_point) + sigma_point.noise[:,np.newaxis]).T for sigma_point in sigma_points])

        measurement_sigma = (
            np.hstack([self.sensor.measurement_map(state) for state in sigma_points])
            + sigma_points.noise
        )
        # TODO Figure out a way to return new sigma point object representing measurements
        measurement_mean, measurement_covariance = reconstruct_sigma_points(
            measurement_sigma, weights=self.weights
        )

        innovation = measurement - measurement_mean

        # cross-covariance
        state_difference = sigma_points.state - state_bar
        measurement_difference = measurement_sigma - measurement_mean

        cross_covariance = np.einsum(
            "ij,j,kj->ik",
            state_difference,
            self.weights_covariance,
            measurement_difference,
        )

        # kalman_gain = np.linalg.solve(measurement_covariance, cross_covariance.T).T
        kalman_gain = np.linalg.solve(measurement_covariance.T, cross_covariance.T).T

        state_estimate = state_bar + kalman_gain @ (measurement - measurement_mean)
        covariance_estimate = (
            covariance_bar
            - cross_covariance @ kalman_gain.T
            - kalman_gain @ cross_covariance.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )

        self.residuals_data[0].append(innovation[0, 0])
        self.residuals_data[1].append(innovation[1, 0])

        self.residual_cov_data[0].append(3 * np.sqrt(measurement_covariance[0, 0]))
        self.residual_cov_data[1].append(-3 * np.sqrt(measurement_covariance[0, 0]))

        self.residual_cov_data[2].append(3 * np.sqrt(measurement_covariance[1, 1]))
        self.residual_cov_data[3].append(-3 * np.sqrt(measurement_covariance[1, 1]))

        self.state_estimate_hist.append(state_estimate)
        self.covariance_estimate_hist.append(covariance_estimate)
        # TODO refactor to handle the memory better. Try saving these values to the filter state
        if return_kalman:
            return (
                state_estimate,
                covariance_estimate,
                measurement_mean,
                measurement_covariance,
                kalman_gain,
            )
        return (
            state_estimate,
            covariance_estimate,
            measurement_mean,
            measurement_covariance,
        )

    def plot_state_res(self, path, truths):

        state_estimates = np.hstack(self.state_estimate_hist)

        covariance_estimates = np.array(self.covariance_estimate_hist)

        state_res = (truths - state_estimates).T

        pos_labels = ["X [KM]", "Y [KM]", "Z [KM]"]
        vel_labels = ["Vx [KM/S^2]", "Vy [KM/S^2]", "Vz [KM/S^2]"]
        # pos_lim = [[],[],[]]
        # vel_lim = [[-0.005,0.005],[-0.005,0.005],[]]
        # Pos--------------------------------------------------
        fig_pos, axs_pos = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):
            # Plot the residual for position variable i
            axs_pos[i].plot(
                state_res[:, i],
                marker="x",
                linestyle="none",
                label=f"{pos_labels[i]} residual".replace("[KM]", ""),
            )

            sigma = 3 * np.sqrt(covariance_estimates[:, i, i])

            axs_pos[i].plot(sigma, color="red", label=r"$\pm3\sigma$")

            axs_pos[i].plot(-sigma, color="red")

            axs_pos[i].set_ylabel(pos_labels[i])
            axs_pos[i].grid(True)
            axs_pos[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs_pos[i].legend(fontsize=14)
            # axs_pos[i].set_ylim([-2.5, 2.5])
            axs_pos[i].set_title(
                f"RMS:{np.sqrt(np.mean(np.square(state_res[:, i]))):.4e} " + "[KM]"
            )

        axs_pos[-1].set_xlabel("Time in Steps of 540 Seconds")
        fig_pos.tight_layout()
        fig_pos.savefig(path.replace(".png", "_position.png"))
        plt.show()

        # Velo-------------------------------------------------------------
        fig_vel, axs_vel = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i in range(3):

            axs_vel[i].plot(
                state_res[:, i + 3],
                marker="x",
                linestyle="none",
                label=f"{vel_labels[i]} residual".replace("[KM/S^2]", ""),
            )

            sigma = 3 * np.sqrt(covariance_estimates[:, i + 3, i + 3])
            axs_vel[i].plot(sigma, color="red")
            axs_vel[i].plot(-sigma, color="red", label=r"$\pm3\sigma$")

            axs_vel[i].set_ylabel(vel_labels[i])
            # axs_vel[i].set_ylim([-0.005, 0.005])
            axs_vel[i].grid(True)
            axs_vel[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axs_vel[i].legend(fontsize=14)
            axs_vel[i].set_title(
                f"RMS:{np.sqrt(np.mean(np.square(state_res[:, i+3]))):.4e} "
                + "[KM/S^2]"
            )

        axs_vel[-1].set_xlabel("Time in Steps of 540 Seconds ")

        fig_vel.tight_layout()
        fig_vel.savefig(path.replace(".png", "_velocity.png"))
        plt.show()

    def plot_res(self, path):

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        ax1.plot(
            self.residuals_data[0], marker="x", linestyle="none", label=r"z - $\bar{z}$"
        )
        ax1.plot(self.residual_cov_data[0], color="red", label=r"$\pm 3\sigma$")
        ax1.plot(self.residual_cov_data[1], color="red")
        # ax1.set_ylim([-1.5e-3, 1.5e-3])
        ax1.set_title(
            r"$\alpha$  "
            + f"RMS:{np.sqrt(np.mean(np.square(self.residuals_data[0]))):.4e} "
            + "[rad]"
        )
        ax1.set_ylabel("RA [rad]")
        ax1.grid()

        ax1.legend(fontsize=14)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax2.plot(self.residuals_data[1], marker="x", linestyle="none")
        ax2.plot(self.residual_cov_data[2], color="red", label=r"$\pm 3\sigma$")
        ax2.plot(self.residual_cov_data[3], color="red")
        # ax2.set_ylim([-5e-4, 5e-4])
        ax2.set_title(
            r"$\delta$  "
            + f"RMS:{np.sqrt(np.mean(np.square(self.residuals_data[1]))):.4e} "
            + "[rad]"
        )
        ax2.set_xlabel("Time Steps in Increments of 540 Seconds")
        ax2.set_ylabel("DEC [rad]")
        ax2.grid()
        ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(path)

        plt.show()


class NoiseDynamic(Dynamic):

    def __init__(self, scenario, agent=None, stm=None, function=None):
        if function is None:
            function = self.noise_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)

    @staticmethod
    def noise_function(state: State, time):
        return State(acceleration=state.process_noise)
