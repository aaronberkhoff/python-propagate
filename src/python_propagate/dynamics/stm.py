import numpy as np
from numpy import sqrt

from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.agents.state import State

# TODO: explore specifying the difference between the classes for normal dynamics and STMs


class STM(Dynamic):

    def __init__(self, scenario: Scenario, agent=None, stm=True):
        super().__init__(scenario, agent, stm)

    def function(self, state: State, time: float):

        stm = state.stm
        A_matrix = self.a_matrix(state)

        stm_dot = A_matrix @ stm

        return State(stm_dot=stm_dot, time=time)

    def a_matrix(self, state: State):

        rx, ry, rz = state.extract_position()
        vx, vy, vz = state.extract_velocity()

        radius = np.sqrt(rx**2 + ry**2 + rz**2)

        radius_body = self.scenario.central_body.radius
        mu = self.scenario.central_body.mu
        j2 = self.scenario.central_body.j2
        j3 = self.scenario.central_body.j3

        rho0, h0, scale_height = self.scenario.central_body.atmosphere_model(radius)
        cd = self.agent.coefficet_of_drag
        area = self.agent.area
        mass = self.agent.mass
        angular_velocity = self.scenario.central_body.angular_velocity

        # automatically generated by sympy
        a_matrix_total = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * (-rx * angular_velocity + vy)
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * rx
                    * (ry * angular_velocity + vx)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rx**2
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 3
                    * j2
                    * radius_body**2
                    * mu
                    * rx**2
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    + (3 / 2)
                    * j2
                    * radius_body**2
                    * mu
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rx**2
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    - 15
                    * j3
                    * radius_body**3
                    * mu
                    * rx**2
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rx**2 / (rx**2 + ry**2 + rz**2) ** (5 / 2)
                    - mu / (rx**2 + ry**2 + rz**2) ** (3 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * (ry * angular_velocity + vx) ** 2
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    - 500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / mass
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * ry
                    * (ry * angular_velocity + vx)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * ry
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 3
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * ry
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * ry
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    - 15
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * ry
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rx * ry / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * rz
                    * (ry * angular_velocity + vx)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * rz
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 12
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * rz**2
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    + 20
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * rz**2
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rx * rz / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * (ry * angular_velocity + vx) ** 2
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    - 500000000.0
                    * area
                    * cd
                    * rho0
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / mass,
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * (-rx * angular_velocity + vy)
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                ],
                [
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * (-rx * angular_velocity + vy) ** 2
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / mass
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * rx
                    * (-rx * angular_velocity + vy)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * ry
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 3
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * ry
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * ry
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    - 15
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * ry
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rx * ry / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * angular_velocity
                    * (-rx * angular_velocity + vy)
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * ry
                    * (-rx * angular_velocity + vy)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * ry**2
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 3
                    * j2
                    * radius_body**2
                    * mu
                    * ry**2
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    + (3 / 2)
                    * j2
                    * radius_body**2
                    * mu
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * ry**2
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    - 15
                    * j3
                    * radius_body**3
                    * mu
                    * ry**2
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * ry**2 / (rx**2 + ry**2 + rz**2) ** (5 / 2)
                    - mu / (rx**2 + ry**2 + rz**2) ** (3 / 2),
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * rz
                    * (-rx * angular_velocity + vy)
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * ry
                    * rz
                    * (-(rx**2) - ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 12
                    * j2
                    * radius_body**2
                    * mu
                    * ry
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * ry
                    * rz**2
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    + 20
                    * j3
                    * radius_body**3
                    * mu
                    * ry
                    * rz**2
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * ry
                    * (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * ry * rz / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * (-rx * angular_velocity + vy)
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * (-rx * angular_velocity + vy) ** 2
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    - 500000000.0
                    * area
                    * cd
                    * rho0
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / mass,
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * (-rx * angular_velocity + vy)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                ],
                [
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * angular_velocity
                    * (-rx * angular_velocity + vy)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * rx
                    * vz
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 2 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 9
                    * j2
                    * radius_body**2
                    * mu
                    * rx
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rx
                    * (
                        7 * rz**4
                        + (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * (
                        2 * rx * (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        + 1.2 * rx * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rx * rz / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * angular_velocity
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    + 500000000.0
                    * area
                    * cd
                    * rho0
                    * ry
                    * vz
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * ry
                    * rz
                    * (-3 * rx**2 - 3 * ry**2 + 2 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    - 9
                    * j2
                    * radius_body**2
                    * mu
                    * ry
                    * rz
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * ry
                    * (
                        7 * rz**4
                        + (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * (
                        2 * ry * (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        + 1.2 * ry * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * ry * rz / (rx**2 + ry**2 + rz**2) ** (5 / 2),
                    500000000.0
                    * area
                    * cd
                    * rho0
                    * rz
                    * vz
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (scale_height * mass * sqrt(rx**2 + ry**2 + rz**2))
                    - 21
                    / 2
                    * j2
                    * radius_body**2
                    * mu
                    * rz**2
                    * (-3 * rx**2 - 3 * ry**2 + 2 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 6
                    * j2
                    * radius_body**2
                    * mu
                    * rz**2
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    + (3 / 2)
                    * j2
                    * radius_body**2
                    * mu
                    * (-3 * rx**2 - 3 * ry**2 + 2 * rz**2)
                    / (rx**2 + ry**2 + rz**2) ** (7 / 2)
                    - 45
                    / 2
                    * j3
                    * radius_body**3
                    * mu
                    * rz
                    * (
                        7 * rz**4
                        + (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (11 / 2)
                    + (5 / 2)
                    * j3
                    * radius_body**3
                    * mu
                    * (
                        28 * rz**3
                        + 2 * rz * (0.6 * rx**2 + 0.6 * ry**2 - 5.4 * rz**2)
                        - 10.8 * rz * (rx**2 + ry**2 + rz**2)
                    )
                    / (rx**2 + ry**2 + rz**2) ** (9 / 2)
                    + 3 * mu * rz**2 / (rx**2 + ry**2 + rz**2) ** (5 / 2)
                    - mu / (rx**2 + ry**2 + rz**2) ** (3 / 2),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * (ry * angular_velocity + vx)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz
                    * (-rx * angular_velocity + vy)
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    ),
                    -500000000.0
                    * area
                    * cd
                    * rho0
                    * vz**2
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / (
                        mass
                        * sqrt(
                            vz**2
                            + (-rx * angular_velocity + vy) ** 2
                            + (ry * angular_velocity + vx) ** 2
                        )
                    )
                    - 500000000.0
                    * area
                    * cd
                    * rho0
                    * sqrt(
                        vz**2
                        + (-rx * angular_velocity + vy) ** 2
                        + (ry * angular_velocity + vx) ** 2
                    )
                    * np.exp(
                        (radius_body + h0 - sqrt(rx**2 + ry**2 + rz**2)) / scale_height
                    )
                    / mass,
                ],
            ]
        )

        return a_matrix_total
