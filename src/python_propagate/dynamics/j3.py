import numpy as np
from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.states import State


class J3(Dynamic):

    def __init__(self, scenario, agent=None, stm=None, function=None):
        if function is None:
            function = self.j3_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)

    def j3_function(self, state: State, time: float):

        rx, ry, rz = state.extract_position()

        # Compute common terms
        r2 = rx**2 + ry**2 + rz**2  # Square of the radial distance
        r = np.sqrt(r2)  # Radial distance
        r7 = r**7  # r^7
        R3 = self.scenario.central_body.radius**3  # Earth's radius cubed

        # Central body's parameters
        J3 = self.scenario.central_body.j3  # J3 coefficient
        mu = self.scenario.central_body.mu  # Gravitational parameter

        alpha = -5 * J3 * mu * R3 / (2 * r7)
        beta = 3 * rz - 7 * rz**3 / r2
        gamma = 6 * rz**2 - 7 * rz**4 / r2 - 3 / 5 * r2

        # Compute accelerations using Vallado's formulation
        ax = alpha * rx * beta
        ay = alpha * ry * beta
        az = alpha * gamma

        return State(acceleration=np.array([ax, ay, az]))
