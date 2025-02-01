import numpy as np
from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.agents.state import State


class J2(Dynamic):
    """
    A class to represent a J2 dynamic.

    Attributes

    scenario : Scenario
        The scenario of the dynamic.
    agent : Agent
        The agent of the dynamic.
    stm : STM
        The state transition matrix of the dynamic.
    """
    def __init__(self, scenario: Scenario, agent=None, stm=None):
        super().__init__(scenario, agent, stm)

    def function(self, state: State, time: float):

        rx, ry, rz = state.extract_position()

        # Compute common terms
        r2 = rx**2 + ry**2 + rz**2  # Square of the radial distance
        r = np.sqrt(r2)  # Radial distance
        r5 = r**5
        R2 = self.scenario.central_body.radius**2  # Earth's radius squared

        # Central body's parameters
        J2 = self.scenario.central_body.j2  # J2 coefficient
        mu = self.scenario.central_body.mu  # Gravitational parameter

        alpha = -3 * J2 * mu * R2
        beta = 1 - 5 * rz**2 / r2
        gamma = 2 * r5

        # Compute accelerations using Vallado's formulation
        ax = alpha * rx / gamma * beta
        ay = alpha * ry / gamma * beta
        az = alpha * rz / gamma * (3 - 5 * rz**2 / r2)

        return State(acceleration=np.array([ax, ay, az]))
