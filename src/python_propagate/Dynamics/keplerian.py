import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Dynamic
from python_propagate.Agents.state import State

class Keplerian(Dynamic):

    def __init__(self, scenario: Scenario, agent = None,stm = None):
        super().__init__(scenario, agent, stm)

    def function(self,state: State, time: float):

        rx, ry, rz = state.extract_position()

        r = np.sqrt(rx**2 + ry**2 + rz**2)

        ax = -self.scenario.central_body.mu * rx /r**3
        ay = -self.scenario.central_body.mu * ry /r**3
        az = -self.scenario.central_body.mu * rz /r**3

        return State(acceleration=np.array([ax, ay, az]))
