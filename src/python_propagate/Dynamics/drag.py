import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Dynamic
from python_propagate.Agents.state import State


class Drag(Dynamic):
    def __init__(self, scenario: Scenario, agent = None,stm = None):
        super().__init__(scenario, agent, stm)
        
    def function(self,state: State, time: float):

        rx, ry, rz = state.extract_position()
        vx, vy, vz = state.extract_velocity()

        r = np.sqrt(rx**2 + ry**2 + rz**2)

        density = self.scenario.central_body.atmosphere_model(r) 

        vax = vx + self.scenario.central_body.angular_velocity * ry
        vay = vy - self.scenario.central_body.angular_velocity * rx

        va = np.sqrt(vax ** 2 + vay ** 2 + vz**2)

        dynamic_pressure = -.5 * self.agent.coefficet_of_drag * density * self.agent.area / self.agent.mass

        ax = dynamic_pressure * vax * va
        ay = dynamic_pressure * vay * va
        az = dynamic_pressure * vz * va

        return State(acceleration=np.array([ax, ay, az]))
        





