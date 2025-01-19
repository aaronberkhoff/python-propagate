import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Agents import Agent
import numpy as np

def drag_motion(state: np.array, time: float, scenario:Scenario, agent:Agent):

    rx = state[0]
    ry = state[1]
    rz = state[2]
    
    vx = state[3]
    vy = state[4]
    vz = state[5]

    r = np.sqrt(rx**2 + ry**2 + rz**2)

    denstity = scenario.central_body.atmosphere_model(r) * 1000**3

    vax = vx + scenario.central_body.angular_velocity * ry
    vay = vy - scenario.central_body.angular_velocity * rx

    va = np.sqrt(vax ** 2 + vay ** 2 + vz**2)

    dynamic_pressure = -.5 * agent.coefficet_of_drag * denstity * agent.area / agent.mass

    ax = dynamic_pressure * vax * va
    ay = dynamic_pressure * vay * va
    az = dynamic_pressure * vz * va

    return ax, ay, az





