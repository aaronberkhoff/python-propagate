import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Agents import Agent


def keplerian_motion(state:np.array,time:np.array, scenario: Scenario, agent:Agent = None): 

    rx  = state[0]
    ry  = state[1]
    rz  = state[2]

    r = np.sqrt(rx**2 + ry**2 + rz**2)

    ax = -scenario.central_body.mu * rx /r**3
    ay = -scenario.central_body.mu * ry /r**3
    az = -scenario.central_body.mu * rz /r**3

    return ax, ay, az
