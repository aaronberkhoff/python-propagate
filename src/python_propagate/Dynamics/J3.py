import numpy as np
from python_propagate.Scenario import Scenario


import numpy as np

def J3_motion(state: np.array, time: np.array, scenario) -> np.array:

    # Extract position components
    rx = state[0]
    ry = state[1]
    rz = state[2]

    # Compute common terms
    r2 = rx**2 + ry**2 + rz**2    # Square of the radial distance
    r = np.sqrt(r2)               # Radial distance
    r7 = r**7                # r^7
    R3 = scenario.central_body.radius**3  # Earth's radius cubed

    # Central body's parameters
    J3 = scenario.central_body.J3       # J3 coefficient
    mu = scenario.central_body.mu       # Gravitational parameter

    alpha = -5 * J3 * mu * R3 / (2 * r7)
    beta  =  3 * rz - 7 *rz**3 / r2
    gamma = 6*rz**2 - 7*rz**4 / r2 - 3/5 * r2
    
    # Compute accelerations using Vallado's formulation
    ax = alpha * rx * beta
    ay = alpha * ry * beta 
    az = alpha *  gamma
    

    return ax, ay, az

