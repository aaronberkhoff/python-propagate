import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Agents import Agent


import numpy as np

def J2_motion(state: np.array, time: np.array, scenario:Scenario, agent:Agent = None):
    """
    Compute the acceleration due to the second zonal harmonic J2.
    
    Parameters:
        state (np.array): Position vector [rx, ry, rz].
        time (np.array): Time (not used in this function but can be relevant for future modifications).
        scenario: Scenario object containing central body properties (radius, J2, and gravitational parameter).
        
    Returns:
        np.array: Accelerations [ax, ay, az] due to J2.
    """
    # Extract position components
    rx = state[0]
    ry = state[1]
    rz = state[2]

    # Compute common terms
    r2 = rx**2 + ry**2 + rz**2    # Square of the radial distance
    r = np.sqrt(r2)               # Radial distance
    r5 = r**5
    R2 = scenario.central_body.radius**2  # Earth's radius squared

    # Central body's parameters
    J2 = scenario.central_body.J2       # J2 coefficient
    mu = scenario.central_body.mu       # Gravitational parameter

    alpha = -3 * J2 * mu * R2
    beta = (1 - 5 * rz**2 / r2)
    gamma = (2 * r5)
    
    # Compute accelerations using Vallado's formulation
    ax = alpha * rx / gamma * beta
    ay = alpha * ry / gamma * beta
    az = alpha * rz / gamma * (3 - 5 * rz**2 / r2)

    return ax,ay,az


# def J2_motion(state:np.array,time:np.array, scenario: Scenario): 

#     r_e = scenario.central_body.radius
#     mu = scenario.central_body.mu
#     J_2 = scenario.central_body.J2

#     rx  = state[0]
#     ry  = state[1]
#     rz  = state[2]
#     r = np.sqrt(rx**2 + ry**2 + rz**2)

        
#     t2 = r_e**2
#     t3 = rx**2
#     t4 = ry**2
#     t5 = rz**2
#     t6 = t3+t4+t5
#     t7 = 1.0/t6
#     t10 = 1.0/np.sqrt(t6)
#     t8 = t7**2
#     t9 = t7**3
#     t11 = t10**3
#     t12 = t5*t7*(3.0/2.0)
#     t13 = t12-1.0/2.0
#     t14 = J_2*t2*t7*t13
#     t15 = t14-1.0


#     ax = mu*t10*(J_2*rx*t2*t5*t9*3.0+J_2*rx*t2*t8*t13*2.0)+mu*rx*t11*t15
#     ay =  mu*t10*(J_2*ry*t2*t5*t9*3.0+J_2*ry*t2*t8*t13*2.0)+mu*ry*t11*t15
#     az = -mu*t10*(J_2*t2*t7*(rz*t7*3.0-rz**3*t8*3.0)-J_2*rz*t2*t8*t13*2.0)+mu*rz*t11*t15

    

#     return ax, ay, az