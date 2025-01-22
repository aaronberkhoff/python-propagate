import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Dynamic

class J2(Dynamic):
    def __init__(self, scenario: Scenario, agent= None,stm = None):
        super().__init__(scenario, agent, stm)

    def function(self,state: np.array, time: np.array):
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
        R2 = self.scenario.central_body.radius**2  # Earth's radius squared

        # Central body's parameters
        J2 = self.scenario.central_body.J2       # J2 coefficient
        mu = self.scenario.central_body.mu       # Gravitational parameter

        alpha = -3 * J2 * mu * R2
        beta = (1 - 5 * rz**2 / r2)
        gamma = (2 * r5)
        
        # Compute accelerations using Vallado's formulation
        ax = alpha * rx / gamma * beta
        ay = alpha * ry / gamma * beta
        az = alpha * rz / gamma * (3 - 5 * rz**2 / r2)

        return np.array([ax, ay, az])


  