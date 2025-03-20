import numpy as np
import spiceypy as spice
from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.states import State
from python_propagate.utilities.load_spice import load_spice

class ThreeBody(Dynamic):
    """
    A class to represent a Keplerian dynamic.

    Attributes
    ----------
    scenario : Scenario
        The scenario of the dynamic.
    agent : Agent
        The agent of the dynamic.
    stm : STM
        The state transition matrix of the dynamic.
    function : function
        The function of the dynamic.
    """

    def __init__(self, scenario, agent=None, stm=None, function=None):
        load_spice()
        if function is None:
            function = self.three_body_all  # Assign the default function

        super().__init__(scenario, agent, stm, function)



    def three_body(self, state, celestial_body):

        # Gravitational parameter (GM) in km^3/sec^2
        mu = celestial_body.mu

        # Get ephemeris time
        current_time = state.time
        et = spice.str2et(current_time.strftime('%Y-%m-%dT%H:%M:%S'))

        # Get the position and velocity of the third body relative to inertial body
        third_body_state, _ = spice.spkezr(celestial_body.name.upper(), et, 'J2000', 'NONE', self.scenario.central_body.name.upper())

        # Extract the position (first 3 elements of the state vector)
        inertial_to_third_body = third_body_state[:3]  # 3rd body position in km (X, Y, Z)

        # Agent position relative to inertial body
        inertial_to_agent = state.position_eci # Agent position in km (X, Y, Z)

        # Agent position relative to third body
        agent_to_third_body = inertial_to_third_body - inertial_to_agent

        # Third body acceleration
        accel_third_body = mu * (agent_to_third_body/(np.linalg.norm(agent_to_third_body)**3) - inertial_to_third_body/(np.linalg.norm(inertial_to_third_body)**3))

        return State(acceleration=accel_third_body, time=current_time)




       

    def three_body_all(self, state: State, time: float = None):
        """
        The function of the Keplerian dynamic.

        Parameters
        ----------
        state : State
            The state of the dynamic.
        time : float
            The time of the dynamic.

        Returns
        -------
        State
            The result of the function.
        """

        accel = State(acceleration=np.zeros(3))
        for body in self.scenario.celestial_bodies:
            accel @= self.three_body(state, body)
            
        return accel