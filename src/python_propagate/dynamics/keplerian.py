import numpy as np

from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.agents.state import State


class Keplerian(Dynamic):
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

    def __init__(self, scenario: Scenario, agent=None, stm=None):
        super().__init__(scenario, agent, stm)
        """
        Constructs all the necessary attributes for the Keplerian object.

        Parameters
        ----------
        scenario : Scenario
            The scenario of the dynamic.
        agent : Agent
            The agent of the dynamic.
        stm : STM
            The state transition matrix of the dynamic.

        """

    def function(self, state: State, time: float):
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
        rx, ry, rz = state.extract_position()

        r = np.sqrt(rx**2 + ry**2 + rz**2)

        ax = -self.scenario.central_body.mu * rx / r**3
        ay = -self.scenario.central_body.mu * ry / r**3
        az = -self.scenario.central_body.mu * rz / r**3

        return State(acceleration=np.array([ax, ay, az]), time=time)
