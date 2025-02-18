import numpy as np

from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.states import State


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

    def __init__(self, scenario, agent=None, stm=None, function=None):
        if function is None:
            function = self.kep_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)

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

    def kep_function(self, state: State, time: float = None):
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
