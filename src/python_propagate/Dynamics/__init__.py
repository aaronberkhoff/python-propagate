"""
Dynamic module.

This module contains the Dynamic class.

Classes:
- Dynamic: A class to represent a dynamic.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

# TODO: Need to fix how dynamics are done, look at the pylinrc for reference
import numpy as np

from python_propagate.Scenario import Scenario
from python_propagate.Agents.state import State


class Dynamic:
    """
    A class to represent a dynamic.

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
        """
        Constructs all the necessary attributes for the Dynamic object.

        Parameters
        ----------
        scenario : Scenario
            The scenario of the dynamic.
        agent : Agent
            The agent of the dynamic.
        stm : STM
            The state transition matrix of the dynamic.
        """
        self.stm = stm
        self.scenario = scenario
        self.agent = agent
        self.function = None

    def __call__(self, state: State, time: np.array):
        """
        Calls the function of the dynamic.

        Parameters
        ----------
        state : State
            The state of the dynamic.
        time : np.array
            The time of the dynamic.

        Returns
        -------
        np.array
            The result of the function.
        """

        return self.function(state, time)
