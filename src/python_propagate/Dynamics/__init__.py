import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Agents.state import State


class Dynamic:

    def __init__(self, scenario: Scenario, agent = None,stm = None):
        self.stm = stm
        self.scenario = scenario
        self.agent = agent
        pass

    def __call__(self,state:State,time:np.array):

        return self.function(state,time)
    
    def set_function(self,function):

        self.function = function




