import numpy as np
from python_propagate.Scenario import Scenario

class Result:

    def __init__(self,accleration = np.array([0,0,0]),stm = None):
        self.acceleration = accleration
        self.stm = stm

    def __add__(self,result):

        acceleration = self.acceleration + result.acceleration

        if self.stm is not None:
            stm = self.stm

        elif result.stm is not None:
            stm = result.stm

        else:
            stm = None

        return Result(accleration=acceleration,stm=stm)

    def compile(self):

        if self.stm is not None:
        
            return np.hstack((self.acceleration,self.stm))
        
        else:

            return self.acceleration
    

class Dynamic:

    def __init__(self, scenario: Scenario, agent = None,stm = None):
        self.stm = stm
        self.scenario = scenario
        self.agent = agent
        pass

    def __call__(self,state:np.array,time:np.array):

        if self.stm is not None:

            return Result(accleration=self.function(state,time),stm = self.stm(state,time))
        
        else: 
            return Result(accleration=self.function(state,time))
    
    def set_function(self,function):

        self.function = function




