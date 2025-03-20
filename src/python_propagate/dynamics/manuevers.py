from datetime import timedelta
import numpy as np
from python_propagate.dynamics import Dynamic
from python_propagate.states import State
from python_propagate.utilities.transforms import inertial_to_ric


class ImpulseManuever(Dynamic):

    def __init__(self,
                 scenario = None,
                 execution_time:float = 30,
                 magnitude: float = 1.0,
                 direction_ric: np.ndarray = np.array([0.0,1.0,0.0]),
                 agent=None,
                 stm=None,
                 function=None):
        
        # magnitude *= 1e-3  # convert from km/s to m/s
        if isinstance(execution_time, dict):
            self.execution_time = timedelta(**execution_time).total_seconds()
        else:
            self.execution_time = execution_time

        if isinstance(direction_ric, list):
            self.direction_ric = np.array(direction_ric)
    
        self.magnitude = magnitude 
        self.execute_bool = True

        if function is None:
            function = self.manuever_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)


    def manuever_function(self,state:State,time:float):
        
        if time > self.execution_time and self.execute_bool:
            print(f'Manuever at time = {time}')
            self.execute_bool = False
            transform = inertial_to_ric(state=state.compile())
            velo_inertial = transform.T @ (self.magnitude * self.direction_ric)
        else:
            velo_inertial = np.array([0.0,0.0,0.0])

        return State(velocity=velo_inertial, time=time)
    
            
class ThrustManuever(Dynamic):

    def __init__(self,
                 scenario = None,
                 execution_time:float = 30,
                 execution_duration: float = 120,
                 magnitude: float = 1.0,
                 direction_ric: np.ndarray = np.array([0.0,1.0,0.0]),
                 agent=None,
                 stm=None,
                 function=None):
        
        # magnitude *= 1e-3  # convert from km/s to m/s
        if isinstance(execution_time, dict):
            self.execution_time = timedelta(**execution_time).total_seconds()
        else:
            self.execution_time = execution_time

        if isinstance(execution_time, dict):
            self.execution_duration = timedelta(**execution_duration).total_seconds()
        else:
            self.execution_duration = execution_duration

        if isinstance(direction_ric, list):
            self.direction_ric = np.array(direction_ric)
    
        self.magnitude = magnitude 
        # self.execute_bool = True

        if function is None:
            function = self.manuever_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)


    def manuever_function(self,state:State,time:float):
        
        if time > self.execution_time and time < (self.execution_time+self.execution_duration):
            # print(f'Thrust Manuever at time = {time}')
            self.execute_bool = False
            transform = inertial_to_ric(state=state.compile())
            acc_inertial = transform.T @ (self.magnitude * self.direction_ric)
        else:
            acc_inertial = np.array([0.0,0.0,0.0])

        return State(acceleration=acc_inertial, time=time)
            


        
