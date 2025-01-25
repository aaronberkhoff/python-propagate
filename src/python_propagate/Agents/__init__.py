
from datetime import datetime, timedelta
from astropy import units as u
import numpy as np
import scipy.integrate as sci_int

from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Dynamic
from python_propagate.Dynamics.keplerian import Keplerian
from python_propagate.Dynamics.J2 import J2
from python_propagate.Dynamics.J3 import J3
from python_propagate.Dynamics.drag import Drag
from python_propagate.Dynamics.stm import STM
from python_propagate.Agents.state import State

        



class Agent:

    def __init__(self, state:State, start_time:datetime, duration:timedelta, dt: timedelta, coefficent_of_drag = None, mass = None, area = None):
        self.state = state
        self._start_time = start_time
        self._duration = duration
        self._dt       = dt
        self.scenario = None
        self.state_data = None
        self.time_data = None
        self._coefficent_of_drag = coefficent_of_drag
        self._mass = mass
        self._area = area
    
        self.dynamics = []

        pass

    @property
    def start_time(self):
        return self._start_time
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def dt(self):
        return self._dt
    
    @property
    def coefficet_of_drag(self):
        return self._coefficent_of_drag
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def area(self):
        return self._area
    
    def add_dynamics(self,dynamics: tuple):
        for dynamic in dynamics:
            if dynamic == 'kepler':
                self.dynamics.append(Keplerian(scenario=self.scenario,agent=self))

            elif dynamic == 'J2':
                self.dynamics.append(J2(scenario=self.scenario,agent=self))

            elif dynamic == 'J3':
                self.dynamics.append(J3(scenario=self.scenario,agent=self))

            elif dynamic == 'drag':
                self.dynamics.append(Drag(scenario=self.scenario,agent=self))

            elif dynamic == 'stm':
                self.dynamics.append(STM(scenario=self.scenario,agent=self))

            elif isinstance(dynamic,Dynamic):
                self.dynamics.append(dynamic)
            else:
                raise NotImplementedError(f'Dynamic <{dynamic}> is not an option or is spelled wrong')

    def set_scenario(self,scenario:Scenario):

        self.scenario = scenario

    def propagator(self,time,state):

        # result = Result()
        #TODO: Specifying an object in the propagation loop will increase comp time
        #TODO THis sTm config is a quick fix
        #TODO Create own propagators
        state = State(position=state[0:3],velocity=state[3:6],acceleration=np.array([0,0,0]))

        for dynamic in self.dynamics:
            # a_x,a_y,a_z = dynamic(state,time,self.scenario,self)
            state.update_acceleration_from_state(dynamic(state,time))

            
        return state.dot()
    
    def stm_propagator(self,time,state):

        # result = Result()
        #TODO: Specifying an object in the propagation loop will increase comp time
        #TODO THis sTm config is a quick fix
        #TODO Create own propagators
        state = State(position=state[0:3],velocity=state[3:6],acceleration=np.array([0,0,0]),stm = np.reshape(state[6:],(6,6)))

        for dynamic in self.dynamics:
            # a_x,a_y,a_z = dynamic(state,time,self.scenario,self)
            state.update_acceleration_from_state(dynamic(state,time))

            
        return state.dot()
    
    def propagate(self,tolerance = 1e-12):

        #TODO Create own propagators instead of using scipy

        time = [0,self.duration.total_seconds()]
        method = 'RK45'

        rtol = tolerance
        atol = tolerance
        t_eval = np.arange(time[0],time[1]+self.dt.seconds,self.dt.seconds)

        if self.state.stm is not None:
            ode_state = sci_int.solve_ivp(self.stm_propagator,time,self.state.compile(),method = method,rtol = rtol,atol = atol,t_eval = t_eval)
            self.state.position = ode_state.y[0:3,-1]
            self.state.velocity = ode_state.y[3:6,-1]
            self.state.stm = np.reshape(ode_state.y[6:,-1],(6,6))
        else:
            ode_state = sci_int.solve_ivp(self.propagator,time,self.state.compile(),method = method,rtol = rtol,atol = atol,t_eval = t_eval)
            self.state.position = ode_state.y[0:3,-1]
            self.state.velocity = ode_state.y[3:6,-1]
            


    def update_state(self,new_state):

        self.state.position = new_state[0:3]
        self.state.velocity = new_state[3:6]

        self.state.stm = np.reshape(new_state[6:],(6,6))


        pass

    def save_state_data(self,ode_state):
        self.state_data = ode_state.y
        self.time_data  = ode_state.t




