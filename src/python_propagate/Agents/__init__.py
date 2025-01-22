
from datetime import datetime, timedelta
from astropy import units as u
import numpy as np
import scipy.integrate as sci_int

from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Result
from python_propagate.Dynamics import Dynamic
from python_propagate.Dynamics.keplerian import Keplerian
from python_propagate.Dynamics.J2 import J2
from python_propagate.Dynamics.J3 import J3
from python_propagate.Dynamics.drag import Drag



class State:
    def __init__(self, position, velocity, stm = np.eye(6), frame="inertial"):
        """
        Initialize the state of an object with explicit units.

        :param position: Position vector (e.g., [x, y, z] in kilometers or km)
        :param velocity: Velocity vector (e.g., [vx, vy, vz] in m/s or km/s)
        :param frame: Reference frame (e.g., "inertial", "body")
        """
        self.position = np.array(position* u.km)   # Assume kilometers by default
        self.velocity = np.array(velocity * (u.km / u.s))  # Assume kilometers per second
        self.stm      = stm
        self.frame = frame

    def __call__(self,stm = None):

        if stm is not None:
            return np.hstack((self.position,self.velocity,self.stm.flatten()))
        else:
            return np.hstack((self.position,self.velocity))



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

            elif isinstance(dynamic,Dynamic):
                self.dynamics.append(dynamic)
            else:
                raise NotImplementedError(f'Dynamic <{dynamic}> is not an option or is spelled wrong')

    def set_scenario(self,scenario:Scenario):

        self.scenario = scenario

    def propagator(self,time,state):


        result = Result()

        for dynamic in self.dynamics:
            # a_x,a_y,a_z = dynamic(state,time,self.scenario,self)
            result += dynamic(state,time)
            
        return np.hstack((state[3:6],result.compile()))
    
    def propagate(self,tolerance = 1e-12):

        time = [0,self.duration.total_seconds()]
        method = 'RK45'

        rtol = tolerance
        atol = tolerance
        t_eval = np.arange(time[0],time[1]+self.dt.seconds,self.dt.seconds)


        ode_state = sci_int.solve_ivp(self.propagator,time,self.state(),method = method,rtol = rtol,atol = atol,t_eval = t_eval)

        self.update_state(new_state=ode_state.y[:,-1])
        self.save_state_data(ode_state=ode_state)

    def update_state(self,new_state):

        self.state.position = new_state[0:3]
        self.state.velocity = new_state[3:6]


        pass

    def save_state_data(self,ode_state):
        self.state_data = ode_state.y
        self.time_data  = ode_state.t




