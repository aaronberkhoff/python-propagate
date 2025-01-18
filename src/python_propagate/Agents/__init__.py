
from datetime import datetime, timedelta
from astropy import units as u
import numpy as np
import scipy.integrate as sci_int

from python_propagate.Scenario import Scenario

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

    def __call__(self,stm = False):

        if stm:
            return np.hstack((self.position,self.velocity,self.stm.flatten()))
        else:
            return np.hstack((self.position,self.velocity))



class Agent:

    def __init__(self, state:State, start_time:datetime, duration:timedelta, dt: timedelta):
        self.state = state
        self._start_time = start_time
        self._duration = duration
        self._dt       = dt
        self.scenario = None
        self.state_data = None
        self.time_data = None
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
    
    def add_dynamics(self,dynamics: tuple):
        for dynamic in dynamics:
            self.dynamics.append(dynamic)

    def set_scenario(self,scenario:Scenario):

        self.scenario = scenario

    def propagator(self,time,state):
 
        vx  = state[3]
        vy  = state[4]
        vz  = state[5]

        ax = 0
        ay = 0
        az = 0

        for dynamic in self.dynamics:
            a_x,a_y,a_z = dynamic(state,time,self.scenario)

            ax += a_x
            ay += a_y
            az += a_z

        return np.array([vx,vy,vz,ax,ay,az])
    
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




