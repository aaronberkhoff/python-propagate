import numpy as np
import spiceypy as spice
from python_propagate.Utilities.load_spice import load_spice
from python_propagate.Utilities.transforms import cart2classical


TARGET = 'EARTH'
ECI = 'J2000'
ECEF = 'ITRF93'



class State:
    def __init__(self, position = None, velocity = None, acceleration = None,
                stm = None, stm_dot = None,
                time = None, dimension = 6, frame="inertial"):

        self.position = position   # Assume kilometers by default
        self.velocity = velocity  # Assume kilometers per second
        self.acceleration = acceleration
        self.frame = frame
        self.dimension = dimension
        self.stm = stm 
        self.time = time
        self.stm_dot = stm_dot
    

    @property
    def position_ECI(self):

        if self.frame == 'inertial':
            return self.position
        elif self.frame == 'ECEF':
            et = spice.str2et(self.time.strftime('%Y-%m-%dT%H:%M:%S'))
            rotation_matix = spice.pxform(ECEF,ECI,et)

            return rotation_matix @ self.position
        
    @property
    def position_ECEF(self):

        if self.frame == 'ECEF':
            return self.position
        elif self.frame == 'inertial':
            et = spice.str2et(self.time.strftime('%Y-%m-%dT%H:%M:%S'))
            rotation_matix = spice.pxform(ECI,ECEF,et)

            return rotation_matix @ self.position
        
    @property
    def velocity_ECI(self):

        if self.frame == 'inertial':
            return self.velocity
        elif self.frame == 'ECEF':
            et = spice.str2et(self.time.strftime('%Y-%m-%dT%H:%M:%S'))
            rotation_matix = spice.pxform(ECEF,ECI,et)

            return rotation_matix @ self.velocity
        
    @property
    def velocity_ECEF(self):

        if self.frame == 'ECEF':
            return self.velocity
        elif self.frame == 'inertial':
            et = spice.str2et(self.time.strftime('%Y-%m-%dT%H:%M:%S'))
            rotation_matix = spice.pxform(ECI,ECEF,et)
            return rotation_matix @ self.velocity
        else:
            raise ValueError(f'Frame <{self.frame}> is spelled wrong or is not supported')

 

    #handle the positions
    @property
    def radius(self):
        return np.sqrt(self.position[0]**2 + self.position[1]**2 + self.position[2]**2)
    @property
    def x_ECI(self):
            return self.position_ECI[0]
    @property
    def y_ECI(self):
        return self.position_ECI[1]
    @property
    def z_ECI(self):
        return self.position_ECI[2]
    @property
    def x_ECEF(self):
            return self.position_ECEF[0]
    @property
    def y_ECEF(self):
        return self.position_ECEF[1]
    @property
    def z_ECEF(self):
        return self.position_ECEF[2]
    
    #handle the velocities
    @property
    def vx_ECI(self):
            return self.velocity_ECI[0]
    @property
    def vy_ECI(self):
        return self.velocity_ECI[1]
    @property
    def vz_ECI(self):
        return self.velocity_ECI[2]
    @property
    def vx_ECEF(self):
            return self.velocity_ECEF[0]
    @property
    def vy_ECEF(self):
        return self.velocity_ECEF[1]
    @property
    def vz_ECEF(self):
        return self.velocity_ECEF[2]
    
    @property
    def latitude(self):
        return np.arctan2(self.z_ECEF,np.sqrt(self.x_ECEF**2 + self.y_ECEF**2))
    @property
    def longitude(self):
        return np.arctan2(self.y_ECEF,self.x_ECEF)
    
    #TODO: ADD frame conversions as properties

    def compile(self):

        if self.stm is not None:
            return np.hstack((self.position,self.velocity,self.stm.flatten()))
        else:
            return np.hstack((self.position,self.velocity))
        
    def to_keplerian(self,mu):

        sma,ecc,inc,raan,arg,nu = cart2classical(self.compile(),mu)

        return sma,ecc,inc,raan,arg,nu

        
    def dot(self):
        if self.stm is not None:
            return np.hstack((self.velocity,self.acceleration,self.stm_dot.flatten()))
        else:
            return np.hstack((self.velocity,self.acceleration))

        
    def extract_position(self):

        return self.position[0],self.position[1],self.position[2]
    
    def extract_velocity(self):

        return self.velocity[0],self.velocity[1],self.velocity[2]
        
    def add_acceleration_from_state(self,state):
        pass

    def update_acceleration_from_state(self,state):

        if state.acceleration is not None:
            self.acceleration = self.acceleration + state.acceleration

        if state.stm_dot is not None:
            self.stm_dot = state.stm_dot


    