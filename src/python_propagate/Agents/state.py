import numpy as np

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
        self.time = None
        
    @property
    def radius(self):
        return np.sqrt(self.position[0]**2 + self.position[1]**2 + self.position[2]**2)
    @property
    def x(self):
        return self.position[0]
    @property
    def y(self):
        return self.position[1]
    @property
    def z(self):
        return self.position[2]
    @property
    def latitude(self):
        return np.arctan2(self.z,np.sqrt(self.x**2 + self.y**2))
    @property
    def longitude(self):
        return np.arctan2(self.y,self.x)
    
    #TODO: ADD frame conversions as properties

    def compile(self):

        if self.stm is not None:
            return np.hstack((self.position,self.velocity,self.stm.flatten()))
        else:
            return np.hstack((self.position,self.velocity))
        
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


    