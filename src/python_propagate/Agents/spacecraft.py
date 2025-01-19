from python_propagate.Agents import Agent

class Spacecraft(Agent):

    def __init__(self, state, start_time, duration, dt, coefficent_of_drag = None, mass = None, area = None):
        super().__init__(state, start_time, duration, dt, coefficent_of_drag, mass, area = area)

        
