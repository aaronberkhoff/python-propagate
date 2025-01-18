from python_propagate.Agents import Agent

class Spacecraft(Agent):

    def __init__(self, state, start_time, duration, dt):
        super().__init__(state, start_time, duration, dt)