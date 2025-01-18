from python_propagate.Environment.Planets import Planet
from python_propagate.Agents import Agent
from datetime import datetime, timedelta
import numpy as np


class Scenario:
    
    def __init__(self,central_body: Planet, start_time:datetime, duration:timedelta, dt: timedelta ):

        self._central_body = central_body
        self._start_time    = start_time
        self._duration      = duration
        self._dt            = dt
        self.agents         = []

    @property
    def central_body(self):
        return self._central_body
    
    @property
    def start_time(self):
        return self._start_time
    
    @property
    def duration(self):
        return self._duration
    
    @property
    def dt(self):
        return self._dt

    def add_agent(self,agent:Agent):
        agent.set_scenario(self)
        self.agents.append(agent)
        pass

    def add_dynamics(self,dynamics: tuple):
        for agent in self.agents:
            agent.add_dynamics(dynamics)

    def run(self):
        for agent in self.agents:
            agent.propagate()
        pass




    