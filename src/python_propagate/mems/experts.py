from python_propagate.agents import Agent

class Expert(Agent):

    def __init__(self, state, start_time, duration, dt, coefficient_of_drag=None, mass=None, area=None, name="Agent", dynamics=[], scenario=None, manuevers=[], bus=None):
        super().__init__(state, start_time, duration, dt, coefficient_of_drag, mass, area, name, dynamics, scenario, manuevers, bus)

        self.probability_data = None