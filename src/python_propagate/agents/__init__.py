"""Agent module for the python_propagate package.

This module contains the Agent class.

Author: Aaron Berkhoff
Date: 2025-01-30

"""

from datetime import timedelta

import numpy as np
import scipy.integrate as sci_int

from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.dynamics.keplerian import Keplerian
from python_propagate.dynamics.j2 import J2
from python_propagate.dynamics.j3 import J3
from python_propagate.dynamics.drag import Drag
from python_propagate.dynamics.stm import STM
from python_propagate.agents.state import State


class Agent:
    """
    Base class for agents in the simulation.

    Attributes
    ----------
    state : State
        The current state of the agent.
    state_data : list
        A list to store the state data of the agent.
    start_time : datetime
        The start time of the simulation.
    """

    def __init__(
        self,
        state,
        start_time,
        duration,
        dt,
        coefficent_of_drag=None,
        mass=None,
        area=None,
        name='Agent'
    ):
        """
        Initializes the Agent with the given parameters.

        Parameters
        ----------
        state : State
            The initial state of the agent.
        start_time : datetime
            The start time of the simulation.
        duration : timedelta
            The duration of the simulation.
        dt : timedelta
            The time step of the simulation.
        coefficent_of_drag : float, optional
            The coefficient of drag of the agent (default is None).
        mass : float, optional
            The mass of the agent (default is None).
        area : float, optional
            The area of the agent (default is None).
        """
        self.state = state
        self._start_time = start_time
        self._duration = duration
        self._dt = dt
        self._coefficent_of_drag = coefficent_of_drag
        self._mass = mass
        self._area = area
        self._name = name
        self.state_data = []
        self.time_data = []
        self.scenario = None
        self.dynamics = []
        

    @property
    def start_time(self):
        """Returns the start time of the simulation."""
        return self._start_time

    @property
    def duration(self):
        """Returns the duration of the simulation."""
        return self._duration

    @property
    def dt(self):
        """Returns the time step of the simulation."""
        return self._dt

    @property
    def coefficet_of_drag(self):
        """Returns the coefficient of drag of the agent."""
        return self._coefficent_of_drag

    @property
    def mass(self):
        """Returns the mass of the agent."""
        return self._mass

    @property
    def area(self):
        """Returns the area of the agent."""
        return self._area
    
    @property
    def name(self):
        """Returns the name of the agent."""
        return self._name

    def add_dynamics(self, dynamics: tuple):
        """Adds dynamics to the agent.
        Parameters
        dynamics : tuple
            A tuple of dynamics to be added to the agent.
        """
        for dynamic in dynamics:
            if dynamic == "kepler":
                self.dynamics.append(Keplerian(scenario=self.scenario, agent=self))

            elif dynamic == "J2":
                self.dynamics.append(J2(scenario=self.scenario, agent=self))

            elif dynamic == "J3":
                self.dynamics.append(J3(scenario=self.scenario, agent=self))

            elif dynamic == "drag":
                self.dynamics.append(Drag(scenario=self.scenario, agent=self))

            elif dynamic == "stm":
                self.dynamics.append(STM(scenario=self.scenario, agent=self))

            elif isinstance(dynamic, Dynamic):
                self.dynamics.append(dynamic)
            else:
                raise NotImplementedError(
                    f"Dynamic <{dynamic}> is not an option or is spelled wrong"
                )

    def set_scenario(self, scenario: Scenario):
        """Sets the scenario for the agent.
        Parameters
        scenario : Scenario
            The scenario to be set for the agent.
        """
        #TODO: Explore weakref to avoid circular dependancies
        self.scenario = scenario

    def propagator(self, time, state):
        """
        Propagates the agent's state using numerical integration.

        Parameters
        time : float
            The current time in seconds.
        state : array-like
            The current state vector of the agent.
        Returns
        array-like
            The derivative of the state vector.
        """

        # result = Result()
        # TODO: Specifying an object in the propagation loop will increase comp time
        # TODO THis sTm config is a quick fix
        # TODO Create own propagators
        state = State(
            position=state[0:3], velocity=state[3:6], acceleration=np.array([0, 0, 0])
        )

        for dynamic in self.dynamics:
            # a_x,a_y,a_z = dynamic(state,time,self.scenario,self)
            state.update_acceleration_from_state(dynamic(state, time))

        return state.dot()

    def stm_propagator(self, time, state):
        """
        Propagates the agent's state and stm using numerical integration.

        Parameters
        ----------
        time : float
            The current time in seconds.
        state : array-like
            The current state vector of the agent.
        Returns
        -------
        array-like
            The derivative of the state vector.


        """

        # result = Result()
        # TODO: Specifying an object in the propagation loop will increase comp time
        # TODO THis sTm config is a quick fix
        # TODO Create own propagators
        state = State(
            position=state[0:3],
            velocity=state[3:6],
            acceleration=np.array([0, 0, 0]),
            stm=np.reshape(state[6:], (6, 6)),
        )

        for dynamic in self.dynamics:
            # a_x,a_y,a_z = dynamic(state,time,self.scenario,self)
            state.update_acceleration_from_state(dynamic(state, time))

        return state.dot()

    def propagate(self, tolerance=1e-12):
        """
        Propagates the agent's state using numerical integration.

        Parameters
        ----------
        tolerance : float, optional
            The tolerance for the numerical integration (default is 1e-12).
        """

        # TODO Create own propagators instead of using scipy

        time = [0, self.duration.total_seconds()]
        method = "RK45"

        rtol = tolerance
        atol = tolerance
        t_eval = np.arange(time[0], time[1] + self.dt.seconds, self.dt.seconds)

        if self.state.stm is not None:
            ode_state = sci_int.solve_ivp(
                self.stm_propagator,
                time,
                self.state.compile(),
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=t_eval,
            )
            self.state.position = ode_state.y[0:3, -1]
            self.state.velocity = ode_state.y[3:6, -1]
            self.state.stm = np.reshape(ode_state.y[6:, -1], (6, 6))
        else:
            ode_state = sci_int.solve_ivp(
                self.propagator,
                time,
                self.state.compile(),
                method=method,
                rtol=rtol,
                atol=atol,
                t_eval=t_eval,
            )
            self.state.position = ode_state.y[0:3, -1]
            self.state.velocity = ode_state.y[3:6, -1]
            self.save_state_data(ode_state=ode_state)

    def update_state(self, new_state):
        """
        Updates the state of the agent.

        Parameters
        ----------
        new_state : array-like
            The new state vector of the agent.
        """
        self.state.position = new_state[0:3]
        self.state.velocity = new_state[3:6]
        self.state.stm = np.reshape(new_state[6:], (6, 6))

    def save_state_data(self, ode_state):
        """
        Saves the state data from the ODE solver.

        Parameters
        ----------
        ode_state : OdeResult
            The result object from the ODE solver containing the state and time data.
        """
        for state, time in zip(ode_state.y.transpose(), ode_state.t):
            delta_time = timedelta(seconds=time)
            self.state_data.append(
                State(
                    position=state[0:3],
                    velocity=state[3:6],
                    time=self.start_time + delta_time,
                )
            )
