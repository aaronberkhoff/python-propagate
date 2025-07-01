from datetime import timedelta
import numpy as np
from python_propagate.dynamics import Dynamic
from python_propagate.states import State
from python_propagate.utilities.transforms import inertial_to_ric
from python_propagate.dynamics.j2 import J2
from python_propagate.dynamics.j3 import J3
from python_propagate.dynamics.drag import Drag
from python_propagate.dynamics.stm import STM
from python_propagate.dynamics.three_body import ThreeBody
from python_propagate.dynamics.srp import SRP

class Manuever(Dynamic):
    """
    Base class for manuevers. This is a placeholder for the manuever class.
    """

    def __init__(
        self,
        scenario=None,
        execution_time: float = 30,
        execution_duration: float = 60,
        magnitude: float = 1.0,
        direction_ric: np.ndarray = np.array([0.0, 1.0, 0.0]),
        agent=None,  # agent added in agent object TODO: Better way?
        stm=None,
        function=None,
    ):

        # magnitude *= 1e-3  # convert from km/s to m/s
        if isinstance(execution_time, dict):
            self.execution_time = timedelta(**execution_time).total_seconds()
        else:
            self.execution_time = execution_time

        if isinstance(execution_duration, dict):
            self.execution_duration = timedelta(**execution_duration).total_seconds()
        else:
            self.execution_duration = execution_duration

        if isinstance(direction_ric, list):
            self.direction_ric = np.array(direction_ric)

        self.magnitude = magnitude
        self.execute_bool = True

        if function is None:
            function = self.manuever_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)

    # def __repr__(self):
    #     return f"{self.__class__}"

    def manuever_function(self, state: State, time: float):
        """
        Default manuever function. This can be overridden by subclasses.
        """
        raise NotImplementedError("Manuever function must be provided or overridden.")


class ImpulseManuever(Manuever):

    def __init__(
        self,
        scenario=None,
        execution_time: float = 30,
        magnitude: float = 1.0,
        direction_ric: np.ndarray = np.array([0.0, 1.0, 0.0]),
        agent=None,
        stm=None,
    ):

        super().__init__(
            scenario=scenario,
            execution_time=execution_time,
            magnitude=magnitude,
            direction_ric=direction_ric,
            agent=agent,
            stm=stm,
            function=self.manuever_function,
        )

    def manuever_function(self, state: State, time: float):

        if time >= self.execution_time and self.execute_bool:
            print(f"Manuever at time = {time}")
            self.execute_bool = False
            transform = inertial_to_ric(state=state.compile())
            velo_inertial = transform.T @ (self.magnitude * self.direction_ric)
        else:
            velo_inertial = np.array([0.0, 0.0, 0.0])

        return State(velocity=velo_inertial, time=time)


class ThrustManuever(Manuever):

    def __init__(
        self,
        scenario=None,
        execution_time: float = 30,
        execution_duration: float = 60,
        magnitude: float = 1.0,
        direction_ric: np.ndarray = np.array([0.0, 1.0, 0.0]),
        agent=None,
        stm=None,
    ):

        super().__init__(
            scenario=scenario,
            execution_time=execution_time,
            execution_duration=execution_duration,
            magnitude=magnitude,
            direction_ric=direction_ric,
            agent=agent,
            stm=stm,
            function=self.manuever_function,
        )

    def manuever_function(self, state: State, time: float):

        if time >= self.execution_time and time < (
            self.execution_time + self.execution_duration
        ):
            # print(f'Thrust Manuever at time = {time}')
            self.execute_bool = False
            transform = inertial_to_ric(state=state.compile())
            acc_inertial = transform.T @ (self.magnitude * self.direction_ric)
        else:
            acc_inertial = np.array([0.0, 0.0, 0.0])

        return State(acceleration=acc_inertial, time=time)


class StationKeepLoss(Manuever):

    def __init__(
        self,
        scenario,
        agent=None,
        execution_time: float = 30,
        execution_duration: float = 120,
        dynamics=["J2"],
        stm=None,
    ):

        self.dynamics = []
        self.scenario = scenario
        self.agent = agent
        self.add_dynamics(dynamics=dynamics)

        # self.execute_bool = True

        super().__init__(
            scenario=scenario,
            execution_duration=execution_duration,
            execution_time=execution_time,
            magnitude=None,
            direction_ric=None,
            agent=agent,
            stm=stm,
            function=self.manuever_function,
        )

    def add_dynamics(self, dynamics: list):
        """Adds dynamics to the manuever.
        Parameters
        dynamics : tuple
            A tuple of dynamics to be added to the agent.

        """

        # TODO: Self referenceing to self is not good practice
        for dynamic in dynamics:

            if isinstance(dynamic, type) and issubclass(dynamic, Dynamic):
                dynamic = dynamic(scenario=self.scenario, agent=self.agent)

            elif dynamic == "J2":
                self.dynamics.append(J2(scenario=self.scenario, agent=self.agent))

            elif dynamic == "J3":
                self.dynamics.append(J3(scenario=self.scenario, agent=self.agent))

            elif dynamic == "drag":
                self.dynamics.append(Drag(scenario=self.scenario, agent=self.agent))

            elif dynamic == "3body":
                self.dynamics.append(ThreeBody(scenario=self.scenario, agent=self.agent))

            elif dynamic == "complex_srp":
                self.dynamics.append(SRP(scenario=self.scenario, agent=self.agent,complex_srp=True))

            elif dynamic == "complex_drag":
                self.dynamics.append(Drag(scenario=self.scenario, agent=self.agent,complex_drag=True))

            elif dynamic == "stm":
                self.dynamics.append(STM(scenario=self.scenario, agent=self.agent))

            elif isinstance(dynamic, Dynamic):
                self.dynamics.append(dynamic)

            elif issubclass(dynamic, Dynamic):

                self.dynamics.append(dynamic)
            else:
                raise NotImplementedError(
                    f"Dynamic <{dynamic}> is not an option in StationKeepLoss or is spelled wrong"
                )

    # TODO add other perturbation

    def manuever_function(self, state: State, time: float):

        if time > self.execution_time and time < (
            self.execution_time + self.execution_duration
        ):
            # print(f'Thrust Manuever at time = {time}')
            self.execute_bool = False
            state2 = State(acceleration=np.zeros(3))
            for dyn in self.dynamics:
                dyn.agent = self.agent  # ensure that the agent is set #TODO Better way?
                state2 @= dyn(state, time)
        else:
            state2 = State(acceleration = np.zeros(3))
        
        return state2


class FreeRotate(Manuever):

    def __init__(
        self,
        scenario=None,
        execution_time=30,
        execution_duration=60,
        agent=None,
        stm=None,
    ):

        function = self.manuever_function
        super().__init__(
            scenario,
            execution_time,
            magnitude=None,
            execution_duration=execution_duration,
            agent=agent,
            stm=stm,
            function=function,
        )

    def manuever_function(self, state, time):

        if time >= self.execution_time and time < (
            self.execution_time + self.execution_duration
        ):

            self.agent.bus.orientation = "free"

        else:
            self.agent.bus.orientation = self.agent.bus.base_orientation

        return State(acceleration=np.zeros(3))
