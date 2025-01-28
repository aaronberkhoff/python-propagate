from python_propagate.Agents import Agent

class Spacecraft(Agent):
    """
    A class to represent a spacecraft.

    Inherits from the Agent class.

    Attributes
    ----------
    state : object
        The initial state of the spacecraft.
    start_time : datetime
        The start time of the simulation.
    duration : float
        The duration of the simulation.
    dt : float
        The time step for the simulation.
    coefficent_of_drag : float, optional
        The coefficient of drag of the spacecraft (default is None).
    mass : float, optional
        The mass of the spacecraft (default is None).
    area : float, optional
        The cross-sectional area of the spacecraft (default is None).

    Methods
    -------
    __init__(self, state, start_time, duration, dt, coefficent_of_drag=None, mass=None, area=None):
        Initializes the Spacecraft with the given parameters.
    """

    def __init__(self, state, 
                 start_time, 
                 duration, dt, 
                 coefficent_of_drag = None, mass = None, area = None):
        """
        Constructs all the necessary attributes for the Spacecraft object.

        Parameters
        ----------
        state : object
            The initial state of the spacecraft.
        start_time : datetime
            The start time of the simulation.
        duration : float
            The duration of the simulation.
        dt : float
            The time step for the simulation.
        coefficent_of_drag : float, optional
            The coefficient of drag of the spacecraft (default is None).
        mass : float, optional
            The mass of the spacecraft (default is None).
        area : float, optional
            The cross-sectional area of the spacecraft (default is None).
        """
        super().__init__(state, start_time, duration, dt, coefficent_of_drag, mass, area = area)
