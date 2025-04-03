"""
planets.py

This module contains the Planet class and its subclasses.

Classes:
- Planet: A class to represent a planet.
- Earth: A class to represent the Earth.

Author: Aaron Berkhoff
Date: 2025-01-30

"""


class Planet:
    """
    A class to represent a planet.

    Attributes
    ----------
    name : str
        The name of the planet.
    radius : float
        The radius of the planet.
    j2 : float
        The second zonal harmonic coefficient of the planet.
    j3 : float
        The third zonal harmonic coefficient of the planet.
    spice_id : int
        The SPICE ID of the planet.
    mu : float
        The gravitational parameter of the planet.
    angular_velocity : float
        The angular velocity of the planet.
    flattening : float
        The flattening factor of the planet.

    Methods
    -------
    __init__(self, name: str, radius: float, j2: float, j3: float, spice_id: int, mu: float, angular_velocity: float, flattening: float):
        Initializes the Planet with the given parameters.
    radius(self):
        Returns the radius of the planet.
    mu(self):
        Returns the gravitational parameter of the planet.
    j2(self):
        Returns the second zonal harmonic coefficient of the planet.
    j3(self):
        Returns the third zonal harmonic coefficient of the planet.
    name(self):
        Returns the name of the planet.
    spice_id(self):
        Returns the SPICE ID of the planet.
    angular_velocity(self):
        Returns the angular velocity of the planet.
    flattening(self):
        Returns the flattening factor of the planet.
    """

    def __init__(
        self,
        name: str,
        radius: float,
        j2: float,
        j3: float,
        spice_id: int,
        mu: float,
        angular_velocity: float,
        flattening_bool: bool = False,
        flattening: float = 0.0,
    ):
        """
        Initializes the Planet with the given parameters.

        Parameters
        ----------
        name : str
            The name of the planet.
        radius : float
            The radius of the planet.
        j2 : float
            The second zonal harmonic coefficient of the planet.
        j3 : float
            The third zonal harmonic coefficient of the planet.
        spice_id : int
            The SPICE ID of the planet.
        mu : float
            The gravitational parameter of the planet.
        angular_velocity : float
            The angular velocity of the planet.
        flattening : bool
            Include flattening.
        """
        self._radius = radius
        self._mu = mu
        self._j2 = j2
        self._j3 = j3
        self._name = name
        self._spice_id = spice_id
        self._angular_velocity = angular_velocity
        self._flattening_bool = flattening_bool
        self._flattening = flattening

    def __repr__(self):
        """
        Returns a string representation of the Planet object.

        Returns
        -------
        str
            A string representation of the Planet object.
        """
        return (
            f"Planet(name={self._name!r}, radius={self._radius}, J2={self.j2}, "
            f"J3={self.j3}, spice_id={self._spice_id}, mu={self._mu}, "
            f"angular_velocity={self._angular_velocity},"
            f"flattening={self._flattening},flattening_bool={self._flattening_bool})"
        )

    @property
    def radius(self):
        """
        Returns the radius of the planet.

        Returns
        -------
        float
            The radius of the planet.
        """
        return self._radius

    @property
    def mu(self):
        """
        Returns the gravitational parameter of the planet.

        Returns
        -------
        float
            The gravitational parameter of the planet.
        """
        return self._mu

    @property
    def j2(self):
        """
        Returns the second zonal harmonic coefficient of the planet.

        Returns
        -------
        float
            The second zonal harmonic coefficient of the planet.
        """
        return self._j2

    @property
    def j3(self):
        """
        Returns the third zonal harmonic coefficient of the planet.

        Returns
        -------
        float
            The third zonal harmonic coefficient of the planet.
        """
        return self._j3

    @property
    def name(self):
        """
        Returns the name of the planet.

        Returns
        -------
        str
            The name of the planet.
        """
        return self._name

    @property
    def spice_id(self):
        """
        Returns the SPICE ID of the planet.

        Returns
        -------
        int
            The SPICE ID of the planet.
        """
        return self._spice_id

    @property
    def angular_velocity(self):
        """
        Returns the angular velocity of the planet.

        Returns
        -------
        float
            The angular velocity of the planet.
        """
        return self._angular_velocity

    @property
    def flattening_bool(self):
        """
        Returns the flattening boolean of the planet.

        Returns
        -------
        bool
            The flattening boolean of the planet.
        """
        return self._flattening_bool

    @property
    def flattening(self):
        """
        Returns the flattening factor of the planet.

        Returns
        -------
        float
            The flattening factor of the planet.
        """
        return self._flattening

    @property
    def eccentricity(self):
        """
        Returns the eccentricity of the planet.

        Returns
        -------
        float
            The eccentricity of the planet.

        """
        return 2 * self._flattening - self._flattening**2


