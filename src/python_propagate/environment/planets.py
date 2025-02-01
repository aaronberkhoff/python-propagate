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
        flattening: float = 0.0
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
        return (f"Planet(name={self._name!r}, radius={self._radius}, J2={self._J2}, "
                f"J3={self._J3}, spice_id={self._spice_id}, mu={self._mu}, "
                f"angular_velocity={self._angular_velocity},"
                f"flattening={self._flattening},flattening_bool={self._flattening_bool})")

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


class Earth(Planet):
    """
    A class to represent the Earth.

    Inherits from Planet.

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
    """

    def __init__(
        self,
        name="Earth",
        radius=6378.1363,
        j2=0.0010826267,
        j3=-0.0000025327,
        spice_id=399,
        mu=398600.4415,
        angular_velocity=7.29211585530066e-5,
        flattening_bool=False
    ):
        """
        Initializes the Earth with the given parameters.

        Parameters
        ----------
        name : str, optional
            The name of the planet (default is 'Earth').
        radius : float, optional
            The radius of the planet (default is 6378.1363).
        j2 : float, optional
            The second zonal harmonic coefficient of the planet (default is 0.0010826267).
        j3 : float, optional
            The third zonal harmonic coefficient of the planet (default is -0.0000025327).
        spice_id : int, optional
            The SPICE ID of the planet (default is 399).
        mu : float, optional
            The gravitational parameter of the planet (default is 398600.4415).
        angular_velocity : float, optional
            The angular velocity of the planet (default is 7.29211585530066e-5).
        flattening_bool : float, optional
            The flattening boolean of the planet (default is 1 / 298.257223563).
        """
        if flattening_bool:
            flattening = 1 / 298.257223563
        else:
            flattening = 0.0

        super().__init__(
            name, radius, j2, j3, spice_id, mu, angular_velocity, flattening=flattening
        )

    def __repr__(self):
        """
        Returns a string representation of the Planet object.

        Returns
        -------
        str
            A string representation of the Planet object.
        """
        return (f"Earth(name={self._name}, radius={self._radius}, J2={self._j2}, "
                f"J3={self._j3}, spice_id={self._spice_id}, mu={self._mu}, "
                f"angular_velocity={self._angular_velocity}, flattening_bool={self._flattening_bool})")

    def atmosphere_model(self, radius_spacecraft):
        """
        Returns the atmospheric density at the given altitude.

        Parameters
        ----------
        radius_spacecraft : float
            The radius of the spacecraft.

        Returns
        -------
        rho0 : float
            The atmospheric density at the given altitude.
        h0 : float
            The altitude of the atmospheric density.
        base_height : float
            The base height of the atmospheric density.
        """

        altitude = radius_spacecraft - self.radius

        if altitude > 1000:
            rho0 = 3.019e-15
            h0 = 1000
            base_height = 268
        elif altitude > 900:
            rho0 = 5.245e-15
            h0 = 900
            base_height = 181.05
        elif altitude > 800:
            rho0 = 1.170e-14
            h0 = 800
            base_height = 124.64
        elif altitude > 700:
            rho0 = 3.614e-14
            h0 = 700
            base_height = 88.667
        elif altitude > 600:
            rho0 = 1.454e-13
            h0 = 600
            base_height = 71.835
        elif altitude > 500:
            rho0 = 6.967e-13
            h0 = 500
            base_height = 63.822
        elif altitude > 450:
            rho0 = 1.585e-12
            h0 = 450
            base_height = 60.828
        elif altitude > 400:
            rho0 = 3.725e-12
            h0 = 400
            base_height = 58.515
        elif altitude > 350:
            rho0 = 9.518e-12
            h0 = 350
            base_height = 53.298
        elif altitude > 300:
            rho0 = 2.418e-11
            h0 = 300
            base_height = 53.628
        elif altitude > 250:
            rho0 = 7.248e-11
            h0 = 250
            base_height = 45.546
        elif altitude > 200:
            rho0 = 2.789e-10
            h0 = 200
            base_height = 37.105
        elif altitude > 180:
            rho0 = 5.464e-10
            h0 = 180
            base_height = 29.740
        elif altitude > 150:
            rho0 = 2.070e-9
            h0 = 150
            base_height = 22.523
        elif altitude > 140:
            rho0 = 3.845e-9
            h0 = 140
            base_height = 16.149
        elif altitude > 130:
            rho0 = 8.484e-9
            h0 = 130
            base_height = 12.636
        elif altitude > 120:
            rho0 = 2.438e-8
            h0 = 120
            base_height = 9.473
        elif altitude > 110:
            rho0 = 9.661e-8
            h0 = 110
            base_height = 7.263
        elif altitude > 100:
            rho0 = 5.297e-7
            h0 = 100
            base_height = 5.877
        elif altitude > 90:
            rho0 = 3.396e-6
            h0 = 90
            base_height = 5.382
        elif altitude > 80:
            rho0 = 1.905e-5
            h0 = 80
            base_height = 5.799
        elif altitude > 70:
            rho0 = 8.770e-5
            h0 = 70
            base_height = 6.549
        elif altitude > 60:
            rho0 = 3.206e-4
            h0 = 60
            base_height = 7.714
        elif altitude > 50:
            rho0 = 1.057e-3
            h0 = 50
            base_height = 8.382
        elif altitude > 40:
            rho0 = 3.972e-3
            h0 = 40
            base_height = 7.554
        elif altitude > 30:
            rho0 = 1.774e-2
            h0 = 30
            base_height = 6.682
        elif altitude > 25:
            rho0 = 3.899e-2
            h0 = 25
            base_height = 6.349
        else:
            rho0 = 1.225
            h0 = 0
            base_height = 7.249

        return rho0, h0, base_height
