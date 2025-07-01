from python_propagate.environment.planets import Planet


class Sun(Planet):
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
        name="Sun",
        radius=6378.1363,
        j2=0.0010826267,
        j3=-0.0000025327,
        spice_id=399,
        mu=132712440018,
        angular_velocity=7.29211585530066e-5,
        flattening_bool=False,
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
        return (
            f"Earth(name={self._name}, radius={self._radius}, J2={self._j2}, "
            f"J3={self._j3}, spice_id={self._spice_id}, mu={self._mu}, "
            f"angular_velocity={self._angular_velocity}, flattening_bool={self._flattening_bool})"
        )
