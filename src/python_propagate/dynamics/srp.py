import numpy as np
from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.states import State
from python_propagate.environment.sun import Sun
from python_propagate.utilities.load_spice import load_spice
from python_propagate.utilities.calculations import calc_shadow
from python_propagate.utilities.constants import AU, C1    


class SRP(Dynamic):

    def __init__(self, scenario: Scenario, agent=None, stm=None, function=None, complex_srp=True):

        if function is None:
            load_spice()
            if complex_srp:
                function =self.complex_srp
            else:
                raise NotImplementedError(
                    "Cannonball srp is not implemented in this class. Use complex_srp=True to use the complex SRP model."
                )

        self.sun = Sun()
        super().__init__(scenario, agent, stm, function)


    def complex_srp(self, state: 'State', time: float):
        """
        Calculate the acceleration due to solar radiation pressure (SRP) on the spacecraft.
        
        This function computes the SRP force by considering the face normals and areas of the spacecraft
        bus and the reflectivity (Cs, Cd) properties of each face. It returns a new State object 
        with the computed acceleration.
        """

        # Get the state of the sun at the current time (assumed to be provided by your sun object)
        state_sun = self.sun.get_state(state.time)

        # Check for shadowing: if in shadow, return zero acceleration.
        shadow_bool, shadow_value = calc_shadow(
            state_agent=state,
            state_sun=state_sun,
            reference_body_radius=self.scenario.central_body.radius
        )
        if shadow_bool:
            return State(acceleration=np.zeros(3))

        # Set the spacecraft bus orientation
        self.agent.bus.set_orientation(state)

        # Get the face normals and areas from the bus shape.
        normals = self.agent.bus.shape.face_normals
        areas = self.agent.bus.shape.area_faces

        # Compute the cosine of the angle between each face normal and the sunlight direction.
        # The sun direction is taken as the unit vector from the spacecraft toward the sun.
        sun_direction = state_sun.position / np.linalg.norm(state_sun.position)
        cos_theta = np.dot(normals, sun_direction)

        # Only consider faces exposed to the sun (cos_theta > 0)
        exposed = cos_theta > 0
        cos_theta = cos_theta[exposed]
        normals = normals[exposed]
        # Also filter the corresponding areas.
        areas_exposed = areas[exposed]

        # Retrieve the face properties for the exposed faces.
        # Note: self.agent.bus.shape.face_properties is a dict indexed by triangle index.
        # We convert the values to a NumPy array and then select the exposed indices.
        all_cs = np.array([value['Cs'] for key, value in self.agent.bus.shape.face_properties.items()])
        all_cd = np.array([value['Cd'] for key, value in self.agent.bus.shape.face_properties.items()])
        cs_data = all_cs[exposed]
        cd_data = all_cd[exposed]

        # Compute reflectivity model coefficients
        mus = 0.5 * cs_data         # Albedo coefficient for SRP
        nu = (1.0 / 3.0) * cd_data    # Drag coefficient (not used in SRP but computed for completeness)
        btheta = 2 * nu * cos_theta + 4 * mus * cos_theta**2

        # Compute the vector from the spacecraft to the sun and its magnitude
        agent_to_sun = state_sun.position - state.position
        r_sun = np.linalg.norm(agent_to_sun)
        agent_to_sun_unit = agent_to_sun / r_sun

        # Compute the distance in AU (note: r_sun is in meters)
        distance_au = r_sun / AU

        # Compute the SRP force per unit area for each exposed face.
        # Note: The formulation below follows Vallado (2013) style, where the force is applied 
        # along the face normal and along the sun direction.
        # The minus sign ensures the force is repulsive (away from the sun).
        force_srp_faces = -C1 / distance_au * (
            btheta[:, None] * normals + (1 - mus[:,None]) * (cos_theta**2)[:, None] * agent_to_sun_unit
        ) * areas_exposed[:, None]

        # Sum the force contributions from all exposed faces
        total_force_srp = np.sum(force_srp_faces, axis=0)

        # Convert the force to acceleration (F = ma)
        acceleration = total_force_srp / self.agent.mass

        # apply the shadow value scaling (if shadow_value is meant to scale the force)
        acceleration *= shadow_value

        # Return the new state with the computed acceleration
        return State(acceleration=acceleration)


        
