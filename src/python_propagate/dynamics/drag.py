import numpy as np
from python_propagate.scenario import Scenario
from python_propagate.dynamics import Dynamic
from python_propagate.states import State


class Drag(Dynamic):
    def __init__(self, scenario: Scenario, agent=None, stm=None, function=None, complex_drag=False):

        if function is None:
            if complex_drag:
                function =self.complex_drag
            else:
                function = self.drag_function  # Assign the default function

        super().__init__(scenario, agent, stm, function)

    def drag_function(self, state: State, time: float):

        rx, ry, rz = state.extract_position()
        vx, vy, vz = state.extract_velocity()

        r = np.sqrt(rx**2 + ry**2 + rz**2)
        alt = r - self.agent.scenario.central_body.radius

        rho0, h0, scale_height = self.agent.scenario.central_body.atmosphere_model(r)

        density = rho0 * np.exp(-(alt - h0) / scale_height) * 1000**3

        vax = vx + self.agent.scenario.central_body.angular_velocity * ry
        vay = vy - self.agent.scenario.central_body.angular_velocity * rx

        va = np.sqrt(vax**2 + vay**2 + vz**2)

        dynamic_pressure = (
            -0.5
            * self.agent.coefficient_of_drag
            * density
            * self.agent.area
            / self.agent.mass
        )

        ax = dynamic_pressure * vax * va
        ay = dynamic_pressure * vay * va
        az = dynamic_pressure * vz * va

        return State(acceleration=np.array([ax, ay, az]))


    def complex_drag(self, state: State, time: float):
        """
        Example of a more complex drag function that could be used.
        This is just a placeholder for demonstration purposes.
        """

        rx, ry, rz = state.extract_position()
        vx, vy, vz = state.extract_velocity()

        r = np.sqrt(rx**2 + ry**2 + rz**2)
        alt = r - self.agent.scenario.central_body.radius

        rho0, h0, scale_height = self.agent.scenario.central_body.atmosphere_model(r)

        density = rho0 * np.exp(-(alt - h0) / scale_height) * 1000**3

        vax = vx + self.agent.scenario.central_body.angular_velocity * ry
        vay = vy - self.agent.scenario.central_body.angular_velocity * rx

        va = np.sqrt(vax**2 + vay**2 + vz**2)

        v_hat = np.array([vax, vay, vz]) / va 

        dynamic_pressure = (
            -0.5
            * self.agent.coefficient_of_drag
            * density
            / self.agent.mass
        )
        #set the orientation of the bus
        self.agent.bus.set_orientation(state)  # Ensure the bus orientation is set based on the current state

        normals = self.agent.bus.shape.face_normals 
        areas = self.agent.bus.shape.area_faces

        # Dot product with flow direction
        cos_theta = np.dot(normals, -v_hat)

        # Only consider faces exposed to the flow
        exposed = cos_theta > 0

        # Projected area per face
        area_proj = areas[exposed] * cos_theta[exposed]

        # Total drag force on the spacecraft
        acceleration = dynamic_pressure * area_proj[:,None] * (np.array([vax, vay, vz]) * va) # Apply drag in the direction of the velocity vector

        acceleration = np.sum(acceleration, axis=0)

        return State(
            acceleration=np.sum(acceleration, axis=0)  # Sum over all exposed faces to get total acceleration
        )
    
