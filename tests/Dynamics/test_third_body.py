from python_propagate.dynamics.three_body import ThreeBody
from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.environment.sun import Sun
from python_propagate.environment.moon import Moon
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State
from datetime import datetime, timedelta
from python_propagate.dynamics.keplerian import Keplerian
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from pathlib import Path
from python_propagate.plots.plot_orbital_elements import plot_orbital_elements


def plot_orbit(agents, output_directory, name, central_body):
        """
        Plots an isometric view of the orbit, showing four different projections:
        - axxy: XY plane
        - axxz: XZ plane
        - axyz: YZ plane
        - ax3d: 3D orbit view

        The central body is represented by a sphere at the origin with a color gradient in the 3D view.
        Its radius is determined by self.central_body.radius.
        """

        # Create a new figure with a grid of subplots
        fig = plt.figure(figsize=(14, 14))
        axxy = fig.add_subplot(221)
        axxz = fig.add_subplot(223)
        axyz = fig.add_subplot(224)
        ax3d = fig.add_subplot(222, projection="3d")

        # Determine sphere properties for the central body
        sphere_radius = central_body.radius
        central_color = "blue"  # Base color for 2D plots
        sphere_alpha = 0.3  # Transparency for the sphere

        # Draw the sphere on the 2D plots as a circle centered at (0, 0)
        sphere_circle = Circle(
            (0, 0), sphere_radius, color=central_color, alpha=sphere_alpha
        )
        axxy.add_patch(sphere_circle)
        sphere_circle_xz = Circle(
            (0, 0), sphere_radius, color=central_color, alpha=sphere_alpha
        )
        axxz.add_patch(sphere_circle_xz)
        sphere_circle_yz = Circle(
            (0, 0), sphere_radius, color=central_color, alpha=sphere_alpha
        )
        axyz.add_patch(sphere_circle_yz)

        # Draw the sphere on the 3D plot using a parametric surface with a color gradient.
        # Generate the mesh for the sphere.
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x_sphere = sphere_radius * np.cos(u) * np.sin(v)
        y_sphere = sphere_radius * np.sin(u) * np.sin(v)
        z_sphere = sphere_radius * np.cos(v)

        # Use a colormap (e.g., 'viridis') to create a color gradient across the surface.
        # The colormap will map the z_sphere values to colors.
        surface = ax3d.plot_surface(
            x_sphere,
            y_sphere,
            z_sphere,
            cmap="viridis",  # Choose your favorite colormap here
            alpha=sphere_alpha,
            rstride=1,
            cstride=1,  # Adjust these parameters for resolution
            linewidth=0,
            antialiased=True,
        )

        # # Optionally, add a color bar for reference
        # fig.colorbar(surface, ax=ax3d, shrink=0.5, aspect=10)

        # Loop through all agents to plot their orbits and key points (start and end)
        for agent in agents:
            # Plot start and end markers along with the trajectory on the XY plane
            axxy.plot(
                agent.state_data[0].position[0],
                agent.state_data[0].position[1],
                "g*",
                label="start",
                fillstyle="none",
            )
            axxy.plot(
                agent.state_data[-1].position[0],
                agent.state_data[-1].position[1],
                "rs",
                label="end",
                fillstyle="none",
            )
            axxy.plot(
                [state.position[0] for state in agent.state_data],
                [state.position[1] for state in agent.state_data],
                label=agent.name,
                linewidth=1.0,
            )
            axxy.set_xlabel("X [KM]")
            axxy.set_ylabel("Y [KM]")

            # Plot on the XZ plane
            axxz.plot(
                agent.state_data[0].position[0],
                agent.state_data[0].position[2],
                "g*",
                label="start",
                fillstyle="none",
            )
            axxz.plot(
                agent.state_data[-1].position[0],
                agent.state_data[-1].position[2],
                "rs",
                label="end",
                fillstyle="none",
            )
            axxz.plot(
                [state.position[0] for state in agent.state_data],
                [state.position[2] for state in agent.state_data],
                label=agent.name,
                linewidth=1.0,
            )
            axxz.set_xlabel("X [KM]")
            axxz.set_ylabel("Z [KM]")

            # Plot on the YZ plane
            axyz.plot(
                agent.state_data[0].position[1],
                agent.state_data[0].position[2],
                "g*",
                label="start",
                fillstyle="none",
            )
            axyz.plot(
                agent.state_data[-1].position[1],
                agent.state_data[-1].position[2],
                "rs",
                label="end",
                fillstyle="none",
            )
            axyz.plot(
                [state.position[1] for state in agent.state_data],
                [state.position[2] for state in agent.state_data],
                label=agent.name,
                linewidth=1.0,
            )
            axyz.set_xlabel("Y [KM]")
            axyz.set_ylabel("Z [KM]")

            # Plot on the 3D view
            ax3d.plot(
                [state.position[0] for state in agent.state_data],
                [state.position[1] for state in agent.state_data],
                [state.position[2] for state in agent.state_data],
                label=agent.name,
                linewidth=1.0,
            )
            ax3d.plot(
                [agent.state_data[0].position[0]],
                [agent.state_data[0].position[1]],
                [agent.state_data[0].position[2]],
                "g*",
                label="start",
                fillstyle="none",
            )
            ax3d.plot(
                [agent.state_data[-1].position[0]],
                [agent.state_data[-1].position[1]],
                [agent.state_data[-1].position[2]],
                "rs",
                label="end",
                fillstyle="none",
            )
            ax3d.set_xlabel("X [KM]")
            ax3d.set_ylabel("Y [KM]")
            ax3d.set_zlabel("Z [KM]")

        # Add legend and grid to the 2D subplots
        axxy.legend()
        axxy.grid(True)
        axxz.grid(True)
        axyz.grid(True)

        axxy.set_aspect("equal", adjustable="datalim")
        axxz.set_aspect("equal", adjustable="datalim")
        axyz.set_aspect("equal", adjustable="datalim")
        ax3d.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()

        # Save the figure to the designated output directory
        saveas = output_directory / f"{name}_orbit_plot.png"
        print(f"Files saved:\n {saveas}: ")
        plt.savefig(saveas, dpi=100)
        plt.close(fig)





earth = Earth()

start_time = datetime.strptime("2025-01-15T12:30:00", "%Y-%m-%dT%H:%M:%S")
duration = timedelta(seconds=7*86400)
dt = timedelta(seconds=30)

celestial_bodies = [Sun(), Moon()]

scenario = Scenario(
    central_body=Earth(), start_time=start_time, duration=duration, dt=dt, celestial_bodies=celestial_bodies
)
position = np.array([1340.745, -6663.403, -132.528])
velocity = np.array([5.457807, 1.368701, -5.614317])

initial_state = State(position=position, velocity=velocity, time=start_time)


third_body = ThreeBody(scenario=scenario)

accel = third_body(initial_state, time=None)

coefficient_of_drag = 2.0
mass = 1350
area = 3.6 

jah_sat = Spacecraft(
    initial_state,
    start_time=start_time,
    duration=duration,
    dt=scenario.dt,
    coefficient_of_drag=coefficient_of_drag,
    mass=mass,
    area=area,
)

jah_sat.set_scenario(scenario=scenario)
dynamics = ["kepler", "J2", "J3", "drag", "3body"]
jah_sat.add_dynamics(dynamics=dynamics)

jah_sat.propagate()
output_directory = "tests/results"
name = "third_body_test_3"
plot_orbit([jah_sat], Path(output_directory), name, earth)
plot_orbital_elements([jah_sat], scenario, Path(output_directory), name = name,legend = True)
plt.show()

pass