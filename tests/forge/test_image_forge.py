from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from python_propagate.scenario import Scenario
from python_propagate.environment.earth import Earth
from python_propagate.agents.spacecraft import Spacecraft
from python_propagate.agents import State
from python_propagate.constructors.yaml_constructors import load_yaml

from python_propagate.utilities.load_spice import load_spice

from python_propagate.sensors.optical import Camera, Image
from matplotlib.animation import FuncAnimation
from python_propagate.plots.plot_ground_track import plot_ground_track

import imageio

import numpy as np



def test_image_unit() -> None:
    config = load_yaml(yaml_file='examples/forge_eo_ir.yaml')

    agents = config['nominal_agents'][0]
    # agents = config['genes1'].agents
    station = config["stations"][0]
    camera = Camera(noise_mean=[0, 0], noise_covariance=[0, 0])

    # Propagate all agents
    agents.propagate()

    # You could return or save the images here if needed

    image_data = []

    for state in agents.state_data:

        image_data.append(camera.measurement_map(state,agents,station))




    
    output_directory = 'tests/results/images'
    name = 'image_test'

    movie = np.array([image.rendered_image for image in image_data])

    plot_ground_track(
                    [agents], [station],output_directory, name=name,legend=True
                )

    imageio.mimsave("tests/results/eoir_test.mp4", movie, fps= 5)
    # fig, ax = plt.subplots()
    
    # im = ax.imshow(image_data[0].data, cmap='gray', animated=True)

    # def _update(frame):
    #     im.set_array(image_data[frame].data)  # Update the image with the current frame
    #     return [im]

    # # Create the animation
    # ani = FuncAnimation(fig, _update, frames=len(image_data), interval=10, blit=True)

    # # Save the animation as a video file
    # # plt.show()
    # ani.save('image_movie.html', writer='html', fps=1)

    # plt.imshow(image.data, cmap='gray')

    # plt.show()

    pass
# Update function for the animation


    

 





if __name__ == "__main__":

    
    """
    Entry point for the test script.
    """
    test_image_unit()