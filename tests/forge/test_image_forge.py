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

import itertools

import imageio

import numpy as np
import cv2



def test_image_unit() -> None:
    config = load_yaml(yaml_file='examples/forge_eo_ir.yaml')

    # agents = config['anomalous_agents']
    # agents = config['anomalous_agents']
    agents = config['leo_agents']
    # agents = config['genes1'].agents
    station = config["stations"][0]
    # fov = 90 - station.minimum_elevation_angle
    fov = .01
    camera = Camera(noise_mean=[0, 0], noise_covariance=[0, 0],fov=fov,shutter_speed=1/20)
    # camera = Camera(noise_mean=[0, 0], noise_covariance=[0, 0],fov=fov, pointing_angles=(90,15))

    # Propagate all agents
    
    

    # You could return or save the images here if needed

    image_data = [[] for _ in range(len(agents))]

    # for state in agents.state_data:
    # for agent in agents:
    for i,agent in enumerate(agents):
        agent.propagate()

        cnt = 0
        for state in agent.state_data:
            cnt += 1
            print(f'Image {cnt}')
            image_data[i].append(camera.measurement_map(state,agents[0],station))

    combined_images = []

    for image in list(zip(*image_data)):
        
        combined_images.append(sum(image))


    
    
    output_directory = 'tests/results/images'
    name = 'image_test'

    # for 
    height, width = (combined_images[0].resolution, combined_images[0].resolution)
    # height, width = (1024,1024)
    fps = 30
    output_path = "tests/results/images/eoir_test.mp4"

    movie = np.array([image.rendered_image for image in combined_images])

    plot_ground_track(
                    agents, [station],output_directory, name=name,legend=True
                )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in movie:
    # Ensure the frame is in uint8 BGR format and size matches
        video_writer.write(frame)

    video_writer.release()
    # imageio.mimsave("tests/results/images/eoir_test.mp4", movie, fps=  1)


    pass
# Update function for the animation


    

 





if __name__ == "__main__":

    
    """
    Entry point for the test script.
    """
    test_image_unit()