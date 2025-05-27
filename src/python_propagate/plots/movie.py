from typing import Iterable

import numpy as np
import os
import cv2


def generate_movie(agent: Iterable,
                   output_directory: str,
                   name: str = "movie",
                   fps = 5,
                ):

    os.makedirs(output_directory, exist_ok=True)
    outpath = os.path.join(output_directory, f"{name}_movie.mp4")


    height, width = agent.state_data[0].metadata['image'].shape[0:2]

    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    for state in agent.state_data:
        frame = state.metadata['image']
        if np.any(frame != -99):
            if frame.dtype != np.uint8:  
                # frame = cv2.normalize(frame,None,0,255,cv2.NORM_MINMAX)
                frame = np.clip(frame,0,255)
                frame = frame.astype(np.uint8)
            video_writer.write(frame)

    video_writer.release()
    print(f"mp4 saved to {outpath}")

def generate_pix(agent: Iterable,
                 output_directory: str,
                 name: str = "frame"):
    
    outpath = output_directory / 'pix'

    os.makedirs(outpath, exist_ok=True)

    for i, state in enumerate(agent.state_data):
        frame = state.metadata['image']
        if np.any(frame != -99):  # Skip empty frames
            frame_filename = os.path.join(outpath, f"{name}_{i:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
    print(f"jpegs saved to {outpath}")
    # plt.close(fig)