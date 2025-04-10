import numpy as np

from python_propagate.sensors import Sensor
from python_propagate.states import State
from python_propagate.agents import Agent
from python_propagate.platforms.station import Station


class Optical(Sensor):

    def __init__(self, noise_mean, noise_covariance):

        super().__init__(
            noise_mean, noise_covariance
        )


    def measurement_map(self,state: State):

        right_ascension = np.arctan2(state.position[1], state.position[0])
        declination = np.arcsin(state.position[2] / np.linalg.norm(state.position))

        return np.vstack([right_ascension, declination])

class Image:

    def __init__(self,resolution = 1024, background = 'dark', scale = 'grey', timestamp = None, stars = 1000):

        self.resolution = resolution
        self.background = background
        self.scale = scale
        self.timestamp = timestamp
        self.stars = stars
        self.centroid = (resolution // 2,) * 2
        self.data = np.zeros((resolution,resolution,3))
        self._render_background(background=background)



        pass

    def _render_background(self, background, blur_length=10, blur_direction=(0, 1)):
        """
        Renders background stars with optional motion blur.

        Args:
            background (str): Type of background ("dark" or "stars").
            blur_length (int): Length of motion blur in pixels.
            blur_direction (tuple): Direction of blur (dx, dy).
        """
        if background == 'dark':
            pass  # default is dark

        elif background == 'stars':
            num_stars = self.stars

            star_x = np.random.randint(0, self.resolution, num_stars)
            star_y = np.random.randint(0, self.resolution, num_stars)

            star_brightness = np.random.random(num_stars) 
            # Normalize the blur direction vector
            dx, dy = blur_direction
            magnitude = np.hypot(dx, dy)
            dx /= magnitude
            dy /= magnitude

            for x, y, brightness in zip(star_x, star_y, star_brightness):
                # for i in range(np.random.randint(0,blur_length)):
                for i in range(blur_length):
                    # Move along the blur direction
                    xi = int(round(x + i * dx))
                    yi = int(round(y + i * dy))

                    # Check if inside the image bounds
                    if 0 <= xi < self.resolution and 0 <= yi < self.resolution:
                        self.data[xi, yi,:] = brightness  # or you could fade brightness
                        if self.scale == 'rgb':
                            self.data[xi, yi,:] *= self._generate_random_color()  # or you could fade brightness
        
        
        


        else:
            raise ValueError(f"{background} not supported")


    def _generate_random_color(self):
        """
        Generates a random RGB color in the visible spectrum.
        Returns:
            A tuple of (r, g, b) with values between 0 and 1.
        """
        r = np.random.uniform(0.1, 1.0)  # red between 0.1 and 1.0
        g = np.random.uniform(0.1, 1.0)  # green between 0.1 and 1.0
        b = np.random.uniform(0.1, 1.0)  # blue between 0.1 and 1.0

        return r, g, b


class Camera(Sensor):

    def __init__(self, noise_mean, noise_covariance, resolution = 1024, reference_range = 30000, reference_area = 1, reference_brightness = 15): #reference range -> 1 pixel at specified range

        reference_area /= 1000
        self.resolution = resolution
        self.reference_range = reference_range
        self.reference_area = reference_area # convert to km
        self.pixel_size = np.sqrt(reference_area) / reference_range
        self.reference_brightness = reference_brightness

        super().__init__(noise_mean, noise_covariance)

    def measurement_map(self, state:State, agent:Agent, station: Station):
        return self.generate_image(state,agent,station, crosslines=True)
    

    def generate_image(self, state: State, agent: Agent, station: Station, crosslines = False):
        # Create the background
        image = Image(resolution=self.resolution, timestamp=None, background='stars', scale='rgb')

        # Generate the agent's image (spacecraft)
        agent_image = self._generate_agent_image(state, agent, station)

        fov_deg = station.minimum_elevation_angle

        # Image parameters
        image_height, image_width = image.data.shape[:2]

        # Get Azimuth and Elevation
        az_rad, el_rad, enu = station.calculate_azimuth_and_elevation(state=state, enu_frame=True)

        # Filter out targets below the horizon
        if np.degrees(el_rad) < station.minimum_elevation_angle:
            return image

        # Normalize azimuth and elevation
        # az_deg = np.degrees(az) % 360
        # el_deg = np.degrees(el)

        # ---- NEW: Map Az/El to (x, y) pixel coordinates ----
        # Convert degrees to radians
        # az_rad = np.radians(az_deg)

        # Normalize radius: 0 at zenith (90 deg), 1 at horizon (0 deg)
        r = ((np.pi / 2 - el_rad) / np.pi / 2) / np.radians(fov_deg) 

        # (x, y) in unit circle
        x = r * np.sin(az_rad)
        y = r * np.cos(az_rad)

        # Map (x, y) to pixel coordinates
        pixel_x = int((x + 1) * (image_width / 2))
        pixel_y = int((1 - y) * (image_height / 2))

        # ---- Done mapping Az/El to pixels ----

        # Agent image size
        agent_height, agent_width, agent_color = agent_image.shape[:3]

        # Compute bounds for placing the agent image
        new_row_start = max(0, pixel_y - agent_height // 2)
        new_row_end = min(image_height, pixel_y + agent_height // 2 + (agent_height % 2))

        new_col_start = max(0, pixel_x - agent_width // 2)
        new_col_end = min(image_width, pixel_x + agent_width // 2 + (agent_width % 2))

        # Resize agent image if it doesn't fit fully
        agent_patch = agent_image[
            :new_row_end - new_row_start,
            :new_col_end - new_col_start, :
        ]

        # Blend or paste the agent image
        image.data[
            new_row_start:new_row_end,
            new_col_start:new_col_end
        ] = agent_patch


        #crosslines
        if crosslines:

            self._draw_crosshairs(image)

        return image
    

    def _draw_crosshairs(self,image):
        """
        Draws vertical and horizontal crosshairs through the center of the image.
        """
        image_height, image_width = image.data.shape[:2]
        center_x = image_width // 2
        center_y = image_height // 2

        # Draw vertical line
        for y in range(image_height):
            image.data[y, center_x,:] = (0.0,1.0,0.0)  

        # Draw horizontal line
        for x in range(image_width):
            image.data[center_y, x, : ] = (0.0,1.0,0.0)  






    
    def _generate_agent_image(self, state, agent: Agent, station: Station):
        _, apparent_mag, areas = station.calculate_light_areas_exposed(state, agent)

        pixel_sizes = areas / self.pixel_size
        nearest_square = round_to_nearest_square(np.sum(pixel_sizes))

        # Create a base image with RGB channels
        base_image = np.ones((int(np.sqrt(nearest_square)), int(np.sqrt(nearest_square)), 3))  # RGB

        pixel_weights = np.floor(np.sum(pixel_sizes)) / nearest_square
        base_image *= pixel_weights

        n_int = [int(np.floor(pixel)) for pixel in pixel_sizes]
        frac = pixel_sizes - n_int
        weights = [[1.0] * n for n in n_int]

        # Distribute fractional pixel values
        for i, w in enumerate(weights):
            if frac[i] > 0:
                w.append(frac[i])

        # Calculate the average apparent magnitude
        adjusted_mag = np.average([np.average(np.array(w) * mag) for w, mag in zip(weights, apparent_mag)])

        # Adjust base image brightness based on the apparent magnitude and reference brightness
        brightness_factor = min(1, self.reference_brightness / adjusted_mag)

        # Create the RGB color based on the adjusted magnitude
        # Assuming the apparent magnitude directly affects brightness, we'll use it to scale RGB channels
        # You can modify this to have more complex color mapping if needed.
        # For now, we'll just use grayscale RGB based on the magnitude.
        agent_color = np.array([brightness_factor] * 3)  # Grayscale color, you could customize it
        # print(brightness_factor)
        # agent_color = np.array([brightness_factor, 0, 0])

        # Apply the color scaling to the base image
        base_image *= agent_color  # Apply the grayscale color across all RGB channels

        # If you want to handle color other than grayscale, you could modify the agent_color for each channel
        # E.g., for red tint:
        # agent_color = np.array([255, 0, 0])

        return base_image


            


    def set_image_background(self):

        pass 

        


def round_to_nearest_square(x):
    # Find the square root
    root = np.sqrt(x)
    # Round the root to the nearest integer
    nearest_int = int(np.round(root))
    # Square it back
    return nearest_int ** 2


