
import numpy as np
from copy import deepcopy
from typing import Iterable
import random 
import cv2
from scipy.ndimage import gaussian_filter
from scipy.ndimage import rotate


from python_propagate.sensors import Sensor
from python_propagate.states import State
from python_propagate.agents import Agent
from python_propagate.platforms.station import Station

from python_propagate.utilities.units import ARC2DEG, RAD2DEG, DEG2RAD, RAD2ARC, ARC2RAD


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

    def __init__(self,resolution = 1024, background = 'dark', scale = 'grey', timestamp = None, n_stars = 1000, high_res = 8000):

        self.resolution = resolution
        self.background = background
        self.scale = scale
        self.timestamp = timestamp
        self.n_stars = n_stars
        self.centroid = (resolution // 2,) * 2
        self.base_image = np.zeros((resolution,resolution,3),dtype=int)
        self.agent_image = np.zeros((resolution,resolution,3),dtype=int)
        self.background_image = np.zeros((resolution,resolution,3),dtype=int)
        self.high_res = high_res

        self._render_background()

        pass

    def _render_background(self, blur_length=10, blur_direction=(0, 1)):
        """
        Renders background stars with optional motion blur.

        Args:
            background (str): Type of background ("dark" or "stars").
            blur_length (int): Length of motion blur in centers.
            blur_direction (tuple): Direction of blur (dx, dy).
        """
        if self.background == 'dark':
            pass  # default is dark

        elif self.background == 'stars':
            num_stars = self.n_stars

            star_x = np.random.randint(0, self.resolution, num_stars)
            star_y = np.random.randint(0, self.resolution, num_stars)

            # Generate random integer brightness values (e.g., between 50 and 255)
            star_brightness = np.random.randint(50, 256, num_stars)  
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
                        self.background_image[xi, yi,:] = brightness  # or you could fade brightness
                        if self.scale == 'rgb':
                            self.background_image[xi, yi,:] *= self._generate_star_color()  # or you could fade brightness
        
        else:
            raise ValueError(f"{self.background} not supported")




    def _generate_star_color(self):
        """
        Generates a realistic star color (blue, white, yellow, or red).
        Returns:
            A tuple of (r, g, b) with integer values between 0 and 255.
        """
        color_temps = {
            'blue': (0.7, 0.8, 1.0),
            'white': (1.0, 1.0, 1.0),
            'yellow': (1.0, 1.0, 0.6),
            'orange': (1.0, 0.6, 0.3),
            'red': (1.0, 0.3, 0.3),
        }

        # Use random.choice instead of np.random.choice
        base_color = np.array(random.choice(list(color_temps.values())))
        jitter = np.random.normal(0, 0.05, 3)
        noisy_color = np.clip(base_color + jitter, 0, 1)

        int_color = (noisy_color * 255).astype(int)
        return tuple(int_color)

    
    @property
    def rendered_image(self):

        data = np.clip(self.base_image,0,255).astype(dtype=np.uint8)
        data[self.background_image > 0] = self.background_image[self.background_image > 0]
        data[self.agent_image > 0] = self.agent_image[self.agent_image > 0]

        return data
    
    def __add__(self,other):
        if isinstance(other, Image):
            self.agent_image[other.agent_image > 0] = other.agent_image[other.agent_image>0]
            return self
        elif other == 0:
            return self
        else:
            return NotImplemented
        
    def __radd__(self, other):
        return self.__add__(other)  # ensures 0 + obj works
        
        

        




class Camera(Sensor):

    # def __init__(self, noise_mean = [0,0], noise_covariance = [0,0], resolution = 1024, reference_range = 30000, reference_area = 1, reference_brightness = 15, shutter_speed = 1 / 250): #reference range -> 1 pixel at specified range
    def __init__(self, 
                 noise_mean = [0,0],
                 noise_covariance = [0,0],
                 resolution = 1024*2,
                 shutter_speed = 1 / 250, 
                 reference_mag = 22,
                 fov = 60,
                 high_res = 1e7,
                 pointing_angles = (0,90)):
        
        """
        Parameters:
        ----------
        noise_mean : list
            Mean of the noise in the image.
        noise_covariance : list
            Covariance of the noise in the image.
        resolution : int
            Resolution of the image.
        shutter_speed : float
            Shutter speed of the camera in Hz.
        """

        # reference_area /= 1000
        self.resolution = resolution
        fov = fov * DEG2RAD
        self.high_res = int(high_res)
        # self.reference_range = reference_range
        # self.reference_area = reference_area # convert to km
        # self.pixel_size = np.sqrt(reference_area) / reference_range
        v_band_magnitude = 3.6e-9 # Approximate V-band magnitude for the Sun in W/m^2, used for apparent magnitude calculation
        self.reference_flux = v_band_magnitude * 10 ** (-reference_mag / 2.5)
        self.shutter_speed = shutter_speed
        self.fov = fov
        self.angular_resolution = fov / resolution
        # self.angular_resolution = resolution / fov  
        # self.resolution_scale = high_res // resolution
        self.az_pointing = pointing_angles[0] * DEG2RAD
        self.el_pointing = pointing_angles[1] * DEG2RAD
        

        super().__init__(noise_mean, noise_covariance)

    def measurement_map(self, state:State, agent:Agent, station: Station):
        return self.generate_image(state,agent,station, crosslines=True)
    

    def generate_image(self, state: State, agent: Agent, station: Station, crosslines = False):
        # Create the background
        image = Image(resolution=self.resolution, timestamp=None, background='stars', scale='rgb')

        # Generate the agent's image (spacecraft)
        self._generate_agent_image(image, state, agent, station)

        
        return image
    

    def _draw_crosshairs(self,image):
        """
        Draws vertical and horizontal crosshairs through the center of the image.
        """
        image_height, image_width = image.base_image.shape[:2]
        center_x = image_width // 2
        center_y = image_height // 2

        # Draw vertical line
        for y in range(image_height):
            image.base_image[y, center_x,:] = (0.0,1.0,0.0)  

        # Draw horizontal line
        for x in range(image_width):
            image.base_image[center_y, x, : ] = (0.0,1.0,0.0)  

    
    def _generate_agent_image(self, image, state, agent: Agent, station: Station):
        # Create base image (initially all zeros or dark)
        # base_image = np.zeros((self.high_res, self.high_res, 3), dtype=np.uint8)
        # base_image = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)

        # Calculate flux, apparent magnitude, areas, etc.
        flux, apparent_mag, areas = station.calculate_light_areas_exposed(state, agent)
        rho, rhodot = station.calculate_range_and_range_rate_from_target(state=state, vectorized=True)
        az_rad, el_rad,enu,enu_transform = station.calculate_azimuth_and_elevation(state=state, enu_frame=True)

        # enu[0] = -enu[0] # reverse the eastward direction

        enu_rhodot = enu_transform @ rhodot
        enu_rho = enu_transform @ rho
        enu_rho[1] *= -1
        enu_rho[0] *= -1

        enu_rhodot[1] *= -1
        enu_rhodot[0] *= -1
        # enu = np.array([1, 0, 1])  # East and Up
        if not areas.size or el_rad < (station.minimum_elevation_angle * DEG2RAD):  # spacecraft not visible
            return None

        # Calculate pixel size and the location on the image
        apparent_pixel_sizes = np.sqrt(areas / (np.linalg.norm(rho)**2))  # in radians
        pixel_sizes = (apparent_pixel_sizes / self.angular_resolution)
        # pixel_sizes_low_res = pixel_sizes / self.resolution_scale

        x = (enu_rho[0] / enu_rho[2]) * self.resolution
        y = (enu_rho[1] / enu_rho[2]) * self.resolution

        x_pixel = int(self.resolution/2 + x)
        y_pixel = int(self.resolution/2 + y)

        # Calculate the total flux, considering the shutter speed
        total_flux = flux / self.shutter_speed
        brightness = total_flux / self.reference_flux * 255

        #Averging
        # Correct size: total full pixels + one for each fractional part
        total_pixels = int(np.sum(np.floor(pixel_sizes))) + np.count_nonzero(pixel_sizes % 1 > 0)
        brightness_map = np.zeros(total_pixels)
        # brightness_area = np.zeros(int(np.ceil(np.sum(pixel_sizes))))

    
        # Define sigmas: major axis (motion) and minor axis (orthogonal)
        # sigma_major = np.linalg.norm(pixel_blur_length) / 2.355  # FWHM to sigma
        # sigma_minor = 0.5  # small value to keep blur narrow across motion

        index = 0
        for i, area in enumerate(pixel_sizes):
            num_pixels = int(np.floor(area))  # Number of whole pixels
            remaining_fraction = area - num_pixels  # Fractional part of pixels

            # Assign the full brightness for the whole pixels
            for _ in range(num_pixels):
                brightness_map[index] = brightness[i]
                index += 1

            # For the fractional part, we can assign the remaining brightness to a single pixel
            if remaining_fraction > 0:
                brightness_map[index] = brightness[i] * remaining_fraction
                index += 1

        # num_pixels
        # Iterate over the faces of the spacecraft
        low_res_area = np.sum(pixel_sizes) 
        low_res_pixels = np.ceil(low_res_area)

        low_res_bright = np.clip(low_res_area * np.sum(brightness_map),0,255)
        
        
        # Assuming each face has a center (x_face, y_face)
        # You need the specific x and y offsets for each face from the spacecraft center (x_pixel, y_pixel)
        
        # For simplicity, we're assuming the faces are square, but you can adjust for non-square shapes (e.g., elliptical or rectangular).
        
        buffer = 3 # two pixel buffer
        # length = int(np.ceil(np.linalg.norm(pixel_blur_length))) + buffer
        
        # angle_deg = np.degrees(angle_rad)  # Optional: ensure 0–360 range

        #Compute the blur
        pixel_blur_vector = (enu_rhodot / np.linalg.norm(enu_rho) * self.shutter_speed) / self.angular_resolution
        # pixel_blur_vector = (enu_rhodot / enu_rho[2]  * self.shutter_speed) / self.angular_resolution
        # blur_lengths, pixel_blur_vector = compute_blur_length(rhodot=rhodot,rho=rho,exposure_time=self.shutter_speed,angular_resolution=self.angular_resolution)
        # pixel_blur_vector = (enu_rhodot) / np.linalg.norm(enu_rhodot) * agent.dt.total_seconds() * self.angular_resolution
        # pixel_blur_vector = (enu_rhodot  * 10000.0 / enu_rho[2] )  * self.angular_resolution
        # pixel_blur_vector = [100,0,0]

        # apply_motion_blur(image,(x_pixel,y_pixel),low_res_pixels,pixel_blur_vector,low_res_bright)
        apply_motion_streak(image,(x_pixel,y_pixel),low_res_pixels,pixel_blur_vector,low_res_bright)
                                                    
        # image.agent_image[y_pixel - half_size:y_pixel + half_size, x_pixel - half_size:x_pixel + half_size] = [low_res_bright,low_res_bright,low_res_bright]




        # image.agent_image = apply_directional_blur(image.agent_image, angle_deg, sigma_major, sigma_minor)
        # low_res_image = cv2.resize(base_image, (self.resolution, self.resolution), interpolation=cv2.INTER_LINEAR)
        # image.agent_image = gaussian_filter(image.agent_image, sigma=1.0).astype(dtype=np.uint8)

        return True



def apply_motion_blur(image, center, pixel_area, pixel_blur_vector, brightness):
    import numpy as np

    x_center, y_center = center
    dx, dy, _ = pixel_blur_vector

    # Compute blur length from motion vector
    blur_length = int(np.ceil(np.sqrt(dx**2 + dy**2)))
    blur_length = max(blur_length, 1)

    # Estimate width from pixel area
    # half_width = int(np.ceil(np.sqrt(pixel_area / blur_length)))
    # width = 2 * half_width + 1

    # Create patch grid (centered)
    # x = np.arange(-blur_length, blur_length + 1)
    # y = np.arange(-half_width, half_width + 1)
    # X, Y = np.meshgrid(x, y)

    # Rotate to align with motion direction
    angle_rad = np.arctan2(dy, dx)
    # Define blur length and perpendicular thickness (in pixels)
    streak_half_length = int(np.ceil(np.sqrt(dx**2 + dy**2)))
    streak_half_width = int(np.ceil(np.sqrt(pixel_area / (2 * streak_half_length + 1))))

    # Create meshgrid around the center (adjust size based on motion and thickness)
    size_y = streak_half_width * 2 + 1
    size_x = streak_half_length * 2 + 1
    Y, X = np.meshgrid(np.arange(-size_y // 2, size_y // 2 + 1),
                    np.arange(-size_x // 2, size_x // 2 + 1))

    # Rotate coordinates into motion-aligned frame
    X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
    Y_rot = -X * np.sin(angle_rad) + Y * np.cos(angle_rad)

    # Create binary mask: full brightness inside streak width
    mask = np.abs(Y_rot) <= streak_half_width

    # Fill brightness values where mask is True
    patch = np.zeros((*mask.shape, 3), dtype=np.uint8)
    brightness_val = np.array(brightness if isinstance(brightness, (list, tuple, np.ndarray)) else [brightness]*3)

    patch[mask] = brightness_val

    # Compute destination bounds in image
    x_start = x_center - streak_half_length
    y_start = y_center - streak_half_width
    x_end = x_start + patch.shape[1]
    y_end = y_start + patch.shape[0]

    # Clip bounds to image size
    h, w = image.agent_image.shape[:2]
    x_start_clip, x_end_clip = max(0, x_start), min(w, x_end)
    y_start_clip, y_end_clip = max(0, y_start), min(h, y_end)

    # Patch bounds
    patch_x1 = x_start_clip - x_start
    patch_x2 = patch_x1 + (x_end_clip - x_start_clip)
    patch_y1 = y_start_clip - y_start
    patch_y2 = patch_y1 + (y_end_clip - y_start_clip)

    # Apply to image
    image.agent_image[y_start_clip:y_end_clip, x_start_clip:x_end_clip] = \
        patch[patch_y1:patch_y2, patch_x1:patch_x2]



def apply_motion_streak(image, center, pixel_area, pixel_blur_vector, brightness):
    """
    Draws a sharp, constant-brightness streak in the direction of motion.

    Parameters:
        image            : np.ndarray (H, W, 3)
        center           : tuple (x, y)
        pixel_area       : float
        pixel_blur_vector: np.ndarray (dx, dy, dz) - motion direction
        brightness       : float or [R, G, B]
    """
    x_center, y_center = center
    dx, dy, _ = pixel_blur_vector
    length = int(np.ceil(np.sqrt(dx**2 + dy**2)))

    if length == 0:
        length = 1  # Avoid zero-length line

    # Normalize direction
    dx /= length
    dy /= length

    # Define streak width based on pixel area
    width = int(np.sqrt(pixel_area / length))
    if width < 1:
        width = 1

    for i in range(length):
        x = int(x_center + i * dx)
        y = int(y_center + i * dy)

        # Draw a square around the line point for "thickness"
        for wx in range(-width // 2, width // 2 + 1):
            for wy in range(-width // 2, width // 2 + 1):
                xi = x + wx
                yi = y + wy
                if 0 <= xi < image.agent_image.shape[1] and 0 <= yi < image.agent_image.shape[0]:
                    image.agent_image[yi, xi] = brightness

    return True



def draw_circle(image, center, radius, color):
    """
    Draw a filled circle onto a NumPy image array.

    Parameters:
        image : np.ndarray (H, W, 3) - The image to draw on.
        center : tuple (x, y) - The center of the circle.
        radius : int - Radius in pixels.
        color : list or tuple - RGB color triplet.
    """
    y_center, x_center = center
    H, W = image.shape[:2]

    y, x = np.ogrid[:H, :W]
    mask = (x_center - x)**2 + (y_center - y)**2 <= radius**2
    image[mask] = color

def compute_blur_length(rhodot, rho, exposure_time, angular_resolution):
    """
    Computes motion blur length in pixels from relative velocity (rhodot).

    Parameters:
        rhodot : (N, 3) np.ndarray - Relative velocity vectors (in m/s)
        rho    : (N, 3) np.ndarray - Line-of-sight vectors (in m)
        exposure_time : float - Camera exposure time (in seconds)
        angular_resolution : float - Angular resolution (in radians per pixel)

    Returns:
        blur_lengths : (N,) np.ndarray - Blur lengths in pixels for each object
        pixel_vectors: (N, 2) np.ndarray - Motion vector in pixel space (dx, dy)
    """
    # Normalize line-of-sight vectors to get unit direction
    rhodot = rhodot[:,np.newaxis]
    rho = rho[:,np.newaxis]
    rho_unit = rho / np.linalg.norm(rho, axis=1, keepdims=True)

    # Project rhodot onto plane orthogonal to rho (perpendicular motion)
    radial_component = np.sum(rhodot * rho_unit, axis=1, keepdims=True) * rho_unit
    perp_velocity = rhodot - radial_component  # Tangential motion causes streaks

    # Angular rate (radians/sec) = perpendicular velocity / distance
    rho_norm = np.linalg.norm(rho, axis=1, keepdims=True)
    angular_rate = perp_velocity / rho_norm  # radians/sec in x, y, z (approx)

    # Total angular displacement over exposure time (in radians)
    angular_disp = angular_rate * exposure_time

    # Convert angular displacement to pixel displacement
    pixel_vectors = angular_disp / angular_resolution  # Drop z (depth)
    blur_lengths = np.linalg.norm(pixel_vectors, axis=1)

    return blur_lengths, pixel_vectors
