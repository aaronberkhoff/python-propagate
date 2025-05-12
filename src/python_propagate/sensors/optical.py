
import numpy as np
import random 

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

    def __init__(self,resolution = (1024,1024), background = 'dark', scale = 'grey', timestamp = None, n_stars = 1000, high_res = 8000):

        self.resolution = resolution
        self.background = background
        self.scale = scale
        self.timestamp = timestamp
        self.n_stars = n_stars
        self.base_image = np.zeros(resolution + (3,),dtype=int)
        self.agent_image = np.zeros(resolution + (3,),dtype=int)
        self.background_image = np.zeros(resolution + (3,),dtype=int)
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

            star_x = np.random.randint(0, self.resolution[0], num_stars)
            star_y = np.random.randint(0, self.resolution[1], num_stars)

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
                    if 0 <= xi < self.resolution[0] and 0 <= yi < self.resolution[1]:
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

        if self.agent_image is not None and np.any(self.agent_image != -99):
            data = np.clip(self.base_image,0,255).astype(dtype=np.int16)
            data[self.background_image > 0] = self.background_image[self.background_image > 0]
            data[self.agent_image > 0] = self.agent_image[self.agent_image > 0]
            return data
        else:
            return np.full(self.resolution + (3,), -99, dtype=np.int16)
    
    def __add__(self, other):
        if isinstance(other, Image):
            np.copyto(self.agent_image, other.agent_image, where=(other.agent_image > 0))       
        elif other == 0:
            return self
        return NotImplemented
        
    def __radd__(self, other):
        return self.__add__(other)  # ensures 0 + obj works
        

class Camera(Sensor):

    def __init__(self, 
                 noise_mean = [0,0],
                 noise_covariance = [0,0],
                 resolution = (1024,1024), 
                 reference_mag = 20,
                 fov = 60,
                 pointing_angles = (0,0),
                 background = 'stars'):
        
        """
        """

        self.resolution = tuple(resolution)
        fov = fov * DEG2RAD
        v_band_magnitude = 3.6e-9 # Approximate V-band magnitude for the Sun in W/m^2, used for apparent magnitude calculation
        self.reference_flux = v_band_magnitude * 10 ** (-reference_mag / 2.5)
        self.fov = fov
        self.angular_resolution =  resolution[0] / fov
        daz_rad, del_rad = pointing_angles[0] * DEG2RAD, pointing_angles[1] * DEG2RAD

        az_pointing_transform = np.array([[np.cos(daz_rad),-np.sin(daz_rad),0],
                                          [np.sin(daz_rad),np.cos(daz_rad),0],
                                          [0,0,1]])
        
        el_pointing_transform = np.array([[1,0,0],
                                          [0, np.cos(del_rad), -np.sin(del_rad)],
                                          [0, np.sin(del_rad), np.cos(del_rad)]])
        
        self.pointing_rotation = az_pointing_transform @ el_pointing_transform
        self.background = background
        

        super().__init__(noise_mean, noise_covariance)

    def measurement_map(self, state:State, agent:Agent, station: Station):
        return self.generate_image(state,agent,station, crosslines=True)
    

    def generate_image(self, state: State, agent: Agent, station: Station, crosslines = False):
        # Create the background
        image = Image(resolution=self.resolution, timestamp=None, background=self.background, scale='rgb')

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

        rhodot_enu = enu_transform @ rhodot
        rho_enu = enu_transform @ rho
 
        # enu = np.array([1, 0, 1])  # East and Up
        if not areas.size or el_rad < (station.minimum_elevation_angle * DEG2RAD):  # spacecraft not visible
            image.agent_image = np.full(self.resolution + (3,), -99, dtype=np.int16)
            return False

        # Calculate pixel size and the location on the image
        apparent_pixel_sizes = np.sqrt(areas / (np.linalg.norm(rho)**2))  # in radians
        pixel_sizes = (apparent_pixel_sizes * self.angular_resolution)


        x_pixel, y_pixel = self._az_el_to_image_coords(rho_enu=rho_enu)

        if not self._check_pixel_within_bounds(x_pixel=x_pixel,y_pixel=y_pixel):
            print('WARNING: Object is outside fov')
            image.agent_image = np.full(self.resolution + (3,), -99, dtype=np.int16)
            return False

        # Calculate the total flux, considering the shutter speed
        total_flux = flux * agent.dt.total_seconds()
        brightness = total_flux / self.reference_flux * 255

        total_brightness = np.clip(np.sum(brightness * pixel_sizes),0,255)
        total_pixels = np.ceil(np.sum(pixel_sizes))
        # total_pixels = 100

    
        pixel_blur_vector = self._blur_vector(rho_enu=rho_enu,rhodot_enu=rhodot_enu,dt = agent.dt.total_seconds())

        # apply_motion_streak(image,(x_pixel,y_pixel),total_pixels,pixel_blur_vector,total_brightness)
        self.apply_motion_blur(image.agent_image,(x_pixel,y_pixel),total_pixels,pixel_blur_vector,total_brightness)
                                                    
        return True

    def _az_el_to_image_coords(self,rho_enu):


        width, height = self.resolution

        rho_enu = self.pointing_rotation @ rho_enu

        # ENU to camera rotation (you can modify this as needed)
        R = np.array([
            [-1,  0,  0],
            [0,  -1, 0],
            [0,  0,  1]
        ])

        cam_dir = R @ rho_enu

        
        cx, cy = width / 2, height / 2

        x_pixel = self.angular_resolution * (cam_dir[0] / cam_dir[2]) + cx
        y_pixel = self.angular_resolution * (cam_dir[1] / cam_dir[2]) + cy


        return int(x_pixel), int(y_pixel)
    
    def _blur_vector(self,rho_enu,rhodot_enu,dt):

        rho_enu = self.pointing_rotation @ rho_enu
        rhodot_enu = self.pointing_rotation @ rhodot_enu

        drho = rhodot_enu * dt
        
        # ENU to camera rotation (you can modify this as needed)
        R = np.array([
            [-1,  0,  0],
            [0,  -1, 0],
            [0,  0,  1]
        ])

        drho = R @ drho
        cam_dir = R @ rho_enu


        dx_pixel = int((self.angular_resolution * (drho[0] / cam_dir[2]))) #TODO make sure this approximation is good rx/rz
        dy_pixel = int((self.angular_resolution * (drho[1] / cam_dir[2]))) 
        dz_pixel = int((self.angular_resolution * (drho[2] / cam_dir[2])))
        
        pixel_blur_vector = np.array([dx_pixel,dy_pixel,dz_pixel])

        return pixel_blur_vector 
    
    
    def _check_pixel_within_bounds(self,x_pixel,y_pixel):

            if x_pixel > self.resolution[0] or y_pixel > self.resolution[1] or x_pixel < 0 or y_pixel < 0:
                return False
            else: 
                return True

    def apply_motion_blur(self,image,center,pixel_area,pixel_blur_vector, brightness):

        x_center, y_center = center
        dx, dy, dz = pixel_blur_vector

        half_area = int(np.sqrt(pixel_area))
        length = int(np.sqrt(dx**2 + dy**2))
        
        dx /= length
        dy /= length
        for i in range(length):

            x = int(dx * i) + x_center
            y = int(dy * i) + y_center   

            if self._check_pixel_within_bounds(x,y):
                image[y - half_area: y + half_area, x - half_area:x + half_area] = [brightness,brightness,brightness]
            else:
                pass
                

    


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


