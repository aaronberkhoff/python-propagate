import numpy as np
class Planet:

    def __init__(self, name : str, 
                 radius: float, 
                 J2: float, 
                 J3: float, 
                 spice_id:int, 
                 mu: float, 
                 angular_velocity:float, 
                 flattening: float):
        self._radius = radius
        self._mu = mu
        self._J2 = J2
        self._J3 = J3
        self._name = name
        self._spice_id = spice_id
        self._angular_velocity = angular_velocity
        self._flattening = flattening


    @property
    def radius(self):
        return self._radius
    
    @property
    def mu(self):
        return self._mu
    
    @property
    def J2(self):
        return self._J2
    
    @property
    def J3(self):
        return self._J3
    
    @property
    def spice_id(self):
        return self._spice_id
    
    @property
    def name(self):
        return self._name
    
    @property
    def angular_velocity(self):
        return self._angular_velocity
    
    @property
    def flattening(self):
        return self._flattening

    @property
    def eccentricity(self):
        return 2 * self._flattening - self._flattening**2




class Earth(Planet):

    def __init__(self, name     = 'Earth',
               radius   = 6378.1363,
               J2       = 0.0010826267,
               J3       = -0.0000025327,
               spice_id = 399,
               mu       = 398600.4415,
               angular_velocity = 7.29211585530066e-5,
               flattening = 1 / 298.257223563):
        
        super().__init__(name, radius, J2, J3, spice_id, mu, angular_velocity,flattening=flattening)

    def atmosphere_model(self,radius_spacecraft):

        altitude = radius_spacecraft - self.radius
            
        if altitude > 1000:
            rho0 = 3.019e-15
            h0 = 1000
            H = 268
        elif altitude > 900:
            rho0 = 5.245e-15
            h0 = 900
            H = 181.05
        elif altitude > 800:
            rho0 = 1.170e-14
            h0 = 800
            H = 124.64
        elif altitude > 700:
            rho0 = 3.614e-14
            h0 = 700
            H = 88.667
        elif altitude > 600:
            rho0 = 1.454e-13
            h0 = 600
            H = 71.835
        elif altitude > 500:
            rho0 = 6.967e-13
            h0 = 500
            H = 63.822
        elif altitude > 450:
            rho0 = 1.585e-12
            h0 = 450
            H = 60.828
        elif altitude > 400:
            rho0 = 3.725e-12
            h0 = 400
            H = 58.515
        elif altitude > 350:
            rho0 = 9.518e-12
            h0 = 350
            H = 53.298
        elif altitude > 300:
            rho0 = 2.418e-11
            h0 = 300
            H = 53.628
        elif altitude > 250:
            rho0 = 7.248e-11
            h0 = 250
            H = 45.546
        elif altitude > 200:
            rho0 = 2.789e-10
            h0 = 200
            H = 37.105
        elif altitude > 180:
            rho0 = 5.464e-10
            h0 = 180
            H = 29.740
        elif altitude > 150:
            rho0 = 2.070e-9
            h0 = 150
            H = 22.523
        elif altitude > 140:
            rho0 = 3.845e-9
            h0 = 140
            H = 16.149
        elif altitude > 130:
            rho0 = 8.484e-9
            h0 = 130
            H = 12.636
        elif altitude > 120:
            rho0 = 2.438e-8
            h0 = 120
            H = 9.473
        elif altitude > 110:
            rho0 = 9.661e-8
            h0 = 110
            H = 7.263
        elif altitude > 100:
            rho0 = 5.297e-7
            h0 = 100
            H = 5.877
        elif altitude > 90:
            rho0 = 3.396e-6
            h0 = 90
            H = 5.382
        elif altitude > 80:
            rho0 = 1.905e-5
            h0 = 80
            H = 5.799
        elif altitude > 70:
            rho0 = 8.770e-5
            h0 = 70
            H = 6.549
        elif altitude > 60:
            rho0 = 3.206e-4
            h0 = 60
            H = 7.714
        elif altitude > 50:
            rho0 = 1.057e-3
            h0 = 50
            H = 8.382
        elif altitude > 40:
            rho0 = 3.972e-3
            h0 = 40
            H = 7.554
        elif altitude > 30:
            rho0 = 1.774e-2
            h0 = 30
            H = 6.682
        elif altitude > 25:
            rho0 = 3.899e-2
            h0 = 25
            H = 6.349
        else:
            rho0 = 1.225
            h0 = 0
            H = 7.249
            
        density = rho0 * np.exp( - (altitude - h0) / H) * 1000**3

        return density
    
    def set_atmosphere_model(self,model):
        self.atmosphere_model = model
        pass

  

