import numpy as np
from scipy.optimize import newton

def classical2cart(a,e,i,arg,raan,mu,nu = None,M = None):  
    
    if nu is None and M is None:
        raise ValueError("A true anomaly (nu) or mean anomaly (M) must be specified")

    if nu is None:
        n = np.sqrt(mu/(a**3))
        # M = M0 + n*tof 
        # M = M % (2*np.pi)
        nu,E = mean2true(M, e)
        # nu = nu % (2*np.pi)
    
    
    p = a*(1-e**2)
    r = p/(1+e*np.cos(nu))
    
    R = np.array([r*np.cos(nu),r*np.sin(nu),0])
    V = np.array([-np.sqrt(mu/p)*np.sin(nu), np.sqrt(mu/p) * (e+np.cos(nu)), 0])
    

    DCM = np.array([
        [
            np.cos(raan) * np.cos(arg) - np.sin(raan) * np.sin(arg) * np.cos(i),
            -np.cos(raan) * np.sin(arg) - np.sin(raan) * np.cos(arg) * np.cos(i),
            np.sin(raan) * np.sin(i)
        ],
        [
            np.sin(raan) * np.cos(arg) + np.cos(raan) * np.sin(arg) * np.cos(i),
            -np.sin(raan) * np.sin(arg) + np.cos(raan) * np.cos(arg) * np.cos(i),
            -np.cos(raan) * np.sin(i)
        ],
        [
            np.sin(arg) * np.sin(i),
            np.cos(arg) * np.sin(i),
            np.cos(i)
        ]
    ])
    
    
    xyz = DCM @ R

    vxyz = DCM @ V

    return np.hstack((xyz,vxyz))

def cart2classical(state,mu,nu_bool = True):
    
    
    x, y, z, vx, vy, vz = state
    
    r = np.sqrt(x**2+y**2+z**2)
    
    v = np.sqrt(vx**2+vy**2+vz**2)
    
    z_hat = [0,0,1]

    h = np.cross([x, y, z], [vx, vy, vz])
    a_vec = np.cross(z_hat,h) / np.linalg.norm(np.cross(z_hat,h))
    e_vector = 1/mu * ((v**2 - mu/r) * np.array([x, y, z]) - np.dot([x, y, z], [vx, vy, vz]) * np.array([vx, vy, vz]))
    
    ecc = np.linalg.norm(e_vector)
    
    
    p = np.linalg.norm(h)**2 / mu
    
    sma = p / (1 - ecc**2)
    
    inc = np.arccos(h[2] / np.linalg.norm(h))
    
    #node line direction
    ahat=(np.cross(z_hat,h))/(np.linalg.norm(np.cross(z_hat,h)))

    #find Omega
    xhat=[1, 0, 0]
    Omega=np.arccos(np.dot(xhat,ahat))

    #Quad check
    if ahat[1] < 0 : Omega=2*np.pi-Omega

    #find w
    arg=np.arccos((np.dot(ahat,e_vector))/np.linalg.norm(e_vector))

    #quad check

    if e_vector[2] <0:
        arg=2*np.pi-arg


    # nu = np.arctan2(z / (r * np.sin(i)), (x * np.cos(Omega) + y * np.sin(Omega)) / r)
    nu = np.arccos(np.dot(e_vector,[x,y,z])/(ecc*r))

    

    # nu = nu - arg
    if np.dot([x,y,z],[vx,vy,vz]) < 0: nu = 2*np.pi - nu 

    if nu_bool:
        return sma, ecc, inc, arg, Omega, nu
    else:
        E = np.arctan2(np.sqrt(1 - ecc**2) * np.sin(nu), ecc + np.cos(nu))
        M = E - ecc * np.sin(E)
        return sma, ecc, inc, arg, Omega, M
    

        # return a, e, np.rad2deg(i), np.rad2deg(arg), np.rad2deg(Omega), np.rad2deg(nu)
        
    
def mean2true(mean,eccentricity):

    eccentric_anomaly = newton(lambda E: E - eccentricity * np.sin(E) - mean, x0=mean)
    true_anomaly = 2*np.arctan((np.sqrt((1+eccentricity)/(1-eccentricity))*np.tan(eccentric_anomaly/2)))

    true_anomaly = true_anomaly % (2*np.pi)
    return true_anomaly, eccentric_anomaly

def true2mean(true_anomaly,eccentricity):
     E = 2 * np.arctan(np.sqrt((1-eccentricity))/np.sqrt(1 + eccentricity)* np.tan(true_anomaly/2))
     M = E - eccentricity*np.sin(E)

     
     return M