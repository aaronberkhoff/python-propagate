import numpy as np

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180

def rad2deg(radians:float):
    return radians * RAD2DEG

def deg2rad(radians:float):
    return radians * DEG2RAD