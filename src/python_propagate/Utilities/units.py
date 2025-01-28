import numpy as np

RAD2DEG = 180/np.pi
DEG2RAD = np.pi/180
DEG2ARC = 3600
ARC2DEG = 1/ 3600
RAD2ARC = RAD2DEG * DEG2ARC
ARC2RAD = ARC2DEG * DEG2RAD

def rad2deg(radians:float):
    return radians * RAD2DEG

def deg2rad(radians:float):
    return radians * DEG2RAD