#!/usr/bin/python2.7
"""
File: ARNOLD_RK4.PY
Author: Uyiosa Philip Amadasun

Brief:
    Implements the Runge-Kutta 4th order approximation method. This script is called in the arnold_logistic_coverage.py script.
This script provides an implementation of the Runge-Kutta 4th order (RK4) method for approximating 
the solution of a system of differential equations. It is designed to calculate the values of DS coordinates
('x', 'y', 'z'), one of which is mapped as an orientation in the calculation of the robot's positions ('X' and 'Y')
using the RK4 method.
"""

import rospy
import math
from rospy.numpy_msg import numpy_msg
import numpy as np

def arnold_rk4(A,B,C,v,x,y,z,X,Y,dt,index):
    """ Runge Kutta 4th Order method for the Arnold and Dynamical Systems.

    Args:
        A (float): Parameter 'A'<----\
        B (float): Parameter 'B'<-------- Arnold system parameters
        C (float): Parameter 'C'<----/
        v (float): Velocity 'v'.
        x (float): Initial angle 'x'<----\
        y (float): Initial angle 'y'<-------- DS coordinates
        z (float): Initial angle 'z'<----/
        X (float): Initial position 'X'.
        Y (float): Initial position 'Y'.
        dt (float): Time step for integration.
        index (int): The index used to select the appropriate angle in the 'dynam_angle' list.

    Returns:
        list: A list containing the updated DS coordinates 'x', 'y', and 'z', as well as the updated positions 'X' and 'Y'.
    """

    K1_x=float(A)*math.sin(z)+C*math.cos(y)
    K1_y=float(B)*math.sin(x)+A*math.cos(z)
    K1_z=float(C)*math.sin(y)+B*math.cos(x)
    
    K2_x = float(A)*math.sin(z+0.5*dt*K1_z)+C*math.cos(y+0.5*dt*K1_y)
    K2_y = float(B)*math.sin(x+0.5*dt*K1_x)+A*math.cos(z+0.5*dt*K1_z)
    K2_z = float(C)*math.sin(y+0.5*dt*K1_y)+B*math.cos(x+0.5*dt*K1_x)
    
    K3_x =  float(A)*math.sin(z+0.5*dt*K2_z)+C*math.cos(y+0.5*dt*K2_y)
    K3_y =  float(B)*math.sin(x+0.5*dt*K2_x)+A*math.cos(z+0.5*dt*K2_z)
    K3_z =  float(C)*math.sin(y+0.5*dt*K2_y)+B*math.cos(x+0.5*dt*K2_x)

    K4_x = float(A)*math.sin(z+dt*K3_z)+C*math.cos(y+dt*K3_y)
    K4_y = float(B)*math.sin(x+dt*K3_x)+A*math.cos(z+dt*K3_z)
    K4_z = float(C)*math.sin(y+dt*K3_y)+B*math.cos(x+dt*K3_x)

    x = x + (dt/float(6))*(K1_x + 2*K2_x + 2*K3_x + K4_x)
    y = y + (dt/float(6))*(K1_y + 2*K2_y + 2*K3_y + K4_y)
    z = z + (dt/float(6))*(K1_z + 2*K2_z + 2*K3_z + K4_z)


    dynam_angle = [x,y,z]
    #print("index: ",index)
    X = X + float(dt)*v*math.cos(dynam_angle[index])
    Y = Y + float(dt)*v*math.sin(dynam_angle[index])

    return [x,y,z,X,Y]
