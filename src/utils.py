from typing import Tuple
import numpy as np

def angle_to_pos_conv(anchor: Tuple, theta, L) -> Tuple:
    """
        Convert angular pos values to x,y coords given an anchor pos
    """

    x_a, y_a = anchor

    x = x_a + L*np.sin(theta)
    y = y_a - L*np.cos(theta)

    return x, y

def get_vel_comps(omega, theta, L):
    """
       Get velocity components
    """

    v_x = L*omega*np.cos(theta)
    v_y = L*omega*np.sin(theta)

    return v_x, v_y

