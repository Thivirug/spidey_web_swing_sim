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

def attached_to_released_state(
    state: np.ndarray,
    anchor,
    L
) -> np.ndarray:
    """
    Convert attached state [theta, omega]
    into released state [x, y, vx, vy].
    """

    theta, omega = state

    x, y = angle_to_pos_conv(
        anchor,
        theta,
        L
    )

    vx, vy = get_vel_comps(
        omega,
        theta,
        L
    )

    return np.array([
        x,
        y,
        vx,
        vy
    ])

# TODO: need the inverse of above func for released -> attached transition

def interpolate_landing(t_prev, t_curr, state_prev, state_curr, y_ground):
    """
        Interpolate landing time and state between two projectile states.
    """
    
    y1 = state_prev[1]
    y2 = state_curr[1]

    if np.isclose(y2, y1):
        raise ValueError(
            "Cannot interpolate landing when y values are equal."
        )

    alpha = (
        (y_ground - y1) / (y2-y1)
    )

    land_t = t_prev + alpha * (t_curr - t_prev)
    land_state = state_prev + alpha * (state_curr - state_prev)

    land_state[1] = y_ground # set this back to avoid floating point mismatches

    return land_t, land_state