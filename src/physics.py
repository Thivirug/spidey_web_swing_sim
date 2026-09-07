# motion phy6 defs

import numpy as np
from typing import Tuple, Optional

# grav accel
g = 9.81

def t_derivatives(state: np.ndarray, L) -> np.ndarray:    # NOTE: During attached state
    """

        Take the current state vector [theta, omega]
        & calc dtheta/dt & domega/dt

        state: 
            - theta: Ang displacement in rad
            - omega: Ang velocity
        L: Web length
    """

    # vaidate state and length
    if not np.isfinite(state).all():
        raise ValueError("Invalid numerical state inputs")

    if L <=0:
        raise ValueError("Invalid web length")
    
    theta, omega = state[0], state[1]


    d_theta = omega
    d_omega = -(g / L) * np.sin(theta)

    return np.array([d_theta, d_omega])

def energy(m, L, state: np.ndarray):
    """
        Calc total mech energy at a given state and time
        m: mass
        L: length of web'
        state: Contains theta, omega
    """

    if L <=0 or m<=0:
            raise ValueError("Invalid params passed")

    theta, omega = state[0], state[1]

    return 0.5*m* np.pow(L*omega, 2) + m*g*L*(1 - np.cos(theta))

def t_derivatives_released(state: np.ndarray, drag_params: Optional[Tuple[float, float, float, float]]=None):     # NOTE: during released state
    """
        Calculate derivatives during released/projectile motion.

        state:
            - x: horizontal displacement
            - y: vertical displacement
            - vx: horizontal velocity
            - vy: vertical velocity

        drag_params:
            Optional tuple (m, rho, Cd, A)
            - m: mass (kg)
            - rho: air density (kg/m^3)
            - Cd: dimensionless drag coefficient
            - A: reference/cross-sectional area (m^2)

        Returns:
            [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    # validate state vec
    if not np.isfinite(state).all():
            raise ValueError("Invalid numerical state inputs")

    x, y, vx, vy = state

    speed = np.sqrt(vx**2 + vy**2)

    dx = vx
    dy = vy
    dv_x = 0.0
    dv_y = -g

    if drag_params is not None:
        m, rho, Cd, A = drag_params

        if m <= 0:
            raise ValueError("Mass must be greater than zero")

        if rho < 0 or Cd < 0 or A < 0:
            raise ValueError("Invalid drag parameters")

        drag_factor = 0.5 * rho * Cd * A

        dv_x -= (drag_factor / m) * speed * vx
        dv_y -= (drag_factor / m) * speed * vy

    return np.array([dx, dy, dv_x, dv_y])
