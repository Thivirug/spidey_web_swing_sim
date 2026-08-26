# motion phy6 defs

import numpy as np

# grav accel
g = 9.81

def t_derivatives(state, L) -> np.ndarray:
    """
        calc dtheta/dt & domega/dt

        state: 
            - theta: Ang displacement in rad
            - omega: Ang velocity
        L: Web length
    """

    # vaidate args
    theta, omega = state

    d_theta = omega
    d_omega = -(g / L) * np.sin(theta)

    return np.array([d_theta, d_omega])