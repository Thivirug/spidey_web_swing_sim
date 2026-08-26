# motion phy6 defs

import numpy as np

# grav accel
g = 9.81

def t_derivatives(theta, omega, L):
    """
        calc dtheta/dt & domega/dt

        theta: Ang displacement in rad
        omega: Ang velocity
        L: Web length
    """

    # vaidate args


    d_theta = omega
    d_omega = -(g / L) * np.sin(theta)

    return d_theta, d_omega