# motion phy6 defs

import numpy as np

# grav accel
g = 9.81

def t_derivatives(state: np.ndarray, L) -> np.ndarray:
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