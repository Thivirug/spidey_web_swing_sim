import numpy as np
from typing import Tuple

from .physics import t_derivatives

class RK4:
    """
        4th order runge-kutta solver
        simulated for `T` time in `dt` timesteps
    """

    def __init__(self, T, dt):
        if T <= 0:
            raise ValueError("T must be greater than 0.")

        if dt <= 0:
            raise ValueError("dt must be greater than 0.")

        if dt > T:
            raise ValueError("dt should not be greater than T.")
        
        self.T = T
        self.dt = dt

    def step(self, state: np.ndarray, L) -> np.ndarray:
        """
            Single RK4 step
        """

        k1 = t_derivatives(state, L)
        k2 = t_derivatives(
            state + 0.5*self.dt*k1, 
            L
        )
        k3 = t_derivatives(
            state + 0.5*self.dt*k2, 
            L
        )
        k4 = t_derivatives(
            state + self.dt*k3, 
            L
        )

        new_state = state + self.dt/6.0*(
            k1 + 2*(k2+k3) + k4
        )

        return new_state

    def simulate(self, theta0, omega0, L) -> Tuple[np.ndarray, np.ndarray]:
        """
            Iterate RK4 step for req simulation time
            starting at angle theta0 and ang vel omega0
        """

        # times array of simulation intervals
        times = np.arange(0, self.T+self.dt, self.dt)

        # init states vectors for each sim interval
        states = np.zeros(shape=(len(times), 2))

        states[0] = np.array([theta0, omega0])

        for i in range(len(times)-1):
            # step
            states[i+1] = self.step(
                states[i], L
            )

        # thetas = states[:, 0]
        # omegas = states[:, 1]

        return times, states



