import numpy as np
from typing import Tuple

from .physics import t_derivatives, t_derivatives_released

class RK4:
    """
        4th order runge-kutta solver
        simulated for `T` time in `dt` timesteps
    """

    def __init__(self, dt):
        # if T <= 0:
        #     raise ValueError("T must be greater than 0.")

        if dt <= 0:
            raise ValueError("dt must be greater than 0.")

        # if dt > T:
        #     raise ValueError("dt should not be greater than T.")
        
        # self.T = T
        self.dt = dt

    def step(self, state: np.ndarray, L) -> np.ndarray:
        """
            Single RK4 step for pendulum(spidey) attached motion
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

    def step_released(self, state: np.ndarray, drag_params=None) -> np.ndarray:
        """
            one RK4 step for projectile/released motion.

            state = [x, y, vx, vy]
        """

        k1 = t_derivatives_released(state, drag_params)
        k2 = t_derivatives_released(
            state + 0.5*self.dt*k1,
            drag_params
        )
        k3 = t_derivatives_released(
            state + 0.5*self.dt*k2,
            drag_params
        )
        k4 = t_derivatives_released(
            state + self.dt*k3,
            drag_params
        )

        return state + self.dt/6.0*(
            k1 + 2*k2 + 2*k3 + k4
        )


    # following sim methods are isolated testing methods to run attached or projectile motion independently

    def simulate(self, theta0, omega0, L, T) -> Tuple[np.ndarray, np.ndarray]:
        """
            Iterate RK4 step for req simulation time T
            starting at angle theta0 and ang vel omega0
            (attached motion)
        """
        if T <= 0:
            raise ValueError(
                "simulataion duration must be greater than 0."
            )
        
        # times array of simulation intervals
        times = np.arange(0, T+self.dt, self.dt)

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

    def simulate_released(self, x0, y0, vx0, vy0, T, y_ground=0.0, drag_params=None) -> Tuple[np.ndarray, np.ndarray]:
        """
            Simulate projectile motion after web release.

            x0: initial horizontal position
            y0: initial vertical position
            vx0: initial horizontal velocity
            vy0: initial vertical velocity
            T: simulation time
            y_ground: ground height
            drag_params: optional drag parameters

            return times and states before spidey hits the ground if it happens during sim time
        """
        if T <= 0:
            raise ValueError(
                "simulataion duration must be greater than 0."
            )
        
        times = np.arange(0, T + self.dt, self.dt)
        states = np.zeros(
            shape=(len(times), 4)
        )

        states[0] = np.array([
            x0, y0, vx0, vy0
        ])

        last_index = len(times)-1 # to store the last index before spidey hits the ground

        for i in range(len(times) - 1):

            states[i + 1] = self.step_released(
                states[i],
                drag_params
            )

            y = states[i + 1, 1]
            vy = states[i + 1, 3]

            if y <= y_ground and vy < 0: # hit ground
                last_index = i + 1
                break

        return (
            times[:last_index + 1],
            states[:last_index + 1]
        )

