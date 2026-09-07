# NOTE: use ursina

import numpy as np

from .solver import RK4
from .utils import attached_to_released_state

# phases
ATTACHED = "attached"
RELEASED = "released"

def simv1(
    theta0,
    omega0,
    L,
    anchor,
    release_time,
    T,
    dt,
    y_ground=0.0,
    drag_params=None
):

    """
        Simulate attached swing until release_time,
        then projectile motion until landing or T.
    """

    solver = RK4(dt)
    t = 0.0
    mode = ATTACHED # starting from attached

    # starting attached state
    state = np.array([
        theta0,
        omega0
    ], dtype=float)

    times = []
    modes = []
    attached_states = []
    released_states = []

    while t<=T:
        times.append(t)
        modes.append(mode)

        if mode ==ATTACHED:
            attached_states.append(
                (t, state.copy())
            )

            if t>= release_time:
                # release
                state = attached_to_released_state(state, anchor, L)

                mode = RELEASED

            else:
                # go to next attached state
                state = solver.step(state, L)

        if mode==RELEASED:
            released_states.append(
                (t, state.copy())
            )

            y, vy = state[1], state[3]

            if y <= y_ground and vy < 0: # ground hit
                break

            # go to next released state
            state = solver.step_released(state, drag_params)

        t += dt

    return (
        np.array(times),
        modes,
        attached_states,
        released_states
    )

if __name__ == '__main__':
    times, modes, attached, released = simv1(
        theta0=np.radians(-45),
        omega0=0.0,
        L=15.3,
        anchor=(0.0, 20.0),
        release_time=2.0,
        T=8.0,
        dt=0.01,
        y_ground=0.1
    )

    print("Last attached:")
    print(attached[-1])

    print("First released:")
    print(released[0])