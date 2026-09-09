# NOTE: use ursina

import numpy as np

from .solver import RK4
from .utils import attached_to_released_state, interpolate_landing
from .events import *

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
    mode = WebAttachedEvent(start_time=t) # starting from attached

    # starting attached state
    state = np.array([
        theta0,
        omega0
    ], dtype=float)

    modes = []

    while t<=T:
        if isinstance(mode, WebAttachedEvent):
            mode.states.append(
                (t, state.copy())
            )

            if t>= release_time:
                # update attached mode 
                mode.end_time = t
                modes.append(mode)

                # release
                state = attached_to_released_state(state, anchor, L)

                # create new released mode
                mode = ReleasedEvent(start_time=t)

            else:
                # go to next attached state
                state = solver.step(state, L)

        if isinstance(mode, ReleasedEvent):
            mode.states.append(
                (t, state.copy())
            )

            y, vy = state[1], state[3]

            if y <= y_ground and vy < 0: # ground hit
                # linearly interpolate for better precision
                if len(mode.states)>=2:
                    t_prev, state_prev = mode.states[-2]
                    t_curr, state_curr = mode.states[-1]

                    inter_land_t, inter_land_state = interpolate_landing(t_prev, t_curr, state_prev, state_curr, y_ground)

                    # update released states and times
                    mode.states[-1] = (inter_land_t, inter_land_state)

                    # update released mode 
                    mode.end_time = inter_land_t
                    
                else: # if only 1 released state present 
                    mode.end_time = t

                modes.append(mode)
                break

            # go to next released state
            state = solver.step_released(state, drag_params)

        t += dt

    # handle the case where spidey never lands be4 T and the case where he never releases bef T
    if mode not in modes:
        mode.end_time = T
        modes.append(mode)

    return modes

if __name__ == '__main__':
    modes = simv1(
        theta0=np.radians(-45),
        omega0=0.0,
        L=11.3,
        anchor=(0.0, 20.0),
        release_time=2.0,
        T=8.0,
        dt=0.01,
        y_ground=0.1
    )

    print(len(modes))

    print("======= ATTACHED =======")
    print(f"Num of states: {len(modes[0].states)}")
    print(f"start t: {modes[0].start_time}")
    print(f"end t: {modes[0].end_time}")

    print("======= RELEASED =======")
    print(f"Num of states: {len(modes[1].states)}")
    print(f"start t: {modes[1].start_time}")
    print(f"end t: {modes[1].end_time}")