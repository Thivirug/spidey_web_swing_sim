# NOTE: use ursina for visualisation

import numpy as np

from .solver import RK4
from .utils import attached_to_released_state, interpolate_landing, released_to_attached_state, choose_first_reachable_anchor
from .events import *
from .controller import WebSwingController, PolicyParams

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
    mode = WebAttachedEvent(start_time=t, anchor=anchor, L=L) # starting from attached

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
                state = attached_to_released_state(state, mode.anchor, mode.L)

                # create new released mode
                mode = ReleasedEvent(start_time=t)

            else:
                # go to next attached state
                state = solver.step(state, mode.L)

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

def simv2(
    theta0,
    omega0,
    L,
    anchor,
    theta_release,
    T,
    dt,
    y_ground=0.0,
    drag_params=None,
    anchors=None,
    L_min=5.0,
    L_max=15.0
):
    """
        Simulate attached -> released -> attached cycle
        with release happening when theta >= theta_release.
    """ 
    solver = RK4(dt)
    t = 0.0
    mode = WebAttachedEvent(start_time=t, anchor=anchor, L=L) # starting from attached

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

            if state[0] >= theta_release and state[1]>0:
                # update attached mode 
                mode.end_time = t
                modes.append(mode)

                prev_anchor = mode.anchor
                # release
                state = attached_to_released_state(state, mode.anchor, mode.L)

                # create new released mode
                mode = ReleasedEvent(start_time=t)

            else:
                # go to next attached state
                state = solver.step(state, mode.L)

        if isinstance(mode, ReleasedEvent):
            mode.states.append(
                (t, state.copy())
            )

            # 1-check for reattachment
            new_anchor = choose_first_reachable_anchor(state, anchors, L_min, L_max, prev_anchor)

            if new_anchor is not None:
                # finish released event
                mode.end_time = t
                modes.append(mode)

                # convert projectile state -> pendulum state
                state, new_L = released_to_attached_state(
                    state,
                    new_anchor
                )

                # create new attached event
                mode = WebAttachedEvent(
                    start_time=t,
                    anchor=new_anchor,
                    L=new_L
                )

                # save the exact reattachment state in new attached mode
                mode.states.append(
                    (t, state.copy())
                )

                # advance attached motion from t -> t + dt
                state = solver.step(state, mode.L)

                t += dt
                continue

            # 2- check Ground collision
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

            #3- otherwise go to next released state
            state = solver.step_released(state, drag_params)

        t += dt

    # handle the case where spidey never lands be4 T and the case where he never releases bef T
    if mode not in modes:
        mode.end_time = T
        modes.append(mode)

    return modes

def simv3(
    theta0,
    omega0,
    L,
    anchor,
    policy_params,
    T,
    dt,
    y_ground=0.0,
    drag_params=None,
    anchors=None,
    L_min=5.0,
    L_max=15.0,
    min_flight_time=0.15
):
    """
        Simulate attached -> released -> attached cycle
        with release happening based on the controller's policy params.
    """ 
    solver = RK4(dt)
    controller = WebSwingController(L_min, L_max)
    t = 0.0
    mode = WebAttachedEvent(start_time=t, anchor=anchor, L=L) # starting from attached

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

            if controller.should_release(state, policy_params):
                # update attached mode 
                mode.end_time = t
                modes.append(mode)

                prev_anchor = mode.anchor
                # release
                state = attached_to_released_state(state, mode.anchor, mode.L)

                # create new released mode
                mode = ReleasedEvent(start_time=t)

            else:
                # go to next attached state
                state = solver.step(state, mode.L)

        if isinstance(mode, ReleasedEvent):
            mode.states.append(
                (t, state.copy())
            )

            new_anchor = None
            flight_time = t - mode.start_time 

            # 1- check Ground collision
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

            # 2-check for reattachment if flight time greater than min req
            if flight_time >= min_flight_time:
                new_anchor = controller.choose_attachment(state, anchors, policy_params, prev_anchor)

            if new_anchor is not None:
                # finish released event
                mode.end_time = t
                modes.append(mode)

                # convert projectile state -> pendulum state
                state, new_L = released_to_attached_state(
                    state,
                    new_anchor
                )

                # create new attached event
                mode = WebAttachedEvent(
                    start_time=t,
                    anchor=new_anchor,
                    L=new_L
                )

                # save the exact reattachment state in new attached mode
                mode.states.append(
                    (t, state.copy())
                )

                # advance attached motion from t -> t + dt
                state = solver.step(state, mode.L)

                t += dt
                continue

            #3- otherwise go to next released state
            state = solver.step_released(state, drag_params)

        t += dt

    # handle the case where spidey never lands be4 T and the case where he never releases bef T
    if mode not in modes:
        mode.end_time = T
        modes.append(mode)

    return modes

if __name__ == '__main__':
    print("\n\t====== Simulation V1 ======\n")
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

    print("\n==============================================================================================================================")
    print("\n\t====== Simulation V2 ======\n")

    anchors = [
        (12.0, 19.0),
        (15.0, 20.0),
        (18.0, 22.0),

        (27.0, 19.0),
        (30.0, 22.0),
        (33.0, 21.0),

        (42.0, 18.0),
        (45.0, 20.0),
        (48.0, 22.0),
    ]

    modes2 = simv2(
        theta0=np.radians(-45),
        omega0=0.0,
        L=11.3,
        anchor=(0.0, 20.0),
        theta_release=np.radians(30),
        T=12.0,
        dt=0.01,
        y_ground=0.1,
        anchors=anchors,
        L_min=5.0,
        L_max=15.0
    )

    for i, mode in enumerate(modes2):
        print(
            i,
            mode.name,
            mode.start_time,
            mode.end_time,
            getattr(mode, "anchor", None),
            getattr(mode, "L", None),
        )

    print("\n==============================================================================================================================")
    print("\n\t====== Simulation V3 ======\n")

    for preferred_L in [6.0, 8.0, 10.0, 12.0, 14.0]:

        params = PolicyParams(
            theta_r=np.radians(30),
            L=preferred_L
        )

        modes = simv3(
            theta0=np.radians(-45),
            omega0=0.0,
            L=11.3,
            anchor=(0.0, 20.0),
            policy_params=params,
            T=12.0,
            dt=0.01,
            y_ground=0.1,
            anchors=anchors,
            L_min=5.0,
            L_max=15.0
        )

        print(f"\n===== Preferred L = {preferred_L} =====")

        for mode in modes:
            if mode.name == "attached":
                print(
                    "anchor =", mode.anchor,
                    "actual L =", round(mode.L, 3),
                    "start =", round(mode.start_time, 3),
                    "end =", round(mode.end_time, 3)
                )
       