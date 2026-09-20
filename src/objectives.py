import numpy as np

from .events import *
from .utils import angle_to_pos_conv, get_vel_comps

# helpers
def event_state_to_cartesian(
    event: WebAttachedEvent | ReleasedEvent,
    state: np.ndarray
):
    """
        Convert either event state's representation into
        [x, y, vx, vy].
    """

    if isinstance(event, ReleasedEvent):
        return state.copy()

    if isinstance(event, WebAttachedEvent):
        theta, omega = state

        x, y = angle_to_pos_conv(
            event.anchor,
            theta,
            event.L
        )

        vx, vy = get_vel_comps(
            omega,
            theta,
            event.L
        )

        return np.array([x, y, vx, vy])

    raise TypeError(f"Unsupported event type: {type(event).__name__}")

def get_final_cartesian_state(
    modes: list[WebAttachedEvent | ReleasedEvent]
):
    if len(modes) == 0: 
        raise ValueError("Simulation returned no events")
    final_mode = modes[-1]

    if len(final_mode.states) == 0:
        raise ValueError(f"{final_mode.name} has no states!")

    _, final_state = final_mode.states[-1]

    return event_state_to_cartesian(
        final_mode,
        final_state
    )

def get_initial_cartesian_state(
    modes: list[WebAttachedEvent | ReleasedEvent]
):
    if len(modes) == 0:
        raise ValueError("Simulation returned no events")

    initial_mode = modes[0]

    if len(initial_mode.states) == 0:
        raise ValueError(f"{initial_mode.name} has no states!")

    _, initial_state = initial_mode.states[0]

    return event_state_to_cartesian(
        initial_mode,
        initial_state
    )

# def objv1(modes: list[WebAttachedEvent | ReleasedEvent]):
#     """
#         Simple version 1 of the objective 
#         > ### Maximise Total horiz dist travelled
#     """
#     initial_state = get_initial_cartesian_state(modes)
#     final_state = get_final_cartesian_state(modes)

#     x_initial = initial_state[0]
#     x_final = final_state[0]

#     return x_final - x_initial

# def objv2(modes: list[WebAttachedEvent | ReleasedEvent], w_dist=0.7, w_vel=0.3, dist_scale=50.0, vel_scale=10.0, failure_penalty=0.0):
#     """
#         Advanced version 2 of the objective 
#         > ### Maximise Total horiz dist travelled + retained horiz vel + failure penalties

#         - w_dist, w_vel and weights for dist and vel params
#         - *_scale are ref scales for normalising 
#     """
#     # TODO: weights, and ref scales need to be configured
#     distance = objv1(modes)

#     final_state = get_final_cartesian_state(modes)
#     vx_final = final_state[2]

#     norm_distance = distance / dist_scale
#     norm_velocity = vx_final / vel_scale

#     return (
#         w_dist * norm_distance
#         + w_vel * norm_velocity
#         - failure_penalty
#     )

def objv1(sim_res: FinalSimResult):
    """
        Simple version 1 of the objective 
        > ### Maximise Total horiz dist travelled
    """
    modes = sim_res.modes
    initial_state = get_initial_cartesian_state(modes)
    final_state = get_final_cartesian_state(modes)

    x_initial = initial_state[0]
    x_final = final_state[0]

    return x_final - x_initial 

def objv2(
    sim_res: FinalSimResult,
    w_dist=0.7,
    w_vel=0.3,
    dist_scale=50.0,
    vel_scale=10.0,
    failure_penalty=5.0,
    min_distance=50.0, # min horiz dist travelled
    min_reattachments=3,
    distance_penalty=1.0,
    attachment_penalty=1.0
):
    """
        Maximise:
            - Horizontal displacement
            - Retained horizontal velocity

        Penalise:
            - Ground collision
            - Insufficient displacement
            - Insufficient reattachments
    """

    modes = sim_res.modes

    # Base performance (horiz dist)
    distance = objv1(sim_res)

    final_state = get_final_cartesian_state(modes)
    vx_final = final_state[2]

    score = (
        w_dist * (distance / dist_scale)
        + w_vel * (vx_final / vel_scale)
    )

    # Ground collision penalty
    if sim_res.termination_reason == "ground_collision":
        score -= failure_penalty

    # Insufficient distance penalty
    distance_shortfall = max(
        0.0,
        min_distance - distance
    )

    score -= ( # subtract a proportional penalty based on how shortcoming the travelled dist is
        distance_penalty
        * distance_shortfall
        / min_distance
    )

    # Insufficient reattachment penalty
    attached_count = sum(
        isinstance(mode, WebAttachedEvent)
        for mode in modes
    )

    reattachments = max(
        0,
        attached_count - 1 # since starting state is always attached
    )

    attachment_shortfall = max(
        0,
        min_reattachments - reattachments
    )

    score -= ( # subtract a proportional penalty ...
        attachment_penalty
        * attachment_shortfall
        / min_reattachments
    )

    return score