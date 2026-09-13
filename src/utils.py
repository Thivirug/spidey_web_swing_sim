import numpy as np

def angle_to_pos_conv(anchor: tuple, theta, L) -> tuple:
    """
        Convert angular pos values to x,y coords 
    """

    x_a, y_a = anchor

    x = x_a + L*np.sin(theta)
    y = y_a - L*np.cos(theta)

    return x, y

def get_vel_comps(omega, theta, L):
    """
       Get velocity components
    """

    v_x = L*omega*np.cos(theta)
    v_y = L*omega*np.sin(theta)

    return v_x, v_y

def attached_to_released_state(
    state: np.ndarray,
    anchor,
    L
) -> np.ndarray:
    """
        Convert attached state [theta, omega]
        into released state [x, y, vx, vy].
    """

    theta, omega = state

    x, y = angle_to_pos_conv(
        anchor,
        theta,
        L
    )

    vx, vy = get_vel_comps(
        omega,
        theta,
        L
    )

    return np.array([
        x,
        y,
        vx,
        vy
    ])

def pos_to_angle_conv(x_, y_, vx, vy, L):
    """ 
        Convert x,y coords to angular pos values
        x_, y_ : x, y values rel to anchor 
    """

    theta = np.arctan2(x_, -y_)

    omega = (
        vx*np.cos(theta) + vy*np.sin(theta)
    )/ L

    return theta, omega

def released_to_attached_state(
    state: np.ndarray,
    anchor
) -> tuple[np.ndarray, float]:
    """
        Convert released state [x, y, vx, vy]
        into attached state [theta, omega].
    """

    x,y,vx,vy = state
    x_a, y_a = anchor

    # get pos rel to anchor
    x_ = x-x_a
    y_ = y-y_a

    L = np.sqrt(x_**2 + y_**2)

    if L <=0:
        raise ValueError("Invalid web length")

    theta, omega = pos_to_angle_conv(x_, y_, vx, vy, L)

    attached_state = np.array([theta, omega])
    return attached_state, L

def interpolate_landing(t_prev, t_curr, state_prev, state_curr, y_ground):
    """
        Interpolate landing time and state between two projectile states.
    """
    
    y1 = state_prev[1]
    y2 = state_curr[1]

    if np.isclose(y2, y1):
        raise ValueError(
            "Cannot interpolate landing when y values are equal."
        )

    alpha = (
        (y_ground - y1) / (y2-y1)
    )

    land_t = t_prev + alpha * (t_curr - t_prev)
    land_state = state_prev + alpha * (state_curr - state_prev)

    land_state[1] = y_ground # set this back to avoid floating point mismatches

    return land_t, land_state