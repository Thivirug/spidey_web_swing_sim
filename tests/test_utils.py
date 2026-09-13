import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import angle_to_pos_conv, get_vel_comps, attached_to_released_state, released_to_attached_state


def test_position_at_bottom():
    """
        At theta = 0, spidey should be directly below the anchor.
    """
    anchor = (0.0, 0.0)
    L = 15.0
    theta = 0.0

    x, y = angle_to_pos_conv(anchor, theta, L)

    assert np.isclose(x, 0.0)
    assert np.isclose(y, -15.0) 

def test_position_at_right_side():
    """
        At theta = pi/2, spidey should be L metres to the right.
    """
    anchor = (0.0, 0.0)
    L = 15.0
    theta = np.pi / 2

    x, y = angle_to_pos_conv(anchor, theta, L)

    assert np.isclose(x, 15.0)
    assert np.isclose(y, 0.0)

def test_position_at_left_side():
    """
        At theta = -pi/2, spidey should be L metres to the left.
    """
    anchor = (0.0, 0.0)
    L = 15.0
    theta = -np.pi / 2

    x, y = angle_to_pos_conv(anchor, theta, L)

    assert np.isclose(x, -15.0)
    assert np.isclose(y, 0.0)

def test_position_distance_from_anchor():
    """
        The distance from spidey to the anchor should always equal L.
    """
    anchor = (3.0, 10.0)
    L = 15.0

    theta_values = np.linspace(-np.pi, np.pi, 100)

    for theta in theta_values:
        x, y = angle_to_pos_conv(anchor, theta, L)

        distance = np.sqrt(
            (x - anchor[0]) ** 2 +
            (y - anchor[1]) ** 2
        )

        assert np.isclose(distance, L)

def test_velocity_at_bottom():
    """
        At theta = 0, velocity should be purely horizontal.
    """
    L = 15.0
    omega = 2.0
    theta = 0.0

    vx, vy = get_vel_comps(omega, theta, L)

    assert np.isclose(vx, 30.0)
    assert np.isclose(vy, 0.0)


def test_velocity_at_right_side():
    """
        At theta = pi/2, velocity should be purely vertical.
    """
    L = 15.0
    omega = 2.0
    theta = np.pi / 2

    vx, vy = get_vel_comps(omega, theta, L)

    assert np.isclose(vx, 0.0, atol=1e-10)
    assert np.isclose(vy, 30.0)


def test_velocity_is_tangent_to_swing():
    """
        Radius vector and velocity vector should be perpendicular.
    """
    anchor = (2.0, 8.0)
    L = 15.0
    omega = 1.7

    theta_values = np.linspace(-1.2, 1.2, 50)

    for theta in theta_values:
        x, y = angle_to_pos_conv(anchor, theta, L)
        vx, vy = get_vel_comps(omega, theta, L)

        radius_x = x - anchor[0]
        radius_y = y - anchor[1]

        dot_product = radius_x * vx + radius_y * vy

        assert np.isclose(dot_product, 0.0, atol=1e-10)

def test_attached_released_attached_round_trip():
    """
        Converting an attached state to released coordinates and back
        using the same anchor should recover theta, omega and L.
    """
    anchor = (3.0, 20.0)
    L = 12.0
    theta = 0.6
    omega = 1.3

    attached_state = np.array([theta, omega])

    released_state = attached_to_released_state(
        attached_state,
        anchor,
        L
    )

    recovered_state, recovered_L = released_to_attached_state(
        released_state,
        anchor
    )

    assert np.isclose(recovered_state[0], theta)
    assert np.isclose(recovered_state[1], omega)
    assert np.isclose(recovered_L, L)

def test_released_to_attached_preserves_negative_omega():
    """
        The conversion should preserve rotational direction.
    """
    anchor = (0.0, 10.0)
    L = 8.0
    theta = 0.4
    omega = -2.0

    attached_state = np.array([theta, omega])

    released_state = attached_to_released_state(
        attached_state,
        anchor,
        L
    )

    recovered_state, recovered_L = released_to_attached_state(
        released_state,
        anchor
    )

    assert np.isclose(recovered_state[0], theta)
    assert np.isclose(recovered_state[1], omega)
    assert np.isclose(recovered_L, L)

def test_pure_radial_velocity_gives_zero_omega():
    """
        If projectile velocity is entirely along the new web,
        its tangential component is zero, so omega should be zero.
    """
    anchor = (0.0, 10.0)

    # Spider-Man is directly below anchor
    # theta = 0, L = 5
    state = np.array([
        0.0,    # x
        5.0,    # y
        0.0,    # vx
        3.0     # vy: moving directly toward anchor
    ])

    attached_state, L = released_to_attached_state(
        state,
        anchor
    )

    theta, omega = attached_state

    assert np.isclose(L, 5.0)
    assert np.isclose(theta, 0.0)
    assert np.isclose(omega, 0.0)