import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.controller import PolicyParams, WebSwingController

def test_should_release_true():
    controller = WebSwingController(
        L_min=5.0,
        L_max=15.0
    )

    params = PolicyParams(
        theta_r=np.radians(30),
        L=10.0
    )

    state = np.array([
        np.radians(35),
        0.5
    ])

    assert controller.should_release(
        state,
        params
    )

def test_should_not_release_below_threshold():
    controller = WebSwingController(
        L_min=5.0,
        L_max=15.0
    )

    params = PolicyParams(
        theta_r=np.radians(30),
        L=10.0
    )

    state = np.array([
        np.radians(20),
        0.5
    ])

    assert not controller.should_release(
        state,
        params
    )

def test_should_not_release_when_moving_backward():
    controller = WebSwingController(
        L_min=5.0,
        L_max=15.0
    )

    params = PolicyParams(
        theta_r=np.radians(30),
        L=10.0
    )

    state = np.array([
        np.radians(35),
        -0.5
    ])

    assert not controller.should_release(
        state,
        params
    )

def test_choose_attachment_closest_to_L_rule():
    controller = WebSwingController(
        L_min=5.0,
        L_max=15.0
    )

    params = PolicyParams(
        theta_r=np.radians(30),
        L=10.0
    )

    # released state
    state = np.array([
        10.0,   # x
        10.0,   # y
        5.0,    # vx
        2.0     # vy
    ])

    anchors = [
        (16.0, 16.0),
        (18.0, 16.0),
        (20.0, 16.0)
    ]

    chosen = controller.choose_attachment(
        state,
        anchors,
        params
    )

    # get distances for each anchor from current state
    distances = [
        np.hypot(
            state[0] - a[0],
            state[1] - a[1]
        )
        for a in anchors
    ]

    expected = anchors[
        np.argmin( # index the anchor which has the lowest err
            [
                np.abs(d - params.L) 
                for d in distances
            ]
        )
    ]

    assert chosen == expected

def test_previous_anchor_is_excluded():
    controller = WebSwingController(
        L_min=5.0,
        L_max=15.0
    )

    params = PolicyParams(
        theta_r=np.radians(30),
        L=10.0
    )

    state = np.array([
        10.0,
        10.0,
        5.0,
        2.0
    ])

    previous_anchor = (16.0, 16.0)

    anchors = [
        previous_anchor,
        (18.0, 16.0)
    ]

    chosen = controller.choose_attachment(
        state,
        anchors,
        params,
        excluded_anchor=previous_anchor
    )

    assert chosen != previous_anchor