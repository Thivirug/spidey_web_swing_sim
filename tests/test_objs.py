import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import WebAttachedEvent, ReleasedEvent
from src.objectives import (
    event_state_to_cartesian,
    get_initial_cartesian_state,
    get_final_cartesian_state,
    objv1,
    objv2,
)

def test_event_state_to_cartesian_released():
    event = ReleasedEvent(start_time=0.0)

    state = np.array([
        10.0,
        20.0,
        5.0,
        -2.0
    ])

    result = event_state_to_cartesian(
        event,
        state
    )

    np.testing.assert_allclose(
        result,
        state
    )

    # make sure a copy was returned
    assert result is not state


def test_event_state_to_cartesian_attached():
    event = WebAttachedEvent(
        start_time=0.0,
        anchor=(0.0, 20.0),
        L=10.0
    )

    state = np.array([
        0.0,   # theta
        1.0    # omega
    ])

    result = event_state_to_cartesian(
        event,
        state
    )

    expected = np.array([
        0.0,   # x
        10.0,  # y
        10.0,  # vx
        0.0    # vy
    ])

    np.testing.assert_allclose(
        result,
        expected
    )


def test_event_state_to_cartesian_invalid_type():
    class DummyEvent:
        pass

    with pytest.raises(TypeError):
        event_state_to_cartesian(
            DummyEvent(),
            np.array([1.0])
        )


def test_get_initial_cartesian_state_attached():
    event = WebAttachedEvent(
        start_time=0.0,
        anchor=(0.0, 20.0),
        L=10.0
    )

    event.states.append(
        (
            0.0,
            np.array([
                0.0,
                1.0
            ])
        )
    )

    result = get_initial_cartesian_state(
        [event]
    )

    expected = np.array([
        0.0,
        10.0,
        10.0,
        0.0
    ])

    np.testing.assert_allclose(
        result,
        expected
    )


def test_get_initial_cartesian_state_released():
    event = ReleasedEvent(
        start_time=0.0
    )

    state = np.array([
        5.0,
        10.0,
        3.0,
        2.0
    ])

    event.states.append(
        (0.0, state)
    )

    result = get_initial_cartesian_state(
        [event]
    )

    np.testing.assert_allclose(
        result,
        state
    )


def test_get_final_cartesian_state_attached():
    event = WebAttachedEvent(
        start_time=0.0,
        anchor=(20.0, 20.0),
        L=10.0
    )

    event.states.append(
        (
            1.0,
            np.array([
                0.0,
                0.5
            ])
        )
    )

    result = get_final_cartesian_state(
        [event]
    )

    expected = np.array([
        20.0,
        10.0,
        5.0,
        0.0
    ])

    np.testing.assert_allclose(
        result,
        expected
    )


def test_get_final_cartesian_state_released():
    event = ReleasedEvent(
        start_time=0.0
    )

    state = np.array([
        25.0,
        8.0,
        6.0,
        -1.0
    ])

    event.states.append(
        (1.0, state)
    )

    result = get_final_cartesian_state(
        [event]
    )

    np.testing.assert_allclose(
        result,
        state
    )


def test_get_initial_state_empty_modes():
    with pytest.raises(
        ValueError,
        match="Simulation returned no events"
    ):
        get_initial_cartesian_state([])


def test_get_final_state_empty_modes():
    with pytest.raises(
        ValueError,
        match="Simulation returned no events"
    ):
        get_final_cartesian_state([])


def test_get_initial_state_empty_event():
    event = ReleasedEvent(
        start_time=0.0
    )

    with pytest.raises(
        ValueError,
        match="has no states"
    ):
        get_initial_cartesian_state(
            [event]
        )


def test_get_final_state_empty_event():
    event = ReleasedEvent(
        start_time=0.0
    )

    with pytest.raises(
        ValueError,
        match="has no states"
    ):
        get_final_cartesian_state(
            [event]
        )


def test_objv1_released_to_released():
    initial = ReleasedEvent(
        start_time=0.0
    )

    initial.states.append(
        (
            0.0,
            np.array([
                5.0,
                10.0,
                3.0,
                0.0
            ])
        )
    )

    final = ReleasedEvent(
        start_time=1.0
    )

    final.states.append(
        (
            2.0,
            np.array([
                25.0,
                8.0,
                6.0,
                -1.0
            ])
        )
    )

    result = objv1(
        [initial, final]
    )

    assert np.isclose(
        result,
        20.0
    )


def test_objv1_attached_to_attached():
    initial = WebAttachedEvent(
        start_time=0.0,
        anchor=(0.0, 20.0),
        L=10.0
    )

    initial.states.append(
        (
            0.0,
            np.array([
                0.0,
                0.0
            ])
        )
    )

    final = WebAttachedEvent(
        start_time=1.0,
        anchor=(20.0, 20.0),
        L=10.0
    )

    final.states.append(
        (
            2.0,
            np.array([
                0.0,
                0.0
            ])
        )
    )

    result = objv1(
        [initial, final]
    )

    assert np.isclose(
        result,
        20.0
    )


def test_objv1_attached_to_released():
    initial = WebAttachedEvent(
        start_time=0.0,
        anchor=(0.0, 20.0),
        L=10.0
    )

    initial.states.append(
        (
            0.0,
            np.array([
                0.0,
                0.0
            ])
        )
    )

    final = ReleasedEvent(
        start_time=1.0
    )

    final.states.append(
        (
            2.0,
            np.array([
                15.0,
                8.0,
                4.0,
                -1.0
            ])
        )
    )

    result = objv1(
        [initial, final]
    )

    assert np.isclose(
        result,
        15.0
    )


def test_objv2():
    initial = ReleasedEvent(
        start_time=0.0
    )

    initial.states.append(
        (
            0.0,
            np.array([
                5.0,
                10.0,
                2.0,
                0.0
            ])
        )
    )

    final = ReleasedEvent(
        start_time=1.0
    )

    final.states.append(
        (
            2.0,
            np.array([
                25.0,
                8.0,
                6.0,
                -1.0
            ])
        )
    )

    score = objv2(
        [initial, final],
        w_dist=0.7,
        w_vel=0.3,
        dist_scale=50.0,
        vel_scale=10.0,
        failure_penalty=0.2
    )

    # distance = 20
    # normalised distance = 20 / 50 = 0.4
    # normalised vx = 6 / 10 = 0.6
    #
    # score =
    # 0.7 * 0.4
    # + 0.3 * 0.6
    # - 0.2
    #
    # = 0.26

    expected = 0.26

    assert np.isclose(
        score,
        expected
    )


def test_objv2_zero_failure_penalty():
    initial = ReleasedEvent(
        start_time=0.0
    )

    initial.states.append(
        (
            0.0,
            np.array([
                0.0,
                10.0,
                1.0,
                0.0
            ])
        )
    )

    final = ReleasedEvent(
        start_time=1.0
    )

    final.states.append(
        (
            1.0,
            np.array([
                50.0,
                10.0,
                10.0,
                0.0
            ])
        )
    )

    score = objv2(
        [initial, final]
    )

    # distance / 50 = 1
    # vx / 10 = 1
    # 0.7 * 1 + 0.3 * 1 = 1

    assert np.isclose(
        score,
        1.0
    )