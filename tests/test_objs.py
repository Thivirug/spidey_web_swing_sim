import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.events import *
from src.objectives import (
    event_state_to_cartesian,
    get_initial_cartesian_state,
    get_final_cartesian_state,
    objv1,
    objv2,
)


def released(t, state):
    event = ReleasedEvent(start_time=t)
    event.states.append((t, np.array(state, dtype=float)))
    return event


def attached(t, anchor, length, state):
    event = WebAttachedEvent(start_time=t, anchor=anchor, L=length)
    event.states.append((t, np.array(state, dtype=float)))
    return event


def sim_result(modes, reason="time_limit"):
    return FinalSimResult(modes=modes, termination_reason=reason)


@pytest.fixture
def sample_modes():
    return [
        released(0.0, [5.0, 10.0, 2.0, 0.0]),
        released(2.0, [25.0, 8.0, 6.0, -1.0]),
    ]


def test_event_state_to_cartesian_released():
    event = ReleasedEvent(start_time=0.0)
    state = np.array([10.0, 20.0, 5.0, -2.0])
    result = event_state_to_cartesian(event, state)
    np.testing.assert_allclose(result, state)
    assert result is not state


def test_event_state_to_cartesian_attached():
    event = WebAttachedEvent(start_time=0.0, anchor=(0.0, 20.0), L=10.0)
    result = event_state_to_cartesian(event, np.array([0.0, 1.0]))
    np.testing.assert_allclose(result, [0.0, 10.0, 10.0, 0.0])


def test_event_state_to_cartesian_invalid_type():
    class DummyEvent:
        pass

    with pytest.raises(TypeError):
        event_state_to_cartesian(DummyEvent(), np.array([1.0]))


def test_get_initial_cartesian_state_attached():
    modes = [attached(0.0, (0.0, 20.0), 10.0, [0.0, 1.0])]
    np.testing.assert_allclose(
        get_initial_cartesian_state(modes), [0.0, 10.0, 10.0, 0.0]
    )


def test_get_initial_cartesian_state_released():
    modes = [released(0.0, [5.0, 10.0, 3.0, 2.0])]
    np.testing.assert_allclose(
        get_initial_cartesian_state(modes), [5.0, 10.0, 3.0, 2.0]
    )


def test_get_final_cartesian_state_attached():
    modes = [attached(1.0, (20.0, 20.0), 10.0, [0.0, 0.5])]
    np.testing.assert_allclose(
        get_final_cartesian_state(modes), [20.0, 10.0, 5.0, 0.0]
    )


def test_get_final_cartesian_state_released():
    modes = [released(1.0, [25.0, 8.0, 6.0, -1.0])]
    np.testing.assert_allclose(
        get_final_cartesian_state(modes), [25.0, 8.0, 6.0, -1.0]
    )


@pytest.mark.parametrize("extractor", [get_initial_cartesian_state, get_final_cartesian_state])
def test_state_extractor_empty_modes(extractor):
    with pytest.raises(ValueError, match="Simulation returned no events"):
        extractor([])


@pytest.mark.parametrize("extractor", [get_initial_cartesian_state, get_final_cartesian_state])
def test_state_extractor_empty_event(extractor):
    with pytest.raises(ValueError, match="has no states"):
        extractor([ReleasedEvent(start_time=0.0)])


def test_objv1_released_to_released():
    result = sim_result([
        released(0.0, [5.0, 10.0, 3.0, 0.0]),
        released(2.0, [25.0, 8.0, 6.0, -1.0]),
    ])
    assert np.isclose(objv1(result), 20.0)


def test_objv1_attached_to_attached():
    result = sim_result([
        attached(0.0, (0.0, 20.0), 10.0, [0.0, 0.0]),
        attached(2.0, (20.0, 20.0), 10.0, [0.0, 0.0]),
    ])
    assert np.isclose(objv1(result), 20.0)


def test_objv1_attached_to_released():
    result = sim_result([
        attached(0.0, (0.0, 20.0), 10.0, [0.0, 0.0]),
        released(2.0, [15.0, 8.0, 4.0, -1.0]),
    ])
    assert np.isclose(objv1(result), 15.0)


def test_objv1_collision_still_reports_distance(sample_modes):
    """A distance-only objective should not silently erase physical displacement."""
    result = sim_result(sample_modes, "ground_collision")
    assert np.isclose(objv1(result), 20.0)


def test_objv1_reverse_displacement():
    result = sim_result([
        released(0.0, [15.0, 10.0, -1.0, 0.0]),
        released(2.0, [5.0, 8.0, -2.0, -1.0]),
    ])
    assert np.isclose(objv1(result), -10.0)

# Objective V2 tests
@pytest.fixture
def collision_modes():
    """
    Initial x = 5, final x = 25, final vx = 6.
    Collision occurs at t = 2 seconds.
    """
    first = released(
        0.0,
        [5.0, 10.0, 2.0, 0.0],
    )

    final = released(
        2.0,
        [25.0, 0.0, 6.0, -1.0],
    )
    final.end_time = 2.0

    return [first, final]


def test_objv2_time_limit_no_penalty(sample_modes):
    result = sim_result(sample_modes, "time_limit")

    score = objv2(
        result,
        T=10.0,
        w_dist=0.7,
        w_vel=0.3,
        dist_scale=50.0,
        vel_scale=10.0,
        failure_penalty=0.2,
        early_collision_penalty=10.0,
    )

    # Distance contribution: 0.7 * (20 / 50) = 0.28
    # Velocity contribution: 0.3 * (6 / 10) = 0.18
    # No collision penalty.
    assert score == pytest.approx(0.46)


def test_objv2_ground_collision_penalty(collision_modes):
    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    score = objv2(
        result,
        T=10.0,
        w_dist=0.7,
        w_vel=0.3,
        dist_scale=50.0,
        vel_scale=10.0,
        failure_penalty=0.2,
        early_collision_penalty=10.0,
    )

    base_score = 0.46

    # Collision at t=2 in a T=10 simulation:
    # penalty = 0.2 + 10 * (1 - 2/10) = 8.2
    expected = base_score - 8.2

    assert score == pytest.approx(expected)


def test_objv2_zero_collision_penalties(collision_modes):
    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    score = objv2(
        result,
        T=10.0,
        w_dist=0.7,
        w_vel=0.3,
        failure_penalty=0.0,
        early_collision_penalty=0.0,
    )

    assert score == pytest.approx(0.46)


def test_objv2_collision_score_difference_equals_penalty(
    collision_modes,
):
    T = 10.0
    base_penalty = 0.5
    early_penalty = 10.0
    collision_time = 2.0

    time_limit_score = objv2(
        sim_result(collision_modes, "time_limit"),
        T=T,
        failure_penalty=base_penalty,
        early_collision_penalty=early_penalty,
    )

    collision_score = objv2(
        sim_result(collision_modes, "ground_collision"),
        T=T,
        failure_penalty=base_penalty,
        early_collision_penalty=early_penalty,
    )

    expected_penalty = (
        base_penalty
        + early_penalty * (1.0 - collision_time / T)
    )

    assert (
        time_limit_score - collision_score
    ) == pytest.approx(expected_penalty)

    assert collision_score < time_limit_score


def test_objv2_penalty_not_applied_twice(collision_modes):
    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    T = 10.0

    base_score = (
        0.8 * (20.0 / 50.0)
        + 0.7 * (6.0 / 10.0)
    )

    expected_penalty = (
        1.0
        + 10.0 * (1.0 - 2.0 / T)
    )

    score = objv2(
        result,
        T=T,
        failure_penalty=1.0,
        early_collision_penalty=10.0,
    )

    assert score == pytest.approx(
        base_score - expected_penalty
    )


# New tests for time-dependent collision penalties

@pytest.mark.parametrize(
    "collision_time, expected_penalty",
    [
        (0.0, 10.1),
        (2.0, 8.1),
        (5.0, 5.1),
        (8.0, 2.1),
        (10.0, 0.1),
    ],
)
def test_objv2_collision_penalty_decreases_with_survival_time(
    collision_modes,
    collision_time,
    expected_penalty,
):
    """
    Earlier ground collisions receive larger penalties.
    """
    collision_modes[-1].end_time = collision_time

    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    base_score = objv2(
        sim_result(collision_modes, "time_limit"),
        T=10.0,
    )

    collision_score = objv2(
        result,
        T=10.0,
    )

    assert (
        base_score - collision_score
    ) == pytest.approx(expected_penalty)


@pytest.mark.parametrize(
    "collision_time, expected_penalty",
    [
        (-5.0, 10.1),
        (15.0, 0.1),
    ],
)
def test_objv2_collision_time_is_clipped(
    collision_modes,
    collision_time,
    expected_penalty,
):
    """
    Penalty calculation clips t_collision / T to [0, 1].
    These are synthetic edge cases testing the objective,
    not physically valid collision histories.
    """
    collision_modes[-1].end_time = collision_time

    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    base_score = objv2(
        sim_result(collision_modes, "time_limit"),
        T=10.0,
    )

    collision_score = objv2(
        result,
        T=10.0,
    )

    assert (
        base_score - collision_score
    ) == pytest.approx(expected_penalty)


def test_objv2_collision_requires_end_time(collision_modes):
    collision_modes[-1].end_time = None

    result = sim_result(
        collision_modes,
        "ground_collision",
    )

    with pytest.raises(
        ValueError,
        match="Collision event has no end_time",
    ):
        objv2(result, T=10.0)


@pytest.mark.parametrize(
    "invalid_T",
    [0.0, -1.0],
)
def test_objv2_rejects_invalid_duration(sample_modes, invalid_T):
    result = sim_result(sample_modes)

    with pytest.raises(
        ValueError,
        match="T must be positive",
    ):
        objv2(result, T=invalid_T)


@pytest.mark.parametrize(
    "dist_scale, vel_scale",
    [
        (0.0, 10.0),
        (-1.0, 10.0),
        (50.0, 0.0),
        (50.0, -1.0),
    ],
)
def test_objv2_rejects_invalid_scales(
    sample_modes,
    dist_scale,
    vel_scale,
):
    result = sim_result(sample_modes)

    with pytest.raises(
        ValueError,
        match="Normalisation scales must be positive",
    ):
        objv2(
            result,
            T=10.0,
            dist_scale=dist_scale,
            vel_scale=vel_scale,
        )