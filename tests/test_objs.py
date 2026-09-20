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


def test_objv2_time_limit_no_penalty(sample_modes):
    result = sim_result(sample_modes, "time_limit")
    score = objv2(
        result, w_dist=0.7, w_vel=0.3,
        dist_scale=50.0, vel_scale=10.0, failure_penalty=0.2,
    )
    # 0.7*(20/50) + 0.3*(6/10) = 0.46
    assert np.isclose(score, 0.46)


def test_objv2_ground_collision_penalty(sample_modes):
    result = sim_result(sample_modes, "ground_collision")
    score = objv2(
        result, w_dist=0.7, w_vel=0.3,
        dist_scale=50.0, vel_scale=10.0, failure_penalty=0.2,
    )
    assert np.isclose(score, 0.26)


def test_objv2_zero_failure_penalty(sample_modes):
    result = sim_result(sample_modes, "ground_collision")
    assert np.isclose(objv2(result, failure_penalty=0.0), 0.46)


def test_objv2_collision_score_difference_equals_penalty(sample_modes):
    penalty = 0.5
    time_limit_score = objv2(sim_result(sample_modes, "time_limit"), failure_penalty=penalty)
    collision_score = objv2(sim_result(sample_modes, "ground_collision"), failure_penalty=penalty)
    assert np.isclose(time_limit_score - collision_score, penalty)
    assert collision_score < time_limit_score


def test_objv2_penalty_not_applied_twice(sample_modes):
    result = sim_result(sample_modes, "ground_collision")
    base_score = 0.7 * (20.0 / 50.0) + 0.3 * (6.0 / 10.0)
    assert np.isclose(objv2(result, failure_penalty=1.0), base_score - 1.0)
