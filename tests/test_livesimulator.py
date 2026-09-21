import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulation import simv3
from src.controller import PolicyParams
from sim.sim_live import LiveSimulator

@pytest.fixture
def scenario():
    """A reproducible scenario shared by both simulators."""
    return dict(
        theta0=np.radians(-45),
        omega0=1.2,
        L=10.0,
        anchor=(0.0, 22.0),
        policy_params=PolicyParams(
            theta_r=np.radians(44.082),
            L=12.064,
        ),
        T=10.0,
        dt=0.01,
        y_ground=0.0,
        drag_params=None,
        anchors=[
            (8.0, 25.0),
            (13.0, 20.0),
            (19.0, 28.0),
            (24.0, 23.0),
            (31.0, 30.0),
            (39.0, 22.0),
            (44.0, 27.0),
            (50.0, 19.0),
        ],
        L_min=5.0,
        L_max=15.0,
        min_flight_time=0.25,
    )


def test_live_matches_simv3(scenario):
    """
    Verify that incremental simulation reproduces
    the original simv3 event history.
    """

    # Original simulation
    original = simv3(**scenario)

    # Incremental simulation
    live = LiveSimulator(**scenario)

    while not live.finished:
        live.step()

    incremental = live.get_result()

    # Compare termination
    assert (
        original.termination_reason
        == incremental.termination_reason
    )

    # Compare number of events
    assert len(original.modes) == len(incremental.modes)

    # Compare each event
    for old_mode, new_mode in zip(
        original.modes,
        incremental.modes,
    ):
        assert type(old_mode) is type(new_mode)

        assert old_mode.start_time == pytest.approx(
            new_mode.start_time
        )

        assert old_mode.end_time == pytest.approx(
            new_mode.end_time
        )

        assert len(old_mode.states) == len(new_mode.states)

        # Compare numerical trajectory
        for (old_t, old_state), (new_t, new_state) in zip(
            old_mode.states,
            new_mode.states,
        ):
            assert old_t == pytest.approx(new_t)

            np.testing.assert_allclose(
                old_state,
                new_state,
                rtol=1e-12,
                atol=1e-12,
            )

        # Compare attached-event properties
        if hasattr(old_mode, "anchor"):
            assert old_mode.anchor == new_mode.anchor

            assert old_mode.L == pytest.approx(new_mode.L)


def test_live_initial_state(scenario):
    """Verify that the live simulation starts correctly."""

    live = LiveSimulator(**scenario)

    state = live.get_state()

    assert state["time"] == pytest.approx(0.0)
    assert state["attached"] is True
    assert state["anchor"] == scenario["anchor"]
    assert state["finished"] is False


def test_live_get_result_before_completion(scenario):
    """A result should only be available after termination."""

    live = LiveSimulator(**scenario)

    with pytest.raises(RuntimeError):
        live.get_result()