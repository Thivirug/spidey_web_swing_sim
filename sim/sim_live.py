# simv3() like but an interface designed for exposing live sim steps for the 3d visualiser (ursina)

import numpy as np

from src.solver import RK4
from src.controller import WebSwingController
from src.events import *
from src.utils import (
    attached_to_released_state,
    interpolate_landing,
    released_to_attached_state,
)

class LiveSimulator:
    """
        Incremental version of simv3 func

        Each call to step() executes one iteration of simv3's simulation loop
    """

    def __init__(
        self,
        theta0,
        omega0,
        L,
        anchor, # initial anchor
        policy_params,
        T,
        dt,
        y_ground=0.0,
        drag_params=None,
        anchors=None,
        L_min=5.0,
        L_max=15.0,
        min_flight_time=0.15,
    ):
        self.solver = RK4(dt)
        self.controller = WebSwingController(L_min, L_max)
        self.T = T
        self.t = 0.0
        self.y_ground = y_ground
        self.min_flight_time = min_flight_time
        self.policy_params = policy_params
        self.drag_params - drag_params
        self.anchors = anchors

        # starting state is always attached
        self.state = np.array([theta0, omega0], dtype=float)

        self.mode = WebAttachedEvent(
            start_time=self.t,
            anchor=anchor,
            L=L
        )

        self.modes = [] # store the events during the sim

        self.prev_anchor = None
        # termination management
        self.finished = False # ground collision / time limit done
        self.termination_reason = "time_limit"

    def step(self):
        """
            Advance the simulation by one dt
        """

        if self.finished:
            return 

        if self.t >= self.T: # current t exceeded sim time
            self._finish_time_limit()
            return

        # ATTACHED motion
        if isinstance(self.mode, WebAttachedEvent):
            self.mode.states.append(
                (self.t, self.state.copy())
            )

            if self.controller.should_release(
                self.state,
                self.policy_params,
            ):
                # Complete the attached event.
                self.mode.end_time = self.t
                self.modes.append(self.mode)

                self.prev_anchor = self.mode.anchor

                # Convert pendulum state to projectile state.
                self.state = attached_to_released_state(
                    self.state,
                    self.mode.anchor,
                    self.mode.L,
                )

                # Start a new released event.
                self.mode = ReleasedEvent(
                    start_time=self.t
                )

            else:
                # Normal attached RK4 step.
                self.state = self.solver.step(
                    self.state,
                    self.mode.L,
                )

        # PROJECTILE/ released motion
        if isinstance(self.mode, ReleasedEvent):
            self.mode.states.append(
                (self.t, self.state.copy())
            )

            # GROUND COLLISION
            y = self.state[1]
            vy = self.state[3]

            if y <= self.y_ground and vy < 0: # ground hit
                if len(self.mode.states) >= 2: # req for interpol.

                    t_prev, state_prev = (
                        self.mode.states[-2]
                    )

                    t_curr, state_curr = (
                        self.mode.states[-1]
                    )

                    landing_t, landing_state = (
                        interpolate_landing(
                            t_prev,
                            t_curr,
                            state_prev,
                            state_curr,
                            self.y_ground,
                        )
                    )

                    self.mode.states[-1] = (
                        landing_t,
                        landing_state,
                    )

                    # Expose the interpolated landing state
                    # to the renderer.
                    self.t = landing_t
                    self.state = landing_state.copy()

                    self.mode.end_time = landing_t
                else:
                    self.mode.end_time = self.t 

                self.modes.append(self.mode)
                self.termination_reason = "ground_collision"
                self.finished = True
                return

            # CHECK if REATTACHMENT POSSIBLE
            new_anchor = None
            flight_time = (
                self.t - self.mode.start_time
            )

            if flight_time >= self.min_flight_time:
                new_anchor = (
                    self.controller.choose_attachment(
                        self.state,
                        self.anchors,
                        self.policy_params,
                        self.prev_anchor,
                    )
                )

            if new_anchor is not None: # reattach success
                # Complete the released event.
                self.mode.end_time = self.t
                self.modes.append(self.mode)

                # Convert projectile state to pendulum.
                self.state, new_L = (
                    released_to_attached_state(
                        self.state,
                        new_anchor,
                    )
                )

                # Start a new attached event.
                self.mode = WebAttachedEvent(
                    start_time=self.t,
                    anchor=new_anchor,
                    L=new_L,
                )

                # Record exact attachment state.
                self.mode.states.append(
                    (self.t, self.state.copy())
                )

                # immediate step with current time
                self.state = self.solver.step(
                    self.state,
                    self.mode.L,
                )

                self.t += self.dt

                if self.t >= self.T:
                    self._finish_time_limit()

                return

            # If no collision or reattachment continue with projectile step
            self.state = self.solver.step_released(
                self.state,
                self.drag_params,
            )

        self.t += self.dt
        if self.t >= self.T:
            self._finish_time_limit()

    def _finish_time_limit(self):
        """
            Finalise the active event at the time limit
        """
        self.mode.end_time = self.T
        self.modes.append(self.mode)

        self.termination_reason = "time_limit"
        self.finished = True

    def get_state(self) -> dict:
        """
            Return the current Cartesian state for Ursina
        """
        attached = isinstance(
            self.mode,
            WebAttachedEvent,
        )

        if attached:
            cartesian = attached_to_released_state(
                self.state,
                self.mode.anchor,
                self.mode.L,
            )
            anchor = self.mode.anchor

        else:
            cartesian = self.state.copy()
            anchor = None

        x, y, vx, vy = cartesian

        return {
            "time": float(self.t),
            "x": float(x),
            "y": float(y),
            "vx": float(vx),
            "vy": float(vy),
            "attached": attached,
            "anchor": anchor,
            "finished": self.finished,
            "termination_reason": (
                self.termination_reason
                if self.finished
                else None
            ),
        }

    def get_result(self) -> FinalSimResult:
        """
            Return the completed result of the full simulation
        """

        if not self.finished:
            raise RuntimeError(
                "Simulation has not finished..."
            )

        return FinalSimResult(
            modes=self.modes,
            termination_reason=self.termination_reason,
        )