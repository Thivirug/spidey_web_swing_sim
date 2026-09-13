from dataclasses import dataclass
import numpy as np
from typing import Optional

from .utils import is_anchor_reachable

@dataclass
class PolicyParams:
    theta_r: float  # Release-angle threshold in radians
    L: float # Preferred web length when choosing a new anchor

class WebSwingController:
    def __init__(self, L_min: float, L_max: float):
        if L_min<=0 or L_max<=0:
            raise ValueError("L's Must be positive values")

        if L_min >= L_max:
            raise ValueError("Min L must be strictly lower than Max L")

        self.L_min = L_min
        self.L_max = L_max

    def should_release(self, state: np.ndarray, params: PolicyParams) -> bool:
        """
            Decide whether to release the current web.
            Release when theta >= theta_r & Spidey is moving forward.

            - state:[theta, omega]
            - params: theta_r, L
        """

        theta, omega = state

        return (
            theta >= params.theta_r and omega >0
        )

    def choose_attachment(
            self, 
            state: np.ndarray, 
            anchors: list[tuple], 
            params:PolicyParams, 
            excluded_anchor=None
        ) -> Optional[None | tuple]:

        """
            Choose a new forward reachable anchor.
            Among all valid anchors, choose the one whose required
            web length is closest to params.L

            - Released state: [x, y, vx, vy]
            - excluded_anchor: previous anchor
        """

        x,y,_, _ = state

        best_err = np.inf
        best_anchor = None

        for anchor in anchors:
            # not connecting to the prev anchor
            if anchor == excluded_anchor:
                continue

            # anchor needs to be ahead of spidey
            if anchor[0] <= x:
                continue

            # anchor needs to be reachable 
            if not is_anchor_reachable(state, anchor, self.L_min, self.L_max):
                continue

            x_a, y_a = anchor

            # dist = np.sqrt(
            #     (x-x_a)**2 + (y-y_a)**2
            # )

            dist = np.hypot(
                x-x_a, y-y_a
            )

            err = np.abs(dist - params.L)

            if err < best_err:
                best_err = err
                best_anchor = anchor


        return best_anchor

