import numpy as np
from typing import Optional

from .controller import PolicyParams
from .simulation import simv3
from .objectives import objv1, objv2

def policy_cost(
    obj_info: tuple[str, Optional[dict]], # obj type version name and needed params
    p,
    theta0,
    omega0,
    initial_L,
    initial_anchor,
    T,
    dt,
    anchors,
    drag_params=None,
    y_ground=0.0,
    L_min=5.0,
    L_max=15.0,
    min_flight_time=0.15
):
    """
        Nelder-Mead cost wrapper.

        p :  p[0] = theta_r in radians p[1] = preferred web length

        float
            Minimisation cost.
    """
    if p.shape != (2,):
        raise ValueError(
            "p must contain [theta_r, preferred_L]"
        )
    
    theta_r, pref_L = p
    params = PolicyParams(theta_r, pref_L)

    modes = simv3(
        theta0=theta0,
        omega0=omega0,
        L=initial_L,
        anchor=initial_anchor,
        policy_params=params,
        T=T,
        dt=dt,
        y_ground=y_ground,
        drag_params=drag_params,
        anchors=anchors,
        L_min=L_min,
        L_max=L_max,
        min_flight_time=min_flight_time
    )

    if obj_info[0] == 'v1':
        score = objv1(modes)
    elif obj_info[0] == 'v2':
        if obj_info[1] is None:
            score = objv2(modes)
        else:
            score = objv2(
                modes,
                **obj_info[1]
            )
    else:
        raise ValueError(f"{obj_info} invalid objective")

    return -score # since we optimise a maximisation, but the algo is a minimiser

if __name__ == '__main__': # test
    print(" ====== INTERNAL TESTING ======")

    anchors = [
            (12.0, 19.0),
            (15.0, 20.0),
            (18.0, 22.0),
            (27.0, 19.0),
            (30.0, 22.0),
            (33.0, 21.0),
            (42.0, 18.0),
            (45.0, 20.0),
            (48.0, 22.0),
        ]
    Ls = [6.0, 10.0, 12.0, 14.0]

    print("\t\n======================= OBJv1 =======================")

    for L_pref in Ls:
        p = np.array([
            np.radians(30),
            L_pref
        ])

        cost = policy_cost(
            ("v1", None),
            p,
            theta0=np.radians(-45),
            omega0=0.0,
            initial_L=11.3,
            initial_anchor=(0.0, 20.0),
            T=12.0,
            dt=0.01,
            anchors=anchors,
            y_ground=0.1
        )

        print(
            "theta =", 30,
            "L =", L_pref,
            "cost =", cost
        )

    print("\t\n======================= OBJv2 =======================")

    for L_pref in Ls:
        p = np.array([
            np.radians(30),
            L_pref
        ])

        obj_info = (
            "v2",
            {
                "w_dist": 0.7,
                "w_vel": 0.3,
                "dist_scale": 50.0,
                "vel_scale": 10.0,
                "failure_penalty": 0.5
            }
        )
        cost = policy_cost(
            obj_info,
            p,
            theta0=np.radians(-45),
            omega0=0.0,
            initial_L=11.3,
            initial_anchor=(0.0, 20.0),
            T=12.0,
            dt=0.01,
            anchors=anchors,
            y_ground=0.1
        )

        print(
            "theta =", 30,
            "L =", L_pref,
            "cost =", cost
        )