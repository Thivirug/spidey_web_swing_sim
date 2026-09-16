import numpy as np
from typing import Optional
import time
import logging
log = logging.getLogger(__name__)
from colorama import Fore, Style, init
init(autoreset=True)

from .controller import PolicyParams
from .simulation import simv3
from .objectives import objv1, objv2
from .nelder_mead import NelderMead

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


# MAIN optimising policies
def optimise_policy(
    initial_p,
    obj_info,
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
    min_flight_time=0.15,
    tol=1e-4
):
    """
        Offline optimisation of controller parameters.

        Returns the optimal reusable PolicyParams.
    """

    def cost(p): # wrapper for nelder mead compatability
        return policy_cost(
            obj_info=obj_info,
            p=p,
            theta0=theta0,
            omega0=omega0,
            initial_L=initial_L,
            initial_anchor=initial_anchor,
            T=T,
            dt=dt,
            anchors=anchors,
            drag_params=drag_params,
            y_ground=y_ground,
            L_min=L_min,
            L_max=L_max,
            min_flight_time=min_flight_time
        )

    nm = NelderMead(
        f=cost,
        initial_point=np.asarray(initial_p, dtype=float),
        tol=tol
    )

    final_simplex = nm()

    best_p = final_simplex[0]
    best_cost = nm.f_values[0]

    optimal_params = PolicyParams(
        theta_r=best_p[0],
        L=best_p[1]
    )

    return optimal_params, best_cost

def optim_policy_robust(initial_p, obj_info, scenarios: list[dict], std_weight=0.25, tol=1e-6):
    """
        Optimise one reusable policy across multiple scenarios.

        - scenarios: list of dicts of param values for all defined scenarios

        Return
        - optimal_params - Robust controller parameters.
        - best_cost - robust aggregate cost across scenarios
    """

    t_start = time.perf_counter()
    eval_count=0 # count how many times nelder evals the cost obj

    def robust_cost(p): # wrapper for nelder mead compatability
        nonlocal eval_count
        eval_count += 1
        eval_start = time.perf_counter()

        costs = []
        n_scenarios = len(scenarios)

        # print(
        #     f"\n[Evaluation {eval_count}] "
        #     f"theta={np.degrees(p[0]):.2f} deg | "
        #     f"L={p[1]:.3f} m",
        #     flush=True
        # )

        for i, scenario_params in enumerate(scenarios,start=1):
            sim_start = time.perf_counter()

            cost = policy_cost( # cost for the particualr scenario
                obj_info=obj_info,
                p=p,
                **scenario_params
            )

            costs.append(cost)

            sim_elapsed = time.perf_counter() - sim_start

            print(
                Fore.CYAN + 
                f"  Scenario {i}/{n_scenarios} "
                f"| Cost: {cost:.4f} "
                f"| Time: {sim_elapsed:.2f}s",
                flush=True
            )

        costs = np.array(costs)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)

        result = np.mean(costs) + std_weight*np.std(costs)

        eval_elapsed = time.perf_counter() - eval_start
        total_elapsed = time.perf_counter() - t_start

        print(
            Fore.GREEN + Style.BRIGHT +
            f"[Evaluation {eval_count} completed] "
            f"Cost: {result:.6f} | "
            f"Evaluation time: {eval_elapsed:.2f}s | "
            f"Total elapsed: {total_elapsed:.1f}s",
            flush=True
        )

        log.info(
            "Evaluation=%d | theta=%.3f deg | L=%.3f m | "
            "mean=%.6f | std=%.6f | robust=%.6f | "
            "time=%.2fs",
            eval_count,
            np.degrees(p[0]),
            p[1],
            mean_cost,
            std_cost,
            result,
            eval_elapsed
        )

        return result

    # run nelder mead with the new cost func
    nm = NelderMead(
        f=robust_cost,
        initial_point=np.asarray(initial_p, dtype=float),
        tol=tol
    )

    final_simplex = nm()

    best_p = final_simplex[0]
    best_cost = nm.f_values[0]

    optimal_params = PolicyParams(
        theta_r=best_p[0],
        L=best_p[1]
    )

    t_total = time.perf_counter() - t_start

    print("\n========== OPTIMISATION COMPLETE ==========")
    log.info(
        "Optimisation finished | evaluations=%d | "
        "theta=%.3f deg | L=%.3f m | "
        "best_cost=%.6f | total_time=%.2fs",
        eval_count,
        np.degrees(best_p[0]),
        best_p[1],
        best_cost,
        t_total
    )

    return optimal_params, best_cost

if __name__ == '__main__': # test for policy_cost
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