import numpy as np

from .optim import optimise_policy
from .simulation import simv3


def main():
    # 1- Fixed simulation setup

    theta0 = np.radians(-30.0)
    omega0 = 1.0

    initial_L = 10.0
    initial_anchor = (0.0, 20.0)

    T = 20.0
    dt = 0.01

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

    drag_params = None

    y_ground = 0.0

    L_min = 5.0
    L_max = 15.0

    min_flight_time = 0.15


    # 2- Initial guess for controller parameters
    # p = [theta_r, preferred_L]

    initial_p = np.array([
        np.radians(30.0),
        12.0
    ])


    # 3- Offline optimisation using objective v1

    optimal_params_v1, best_cost_v1 = optimise_policy(
        initial_p=initial_p,
        obj_info=(
            "v1",
            None
        ),
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
        min_flight_time=min_flight_time,
        tol=1e-4
    )


    print()
    print("========================================")
    print("Objective v1 optimisation")
    print("========================================")

    print(
        "Optimal release angle:",
        np.degrees(optimal_params_v1.theta_r),
        "deg"
    )

    print(
        "Optimal preferred L:",
        optimal_params_v1.L
    )

    print(
        "Best cost:",
        best_cost_v1
    )

    print(
        "Best objective score:",
        -best_cost_v1
    )


    # 4- Offline optimisation using objective v2

    objv2_args = {
        "w_dist": 0.7,
        "w_vel": 0.3,

        "dist_scale": 50.0,
        "vel_scale": 10.0,

        "failure_penalty": 0.0
    }


    optimal_params_v2, best_cost_v2 = optimise_policy(
        initial_p=initial_p,
        obj_info=(
            "v2",
            objv2_args
        ),
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
        min_flight_time=min_flight_time,
        tol=1e-4
    )


    print()
    print("========================================")
    print("Objective v2 optimisation")
    print("========================================")

    print(
        "Optimal release angle:",
        np.degrees(optimal_params_v2.theta_r),
        "deg"
    )

    print(
        "Optimal preferred L:",
        optimal_params_v2.L
    )

    print(
        "Best cost:",
        best_cost_v2
    )

    print(
        "Best objective score:",
        -best_cost_v2
    )


    # 5- Choose final policy
    # use objective v2

    final_policy = optimal_params_v2


    # 6- Final simulation
   
    # NOTE: Nelder-Mead is finished before this simulation starts.
    # This simulation uses the fixed optimal parameters.

    final_modes = simv3(
        theta0=theta0,
        omega0=omega0,
        L=initial_L,
        anchor=initial_anchor,
        T=T,
        dt=dt,
        anchors=anchors,
        policy_params=final_policy,
        drag_params=drag_params,
        y_ground=y_ground,
        L_min=L_min,
        L_max=L_max,
        min_flight_time=min_flight_time
    )


    # 7- Final result

    print()
    print("========================================")
    print("Final fixed controller")
    print("========================================")

    print(
        "Release angle:",
        np.degrees(final_policy.theta_r),
        "deg"
    )

    print(
        "Preferred web length:",
        final_policy.L
    )

    print(
        "Number of trajectory modes:",
        len(final_modes)
    )


    # 8. Plot / animate final_modes 

    # plot_simulation(final_modes, anchors)
    # animate_simulation(final_modes, anchors)


if __name__ == "__main__":
    main()