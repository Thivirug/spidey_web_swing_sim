from .diagnostics import (
    energy_diagnostic,
    plot_energy_err,
    sim_swing_traj,
    plot_swing_traj
)

from .objectives import (
    objv1,
    objv2,
    get_initial_cartesian_state,
    get_final_cartesian_state
)

from .simulation import simv3
from .events import WebAttachedEvent

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# -------------------------------------------------------
def show_energy_plot():
    times, _, rel_errors = energy_diagnostic(
        T=5.0,
        dt=0.01,
        the0=45.0,
        omeg0=0.0,
        L=15.0,
        m=1.0
    )

    plot_energy_err(times, rel_errors)

# -------------------------------------------------------

anchor = (0.0, 20.0)

def show_swing_traj_plot():
    l = 20.9
    thetas = sim_swing_traj(
        L=l,
        theta0=-45,
        omega0=0.0,
        T=10,
        dt=0.01
    )

    anchor = (-5.4, 7.9)
    plot_swing_traj(thetas, anchor, l)

# -------------------------------------------------------
def plot_convergence(history, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluations = np.array([
        h["evaluation"] for h in history
    ])

    costs = np.array([
        h["robust_cost"] for h in history
    ])

    # Best cost discovered up to each evaluation
    best_costs = np.minimum.accumulate(costs)

    plt.figure(figsize=(9, 5))

    plt.plot(
        evaluations,
        costs,
        alpha=0.35,
        label="Candidate cost"
    )

    plt.plot(
        evaluations,
        best_costs,
        linewidth=2,
        label="Best cost so far"
    )

    plt.xlabel("Objective evaluation")
    plt.xticks(rotation=90)
    plt.ylabel("Robust cost")
    plt.title("Nelder-Mead Optimisation Convergence")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "convergence.png",
        dpi=300
    )

    plt.close()

def evaluate_scenarios(
    optimal_params,
    obj_info,
    scenarios,
    scenario_names=None
):
    """
        Evaluate the fixed optimal policy across all scenarios.

        Each scenario is simulated exactly once.

        Returns:
            scores:
                np.ndarray of objective scores.

            diagnostics:
                List of dictionaries containing detailed simulation outcomes.
    """

    scores = []
    diagnostics = []

    if scenario_names is None:
        scenario_names = [
            f"scenario_{i}"
            for i in range(1, len(scenarios) + 1)
        ]

    if len(scenario_names) != len(scenarios):
        raise ValueError(
            "scenario_names and scenarios must have equal length"
        )

    for i, (name, scenario) in enumerate(
        zip(scenario_names, scenarios),
        start=1
    ):

        # 1. Run simulation once

        sim_result = simv3(
            theta0=scenario["theta0"],
            omega0=scenario["omega0"],

            L=scenario["initial_L"],
            anchor=scenario["initial_anchor"],

            policy_params=optimal_params,

            T=scenario["T"],
            dt=scenario["dt"],

            y_ground=scenario.get("y_ground", 0.0),
            drag_params=scenario.get("drag_params"),

            anchors=scenario["anchors"],

            L_min=scenario.get("L_min", 5.0),
            L_max=scenario.get("L_max", 15.0),

            min_flight_time=scenario.get(
                "min_flight_time",
                0.15
            )
        )

        modes = sim_result.modes

        # 2. Calculate objective score

        if obj_info[0] == "v1":

            score = objv1(sim_result)

        elif obj_info[0] == "v2":

            if obj_info[1] is None:
                score = objv2(sim_result, scenario["T"])

            else:
                score = objv2(
                    sim_result, scenario["T"],
                    **obj_info[1]
                )

        else:
            raise ValueError(
                f"Invalid objective: {obj_info[0]}"
            )

        scores.append(score)

        # 3. Extract initial and final states

        initial_state = get_initial_cartesian_state(
            modes
        )

        final_state = get_final_cartesian_state(
            modes
        )

        horizontal_displacement = (
            final_state[0] - initial_state[0]
        )

        final_horizontal_velocity = final_state[2]

        # 4. Termination information

        termination_reason = (
            sim_result.termination_reason
        )

        termination_time = modes[-1].end_time

        # 5. Count successful reattachments

        # Every attached event after the initial
        # attached event represents a reattachment.

        attached_count = sum(
            isinstance(mode, WebAttachedEvent)
            for mode in modes
        )

        reattachments = max(
            0,
            attached_count - 1
        )

        # 6. Store scenario diagnostics

        record = {
            "scenario": i,
            "name": name,
            "objective_score": float(score),
            "termination_reason": termination_reason,
            "termination_time": float(termination_time),
            "horizontal_displacement": float(
                horizontal_displacement
            ),
            "final_horizontal_velocity": float(
                final_horizontal_velocity
            ),
            "reattachments": reattachments
        }

        diagnostics.append(record)

        # 7. Print individual scenario summary

        print(
            f"Scenario {i}: {name} | "
            f"Score={score:.4f} | "
            f"Termination={termination_reason} | "
            f"Time={termination_time:.2f}s | "
            f"Distance={horizontal_displacement:.2f}m | "
            f"Vx={final_horizontal_velocity:.2f}m/s | "
            f"Reattachments={reattachments}"
        )

    return np.asarray(scores), diagnostics

def plot_scenario_scores(scores, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.bar(
        np.arange(1, len(scores) + 1),
        scores
    )

    plt.axhline(
        np.mean(scores),
        linestyle="--",
        label="Mean score"
    )

    plt.xlabel("Scenario")
    plt.ylabel("Objective score")
    plt.title("Robust Controller Performance Across Scenarios")

    plt.xticks(
        np.arange(1, len(scores) + 1), rotation="vertical", fontsize=7
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "scenario_performance.png",
        dpi=300
    )

    plt.close()

def export_scenario_diagnostics(
    diagnostics,
    output_dir
):
    """
        Export scenario metrics to a CSV file.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "scenario_diagnostics.csv"
    )

    fieldnames = [
        "scenario",
        "name",
        "objective_score",
        "termination_reason",
        "termination_time",
        "horizontal_displacement",
        "final_horizontal_velocity",
        "reattachments"
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(diagnostics)

    return output_path

def print_scenario_summary(diagnostics):

    total = len(diagnostics)

    if total == 0:
        print("No scenarios evaluated.")
        return

    collisions = sum(
        d["termination_reason"] == "ground_collision"
        for d in diagnostics
    )

    time_limits = sum(
        d["termination_reason"] == "time_limit"
        for d in diagnostics
    )

    collision_rate = collisions / total

    avg_distance = np.mean([
        d["horizontal_displacement"]
        for d in diagnostics
    ])

    avg_reattachments = np.mean([
        d["reattachments"]
        for d in diagnostics
    ])

    print("\n========== SCENARIO SUMMARY ==========")

    print(f"Total scenarios: {total}")

    print(f"Ground collisions: {collisions}")

    print(f"Time-limit completions: {time_limits}")

    print(
        f"Collision rate: "
        f"{collision_rate * 100:.2f}%"
    )

    print(
        f"Average displacement: "
        f"{avg_distance:.3f} m"
    )

    print(
        f"Average reattachments: "
        f"{avg_reattachments:.2f}"
    )

    print("======================================\n")

if __name__ == '__main__':
    show_energy_plot()
    show_swing_traj_plot()