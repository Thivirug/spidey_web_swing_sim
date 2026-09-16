from .diagnostics import energy_diagnostic, plot_energy_err, sim_swing_traj, plot_swing_traj
from .optim import policy_cost
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
    scenarios
):

    p = np.array([
        optimal_params.theta_r,
        optimal_params.L
    ])

    scores = []

    for scenario in scenarios:

        cost = policy_cost(
            obj_info=obj_info,
            p=p,
            **scenario
        )

        scores.append(-cost)

    return np.asarray(scores)


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
        np.arange(1, len(scores) + 1)
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "scenario_performance.png",
        dpi=300
    )

    plt.close()

if __name__ == '__main__':
    show_energy_plot()
    show_swing_traj_plot()