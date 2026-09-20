import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from hydra.core.hydra_config import HydraConfig

from .plot import (
    plot_convergence,
    evaluate_scenarios,
    plot_scenario_scores,
    export_scenario_diagnostics,
    print_scenario_summary,
    plot_trajectory_history
)
from .simulation import simv3
from .events import WebAttachedEvent
from .optim import optim_policy_robust

@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg: DictConfig):

    # prep initial policy for nelder mead
    init_p = np.array([
            np.radians(cfg.optim.initial_theta_r_deg),
            cfg.optim.initial_preferred_L
        ])

    # prep objective
    obj_cfg = cfg.optim.objective

    obj_info = (
        obj_cfg.version,
        OmegaConf.to_container(
            obj_cfg.args,
            resolve=True
        )
    )

    # prep shared sim params
    common_params = OmegaConf.to_container(
        cfg.simulation,
        resolve=True
    )
    # convert anchors from yaml lists to tuples
    common_params['anchors'] = [tuple(anchor) for anchor in common_params['anchors']]

    # build scenario params
    scenarios = []
    # preserve YAML scenario names
    scenario_names = []

    for s in cfg.scenarios:
        params = common_params.copy()

        params.update({
            'theta0': np.radians(s.theta0_deg),
            'omega0': s.omega0,
            'initial_L': s.initial_L,
            'initial_anchor': tuple(s.initial_anchor)
        })

        scenarios.append(params)
        scenario_names.append(s.name)
        
    # optim - run nelder mead 
    optim_params, best_cost, history = optim_policy_robust(
        initial_p=init_p,
        obj_info=obj_info,
        scenarios=scenarios,
        std_weight=cfg.optim.std_weight,
        tol=cfg.optim.tol
    )

    print("Optimal release angle:", np.degrees(optim_params.theta_r))
    print("Optimal preferred L:", optim_params.L)
    print("Best robust cost:", best_cost)

    output_dir = (
        Path(HydraConfig.get().runtime.output_dir)
        / "figures"
    )

    plot_convergence(
        history,
        output_dir
    )

    scores, diagnostics = evaluate_scenarios(
        optim_params,
        obj_info,
        scenarios,
        scenario_names=scenario_names
    )

    plot_scenario_scores(
        scores,
        output_dir
    )

    csv_path = export_scenario_diagnostics(
        diagnostics,
        output_dir
    )

    print_scenario_summary(
        diagnostics
    )

    print(
        f"Scenario diagnostics saved to: {csv_path}"
    )


    # ==========================================================
    # INDEPENDENT TEST: Optimised policy under NEW conditions
    # ==========================================================

    # TODO: Move this entire section out to the final sim file for 3D vis
    print("\n========== INDEPENDENT TEST ==========") 

    test_params = common_params.copy()

    # Change simulation conditions independently of optimisation
    test_params.update({
        "theta0": np.radians(-45.0),
        "omega0": 1.2,
        "initial_L": 10.0,
        "initial_anchor": (0.0, 22.0),

        # Longer simulation
        "T": 50.0,

        # Independent city layout: different from the config the controller was optimised for
        "anchors": [
            (8.0, 25.0),
            (13.0, 20.0),
            (19.0, 28.0),
            (24.0, 23.0),
            (31.0, 30.0),
            (39.0, 22.0),
            (44.0, 27.0),
            (50.0, 19.0),
            (57.0, 26.0),
            (64.0, 31.0),
            (70.0, 23.0),
            (77.0, 28.0),
            (85.0, 20.0),
            (91.0, 26.0),
            (98.0, 32.0),
            (105.0, 24.0),
            (113.0, 29.0),
            (120.0, 21.0),
            (128.0, 27.0),
            (135.0, 33.0),
            (143.0, 25.0),
            (150.0, 30.0),
            (158.0, 22.0),
            (166.0, 28.0),
        ],
    })

    # Run once with the already-optimised controller.
    test_scores, test_diagnostics = evaluate_scenarios(
        optimal_params=optim_params,
        obj_info=obj_info,
        scenarios=[test_params],
        scenario_names=["independent_test"]
    )

    # Save test outputs separately
    test_output_dir = (
        Path(HydraConfig.get().runtime.output_dir)
        / "independent_test"
    )

    # plot_scenario_scores(
    #     test_scores,
    #     test_output_dir
    # )

    test_csv_path = export_scenario_diagnostics(
        test_diagnostics,
        test_output_dir
    )

    print_scenario_summary(test_diagnostics)

    print(f"Independent test saved to: {test_csv_path}")

    test_result = simv3(
        theta0=test_params["theta0"],
        omega0=test_params["omega0"],
        L=test_params["initial_L"],
        anchor=test_params["initial_anchor"],
        policy_params=optim_params,
        T=test_params["T"],
        dt=test_params["dt"],
        y_ground=test_params["y_ground"],
        drag_params=test_params["drag_params"],
        anchors=test_params["anchors"],
        L_min=test_params["L_min"],
        L_max=test_params["L_max"],
        min_flight_time=test_params["min_flight_time"]
    )

    print("\n========== EVENT HISTORY ==========")
    
    for i, mode in enumerate(test_result.modes, start=1):

        print(
            f"Event {i}: "
            f"{type(mode).__name__} | "
            f"start={mode.start_time:.3f}s | "
            f"end={mode.end_time}"
        )

        if isinstance(mode, WebAttachedEvent):
            print(
                f"  Anchor: {mode.anchor} | "
                f"Web length: {mode.L:.3f}m"
            )

            theta_values = [
                state[0]
                for _, state in mode.states
            ]

            print(
                f"  Theta range: "
                f"{np.degrees(min(theta_values)):.2f}° "
                f"to "
                f"{np.degrees(max(theta_values)):.2f}°"
            )

    plot_trajectory_history(
        test_result,
        test_output_dir
    )

    print(
        f"Trajectory plots saved to: {test_output_dir}"
    )

if __name__ == "__main__":
    main()