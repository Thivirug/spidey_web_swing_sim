import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

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

    for s in cfg.scenarios:
        params = common_params.copy()

        params.update({
            'theta0': np.radians(s.theta0_deg),
            'omega0': s.omega0,
            'initial_L': s.initial_L,
            'initial_anchor': tuple(s.initial_anchor)
        })

        scenarios.append(params)

    # optim - run nelder mead 
    optim_params, best_cost = optim_policy_robust(
        initial_p=init_p,
        obj_info=obj_info,
        scenarios=scenarios,
        std_weight=cfg.optim.std_weight,
        tol=cfg.optim.tol
    )

    print("Optimal release angle:", np.degrees(optim_params.theta_r))
    print("Optimal preferred L:", optim_params.L)
    print("Best robust cost:", best_cost)

if __name__ == "__main__":
    main()