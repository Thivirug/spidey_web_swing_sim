## Running the Project

### 1. Run the live simulation

From the project root:

```powershell
python -m sim.main
```

This launches the Ursina visualisation using the currently configured controller parameters in `sim/main.py`.

---

### 2. Re-run the offline optimisation

If you want to optimise the controller parameters yourself, use the provided PowerShell script:

```powershell
.\scripts\run_optim.ps1 <scenario>
```

For example:

```powershell
.\scripts\run_optim.ps1 challenging
```

The script runs:

```powershell
python -m src.mainV2 scenarios=challenging
```

and performs the offline Nelder–Mead optimisation for the selected Hydra scenario configuration.

Available scenario names correspond to the configuration files in the project scenario/config directory.

The optimisation may take several minutes because each Nelder–Mead objective evaluation runs the simulation across multiple scenarios.

After optimisation, the terminal output reports the optimal:

- release angle
- preferred web length
- robust objective cost

These parameters can then be placed in `sim/main.py` to run the optimised controller in the live Ursina simulation.

### Conda note

The supplied PowerShell script contains:

```powershell
conda activate spidey
```

so it expects a Conda environment named `spidey`.

If you are using a different environment, either activate your environment manually before running the command or change this line in `scripts/run_optim.ps1`.

For example:

```powershell
conda activate your_environment
python -m src.mainV2 scenarios=challenging
```