from .solver import RK4
from .physics import energy

import numpy as np
import matplotlib.pyplot as plt

def energy_diagnostic(T, dt, the0, omeg0, L, m):
    """
        Run a RK4 simulation and get energy values for each time and relative errors

        the0 - in degrees
    """

    rk4 = RK4(T, dt)

    times, states = rk4.simulate(
        np.radians(the0), 
        omeg0,
        L
    )

    energies = np.array([
        energy(m, L, state)
        for state in states
    ])

    if np.isclose(energies[0], 0):
        raise ValueError("Initial energy must be non-zero for relative energy error")
    rel_en_errs = (energies - energies[0]) / energies[0]

    return times, energies, rel_en_errs

def plot_energy_err(times, relative_error):
    plt.figure()
    plt.plot(times, relative_error)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative energy error")
    plt.title("RK4 Energy Conservation")
    plt.axhline(0)
    plt.savefig(
        "output/figures/rk4_energy_conservation.png",
        dpi=300,
        bbox_inches="tight"
    )
