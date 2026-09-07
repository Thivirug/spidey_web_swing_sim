from .solver import RK4
from .physics import energy
from .utils import angle_to_pos_conv

import numpy as np
import matplotlib.pyplot as plt

def energy_diagnostic(T, dt, the0, omeg0, L, m):
    """
        Run a RK4 simulation and get energy values for each time and relative errors

        the0 - in degrees
    """

    rk4 = RK4(dt)

    times, states = rk4.simulate(
        np.radians(the0), 
        omeg0,
        L,
        T
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

# ---------------------------------------------------------------------------------------------------

def convert_angles_to_xy(theta_values, anchor, L):
    """
        Convert an array of angular positions into x,y coordinate
    """

    x_vals, y_vals = [], []

    for theta in theta_values:

        x, y = angle_to_pos_conv(anchor, theta, L)

        x_vals.append(x)
        y_vals.append(y)

    return np.array(x_vals), np.array(y_vals)

def sim_swing_traj(L, theta0, omega0, T, dt) -> np.ndarray:
    """ 
        Simulate the swing trajectory and return angle values
    """
    rk4 = RK4(dt)
    
    _, states = rk4.simulate(
        np.radians(theta0), 
        omega0,
        L,
        T
    )

    return states[:, 0]

def plot_swing_traj(theta_values, anchor, L):
    """
        PLot the swinging trajectory in 2D cartesian space
    """

    x_values, y_values = convert_angles_to_xy(
        theta_values,
        anchor,
        L
    )

    plt.figure()

    plt.plot( # plot x, y coords
        x_values,
        y_values,
        label="Swing trajectory"
    )
    plt.plot( # show anchor point
        anchor[0],
        anchor[1],
        marker="x",
        label="Anchor"
    )
    # Draw web at initial position
    plt.plot(
        [anchor[0], x_values[0]],
        [anchor[1], y_values[0]],
        linestyle="--",
        label="Web"
    )

    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.title("Spider-Man 2D Swing Trajectory")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.savefig(
            "output/figures/2d_swing_traj.png",
            dpi=300,
            bbox_inches="tight"
        )
    
