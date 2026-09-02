from .diagnostics import energy_diagnostic, plot_energy_err, sim_swing_traj, plot_swing_traj

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

if __name__ == '__main__':
    show_energy_plot()
    show_swing_traj_plot()