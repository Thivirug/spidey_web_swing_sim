from .diagnostics import energy_diagnostic, plot_energy_err

def show_energy_plot():
    times, energies, rel_errors = energy_diagnostic(
        T=5.0,
        dt=0.01,
        the0=45.0,
        omeg0=0.0,
        L=15.0,
        m=1.0
    )

    plot_energy_err(times, rel_errors)



if __name__ == '__main__':
    show_energy_plot()