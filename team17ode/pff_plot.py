from .polar_factor_flow import pff_sv_exact, PFFIVPSpec


def plot_pff_sv_flow_lines(ax: plt.Axes, num_lines, t_resolution=129, **plot_kwargs):
    t_left, t_right = ax.get_xlim()

    ts = np.linspace(t_left, t_right, t_resolution)
    t0s = np.linspace(t_left, t_right, num_lines + 2)[1:-1]
    s0s = (t0s - t_left) / (t_right - t_left)

    y_bottom, y_top = ax.get_ylim()
    plot_kwargs = plot_kwargs or {}

    if y_bottom < -1.0:
        for t0, s0 in zip(t0s, s0s):
            y0 = -1.0 - (-1.0 - y_bottom) * s0
            ye = pff_sv_exact(ts - t0, y0)
            ax.plot(ts, ye, **plot_kwargs)

    if -1.0 < y_top or y_bottom < 0.0:
        for t0, s0 in zip(t0s, s0s):
            y0 = -1.0 + s0
            ye = pff_sv_exact(ts - t0, y0)
            ax.plot(ts, ye, **plot_kwargs)

    if 0.0 < y_top or y_bottom < 1.0:
        for t0, s0 in zip(t0s, s0s):
            y0 = 1.0 - s0
            ye = pff_sv_exact(ts - t0, y0)
            ax.plot(ts, ye, **plot_kwargs)

    if 1.0 < y_top:
        for t0, s0 in zip(t0s, s0s):
            y0 = 1.0 + (y_top - 1.0) * s0
            ye = pff_sv_exact(ts - t0, y0)
            ax.plot(ts, ye, **plot_kwargs)


def plot_polygon_of_pff_soln_svs(
    ax: plt.Axes,
    spec: PFFIVPSpec,
    soln: itg.OdeSolution,
    plot_args_getter=lambda i: dict(color=f"C{i}", marker=".", linestyle="-"),
    should_plot_exact=True,
    exact_args_getter=lambda i: dict(color=f"C{i}", linestyle=":", linewidth=1.0),
    exact_t_resolution=129,
):

    soln_svss = np.diagonal(
        spec.U.T @ soln.y.T.reshape((soln.y.shape[1], *spec.X0.shape)) @ spec.V,
        axis1=1,
        axis2=2,
    ).T

    for i, soln_svs in enumerate(soln_svss):
        ax.plot(soln.t, soln_svs, **plot_args_getter(i))

    if should_plot_exact:
        t_left = soln.t[0]
        t_right = soln.t[-1]
        ts = np.linspace(t_left, t_right, exact_t_resolution)
        for i, soln_svs in enumerate(soln_svss):
            ax.plot(ts, pff_sv_exact(ts, soln_svs[0]), **exact_args_getter(i))
