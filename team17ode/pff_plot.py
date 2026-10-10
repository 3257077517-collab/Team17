import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.parasite_axes import HostAxes
from .polar_factor_flow import pff_sv_fun, pff_sv_exact, PFFIVPSpec

log_scale_kwargs = dict(base=10)


def hack_off_autoscale(ax, x_clip=None, y_clip=None):
    # I want to do this but it seems buggy???
    # ax.set_autoscale_on(False)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if x_clip is not None:
        xlim = np.clip(xlim, *x_clip)
    if y_clip is not None:
        ylim = np.clip(ylim, *y_clip)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def plot_pff_sv_flow_lines(
    ax: plt.Axes, num_lines, t_resolution=129, t_left=None, t_right=None, **plot_kwargs
):
    if t_left is None or t_right is None:
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
    t_left = soln.t[0]
    t_right = soln.t[-1]
    ts = np.linspace(t_left, t_right, exact_t_resolution)

    if soln.t.shape[0] <= exact_t_resolution:
        soln_svss = np.diagonal(
            spec.U.T @ soln.y.T.reshape((soln.y.shape[1], *spec.X0.shape)) @ spec.V,
            axis1=1,
            axis2=2,
        ).T

        for i, soln_svs in enumerate(soln_svss):
            ax.plot(soln.t, soln_svs, **plot_args_getter(i))
    else:
        soln_svss = np.diagonal(
            spec.U.T
            @ soln.sol(ts).T.reshape((exact_t_resolution, *spec.X0.shape))
            @ spec.V,
            axis1=1,
            axis2=2,
        ).T

        for i, soln_svs in enumerate(soln_svss):
            ax.plot(ts, soln_svs, **plot_args_getter(i))

    if should_plot_exact:
        for i, soln_svs in enumerate(soln_svss):
            ax.plot(ts, pff_sv_exact(ts, soln_svs[0]), **exact_args_getter(i))


def make_sv_streamplot_fig() -> plt.Figure:
    fig: plt.Figure
    ax_flow: plt.Axes
    ax_strip: plt.Axes
    fig, (ax_flow, ax_strip) = plt.subplots(
        1, 2, figsize=plt.figaspect(1), width_ratios=(15, 1), sharey=True
    )
    fig.set_tight_layout(True)

    ax_flow.set_xlabel("$t$")
    ax_flow.set_ylabel("$\\pm\\sigma_i$", labelpad=12, rotation=0)
    ax_flow.set_ylim(-2.0, 2.0)
    ax_flow.set_aspect("equal", adjustable="datalim")
    plot_pff_sv_flow_lines(
        ax_flow, 16, t_left=0.0, t_right=3.5, linewidth=0.5, color="C1", zorder=-1
    )

    ax_flow.axhline(-1.0, linewidth=1.0, color="C0")
    ax_flow.axhline(0.0, linewidth=1.0, color="C0")
    ax_flow.axhline(1.0, linewidth=1.0, color="C0")

    ax_strip.set_xlabel("$\\dot{\\sigma}_i$", labelpad=12, rotation=0)
    ax_strip.tick_params(labelbottom=False, which="both")

    sigmas = np.linspace(*ax_flow.get_ylim(), 33)[1:-1]
    rise = pff_sv_fun(0.0, sigmas)
    run = np.ones_like(sigmas)
    norms = np.linalg.vector_norm(np.array([run, rise]), axis=0)
    ax_strip.quiver(
        np.zeros_like(sigmas),
        sigmas,
        run / norms,
        rise / norms,
        scale=np.exp2(2.0),
        width=np.exp2(-5.0),
        headlength=0,
        headaxislength=0,
        headwidth=0,
        pivot="middle",
        color="C2",
    )
    return fig


def make_polygon_fig(ivp_spec, soln, **fig_kwargs) -> plt.Figure:
    fig: plt.Figure = plt.figure(**fig_kwargs)

    ax: plt.Axes = fig.add_subplot()

    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\pm\\sigma_i$", labelpad=12, rotation=0)

    plot_polygon_of_pff_soln_svs(ax, ivp_spec, soln)

    hack_off_autoscale(ax, y_clip=(-2.5, 2.5))
    plot_pff_sv_flow_lines(ax, 16, color="C6", linewidth=0.125, zorder=-1)

    ax.axhline(-1.0, linewidth=0.5, linestyle="-.", color="C3")
    ax.axhline(0.0, linewidth=0.5, linestyle="-.", color="C3")
    ax.axhline(1.0, linewidth=0.5, linestyle="-.", color="C3")

    return fig


def make_error_analysis_fig(
    ivp_spec: PFFIVPSpec,
    result,
    cmap,
    t_resolution=129,
    figkwargs=dict(figsize=7.2 * np.array([16.0 / 9.0, 1.0])),
) -> plt.Figure:
    fig: plt.Figure = plt.figure(**figkwargs)
    fig.set_constrained_layout(True)

    ax_eigen = fig.add_subplot(221)
    ax_fevals = fig.add_subplot(222)
    ax_lerror = fig.add_subplot(223)
    ax_errsum: HostAxes = fig.add_subplot(224, axes_class=HostAxes)
    twin_errsum_lin = ax_errsum.get_aux_axes(
        viewlim_mode=None, axes_class=mpl.axes.Axes
    )
    ax_eigen.sharex(ax_lerror)
    ax_errsum.sharey(ax_lerror)
    ax_errsum.sharex(ax_fevals)

    ax_eigen.set_xlabel("$t$")
    ax_eigen.xaxis.set_label_position("top")
    ax_eigen.set_ylabel("$\\pm\\sigma$", labelpad=12, rotation=0)
    ax_eigen.tick_params(labeltop=True)
    ax_eigen.tick_params(top=True, labelbottom=False, which="both")

    ax_lerror.set_xlabel("$t$")
    ax_lerror.set_ylabel("$\\left\\Vert E_L \\right\\Vert$", labelpad=12, rotation=0)
    ax_lerror.set_yscale("log", **log_scale_kwargs)
    ax_lerror.tick_params(top=True, right=True, which="both")

    ax_errsum.set_xlabel("$h$")
    ax_errsum.set_xscale("log", **log_scale_kwargs)
    ax_errsum.set_ylabel("$\\left\\Vert E \\right\\Vert$", labelpad=12)
    ax_errsum.axis["left"].toggle(ticks=True, ticklabels=False, label=False)
    ax_errsum.axis["right"].toggle(all=True)
    ax_errsum.axis["right"].label.set_pad(12)
    ax_errsum.axis["right"].label._text_follow_ref_angle = False
    ax_errsum.axis["bottom"].minor_ticklabels.set_visible(False)

    ax_fevals.set_xlabel("$h$")
    ax_fevals.xaxis.set_label_position("top")
    ax_fevals.set_ylabel("$n_{\\mathrm{ev}}$", labelpad=12, rotation=0)
    ax_fevals.set_yscale("log", **log_scale_kwargs)
    ax_fevals.yaxis.set_label_position("right")
    ax_fevals.tick_params(labeltop=True, labelright=True)
    ax_fevals.tick_params(
        top=True,
        labelbottom=False,
        left=False,
        right=True,
        which="both",
        labelleft=False,
    )

    ts = np.linspace(*result.solver_kwargs_list[0]["t_span"], t_resolution)

    num_solns = len(result.solns)
    colors = np.array(
        [cmap(1.0 - i / ((num_solns - 1) or 1)) for i in range(num_solns)]
    )
    for i in range(num_solns):
        soln = result.solns[i]
        color = colors[i]
        step_size_seq = result.step_size_seqs[i]
        local_error_curve = result.local_error_curves[i]

        plot_polygon_of_pff_soln_svs(
            ax_eigen,
            ivp_spec,
            soln,
            plot_args_getter=lambda k: dict(
                color=color,
                linewidth=0.5,
                fillstyle="none",
                marker=("x", "+", "o")[k],
                markersize=3,
                zorder=-i,
            ),
            should_plot_exact=False,
            exact_t_resolution=t_resolution,
        )

        ax_lerror.step(
            soln.t[:-1],
            np.concat((local_error_curve[:1], local_error_curve[:-1])),
            linewidth=1.0,
            color=color,
            label="$\\left\\Vert E_L \\right\\Vert$" if i == 0 else None,
        )

    hack_off_autoscale(ax_eigen, y_clip=(-1.125, 2.5))
    plot_pff_sv_flow_lines(
        ax_eigen, 17, t_resolution=t_resolution, color="C6", linewidth=0.125, zorder=-1
    )

    _, max_local_err = ax_lerror.get_ylim()
    if max_local_err >= 8.0:
        ax_lerror.set_ylim(top=8.0)

    log2_avg_h = np.log2(result.avg_step_sizes)
    violinparts = twin_errsum_lin.violinplot(
        [np.log2(curve[:-1]) for curve in result.local_error_curves],
        log2_avg_h,
        widths=0.0625 * (np.max(log2_avg_h) - np.min(log2_avg_h)),
        linecolor=colors,
        facecolor=colors * np.array([1.0, 1.0, 1.0, 0.25]),
    )
    for lc in violinparts.values():
        if isinstance(lc, mpl.collections.LineCollection):
            lc.set_linewidth(1.0)
    ax_errsum.set_xlim(tuple(np.exp2(bound) for bound in twin_errsum_lin.get_xlim()))

    ax_errsum.scatter(
        result.avg_step_sizes,
        result.avg_local_errors,
        marker=mpl.markers.MarkerStyle("o", fillstyle="none"),
        c=np.clip(colors, 0.0, 1.0),
        label="$\\overline{\\left\\Vert E_L \\right\\Vert}$",
    )
    ax_errsum.scatter(
        result.avg_step_sizes,
        result.global_errors,
        marker="x",
        c=colors,
        label="$\\left\\Vert E_G \\right\\Vert$",
    )
    ax_errsum.legend()
    twin_errsum_lin.set_ylim(*np.log2(np.array(ax_errsum.get_ylim())))

    default_marker_size = mpl.rcParams["lines.markersize"] ** 2
    ax_fevals.scatter(
        result.avg_step_sizes,
        [soln.nfev for soln in result.solns],
        s=0.5 * default_marker_size,
        marker="D",
        c=colors,
        label="$n_{\\mathrm{fev}}$",
    )
    if any(soln.njev > 0 for soln in result.solns):
        ax_fevals.scatter(
            result.avg_step_sizes,
            [soln.njev for soln in result.solns],
            s=default_marker_size,
            marker="P",
            c=colors,
            label="$n_{\\mathrm{jev}}$",
        )
    if any(soln.nlu > 0 for soln in result.solns):
        ax_fevals.scatter(
            result.avg_step_sizes,
            [soln.nlu for soln in result.solns],
            s=0.5 * default_marker_size,
            marker="s",
            c=colors,
            label="$n_{\\mathrm{lu}}$",
        )

    ax_fevals.legend()

    return fig


def make_work_precision_fig(
    error_analyses: dict, jev_scale: float, lu_scale: float, markerset="xxxxoooo+++"
) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.set_xlabel("$\\left\\Vert E_G \\right\\Vert$")
    ax.set_xscale("log", **log_scale_kwargs)
    ax.set_xinverted(True)
    ax.set_ylabel("$n_{\\mathrm{eff}}$", labelpad=12, rotation=0)
    ax.set_yscale("log", **log_scale_kwargs)

    for i, (name, result) in enumerate(error_analyses.items()):
        work = [
            soln.nfev + jev_scale * soln.njev + lu_scale * soln.nlu
            for soln in result.solns
        ]
        ax.plot(
            result.global_errors,
            work,
            f"--{markerset[i % len(markerset)]}",
            linewidth=0.75,
            fillstyle="none",
            label=name,
        )

    hack_off_autoscale(ax, x_clip=(0.5 * np.finfo(float).eps, 8.0))
    ax.axvline(np.finfo(float).eps, zorder=-1, linestyle=":")
    ax.legend()
    return fig
