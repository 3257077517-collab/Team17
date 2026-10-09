from . import *

import colorsys
import numpy as np
from numpy.random import default_rng
import scipy.integrate as itg
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cycler import cycler

plt.style.use("petroff8")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = True
mpl.rcParams["savefig.transparent"] = True

t_bounds = (0.0, 2.0)
ivp_spec = random_pff_ivp(default_rng(0x666666), out_dim=4, in_dim=3)

runs_per_method = 6
fixed_step_kwargs_list = [
    dict(
        fun=ivp_spec.pff_fun,
        t_span=t_bounds,
        y0=np.ravel(ivp_spec.X0),
        num_steps=(1 << (2 * i)) * 4,
    )
    for i in range(runs_per_method)
]
adaptive_kwargs_list = [
    dict(
        fun=ivp_spec.pff_fun,
        jac=ivp_spec.pff_jac,
        t_span=t_bounds,
        y0=np.ravel(ivp_spec.X0),
        rtol=0.0,
        atol=np.exp2(-6 * i - 4),
    )
    for i in range(runs_per_method)
]

error_analyses = {
    name: do_error_analysis(
        method, kwargs_list, ivp_spec.pff_exact_sol_fn, ivp_spec.pff_norm_fn
    )
    for name, method, kwargs_list in [
        *[
            (name, method, fixed_step_kwargs_list)
            for name, method in (
                ("fs RK1", NaiveRK11),
                ("fs RK3", NaiveRK23),
                ("fs RK5", NaiveRK45),
            )
        ],
        *[
            (name, method, adaptive_kwargs_list)
            for name, method in (
                ("as RK1(1)", RK11),
                ("as RK3(2)", itg.RK23),
                ("as RK5(4)", itg.RK45),
                ("Radau IIA", itg.Radau),
                ("BDF", itg.BDF),
                ("Adams/BDF", itg.LSODA),
            )
        ],
    ]
}

(
    naive_rk11_error_analysis_result,
    naive_rk23_error_analysis_result,
    naive_rk45_error_analysis_result,
    rk11_error_analysis_result,
    rk23_error_analysis_result,
    rk45_error_analysis_result,
    radau_error_analysis_result,
    *_,
) = error_analyses.values()

jev_scale, lu_scale = measure_neff_scales(ivp_spec)

make_sv_streamplot_fig()
plt.savefig("figures/sv_streamplot.svg")

make_polygon_fig(
    ivp_spec,
    naive_rk11_error_analysis_result.solns[0],
)
plt.savefig("figures/fixed_euler_course_run.svg")

make_polygon_fig(
    ivp_spec,
    radau_error_analysis_result.solns[1],
)
plt.savefig("figures/radau_course_run.svg")


make_error_analysis_fig(
    ivp_spec, naive_rk11_error_analysis_result, mpl.colormaps["summer"]
)
plt.savefig("figures/fixed_euler_err.svg")

make_error_analysis_fig(ivp_spec, rk11_error_analysis_result, mpl.colormaps["autumn"])
plt.savefig("figures/adapt_euler_err.svg")

make_error_analysis_fig(
    ivp_spec,
    naive_rk23_error_analysis_result,
    mpl.colormaps["winter"],
)
plt.savefig("figures/fixed_rk23_err.svg")
make_error_analysis_fig(
    ivp_spec,
    rk23_error_analysis_result,
    mpl.colormaps["spring"],
)
plt.savefig("figures/adapt_rk23_err.svg")

make_work_precision_fig(error_analyses, jev_scale, lu_scale)
plt.savefig("figures/work_precision.svg")
