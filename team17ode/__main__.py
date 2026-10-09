from . import *

from collections import namedtuple
import colorsys
import warnings
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
warnings.filterwarnings("ignore")

np_orig_err_settings = np.seterr(all="ignore")

t_bounds = (0.0, 2.0)
ivp_spec = random_pff_ivp(default_rng(0x666666), out_dim=4, in_dim=3)

MethodSpec = namedtuple(
    "MethodSpec",
    [
        "name",
        "method",
        "order",
        "adaptive",
        "uses_jac",
    ],
)

method_specs = [
    MethodSpec("fs RK1", NaiveRK11, order=1, adaptive=False, uses_jac=False),
    MethodSpec("fs RK3", NaiveRK23, order=3, adaptive=False, uses_jac=False),
    MethodSpec("fs RK5", NaiveRK45, order=5, adaptive=False, uses_jac=False),
    MethodSpec("as RK1(1)", RK11, order=1, adaptive=True, uses_jac=False),
    MethodSpec("as RK3(2)", itg.RK23, order=3, adaptive=True, uses_jac=False),
    MethodSpec("as RK5(4)", itg.RK45, order=5, adaptive=True, uses_jac=False),
    MethodSpec("Radau IIA", itg.Radau, order=5, adaptive=True, uses_jac=True),
    MethodSpec("BDF", itg.BDF, order=5, adaptive=True, uses_jac=True),
    MethodSpec("Adams/BDF", itg.LSODA, order=5, adaptive=True, uses_jac=True),
]

runs_per_method = 2


def get_kwargs_list(method_spec: MethodSpec) -> dict:
    def get_extra_kwargs(i):
        if method_spec.adaptive:
            extra_kwargs = dict(
                rtol=np.exp2(-3 * i - 2),
                atol=np.exp2(-6 * i - 4),
            )
        else:
            extra_kwargs = dict(
                num_steps=(1 << (2 * i)) * 4,
            )

        if method_spec.uses_jac:
            extra_kwargs.update(jac=ivp_spec.pff_jac)

        return extra_kwargs

    return [
        dict(
            fun=ivp_spec.pff_fun,
            t_span=t_bounds,
            y0=np.ravel(ivp_spec.X0),
            **get_extra_kwargs(i),
        )
        for i in range(runs_per_method)
    ]


error_analyses = {
    method_spec.name: do_error_analysis(
        method_spec.method,
        get_kwargs_list(method_spec),
        ivp_spec.pff_exact_sol_fn,
        ivp_spec.pff_norm_fn,
    )
    for method_spec in method_specs
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
