print("Initializing.", end="")

from . import *

from collections import namedtuple
import colorsys
import os.path
import warnings
import numpy as np
from numpy.random import default_rng
import scipy.integrate as itg
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from cycler import cycler
from colorspacious import cspace_convert
from progress.bar import ShadyBar
from progress.spinner import Spinner

print(".", end="")

plt.style.use("petroff8")
mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["text.usetex"] = True
mpl.rcParams["savefig.transparent"] = True
warnings.filterwarnings("ignore")

output_folder = "figures"
output_ext = ".svg"
figure_files = []


def record_figure(file):
    plt.savefig(os.path.join(output_folder, file))
    figure_files.append(file)


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
    MethodSpec("fs RK1", FixedRK1, order=1, adaptive=False, uses_jac=False),
    MethodSpec("fs RK3", FixedRK3, order=3, adaptive=False, uses_jac=False),
    MethodSpec("fs RK5", FixedRK5, order=5, adaptive=False, uses_jac=False),
    MethodSpec("fs RK8", FixedRK8, order=8, adaptive=False, uses_jac=False),
    MethodSpec("as RK1(1)", RK11, order=1, adaptive=True, uses_jac=False),
    MethodSpec("as RK3(2)", itg.RK23, order=3, adaptive=True, uses_jac=False),
    MethodSpec("as RK5(4)", itg.RK45, order=5, adaptive=True, uses_jac=False),
    MethodSpec("as DOP853", itg.DOP853, order=8, adaptive=True, uses_jac=False),
    MethodSpec("Radau IIA", itg.Radau, order=5, adaptive=True, uses_jac=True),
    MethodSpec("BDF", itg.BDF, order=5, adaptive=True, uses_jac=True),
    MethodSpec("Adams/BDF", itg.LSODA, order=5, adaptive=True, uses_jac=True),
]

runs_per_method = 6


def get_kwargs_list(method_spec: MethodSpec) -> dict:
    def get_extra_kwargs(i):
        if method_spec.adaptive:
            extra_kwargs = dict(
                rtol=np.exp2(-4 * i - 2),
                atol=np.exp2(-8 * i - 4),
            )
        else:
            steps_div = np.max([method_spec.order - 1, 1])
            num_steps = int(np.exp2(2 * i / method_spec.order) * 4)
            extra_kwargs = dict(
                num_steps=np.max([num_steps, 1]),
            )

        if method_spec.uses_jac:
            extra_kwargs.update(jac=ivp_spec.pff_jac)

        return extra_kwargs

    return [
        dict(
            fun=ivp_spec.pff_fun,
            t_span=t_bounds,
            y0=np.ravel(ivp_spec.X0),
            dense_output=True,
            **get_extra_kwargs(i),
        )
        for i in range(runs_per_method)
    ]


def darken_cmap(cmap: mpl.colors.Colormap) -> mpl.colors.Colormap:
    colors = cmap(np.linspace(0.0, 1.0, 13))
    colors_JCh = cspace_convert(colors[:, :3], "sRGB1", "JCh")
    new_colors = np.clip(
        cspace_convert(colors_JCh * np.array([0.75, 1.0, 1.0]), "JCh", "sRGB1"),
        0.0,
        1.0,
    )
    colors = np.column_stack([new_colors, colors[:, 3:]])
    return mpl.colors.ListedColormap(colors)


print(". done.")
analysis_bar = ShadyBar("Doing analysis", max=len(method_specs) * runs_per_method)

error_analyses = {
    method_spec.name: do_error_analysis(
        method_spec.method,
        get_kwargs_list(method_spec),
        ivp_spec.pff_exact_sol_fn,
        ivp_spec.pff_norm_fn,
        bar=analysis_bar,
    )
    for method_spec in method_specs
}

analysis_bar.finish()

num_random_vecs = 1000
neff_bar = ShadyBar("Measuring neff scales", max=num_random_vecs * 3)
jev_scale, lu_scale = measure_neff_scales(ivp_spec, bar=neff_bar)
neff_bar.finish()

print(f"    jev_scale = {jev_scale}\n    lu_scale = {lu_scale}")

graphs_bar = ShadyBar("Making graphs", max=(2 + 2 * len(method_specs)))

make_sv_streamplot_fig()
record_figure(f"sv_streamplot{output_ext}")
graphs_bar.next()

err_cmaps = [
    darken_cmap(mpl.colormaps[cmap_name])
    for cmap_name in [
        "YlOrBr",
        "PuBuGn",
        "YlGnBu",
        "PuRd",
    ]
]

for i, (name, analysis) in enumerate(error_analyses.items()):
    fname = "".join(map(str.lower, filter(str.isalnum, name)))

    make_polygon_fig(
        ivp_spec,
        analysis.solns[np.argmax(analysis.global_errors)],
    )
    record_figure(f"{fname}_course_run{output_ext}")
    graphs_bar.next()

    make_error_analysis_fig(ivp_spec, analysis, err_cmaps[i % len(err_cmaps)])
    record_figure(f"{fname}_err{output_ext}")
    graphs_bar.next()


make_work_precision_fig(error_analyses, jev_scale, lu_scale)
record_figure(f"work_precision{output_ext}")
graphs_bar.next()
graphs_bar.finish()

with open(os.path.join(output_folder, f"index.html"), "w") as file:
    file.write(
        f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Figure Gallery</title>
  </head>
  <body>
    <main>
      <h1>Figure Gallery</h1>{''.join(f'''
        <p><code>{file}</code><br><img src={file}></p>''' for file in figure_files)}
    </main>
  </body>
</html>
"""
    )

print(f"Figures saved in {output_folder}")
