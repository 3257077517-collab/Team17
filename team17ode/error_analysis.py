from collections import namedtuple
import numpy as np
import scipy.integrate as itg


ErrorAnalysisResult = namedtuple(
    "ErrorAnalysisResult",
    [
        "solver_kwargs_list",
        "exact_sol_fn",
        "norm_fn",
        "avg_step_sizes",
        "global_errors",
        "avg_local_errors",
        "step_size_seqs",
        "local_error_curves",
        "solns",
    ],
)


def do_error_analysis(method, solver_kwargs_list, exact_sol_fn, norm_fn):
    num_times_to_solve = len(solver_kwargs_list)
    avg_step_sizes = np.empty(num_times_to_solve)
    global_errors = np.empty(num_times_to_solve)
    avg_local_errors = np.empty(num_times_to_solve)
    step_size_seqs = []
    local_error_curves = []
    solns = []

    for i, solver_kwargs in enumerate(solver_kwargs_list):
        soln = itg.solve_ivp(method=method, **solver_kwargs)
        if "atol" in solver_kwargs:
            # make first step informed
            step_size_seq = soln.t[1:] - soln.t[:-1]
            solver_kwargs = dict(
                **solver_kwargs,
                first_step=(soln.t[-1] - soln.t[0]) / (step_size_seq.shape[0])
            )
            soln = itg.solve_ivp(method=method, **solver_kwargs)
        t0 = soln.t[0]
        y0 = soln.y.T[0]
        tf = soln.t[-1]
        approx_yf = soln.y.T[-1]
        exact_yf = exact_sol_fn(t0, y0, tf)

        global_error = norm_fn(exact_yf - approx_yf)

        step_size_seq = soln.t[1:] - soln.t[:-1]
        local_error_curve = np.empty_like(step_size_seq)

        for k in range(len(step_size_seq)):
            tk = soln.t[k]
            approx_yk = soln.y.T[k]
            tkp1 = soln.t[k + 1]
            approx_ykp1 = soln.y.T[k + 1]
            extend_ykp1 = exact_sol_fn(tk, approx_yk, tkp1)
            local_error_curve[k] = norm_fn(extend_ykp1 - approx_ykp1)

        avg_step_sizes[i] = (tf - t0) / len(step_size_seq)
        global_errors[i] = global_error
        avg_local_errors[i] = np.average(local_error_curve, weights=step_size_seq)
        step_size_seqs.append(step_size_seq)
        local_error_curves.append(local_error_curve)
        solns.append(soln)

    return ErrorAnalysisResult(
        solver_kwargs_list=solver_kwargs_list,
        exact_sol_fn=exact_sol_fn,
        norm_fn=norm_fn,
        avg_step_sizes=avg_step_sizes,
        global_errors=global_errors,
        avg_local_errors=avg_local_errors,
        step_size_seqs=step_size_seqs,
        local_error_curves=local_error_curves,
        solns=solns,
    )
