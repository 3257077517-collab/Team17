import numpy as np
import scipy.integrate as itg
from scipy.integrate._ivp.rk import RungeKutta, rk_step
from .imp_euler import ImplicitEuler


class RK21(RungeKutta):
    order = 1
    error_estimator_order = 1
    n_stages = 1
    C = np.array([0.0])
    A = np.array([[0.0]])
    B = np.array([1.0])
    E = np.array([-0.5, 0.5])
    P = np.array([[1.0, 1, -1], [0, -1, 1]])



class FixedRungeKutta(RungeKutta):
    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        num_steps=50,
        rtol=1e-3,
        atol=1e-6,
        vectorized=False,
        **extraneous
    ):
        itg._ivp.common.warn_extraneous(extraneous)

        self.num_steps = num_steps
        self.steps_taken = 0
        self.step_ts = np.linspace(t0, t_bound, num_steps + 1)
        max_step = np.max(np.abs(self.step_ts[1:] - self.step_ts[:-1]))
        first_step = np.abs(self.step_ts[1] - self.step_ts[0])

        super().__init__(
            fun,
            t0,
            y0,
            t_bound,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
            vectorized=vectorized,
            first_step=first_step,
        )

    def _step_impl(self):
        t = self.t
        y = self.y

        rtol = self.rtol
        atol = self.atol

        t_new = self.step_ts[self.steps_taken + 1]
        h = t_new - t
        h_abs = np.abs(h)

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if h_abs < min_step:
            return False, self.TOO_SMALL_STEP

        y_new, f_new = rk_step(
            self.fun, t, y, self.f, h, self.A, self.B, self.C, self.K
        )
        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        error_norm = self._estimate_error_norm(self.K, h, scale)

        self.h_previous = h
        self.y_old = y

        self.t = t_new
        self.y = y_new

        self.h_abs = h_abs
        self.f = f_new
        self.steps_taken += 1

        return True, None


class FixedRK1(FixedRungeKutta, RK21):
    pass


class FixedRK3(FixedRungeKutta, itg.RK23):
    pass


class FixedRK5(FixedRungeKutta, itg.RK45):
    pass


class FixedRK8(FixedRungeKutta, itg.DOP853):
    pass
