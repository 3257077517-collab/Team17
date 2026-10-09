from collections import namedtuple
from timeit import timeit
import numpy as np
from scipy import linalg as spla
from .random import random_O


def pff_sv_fun(t, s):
    return s * (1.0 - s * s)


def pff_sv_exact(t, y0: float):
    if y0 in (0.0, 1.0, -1.0):
        return np.full_like(t, y0)
    determinant = 1.0 + (1.0 - y0 * y0) * np.expm1(-2.0 * t)
    return y0 * np.reciprocal(np.sqrt(determinant))


def make_pff_fun(out_dim: int):
    def pff_fun(_t, vecX):
        X = np.reshape(vecX, shape=(out_dim, -1))
        XTX = X.T @ X
        return np.reshape(X @ (np.eye(*np.shape(XTX)) - XTX), shape=(-1,))

    return pff_fun


def make_pff_jac(out_dim: int):
    def pff_jac(_t, vecX):
        X = np.reshape(vecX, shape=(out_dim, -1))
        n, m = X.shape
        dac_dbd = np.multiply.outer(np.eye(n), np.eye(m))
        dbd_Xaj_Xcj = np.multiply.outer(np.eye(m), X @ X.T)
        dac_Xkb_Xkd = np.multiply.outer(np.eye(n), X.T @ X)
        Xad_Xcb = np.multiply.outer(X, X)

        return (
            np.transpose(dac_dbd, (0, 2, 1, 3))
            - np.transpose(dbd_Xaj_Xcj, (2, 0, 3, 1))
            - np.transpose(dac_Xkb_Xkd, (0, 2, 1, 3))
            - np.transpose(Xad_Xcb, (0, 3, 2, 1))
        ).reshape((n * m, n * m))

    return pff_jac


def pff_exact_helped(t, sv0s, U: np.array, V: np.array) -> np.array:
    Sigma = np.zeros((U.shape[1], V.shape[1]))
    np.fill_diagonal(Sigma, [pff_sv_exact(t, sv0) for sv0 in sv0s])
    return U @ Sigma @ V.T


PFFIVPSpec = namedtuple(
    "PFFIVPSpec",
    [
        "U",
        "Sigma0",
        "V",
        "sv0s",
        "X0",
        "pff_fun",
        "pff_jac",
        "pff_exact_sol_fn",
        "pff_norm_fn",
    ],
)


def random_pff_ivp(rng: np.random.Generator, out_dim: int, in_dim: int) -> PFFIVPSpec:
    U = random_O(rng, out_dim)
    sv0s = rng.uniform(0.0, 2.0, np.minimum(in_dim, out_dim))
    V = random_O(rng, in_dim)
    U, sv0s, V
    Sigma0 = np.zeros((out_dim, in_dim))
    np.fill_diagonal(Sigma0, sv0s)
    X0 = U @ Sigma0 @ V.T

    pff_fun = make_pff_fun(out_dim)
    pff_jac = make_pff_jac(out_dim)

    def pff_exact_sol_fn(t0, y0, t):
        X0 = np.reshape(y0, (out_dim, in_dim))
        X = pff_exact_helped(t - t0, np.diagonal(U.T @ X0 @ V), U, V)
        return np.ravel(X)

    def pff_norm_fn(dy):
        dX = np.reshape(dy, (out_dim, in_dim))
        return np.linalg.matrix_norm(dX)

    return PFFIVPSpec(
        U=U,
        Sigma0=Sigma0,
        V=V,
        sv0s=sv0s,
        X0=X0,
        pff_fun=pff_fun,
        pff_jac=pff_jac,
        pff_exact_sol_fn=pff_exact_sol_fn,
        pff_norm_fn=pff_norm_fn,
    )


def measure_neff_scales(
    ivp_spec: PFFIVPSpec,
    rng=np.random.default_rng(0xE1E100),
    num_random_vecs=1000,
    num_timeit_reps=100,
):
    random_vecs = rng.uniform(
        -2.0, 2.0, size=num_random_vecs * ivp_spec.X0.shape[0] * ivp_spec.X0.shape[1]
    ).reshape((num_random_vecs, -1))
    fev_time = sum(
        timeit(lambda: ivp_spec.pff_fun(0.0, vec), number=num_timeit_reps)
        for vec in random_vecs
    )
    jev_scale = (
        sum(
            timeit(lambda: ivp_spec.pff_jac(0.0, vec), number=num_timeit_reps)
            for vec in random_vecs
        )
        / fev_time
    )
    random_jacs = [ivp_spec.pff_jac(0.0, vec) for vec in random_vecs]
    lu_scale = (
        sum(
            timeit(lambda: spla.lu_factor(jac), number=num_timeit_reps)
            for jac in random_jacs
        )
        / fev_time
    )
    return jev_scale, lu_scale
