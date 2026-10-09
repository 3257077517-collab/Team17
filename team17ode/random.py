import numpy as np
import scipy.linalg as spla


def random_hypersphere(rng: np.random.Generator, dim: int):
    """https://dl.acm.org/doi/epdf/10.1145/377939.377946"""
    normpt = 0.0
    while normpt == 0.0:
        pt = rng.standard_normal(dim + 1)
        normpt = np.linalg.vector_norm(pt)
    pt /= normpt
    return pt


def random_SO(rng: np.random.Generator, n: int):
    """Matrix exponential of skew symmetric matrices are special orthogonal matrices thanks to 2-forms generating rotations."""
    mex = np.zeros((n, n))

    rot = (np.pi - rng.uniform(-np.pi, np.pi)) * random_hypersphere(
        rng, n * (n - 1) // 2 - 1
    )

    for i in range(n):
        for j in range(i + 1, n):
            entry = rot[j - 1 + i * (2 * n - i - 3) // 2]
            mex[i, j] = entry
            mex[j, i] = -entry

    return spla.expm(mex)


def random_O(rng: np.random.Generator, n: int):
    U = random_SO(rng, n)
    if n > 0 and rng.integers(0, 2) == 1:
        U[:, 0] *= -1.0
    return U
