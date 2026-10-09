"""AI-assisted starter check for the polar-factor flow; not a full solver."""

import numpy as np


def rhs(x):
    """Return X(I - X^T X) for a real square matrix X."""
    return x @ (np.eye(x.shape[0]) - x.T @ x)


def potential(x):
    """Measure departure from orthogonality."""
    defect = x.T @ x - np.eye(x.shape[0])
    return np.linalg.norm(defect, "fro") ** 2 / 4


def main():
    x = np.diag([0.2, 1.4, 3.0])
    np.testing.assert_allclose(rhs(x), np.diag([0.192, -1.344, -24.0]))

    q = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(rhs(q), np.zeros((3, 3)), atol=1e-14)

    # Check this one small step; this is not a proof of general stability.
    x_next = x + 0.001 * rhs(x)
    if not potential(x_next) < potential(x):
        raise AssertionError("Potential should decrease for this small step")
    print("PASS: diagonal RHS, orthogonal equilibrium, small Euler step")


if __name__ == "__main__":
    main()
