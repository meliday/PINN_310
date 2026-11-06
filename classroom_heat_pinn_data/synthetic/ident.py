"""Local sensitivity helpers. Callers must supply the known corridor temperature."""

import numpy as np

if __package__:
    from .solver import solve, sample
else:
    from solver import solve, sample


LAYOUT_A = np.array([
    (0.3, 0.04), (0.7, 0.04), (0.95, 0.25), (0.95, 0.55),
    (0.3, 0.76), (0.7, 0.76), (0.3, 0.4), (0.5, 0.25),
    (0.7, 0.4), (0.5, 0.55),
])


def run(Bi, heaters, thc, nx=101):
    """Solve with thc supplied explicitly from known['theta_inf']['N']."""
    Bi = np.asarray(Bi, dtype=float)
    if Bi.shape != (3,) or not np.all(np.isfinite(Bi)) or np.any(Bi <= 0):
        raise ValueError("Bi must contain three finite positive values (S, E, N)")
    if not np.isfinite(thc):
        raise ValueError("thc must be the finite corridor temperature from known config")
    return solve(nx, dict(zip("SEN", Bi)), {"N": thc}, heaters=heaters)


def covariance_from_jacobian(jacobian, sigma=1.0):
    """Return (local covariance or None, SVD diagnostics).

    A deficient Jacobian has no identifiable covariance; no pseudoinverse is
    substituted. The caller can report rank and singular values before stopping.
    """
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2 or min(J.shape) == 0 or not np.all(np.isfinite(J)):
        raise ValueError("Jacobian must be a nonempty finite matrix")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    _, singular_values, vt = np.linalg.svd(J, full_matrices=False)
    tolerance = max(J.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    n_parameters = J.shape[1]
    condition_number = (float(singular_values[0] / singular_values[-1])
                        if rank == n_parameters else float("inf"))
    diagnostics = {
        "rank": rank,
        "n_parameters": n_parameters,
        "singular_values": singular_values,
        "condition_number": condition_number,
    }
    if rank < n_parameters:
        return None, diagnostics
    covariance = (vt.T * (sigma / singular_values) ** 2) @ vt
    return covariance, diagnostics


def ident(Bi, heaters, pts=LAYOUT_A, sigC=0.1, dT=20, *, thc,
          return_diagnostics=False):
    """Return relative Bi std and S/N correlation under a local noise model.

    thc is required; pass known['theta_inf']['N']. Set return_diagnostics=True
    to also receive rank, singular values and condition number. This function
    loads no config or truth files.
    """
    Bi = np.asarray(Bi, dtype=float)
    if not np.isfinite(dT) or dT <= 0 or not np.isfinite(sigC) or sigC < 0:
        raise ValueError("dT must be positive and sigC nonnegative, both finite")
    if Bi.shape != (3,) or not np.all(np.isfinite(Bi)) or np.any(Bi <= 0):
        raise ValueError("Bi must contain three finite positive values (S, E, N)")
    J = np.zeros((len(pts), 3))
    for p in range(3):
        d = np.zeros(3)
        d[p] = 1e-4 * Bi[p]
        up = run(Bi + d, heaters, thc)
        dn = run(Bi - d, heaters, thc)
        J[:, p] = (sample(*up[:3], pts) - sample(*dn[:3], pts)) / (2 * d[p])
    C, diagnostics = covariance_from_jacobian(J * Bi)
    if C is None:
        raise np.linalg.LinAlgError(
            f"log-Bi Jacobian rank {diagnostics['rank']}/3; "
            f"singular values {diagnostics['singular_values']}; covariance unavailable"
        )
    sd = np.sqrt(np.diag(C)) * sigC / dT
    correlation = C[0, 2] / np.sqrt(C[0, 0] * C[2, 2])
    result = (sd, correlation)
    return (*result, diagnostics) if return_diagnostics else result
