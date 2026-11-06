"""Truth-based tool diagnostics, separate from fitting and training."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

if __package__:
    from .solver import solve, sample
    from .ident import LAYOUT_A, covariance_from_jacobian
else:
    from solver import solve, sample
    from ident import LAYOUT_A, covariance_from_jacobian


DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--skip-global-scan", action="store_true")
    parser.add_argument("--scan-points", type=int, default=14,
                        help="Values per Bi axis in the optional coarse-grid scan")
    args = parser.parse_args(argv)
    if args.scan_points < 2:
        parser.error("--scan-points must be at least 2")
    data_dir = args.data_dir.resolve()
    known = json.loads((data_dir / "known.json").read_text())
    truth_path = data_dir / "config.json"
    truth = json.loads(truth_path.read_text())
    print(f"TOOL DIAGNOSTIC ONLY; truth access: {truth_path}")
    BI = np.array([truth["Bi_true"][w] for w in "SEN"])
    H, TI = known["heater"], known["theta_inf"]
    dT = known["temperature_scaling"]["DT_K"]

    def run(nx, Bi=BI, A=H["A"], k=None):
        return solve(nx, dict(zip("SEN", Bi)), TI, heaters=[dict(H, A=A)], k=k)

    print("(a) actual-problem convergence (Robin-Neumann corners, truncated Gaussian)")
    sols = {nx: run(nx)[2] for nx in (51, 101, 201, 401)}
    differences = [np.abs(sols[nx] - sols[2 * nx - 1][::2, ::2]).max()
                   for nx in (51, 101, 201)]
    d1, d2, d3 = differences
    print(f"   successive max diffs (degC): {dT*d1:.2e} {dT*d2:.2e} {dT*d3:.2e}"
          f" -> orders {np.log2(d1/d2):.2f}, {np.log2(d2/d3):.2f}")
    e = np.abs(sols[201] - sols[401][::2, ::2])
    i, j = np.unravel_index(e.argmax(), e.shape)
    print(f"   largest 201-vs-401 diff at x={i/200:.3f}, y={j/200:.3f}")
    print(f"   estimated reference error (Richardson, p=2): {dT*d3/3:.2e} degC")

    def jacobian(pts, eps, include_A=False):
        base = np.r_[BI, H["A"]]
        columns = []
        for p in range(4 if include_A else 3):
            d = np.zeros(4)
            d[p] = eps * base[p]
            up = run(101, Bi=(base + d)[:3], A=(base + d)[3])
            dn = run(101, Bi=(base - d)[:3], A=(base - d)[3])
            columns.append((sample(*up[:3], pts) - sample(*dn[:3], pts))
                           / (2 * d[p]) * base[p])
        return np.array(columns).T

    print("(b) Jacobian finite-difference step check")
    J1, J2 = jacobian(LAYOUT_A, 1e-4), jacobian(LAYOUT_A, 1e-3)
    print(f"   max rel diff, eps=1e-4 vs 1e-3: {np.abs(J1-J2).max()/np.abs(J1).max():.1e}")
    C3, diag3 = covariance_from_jacobian(J1, sigma=0.1 / dT)
    print(f"   log-Bi Jacobian rank {diag3['rank']}/3; singular values "
          f"{np.round(diag3['singular_values'], 4)}; condition {diag3['condition_number']:.1f}")
    if C3 is not None:
        print("   rel std @0.1C (Bi_S, Bi_E, Bi_N):", np.round(np.sqrt(np.diag(C3))*100, 1))
    else:
        print("   covariance unavailable: rank-deficient Jacobian")

    print("(c) identifiability with heater strength A also unknown (4 params)")
    C4, diag4 = covariance_from_jacobian(jacobian(LAYOUT_A, 1e-4, include_A=True),
                                       sigma=0.1 / dT)
    print(f"   Jacobian rank {diag4['rank']}/4; singular values "
          f"{np.round(diag4['singular_values'], 4)}; condition {diag4['condition_number']:.1f}")
    if C4 is not None:
        print("   rel std @0.1C (Bi_S, Bi_E, Bi_N, A):", np.round(np.sqrt(np.diag(C4))*100, 1))
        corr = C4 / np.sqrt(np.outer(np.diag(C4), np.diag(C4)))
        print("   corr(A, Bi_S/E/N):", np.round(corr[3, :3], 3))
    else:
        print("   covariance unavailable: rank-deficient Jacobian")

    if args.skip_global_scan:
        print("(d) coarse-grid scan: not_run (--skip-global-scan)")
    else:
        print("(d) coarse-grid misfit scan; not a global uniqueness proof")
        t = time.perf_counter()
        xs, ys, Tt, _ = run(51)
        obs = sample(xs, ys, Tt, LAYOUT_A)
        grid = np.geomspace(0.05, 10, args.scan_points)
        best = []
        for a in grid:
            for b in grid:
                for c in grid:
                    x, y, T, _ = run(51, Bi=np.array([a, b, c]))
                    rms = dT * np.sqrt(np.mean((sample(x, y, T, LAYOUT_A) - obs) ** 2))
                    best.append((rms, a, b, c))
        best.sort()
        print(f"   {len(best)} points in [0.05,10]^3, {time.perf_counter()-t:.1f}s;"
              " bilinear sampling at original sensor coordinates")
        for row in best[:5]:
            print(f"   {row[0]:.4f} degC RMS, Bi=({row[1]:.3f},{row[2]:.3f},{row[3]:.3f})")
        far = [row for row in best if np.max(np.abs(np.log(np.array(row[1:]) / BI))) > np.log(2)]
        if far:
            print(f"   smallest misfit at a scanned point >2x away: {far[0][0]:.3f} degC")

    print("(e) mismatch model: k on walls")
    LY = known["domain"]["LY"]
    Kv = lambda X, Y: 1 + 0.25 * np.sin(2*np.pi*X) * np.sin(np.pi*Y/LY)
    s = np.linspace(0, 1, 11)
    print("   max|k-1|:", max(np.abs(Kv(s, 0*s)-1).max(), np.abs(Kv(s, LY+0*s)-1).max(),
                              np.abs(Kv(0*s, s*LY)-1).max(), np.abs(Kv(1+0*s, s*LY)-1).max()))


if __name__ == "__main__":
    main()
