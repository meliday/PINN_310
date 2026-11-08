"""X1-c: PINN support for internal adiabatic walls (LOG-083). The no-wall path is checked
bit-for-bit by the fingerprint recorded in LOG-084, not here."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from classroom_heat_pinn_data.src import common, fd, pinn, run_experiments as rx
from classroom_heat_pinn_data.synthetic import solver_wall

DT = pinn.DTYPE
BI = {"S": 1.0, "E": 0.4, "N": 0.25}  # fixed test values (not read from truth files)


class _Field(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, xy):
        return self.fn(xy)


def _model(known, fn=None):
    torch.manual_seed(0)
    m = pinn.HeatPINN(known, hidden=8, depth=1, bi_fixed=BI)
    if fn is not None:
        m.net = _Field(fn)
    return m


class X1SamplingTests(unittest.TestCase):
    def setUp(self):
        self.known = common.load_known_x1()

    def test_interior_points_avoid_the_wall_and_keep_the_count(self):
        g = torch.Generator().manual_seed(0)
        pts = pinn.sample_interior(1000, self.known, g)
        self.assertEqual(pts.shape, (1000, 2))
        self.assertFalse(common.inside_solid(pts.numpy(), self.known).any())
        # same generator draws as the open room until the top-up, so most points coincide
        base = pinn.sample_interior(1000, common.load_known(), torch.Generator().manual_seed(0))
        kept = ~common.inside_solid(base.numpy(), self.known)
        torch.testing.assert_close(pts[:int(kept.sum())], base[torch.from_numpy(kept)], rtol=0, atol=0)

    def test_wall_face_points_and_normals(self):
        walls = pinn.sample_walls(100, self.known, torch.Generator().manual_seed(0))
        self.assertEqual(set(walls), {"S", "E", "N", "W", "X"})
        X = walls["X"]
        self.assertEqual(len(X), 48 + 48 + 10)  # faces 0.6, 0.6, 0.04 of total 1.24; at least 10 per face
        n = pinn.solid_face_normals(X, common.solids(self.known))
        bottom, top, tip = (X[:, 1] == 0.30), (X[:, 1] == 0.34), (X[:, 0] == 0.60)
        self.assertEqual((int(bottom.sum()), int(top.sum()), int(tip.sum())), (48, 48, 10))
        torch.testing.assert_close(n[bottom], torch.tensor([[0.0, 1.0]], dtype=DT).expand(48, 2))
        torch.testing.assert_close(n[top], torch.tensor([[0.0, -1.0]], dtype=DT).expand(48, 2))
        torch.testing.assert_close(n[tip], torch.tensor([[-1.0, 0.0]], dtype=DT).expand(10, 2))
        self.assertTrue(((X[tip, 1] > 0.30) & (X[tip, 1] < 0.34)).all())
        self.assertNotIn("X", pinn.sample_walls(100, common.load_known(), torch.Generator().manual_seed(0)))

    def test_outer_wall_points_skip_the_stretch_covered_by_the_wall(self):
        # X1-PINN-OCCLUDED-W-01: the solid covers W for y in (0.30, 0.34); no boundary point there.
        walls = pinn.sample_walls(100, self.known, torch.Generator().manual_seed(0))
        for w in ("S", "E", "N", "W"):
            with self.subTest(wall=w):
                self.assertEqual(len(walls[w]), 100)
                self.assertFalse(pinn.outer_wall_covered(walls[w].numpy(), self.known, w).any())
        W = walls["W"][:, 1]
        self.assertFalse(((W > 0.30) & (W < 0.34)).any())

    def test_point_off_the_faces_is_refused(self):
        with self.assertRaisesRegex(ValueError, "exactly one solid face"):
            pinn.solid_face_normals(torch.tensor([[0.3, 0.32], [0.6, 0.30]], dtype=DT), common.solids(self.known))


class X1ResidualTests(unittest.TestCase):
    def setUp(self):
        self.known = common.load_known_x1()

    def test_wall_residual_is_normal_derivative(self):
        # theta = y: grad = (0, 1). Bottom face (normal +y) -> 1, top (-y) -> -1, tip (-x) -> 0.
        m = _model(self.known, lambda xy: xy[:, 1])
        X = pinn.sample_walls(100, self.known, torch.Generator().manual_seed(1))["X"]
        r = pinn.bc_residuals(m, {"X": X})["X"].detach()
        expected = torch.where(X[:, 1] == 0.30, 1.0, torch.where(X[:, 1] == 0.34, -1.0, 0.0)).to(DT)
        torch.testing.assert_close(r, expected)

    def test_losses_average_five_wall_groups(self):
        m = _model(self.known)
        g = torch.Generator().manual_seed(2)
        interior, walls = pinn.sample_interior(200, self.known, g), pinn.sample_walls(20, self.known, g)
        _, parts = pinn.losses(m, self.known, interior, walls)
        groups = [parts[f"bc_{w}"] for w in ("S", "E", "N", "W", "X")]
        torch.testing.assert_close(parts["bc"], sum(groups) / 5)

    def test_heat_flows_exclude_solid_source_and_report_wall_flux(self):
        m = _model(self.known, lambda xy: 0.0 * xy[:, 0] + 1.0)  # constant field: no gradient flux
        fl = pinn.heat_flows(m, self.known)
        self.assertEqual(fl["flux_grad"]["X"], 0.0)
        fv = solver_wall.solve(401, BI, common.theta_inf(self.known), solids=common.solids(self.known),
                               heaters=common.heaters(self.known))[3]["q_in"]
        self.assertLess(abs(fl["source_integral"] / fv - 1), 2e-3)
        open_room = pinn.heat_flows(_model(common.load_known(), lambda xy: 0.0 * xy[:, 0] + 1.0), common.load_known())
        self.assertLess(fl["source_integral"], open_room["source_integral"])
        self.assertNotIn("X", open_room["flux_grad"])

    def test_wall_gradient_flux_excludes_covered_outer_stretch(self):
        # theta = x: -grad(theta).n = 1 on W, so the W flux is the fluid-exposed length 0.8 - 0.04.
        fl = pinn.heat_flows(_model(self.known, lambda xy: xy[:, 0]), self.known)
        # each exposed stretch is integrated on its own grid, so the end points are exact (LOG-109)
        self.assertAlmostEqual(fl["flux_grad"]["W"], 0.76, places=12)
        open_room = pinn.heat_flows(_model(common.load_known(), lambda xy: xy[:, 0]), common.load_known())
        self.assertAlmostEqual(open_room["flux_grad"]["W"], 0.8, places=10)

    def test_fully_covered_outer_wall_has_no_group_and_zero_flux(self):
        # LOG-107 X1-PINN-FULL-COVER-01: a solid covering all of W must not hang the sampler.
        known = {**self.known, "solids": [{"x0": 0.0, "x1": 0.05, "y0": 0.0, "y1": 0.8}]}
        self.assertEqual(pinn.exposed_intervals(known, "W"), [])
        walls = pinn.sample_walls(100, known, torch.Generator().manual_seed(0))
        self.assertNotIn("W", walls)
        self.assertEqual(len(walls["S"]), 100)
        self.assertTrue((walls["S"][:, 0] > 0.05).all())
        fl = pinn.heat_flows(_model(known, lambda xy: xy[:, 0]), known)
        self.assertEqual(fl["flux_grad"]["W"], 0.0)
        g = torch.Generator().manual_seed(2)
        _, parts = pinn.losses(_model(known), known, pinn.sample_interior(100, known, g), pinn.sample_walls(20, known, g))
        self.assertNotIn("bc_W", parts)

    def test_solids_outside_the_room_are_clipped_like_fd(self):
        # LOG-111 PINN-DOMAIN-CLIP-01: a solid entirely above the room changes nothing, as in FD.
        open_room = common.load_known()
        outside = {**open_room, "solids": [{"x0": 0.0, "x1": 0.6, "y0": 0.9, "y1": 1.0}]}
        self.assertEqual(pinn.exposed_intervals(outside, "W"), [(0.0, 0.8)])
        walls = pinn.sample_walls(100, outside, torch.Generator().manual_seed(0))
        self.assertTrue(((walls["W"][:, 1] > 0) & (walls["W"][:, 1] < 0.8)).all())
        self.assertNotIn("X", walls)
        self.assertEqual(pinn.fluid_heater_power(outside), pinn.fluid_heater_power(open_room))
        fd_open = fd.solve_field(BI, open_room, nx=101)[2]
        fd_out = fd.solve_field(BI, outside, nx=101)[2]
        np.testing.assert_array_equal(fd_open, fd_out)
        # LOG-114 P3: same random draws, wall points and heater normalisation as the open room
        g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
        a, b = pinn.sample_walls(50, outside, g1), pinn.sample_walls(50, open_room, g2)
        self.assertEqual(set(a), set(b))
        for w in a:
            torch.testing.assert_close(a[w], b[w], rtol=0, atol=0)
        torch.testing.assert_close(torch.rand(3, generator=g1), torch.rand(3, generator=g2), rtol=0, atol=0)
        e5b = common.load_known_e5b()
        geom = common.sink_geometry(e5b)
        self.assertEqual(pinn._domain_integrals({**e5b, "solids": outside["solids"]}, geom), pinn._domain_integrals(e5b, geom))
        # partly outside: only the in-room part counts
        partly = {**open_room, "solids": [{"x0": -0.2, "x1": 0.6, "y0": 0.30, "y1": 0.34}]}
        self.assertEqual(pinn.exposed_intervals(partly, "W"), pinn.exposed_intervals(self.known, "W"))
        self.assertEqual(pinn.fluid_heater_power(partly), pinn.fluid_heater_power(self.known))
        self.assertEqual(pinn.solid_faces(partly), pinn.solid_faces(self.known))

    def test_x2_sink_removes_fraction_of_fluid_heater_power(self):
        # LOG-107: with solids the heater and sink integrals are exact over the fluid region.
        known = common.load_known_x2()
        torch.manual_seed(0)
        m = pinn.HeatPINN(known, hidden=8, depth=1, bi_fixed=BI, sink=0.1)
        sink = known["sink"]
        S = pinn.fluid_gaussian_integral(known, sink["x"], sink["y"], sink["s"])
        H = pinn.fluid_heater_power(known)
        self.assertAlmostEqual(float(m.heater_power), H, places=14)
        self.assertAlmostEqual(float(m.sink_fraction() * m.heater_power / m.sink_shape_integral) * S / H, 0.1, places=12)
        fd_info = fd.solve_field(BI, known, nx=201, sink_fraction=0.1)[3]
        self.assertLess(abs(H / fd_info["heater_power"] - 1), 1e-3)
        # open room: exact integral agrees with the existing trapezoid to its accuracy
        base = common.load_known()
        h = base["heater"]
        self.assertLess(abs(pinn.fluid_heater_power(base) / pinn._domain_integrals(base, {"x": .5, "y": .4, "s": .1})[0] - 1), 1e-3)

    def test_short_training_runs_with_the_wall(self):
        m = _model(self.known)
        hist, status = pinn.train(m, self.known, {"hidden": 8, "depth": 1, "adam_steps": 20, "log_every": 10,
                                                  "n_interior": 100, "n_wall": 20, "threads": 1})
        self.assertEqual(status["status"], "completed")
        self.assertIn("bc_X", hist[-1])
        self.assertTrue(np.isfinite(hist[-1]["loss"]))


class X1ForwardRunnerTests(unittest.TestCase):
    """run_x1_forward with a tiny network into a temporary results directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        for patcher in (mock.patch.object(common, "RESULTS_DIR", self.tmp),
                        mock.patch.dict(pinn.DEFAULT_EXPERIMENT, {"hidden": 8, "depth": 1, "n_interior": 60,
                                                                   "n_wall": 10, "log_every": 1, "threads": 1}),
                        mock.patch.dict(rx.E2_CONFIGS, {"C1": {"adam_steps": 3}}),
                        mock.patch.dict(rx.E2_EXPERIMENT, {"lbfgs_iters": 0})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _log(self):
        path = self.tmp / "logs" / "truth_access.log"
        return path.read_text() if path.exists() else ""

    def test_truth_order_outputs_and_targets(self):
        real_train, seen = pinn.train, {}

        def train(*args, **kwargs):  # reference must not be read before or during training
            seen["log_at_train"] = self._log()
            return real_train(*args, **kwargs)

        with mock.patch.object(pinn, "train", side_effect=train):
            s = rx.run_x1_forward(0)
        self.assertIn("config_x1.json", seen["log_at_train"])
        self.assertNotIn("reference_x1.npz", seen["log_at_train"])
        self.assertIn("reference_x1.npz", self._log())
        out = self.tmp / "x1" / "forward_seed0"
        for name in ("checkpoint.pt", "summary.json", "history.json"):
            self.assertTrue((out / name).exists(), name)
        self.assertTrue((self.tmp / "figures" / "x1_forward_seed0.png").exists())
        self.assertEqual(s["experiment"]["adam_steps"], 3)
        self.assertEqual(s["experiment"]["lambda_data"], 0.0)
        self.assertEqual(set(s["targets_met"]), {"rel_l2", "energy"})
        self.assertIn("X", s["heat_flows"]["flux_grad"])
        self.assertEqual(s["field"]["n_eval_points"], 201 * 161 - 7 * 120)  # fluid nodes of the 201 grid
        rows = (self.tmp / "metrics.csv").read_text()
        self.assertIn("PINN_forward_C1", rows)

    def test_existing_run_is_refused_by_cli(self):
        rx.run_x1_forward(0)
        with self.assertRaises(SystemExit) as stopped, mock.patch("sys.stderr"):
            rx.main(["--exp", "X1_forward"])
        self.assertEqual(stopped.exception.code, 3)


if __name__ == "__main__":
    unittest.main()
