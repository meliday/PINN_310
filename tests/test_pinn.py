"""PINN unit checks (plan section 13, items 1-3): derivatives, wall normals, beta gradients."""

import unittest

import numpy as np
import torch

from classroom_heat_pinn_data.src import common, pinn

DT = pinn.DTYPE


class _Field(torch.nn.Module):
    """Analytic stand-in for the network, so residuals have known values."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, p):
        return self.fn(p)


def _pts(n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 2, generator=g, dtype=DT) * torch.tensor([1.0, 0.8], dtype=DT)


class DerivativeTests(unittest.TestCase):
    def test_autograd_matches_analytic_function(self):
        f = lambda p: torch.sin(np.pi * p[:, 0]) * torch.cos(2 * p[:, 1]) + p[:, 0] ** 2 * p[:, 1]
        xy = _pts()
        _, grad, lap = pinn.derivatives(f, xy)
        x, y = xy[:, 0], xy[:, 1]
        gx = np.pi * torch.cos(np.pi * x) * torch.cos(2 * y) + 2 * x * y
        gy = -2 * torch.sin(np.pi * x) * torch.sin(2 * y) + x ** 2
        lap_ref = -(np.pi ** 2 + 4) * torch.sin(np.pi * x) * torch.cos(2 * y) + 2 * y
        torch.testing.assert_close(grad, torch.stack([gx, gy], 1), atol=1e-12, rtol=0)
        torch.testing.assert_close(lap, lap_ref, atol=1e-11, rtol=0)

    def test_network_derivatives_use_raw_coordinates(self):
        # Finite differences of the actual network (with its input transform) in raw (x, y).
        known = common.load_known()
        torch.manual_seed(0)
        model = pinn.HeatPINN(known)
        xy = _pts(20, seed=1) * 0.9 + 0.04
        _, grad, lap = pinn.derivatives(model, xy)
        h = 1e-4
        ex, ey = torch.tensor([h, 0.0], dtype=DT), torch.tensor([0.0, h], dtype=DT)
        with torch.no_grad():
            f0 = model(xy)
            fx = (model(xy + ex) - model(xy - ex)) / (2 * h)
            fy = (model(xy + ey) - model(xy - ey)) / (2 * h)
            lap_fd = (model(xy + ex) + model(xy - ex) + model(xy + ey) + model(xy - ey) - 4 * f0) / h ** 2
        torch.testing.assert_close(grad, torch.stack([fx, fy], 1), atol=1e-7, rtol=0)
        torch.testing.assert_close(lap, lap_fd, atol=1e-4, rtol=0)


class NormalSignTests(unittest.TestCase):
    def test_bc_residual_for_linear_field(self):
        # theta = a x + b y + c: grad.n is -b (S), a (E), b (N), -a (W).
        known = common.load_known()
        a, b, c = 0.7, -1.3, 0.5
        model = pinn.HeatPINN(known, bi_fixed=(1.0, 0.4, 0.25))
        model.net = _Field(lambda p: a * p[:, 0] + b * p[:, 1] + c)
        walls = pinn.sample_walls(30, known, torch.Generator().manual_seed(0))
        r = pinn.bc_residuals(model, walls)
        bi = dict(zip("SEN", (1.0, 0.4, 0.25)))
        tinf = common.theta_inf(known)
        expected_n = {"S": -b, "E": a, "N": b, "W": -a}
        for w, xy in walls.items():
            theta = a * xy[:, 0] + b * xy[:, 1] + c
            exp = expected_n[w] + (bi[w] * (theta - tinf[w]) if w != "W" else 0.0)
            torch.testing.assert_close(r[w], torch.full_like(r[w], 0.0) + exp, atol=1e-12, rtol=0)

    def test_wall_points_lie_on_their_wall(self):
        known = common.load_known()
        walls = pinn.sample_walls(50, known, torch.Generator().manual_seed(0))
        self.assertTrue(torch.all(walls["S"][:, 1] == 0) and torch.all(walls["N"][:, 1] == 0.8))
        self.assertTrue(torch.all(walls["W"][:, 0] == 0) and torch.all(walls["E"][:, 0] == 1.0))
        for w, xy in walls.items():  # corners excluded
            along = xy[:, 0] if w in ("S", "N") else xy[:, 1]
            L = 1.0 if w in ("S", "N") else 0.8
            self.assertTrue(torch.all((along > 0) & (along < L)), w)

    def test_heat_flow_sign_for_linear_field(self):
        # theta = 1 - x: heat flows toward +x, so E has outward gradient flux +1*LY, W -1*LY.
        known = common.load_known()
        model = pinn.HeatPINN(known, bi_fixed=(1.0, 1.0, 1.0))
        model.net = _Field(lambda p: 1.0 - p[:, 0])
        fl = pinn.heat_flows(model, known, n=101)
        self.assertAlmostEqual(fl["flux_grad"]["E"], 0.8, places=10)
        self.assertAlmostEqual(fl["flux_grad"]["W"], -0.8, places=10)
        self.assertAlmostEqual(fl["flux_grad"]["S"], 0.0, places=10)


class BetaGradientTests(unittest.TestCase):
    def test_inverse_beta_registered_and_receives_gradient(self):
        known = common.load_known()
        torch.manual_seed(0)
        model = pinn.HeatPINN(known)
        self.assertTrue(model.inverse)
        torch.testing.assert_close(model.beta.detach(), torch.zeros(3, dtype=DT))
        self.assertIn("beta", dict(model.named_parameters()))
        gen = torch.Generator().manual_seed(0)
        interior = pinn.sample_interior(100, known, gen)
        walls = pinn.sample_walls(20, known, gen)
        obs = common.load_observations("A_clean", known)
        total, _ = pinn.losses(model, known, interior, walls,
                               torch.tensor(obs.xy, dtype=DT), torch.tensor(obs.theta, dtype=DT))
        total.backward()
        g = model.beta.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertTrue((g.abs() > 0).all())

    def test_forward_mode_keeps_bi_fixed(self):
        known = common.load_known()
        model = pinn.HeatPINN(known, bi_fixed={"S": 1.0, "E": 0.4, "N": 0.25})
        self.assertFalse(model.inverse)
        self.assertNotIn("beta", dict(model.named_parameters()))
        _, status = pinn.train(model, known, {"adam_steps": 3, "n_interior": 50, "n_wall": 10,
                                              "log_every": 1})
        torch.testing.assert_close(model.bi(), torch.tensor([1.0, 0.4, 0.25], dtype=DT))
        self.assertEqual(status["status"], "completed")

    def test_short_training_is_deterministic_for_a_seed(self):
        known = common.load_known()
        obs = common.load_observations("A_clean", known)
        runs = []
        for _ in range(2):
            torch.manual_seed(123)  # network init
            model = pinn.HeatPINN(known)
            hist, _ = pinn.train(model, known, {"adam_steps": 5, "n_interior": 50, "n_wall": 10,
                                                "log_every": 1, "seed": 7}, obs=obs)
            runs.append([h["loss"] for h in hist])
        self.assertEqual(runs[0], runs[1])


if __name__ == "__main__":
    unittest.main()
