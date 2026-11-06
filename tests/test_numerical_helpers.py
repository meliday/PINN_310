"""Regression checks for existing tools; no generation or training is run."""

import json
from pathlib import Path
import unittest

import numpy as np

from classroom_heat_pinn_data.synthetic import ident, solver


DATA = Path(__file__).resolve().parents[1] / "classroom_heat_pinn_data/synthetic/data"


class NumericalHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.known = json.loads((DATA / "known.json").read_text())
        # Truth is confined to these tool diagnostics, not an inverse method.
        cls.bi = json.loads((DATA / "config.json").read_text())["Bi_true"]

    def test_corridor_temperature_is_explicit(self):
        Bi = [self.bi[w] for w in "SEN"]
        with self.assertRaises(TypeError):
            ident.run(Bi, [self.known["heater"]])
        with self.assertRaises(TypeError):
            ident.ident(Bi, [self.known["heater"]])
        sd, _, diagnostics = ident.ident(
            Bi, [self.known["heater"]], thc=self.known["theta_inf"]["N"],
            return_diagnostics=True,
        )
        self.assertEqual(diagnostics["rank"], 3)
        np.testing.assert_allclose(sd * 100, [1.57, 3.79, 7.88], atol=0.02, rtol=0)

    def test_rank_deficiency_has_no_covariance(self):
        C, diagnostics = ident.covariance_from_jacobian([[1, 2], [2, 4], [3, 6]])
        self.assertIsNone(C)
        self.assertEqual(diagnostics["rank"], 1)
        self.assertTrue(np.isinf(diagnostics["condition_number"]))
        with self.assertRaisesRegex(np.linalg.LinAlgError, "rank"):
            ident.ident(
                [self.bi[w] for w in "SEN"], [self.known["heater"]],
                pts=np.tile([[0.5, 0.4]], (10, 1)), thc=0.7,
            )

    def test_covariance_matches_analytic_diagonal_case(self):
        C, diagnostics = ident.covariance_from_jacobian([[2, 0], [0, 4]], sigma=0.1)
        self.assertEqual(diagnostics["rank"], 2)
        np.testing.assert_allclose(C, np.diag([0.0025, 0.000625]), atol=1e-15, rtol=0)

    def test_energy_balance_includes_prescribed_boundary_flux(self):
        x, y, _ = solver.make_grid(51)
        X, _ = np.meshgrid(x, y, indexing="ij")
        exact = lambda s: 1 + s - 0.5 * s ** 2
        bi = self.bi
        g = {
            "S": lambda s: -bi["S"] * exact(s),
            "N": lambda s: -bi["N"] * (exact(s) - 0.7),
            "E": lambda s: np.full_like(s, -bi["E"] * 1.5),
            "W": lambda s: np.ones_like(s),
        }
        _, _, theta, info = solver.solve(
            51, bi, self.known["theta_inf"], Q=np.ones_like(X), g=g,
        )
        np.testing.assert_allclose(theta, exact(X), atol=1e-10, rtol=0)
        self.assertLess(info["balance_rel_err"], 1e-10)
        self.assertGreater(abs(sum(info["loss"].values()) - info["q_in"]), 0.1)
        self.assertAlmostEqual(sum(info["total_flux"].values()), info["q_in"], places=10)

    def test_existing_production_field_and_mms_are_preserved(self):
        _, _, orders = solver.mms_test(verbose=False)
        self.assertGreater(min(orders), 1.9)
        coarse = solver.solve(101, self.bi, self.known["theta_inf"],
                              heaters=[self.known["heater"]])
        fine = solver.solve(201, self.bi, self.known["theta_inf"],
                            heaters=[self.known["heater"]])
        with np.load(DATA / "reference.npz") as ref:
            np.testing.assert_allclose(fine[2], ref["theta"], atol=1e-10, rtol=0)
        self.assertLess(20 * np.max(np.abs(coarse[2] - fine[2][::2, ::2])), 0.01)
        self.assertLess(fine[3]["balance_rel_err"], 1e-8)
        self.assertEqual(fine[3]["total_flux"], fine[3]["loss"])


if __name__ == "__main__":
    unittest.main()
