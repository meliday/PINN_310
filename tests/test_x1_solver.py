"""X1-a: FD solver with an internal adiabatic wall (criteria pre-registered in LOG-073)."""
import json
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from classroom_heat_pinn_data.synthetic import solver, solver_wall as sw

KNOWN = json.loads((Path(__file__).resolve().parents[1] / "classroom_heat_pinn_data/synthetic/data/known.json")
                   .read_text())
HEATERS = [KNOWN["heater"]]
TINF = {"S": 0.0, "E": 0.0, "N": 0.7}
BI = {"S": 1.0, "E": 0.4, "N": 0.25}  # fixed test values (LOG-073), not read from truth files
WALL = {"x0": 0.0, "x1": 0.60, "y0": 0.30, "y1": 0.34}


class WallSolverTests(unittest.TestCase):
    def test_no_solid_reduces_to_base_solver(self):
        _, _, base, _ = solver.solve(101, BI, TINF, heaters=HEATERS)
        _, _, wall, info = sw.solve(101, BI, TINF, solids=[], heaters=HEATERS)
        self.assertTrue(info["active"].all())
        self.assertLessEqual(np.abs(base - wall).max(), 1e-12)

    def test_energy_balance_with_wall(self):
        for nx in (101, 201):
            _, _, theta, info = sw.solve(nx, BI, TINF, solids=[WALL], heaters=HEATERS)
            self.assertLessEqual(info["balance_rel_err"], 1e-10)
            self.assertTrue(np.isfinite(theta[info["active"]]).all())

    def test_wall_mms_second_order(self):
        hs, errs, orders = sw.mms_wall_test(WALL)
        self.assertGreaterEqual(min(orders[-2:]), 1.8, (errs, orders))

    def test_solid_nodes_removed_and_faces_not_connected(self):
        x, y, theta, info = sw.solve(101, BI, TINF, solids=[WALL], heaters=HEATERS)
        X, Y = np.meshgrid(x, y, indexing="ij")
        inside = (X < 0.6 - 1e-9) & (Y > 0.30 + 1e-9) & (Y < 0.34 - 1e-9)
        np.testing.assert_array_equal(~info["active"], inside)
        self.assertTrue(np.isnan(theta[inside]).all())
        # Graph distance across the wall: below-face and above-face nodes at x < 0.6 have
        # no direct coupling; the only path runs through the passage.
        M, idx = info["matrix"].tocsr(), info["index"]
        below = idx[(np.abs(Y - 0.30) < 1e-9) & (X < 0.6 - 1e-9)]
        above = idx[(np.abs(Y - 0.34) < 1e-9) & (X < 0.6 - 1e-9)]
        self.assertEqual(M[below][:, above].nnz, 0)
        # Each below-face node couples only to nodes at y <= 0.30.
        ycoord = np.full(M.shape[0], np.nan)
        ycoord[idx[info["active"]]] = Y[info["active"]]
        for r in below:
            nbrs = M[r].indices
            self.assertTrue((ycoord[nbrs] <= 0.30 + 1e-9).all())

    def test_heat_crosses_wall_only_through_passage(self):
        # A source confined below the wall at x < 0.6 must still warm the room above it, via the passage.
        # All ambient temperatures are 0 so the only heat input is the source (with the default
        # corridor at 0.7 the region above the wall is corridor-heated and the wall warms it instead).
        heater = [{"x": 0.3, "y": 0.1, "A": 8.0, "s": 0.05}]
        cold = {"S": 0.0, "E": 0.0, "N": 0.0}
        _, _, open_room, _ = sw.solve(101, BI, cold, solids=[], heaters=heater)
        _, _, walled, _ = sw.solve(101, BI, cold, solids=[WALL], heaters=heater)
        # node (0.3, 0.36) just above the wall: cooler with the wall, but still warmed via the passage
        i, j = 30, 36
        self.assertLess(walled[i, j], open_room[i, j])
        self.assertGreater(walled[i, j], 0.0)

    def test_x1_grid_convergence_is_wall_tip_limited(self):
        # LOG-073 criterion 5 (201 vs 401 <= 0.0025) is reported in LOG-074, not asserted here.
        # These checks pin what was observed: the difference is largest at the wall tip and
        # shrinks with refinement; away from the tip it is much smaller.
        fields = {nx: sw.solve(nx, BI, TINF, solids=[WALL], heaters=HEATERS)[:3] for nx in (201, 401, 801)}

        def diff(a, b):
            x, y, ta = fields[a]
            d = np.abs(ta - fields[b][2][::2, ::2])
            X, Y = np.meshgrid(x, y, indexing="ij")
            return d, np.hypot(X - 0.6, Y - 0.32)

        d1, r1 = diff(201, 401)
        d2, _ = diff(401, 801)
        k = np.nanargmax(d1)
        self.assertLess(r1.ravel()[k], 0.03)            # max sits at the re-entrant tip
        self.assertLess(np.nanmax(d2), np.nanmax(d1))   # and decreases with refinement
        self.assertLess(np.nanmax(d1[r1 > 0.1]), 0.25 * np.nanmax(d1))

    def test_misaligned_solid_rejected(self):
        with self.assertRaisesRegex(ValueError, "not on the grid"):
            sw.solve(101, BI, TINF, solids=[{**WALL, "y0": 0.305}], heaters=HEATERS)

    def test_wall_sensitive_two_compartment_exact(self):
        # LOG-075 noted the smooth MMS also passes with the wall ignored. Here a full-height wall
        # splits the room; with only W (theta_inf 1) and E (theta_inf 2) Robin walls and no source,
        # each side is exactly constant at its own ambient value. Ignoring the wall cannot match it.
        split = {"x0": 0.4, "x1": 0.6, "y0": 0.0, "y1": 0.8}
        bi, tinf = {"W": 1.0, "E": 1.0}, {"W": 1.0, "E": 2.0}
        x, y, theta, info = sw.solve(101, bi, tinf, solids=[split], Q=np.zeros((101, 81)))
        X = np.meshgrid(x, y, indexing="ij")[0]
        exact = np.where(X <= 0.4, 1.0, 2.0)
        act = info["active"]
        self.assertLess(np.abs(theta[act] - exact[act]).max(), 1e-10)
        _, _, no_wall, _ = solver.solve(101, bi, tinf, Q=np.zeros((101, 81)))
        self.assertGreater(np.abs(no_wall[act] - exact[act]).max(), 0.3)

    def test_point_contact_rejected(self):
        # X1-POINT-CONTACT-01: two fluid quadrants meeting only at (0.5, 0.4).
        pinch = [{"x0": 0.0, "x1": 0.5, "y0": 0.4, "y1": 0.8}, {"x0": 0.5, "x1": 1.0, "y0": 0.0, "y1": 0.4}]
        with self.assertRaisesRegex(ValueError, "corner"):
            sw.solve(101, BI, TINF, solids=pinch, heaters=HEATERS)

    def test_local_point_contact_in_connected_room_rejected(self):
        # LOG-080 counterexample: solids touch at (0.5, 0.4) while the room stays connected
        # around them, so cell and node region counts are both 1. Must be rejected before solving.
        solids = [{"x0": 0.2, "x1": 0.5, "y0": 0.4, "y1": 0.6}, {"x0": 0.5, "x1": 0.8, "y0": 0.2, "y1": 0.4}]
        with mock.patch.object(sw.spla, "spsolve", side_effect=AssertionError("solved")), \
                self.assertRaisesRegex(ValueError, "corner"):
            sw.solve(51, BI, TINF, solids=solids, heaters=HEATERS)
        # Solids sharing a full edge (no pinch) remain allowed.
        sw.solve(51, BI, TINF, solids=[{"x0": 0.2, "x1": 0.5, "y0": 0.4, "y1": 0.6},
                                       {"x0": 0.5, "x1": 0.8, "y0": 0.4, "y1": 0.6}], heaters=HEATERS)

    def test_unanchored_region_rejected(self):
        # X1-UNANCHORED-01: the left compartment has no Robin boundary (only E has Bi > 0).
        split = {"x0": 0.4, "x1": 0.6, "y0": 0.0, "y1": 0.8}
        with self.assertRaisesRegex(ValueError, "no Robin boundary"):
            sw.solve(101, {"E": 1.0}, {"E": 0.0}, solids=[split], heaters=[{"x": 0.2, "y": 0.4, "A": 1.0, "s": 0.05}])

    def test_reversed_or_zero_area_solid_rejected(self):
        for bad in ({**WALL, "x0": 0.6, "x1": 0.0}, {**WALL, "y1": 0.30}):
            with self.subTest(solid=bad), self.assertRaisesRegex(ValueError, "zero area"):
                sw.solve(101, BI, TINF, solids=[bad], heaters=HEATERS)

    def test_sample_refuses_points_next_to_solid(self):
        x, y, theta, _ = sw.solve(101, BI, TINF, solids=[WALL], heaters=HEATERS)
        sensors = np.loadtxt(Path(__file__).resolve().parents[1] / "classroom_heat_pinn_data/synthetic/data/"
                             "sensors_A_clean.csv", delimiter=",", skiprows=1)[:, 1:3]
        self.assertTrue(np.isfinite(sw.sample(x, y, theta, sensors)).all())
        # Above the top surface y = 0.34 the cell's nodes are all fluid (surface nodes are unknowns).
        self.assertTrue(np.isfinite(sw.sample(x, y, theta, np.array([[0.3, 0.345]]))).all())
        for pt in ([0.3, 0.30], [0.3, 0.335]):  # bottom surface (stencil uses y=0.31), inside the wall
            with self.subTest(pt=pt), self.assertRaisesRegex(ValueError, "next to a solid"):
                sw.sample(x, y, theta, np.array([pt]))


if __name__ == "__main__":
    unittest.main()
