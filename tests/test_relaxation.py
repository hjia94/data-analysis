"""Unit tests for hollow-profile relaxation metrics and Jun2026_relaxation helpers."""
import os
import sys
import unittest

import numpy as np

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "experiments", "ucla-lapd", "Jun-2026"))

from data_analysis.plasma.profiles import HollowProfileMetrics, hollow_profile_metrics
import Jun2026_relaxation as jrx


class TestHollowProfileMetrics(unittest.TestCase):
    def test_1d_analytic_exponential_wall(self):
        # f(r) = 1.0 for r <= 2, exponential exp(0.4 * (r - 1.5)) on wall, flat 3.0 on ring
        r = np.linspace(0.0, 8.0, 81)
        prof = np.where(r <= 2.0, 1.0, np.where(r >= 4.5, 3.0, np.exp(0.4 * (r - 1.5))))
        # Exact exponential across [1.5, 4.5]: override [1.5, 4.5] explicitly
        m_wall = (r >= 1.5) & (r <= 4.5)
        prof[m_wall] = np.exp(0.4 * (r[m_wall] - 1.5))
        # Core median stays 1.0 (0..1.4 has 15 pts at 1.0 vs 6 pts in 1.5..2.0)
        m = hollow_profile_metrics(
            r, prof, core_r_max_cm=2.0, ring_r_cm=(4.5, 7.0), wall_r_cm=(1.5, 4.5),
            radial_edges_cm=np.array([0.0, 2.0, 4.5, 7.0]),
        )
        self.assertIsInstance(m, HollowProfileMetrics)
        self.assertAlmostEqual(m.core_val, 1.0, places=6)
        self.assertAlmostEqual(m.ring_val, 3.0, places=6)
        self.assertAlmostEqual(m.hollowness, 3.0, places=6)
        self.assertAlmostEqual(m.wall_inv_scale_cm, 0.4, places=6)
        self.assertEqual(m.radial_profile.shape, (3,))
        self.assertAlmostEqual(m.radial_profile[0], 1.0, places=6)
        self.assertAlmostEqual(m.radial_profile[2], 3.0, places=6)

    def test_2d_time_series_and_nonpositive_masking(self):
        # 2-D scattered coordinates over 3 time bins, including NaN and negative outlier in wall
        rng = np.random.default_rng(7)
        x = rng.uniform(-7.5, 7.5, 250)
        y = rng.uniform(-7.5, 7.5, 250)
        r = np.hypot(x, y)
        slopes = np.array([0.0, 0.25, 0.50])
        prof = np.exp(r[:, None] * slopes[None, :])
        # Inject invalid samples in time bin 1
        prof[0, 1] = np.nan
        prof[1, 1] = -2.0
        m = hollow_profile_metrics(r, prof, core_r_max_cm=2.0, ring_r_cm=(4.5, 7.0), wall_r_cm=(1.5, 4.5))
        self.assertEqual(m.hollowness.shape, (3,))
        np.testing.assert_allclose(m.wall_inv_scale_cm, slopes, atol=1e-6)
        self.assertAlmostEqual(m.hollowness[0], 1.0, places=6)
        self.assertGreater(m.hollowness[2], m.hollowness[1])
        self.assertGreater(m.hollowness[1], m.hollowness[0])

    def test_nonpositive_core_returns_nan_hollowness(self):
        r = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        prof = np.array([-0.1, 0.0, 0.0, 2.0, 2.2])
        m = hollow_profile_metrics(r, prof)
        self.assertTrue(np.isnan(m.hollowness))

    def test_window_bounds_and_2d_center(self):
        # Stable run (NaN onset) -> fixed mid/late reference windows
        w_stable = jrx._window_bounds(np.nan)
        self.assertEqual(w_stable["pre_onset"], (1.50, 2.50))
        self.assertEqual(w_stable["late_sat"], (3.50, 4.80))

        # Very late onset (Run 10 at 4.20 ms) -> late_sat must be None (no silent fallback)
        w_late = jrx._window_bounds(4.20)
        self.assertIsNone(w_late["late_sat"])
        self.assertTrue(np.isnan(jrx._window_median(np.linspace(0, 5, 51), np.ones(51), w_late["late_sat"])))

        # 2-D core center recovery on off-center Gaussian hollow profile
        gx, gy = np.meshgrid(np.linspace(-6, 6, 25), np.linspace(-6, 6, 25))
        xu, yu = gx.ravel(), gy.ravel()
        vals = 1.0 - 0.6 * np.exp(-((xu - 1.5) ** 2 + (yu + 1.0) ** 2) / 4.0)
        xc, yc = jrx._find_2d_core_center(xu, yu, vals, r_search_cm=3.5, k_lowest=5)
        self.assertAlmostEqual(xc, 1.5, delta=0.15)
        self.assertAlmostEqual(yc, -1.0, delta=0.15)

    def test_batch_all_respects_overwrite_false(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, jrx.OUT_NAME)
            np.savez(p, sentinel=np.array(42))
            loaded = jrx.batch_all(data_dir=tmp, overwrite=False)
            self.assertEqual(int(loaded["sentinel"]), 42)


if __name__ == "__main__":
    unittest.main()
