"""Unit tests for hollow-profile relaxation metrics and Jun2026_relaxation helpers."""
import os
import sys
import unittest

import numpy as np

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "experiments", "ucla-lapd", "Jun-2026"))

from data_analysis.plasma.profiles import (
    HollowProfileMetrics,
    hollow_profile_metrics,
    orbit_smeared_profile,
)
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

    def test_orbit_smeared_profile_identity_and_apparent_filling(self):
        r = np.linspace(0.0, 8.0, 161)
        prof = 1.0 - 0.6 * np.exp(-(r / 2.5) ** 2)
        # Zero displacement is exact identity
        np.testing.assert_allclose(orbit_smeared_profile(r, prof, 0.0), prof)
        # Rigid circular orbit of radius dr raises r=0 to prof(dr) and lowers hollowness
        dr = 1.5
        smeared = orbit_smeared_profile(r, prof, dr)
        self.assertAlmostEqual(smeared[0], float(np.interp(dr, r, prof)), places=6)
        h0 = hollow_profile_metrics(r, prof).hollowness
        hs = hollow_profile_metrics(r, smeared).hollowness
        self.assertLess(hs, h0)

    def test_analyze_mach32_arrays_recovers_offcenter_rotation_and_sign(self):
        # Synthetic pure counter-clockwise rigid-like rotation about (cx, cy) = (1.0, -0.5)
        gx, gy = np.meshgrid(np.linspace(-10.0, 10.0, 41), np.linspace(-10.0, 10.0, 41))
        px, py = gx.ravel(), gy.ravel()
        cx_true, cy_true = 1.0, -0.5
        dx, dy = px - cx_true, py - cy_true
        rr = np.hypot(dx, dy)
        # Purely azimuthal flow: v_theta = +0.085 (counter-clockwise) -> vx = -v_th * dy/r, vy = +v_th * dx/r
        with np.errstate(invalid="ignore", divide="ignore"):
            ux_unit = np.where(rr > 0.0, -dy / rr, 0.0)
            uy_unit = np.where(rr > 0.0, dx / rr, 0.0)
        t32 = np.linspace(-1.0, 6.0, 71)
        amp_t = np.where((t32 >= 0.0) & (t32 <= 5.0), 0.085, 0.0)
        mx = ux_unit[:, None] * amp_t[None, :]
        my = uy_unit[:, None] * amp_t[None, :]
        # Core axial flow = -0.35, ring = -0.15
        mz_base = np.where(rr <= 2.0, -0.35, -0.15)
        mz = np.broadcast_to(mz_base[:, None], mx.shape).copy()

        res = jrx._analyze_mach32_arrays(t32, px, py, mx, my, mz, t_on_ref_ms=1.30)
        np.testing.assert_allclose(res["mach32__center_xy_cm"], [cx_true, cy_true], atol=0.05)
        self.assertLess(res["mach32__center_ratio"], 0.05)
        self.assertAlmostEqual(res["mach32__M_theta_wall_pre_onset"], +0.085, places=3)
        self.assertAlmostEqual(res["mach32__M_theta_wall_late_sat"], +0.085, places=3)
        self.assertAlmostEqual(res["mach32__M_theta_late_over_pre"], 1.0, places=3)
        self.assertAlmostEqual(res["mach32__M_z_core_late_sat"], -0.35, places=3)
        self.assertAlmostEqual(res["mach32__M_z_ring_late_sat"], -0.15, places=3)

    def test_2d_plane_per_window_centering_and_late_onset_no_crash(self):
        # Rigid hollow core shifts from (0.0, -1.0) pre-onset to (2.0, 1.0) late-bias without filling
        gx, gy = np.meshgrid(np.linspace(-8.0, 8.0, 17), np.linspace(-8.0, 8.0, 17))
        xu, yu = gx.ravel(), gy.ravel()
        vals = {
            "pre_bias": np.ones_like(xu),
            "pre_onset": 1.0 - 0.6 * np.exp(-((xu - 0.0) ** 2 + (yu + 1.0) ** 2) / 6.0),
            "early_post": 1.0 - 0.6 * np.exp(-((xu - 1.0) ** 2 + (yu - 0.0) ** 2) / 6.0),
            "late_sat": 1.0 - 0.6 * np.exp(-((xu - 2.0) ** 2 + (yu - 1.0) ** 2) / 6.0),
            "post_off": 1.0 - 0.3 * np.exp(-((xu - 2.0) ** 2 + (yu - 1.0) ** 2) / 6.0),
        }
        res = jrx._summarize_2d_plane_windows(xu, yu, vals, t_on_ms=1.40)
        # Per-window centering must track the 2.83 cm shift and keep H_late == H_preOn!
        np.testing.assert_allclose(res["center_pre_xy_cm"], [0.0, -1.0], atol=0.1)
        np.testing.assert_allclose(res["center_late_xy_cm"], [2.0, 1.0], atol=0.1)
        self.assertAlmostEqual(res["H_late_over_pre"], 1.0, delta=0.01)

        # Late onset (e.g. t_on = 4.20 ms) -> late_sat omitted from vals must NOT crash
        vals_no_late = {k: v for k, v in vals.items() if k != "late_sat"}
        res_nl = jrx._summarize_2d_plane_windows(xu, yu, vals_no_late, t_on_ms=4.20)
        self.assertTrue(np.isnan(res_nl["H_late_sat"]))
        self.assertTrue(np.isnan(res_nl["H_late_over_pre"]))
        self.assertTrue(np.all(np.isnan(res_nl["center_late_xy_cm"])))
        self.assertTrue(np.isfinite(res_nl["H_pre_onset"]))

    def test_local_orbit_displacement_removes_radial_bias(self):
        # Synthetic curved hollow core: f(r) = 1 - 0.65 * exp(-(r / 2.8)^2) wobbling rigidly by dr_true = 1.0 cm
        r_grid = np.linspace(0.0, 8.0, 321)
        prof = 1.0 - 0.65 * np.exp(-(r_grid / 2.8) ** 2)
        dr_true = 1.0
        phi = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        r_test = np.array([2.0, 3.0, 4.0])
        # Exact linear/weakly-nonlinear RMS of f(sqrt(r^2 + dr^2 - 2 r dr cos phi)) / mean
        rms_test = np.empty_like(r_test)
        for i, rt in enumerate(r_test):
            # Use small-amplitude perturbation check (dr = 0.2 cm) for exact linearization AND dr = 1.0 cm
            r_orb = np.sqrt(rt ** 2 + dr_true ** 2 - 2.0 * rt * dr_true * np.cos(phi))
            vals = np.interp(r_orb, r_grid, prof)
            rms_test[i] = float(np.std(vals) / np.mean(vals))
        dr_local = jrx._local_orbit_displacement(r_grid, prof, r_test, rms_test)
        # Local gradient removes the factor-of-3.8 bias (1.55 / 0.93 / 0.41 cm) seen with band-averaged slope
        np.testing.assert_allclose(dr_local, dr_true, rtol=0.18)


if __name__ == "__main__":
    unittest.main()
