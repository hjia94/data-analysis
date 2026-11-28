"""Unit tests for data_analysis.plasma.mach and data_analysis.plasma.flow physics & sign conventions."""

import unittest

import numpy as np

from data_analysis.plasma.flow import (
    antisymmetric_rotation_profile,
    find_flow_centre,
    fit_exponential_relaxation,
    polar_components,
    radial_flow_shear,
)
from data_analysis.plasma.mach import (
    FLOW_SIGN_PLUS_FACING_AXIS,
    K_HUTCHINSON,
    area_ratio,
    binned_face_ratio,
    fit_calibration,
    flow_velocity,
    mach_number,
    mach_single,
    rescale_mach_velocity,
)


class TestMachAndFlow(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.plus = rng.uniform(1.0, 5.0, size=(5, 100))
        self.minus = rng.uniform(1.0, 5.0, size=(5, 100))

    def test_contiguous_bins_match_direct_sums(self):
        edges = np.array([0, 20, 50, 80, 100])
        ratio, counts, amp = binned_face_ratio(self.plus, self.minus, edges)
        for k in range(len(edges) - 1):
            s0, s1 = edges[k], edges[k + 1]
            self.assertEqual(counts[k], 5 * (s1 - s0))
            self.assertAlmostEqual(ratio[k], self.plus[:, s0:s1].sum() / self.minus[:, s0:s1].sum(), places=12)
            self.assertAlmostEqual(
                amp[k],
                (self.plus[:, s0:s1].sum() + self.minus[:, s0:s1].sum()) / (5 * (s1 - s0)),
                places=12,
            )

    def test_zero_width_tail_and_non_positive_bins(self):
        edges = np.array([0, 0, 0, 10, 20, 100, 100, 100])
        ratio, counts, amp = binned_face_ratio(self.plus, self.minus, edges)
        for k in (0, 1, 5, 6):
            self.assertEqual(counts[k], 0)
            self.assertTrue(np.isnan(ratio[k]) and np.isnan(amp[k]))
        self.assertAlmostEqual(ratio[2], self.plus[:, 0:10].sum() / self.minus[:, 0:10].sum(), places=12)

        plus = np.ones((3, 20))
        plus[:, 10:] = 999.0
        r2, c2, a2 = binned_face_ratio(plus, np.ones((3, 20)), np.array([0, 5, 10]))
        np.testing.assert_array_equal(c2, [15, 15])
        np.testing.assert_allclose(r2, [1.0, 1.0])
        np.testing.assert_allclose(a2, [2.0, 2.0])

        plus_bad = self.plus.copy()
        plus_bad[:, 10:20] = -1.0
        r3, c3, _ = binned_face_ratio(plus_bad, self.minus, np.array([0, 10, 20, 30]))
        self.assertEqual((c3[0], c3[1], c3[2]), (50, 0, 50))
        self.assertTrue(np.isnan(r3[1]))

    def test_mach_lab_axis_sign_convention(self):
        self.assertEqual(FLOW_SIGN_PLUS_FACING_AXIS, -1)
        self.assertGreater(mach_single(R=0.5, kappa=1.0), 0.0)
        self.assertLess(mach_single(R=2.0, kappa=1.0), 0.0)
        self.assertGreater(mach_number(R_a=0.5, R_b=2.0), 0.0)
        self.assertAlmostEqual(mach_number(R_a=0.5, R_b=2.0), mach_single(R=0.5, kappa=1.0), places=12)
        self.assertAlmostEqual(
            mach_single(R=2.0, kappa=1.0, flow_sign=+1),
            -mach_single(R=2.0, kappa=1.0),
            places=12,
        )

    def test_hutchinson_ratio_calibration_exact_factor(self):
        self.assertAlmostEqual(K_HUTCHINSON, 0.45, places=12)
        self.assertAlmostEqual(mach_single(R=np.exp(1.0 / K_HUTCHINSON), kappa=1.0), -1.0, places=12)
        self.assertAlmostEqual(mach_single(R=np.exp(-1.0 / K_HUTCHINSON), kappa=1.0), +1.0, places=12)
        self.assertAlmostEqual(mach_single(R=np.exp(1.0 / K_HUTCHINSON), kappa=1.0, flow_sign=+1), +1.0, places=12)

        kappa_true, M_true = 1.35, 0.42
        R_a = kappa_true * np.exp(FLOW_SIGN_PLUS_FACING_AXIS * M_true / K_HUTCHINSON)
        R_b = kappa_true * np.exp(-FLOW_SIGN_PLUS_FACING_AXIS * M_true / K_HUTCHINSON)
        self.assertAlmostEqual(area_ratio(R_a, R_b), kappa_true, places=12)
        self.assertAlmostEqual(mach_number(R_a, R_b), M_true, places=12)
        self.assertAlmostEqual(mach_single(R_a, kappa_true), M_true, places=12)

        k_y, k_z, m1, m2 = 1.18, 0.92, 0.25, -0.15
        s = FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON
        design = np.array([
            [1.0, 0.0, s, 0.0], [1.0, 0.0, -s, 0.0],
            [1.0, 0.0, 0.0, s], [1.0, 0.0, 0.0, -s],
            [0.0, 1.0, 0.0, s], [0.0, 1.0, 0.0, -s],
            [0.0, 1.0, s, 0.0], [0.0, 1.0, -s, 0.0],
        ])
        truth = np.array([np.log(k_y), np.log(k_z), m1, m2])
        params, _, residuals, _ = fit_calibration(design @ truth, np.full(8, 0.01), design)
        np.testing.assert_allclose(params, truth, atol=1e-12)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_rescale_mach_velocity_errors_and_invalid_te(self):
        from data_analysis.plasma.profiles import local_drift_scales

        M = np.array([0.0, 0.25, -0.5])
        Te = np.array([4.0, 4.0, -1.0])
        v, v_err = rescale_mach_velocity(M, Te, mu=4.0, te_err_ev=0.4, M_err=0.05)
        cs = flow_velocity(1.0, 4.0, 4.0)
        cs_prof = float(local_drift_scales(1e12, 4.0, 800.0, mu=4.0)["cs_m_s"]) * 1e-3
        self.assertAlmostEqual(cs, cs_prof, delta=0.01)
        self.assertAlmostEqual(v[0], 0.0, places=12)
        self.assertAlmostEqual(v_err[0], cs * 0.05, places=12)
        self.assertAlmostEqual(v[1], 0.25 * cs, places=12)
        expected_err1 = np.sqrt((cs * 0.05) ** 2 + (0.5 * v[1] * 0.1) ** 2)
        self.assertAlmostEqual(v_err[1], expected_err1, places=12)
        self.assertTrue(np.isnan(v[2]) and np.isnan(v_err[2]))

    def test_antisymmetric_rotation_and_radial_flow_shear(self):
        x = np.arange(-6.0, 8.1, 1.0)
        xc = 1.0
        r, v_rot, v_sym = antisymmetric_rotation_profile(x, 0.85 + 0.4 * (x - xc), x_center_cm=xc)
        np.testing.assert_allclose(r, np.arange(1.0, 7.1, 1.0))
        np.testing.assert_allclose(v_rot, 0.4 * r, atol=1e-12)
        np.testing.assert_allclose(v_sym, 0.85, atol=1e-12)

        r2 = np.arange(1.0, 10.1, 1.0)
        sh_rigid = radial_flow_shear(r2, 0.5 * r2, smooth_window=5, polyorder=2)
        np.testing.assert_allclose(sh_rigid.dv_dr_s, 5e4, rtol=1e-10)
        np.testing.assert_allclose(sh_rigid.omega_rad_s, 5e4, rtol=1e-10)
        np.testing.assert_allclose(sh_rigid.gamma_E_s, 0.0, atol=1e-7)

        sh_flat = radial_flow_shear(r2, np.full_like(r2, 2.0), smooth_window=1)
        np.testing.assert_allclose(sh_flat.dv_dr_s, 0.0, atol=1e-12)
        np.testing.assert_allclose(sh_flat.gamma_E_s, -(2.0 / r2) * 1e5, rtol=1e-12)

        with self.assertRaises(ValueError):
            radial_flow_shear(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), np.ones(5))

    def test_find_flow_centre_and_polar_components_offcenter_rotation(self):
        gx, gy = np.meshgrid(np.linspace(-10.0, 10.0, 41), np.linspace(-10.0, 10.0, 41))
        px, py = gx.ravel(), gy.ravel()
        cx_true, cy_true = 1.0, -0.5
        dx, dy = px - cx_true, py - cy_true
        rr = np.hypot(dx, dy)
        with np.errstate(invalid="ignore", divide="ignore"):
            ux_unit = np.where(rr > 0.0, -dy / rr, 0.0)
            uy_unit = np.where(rr > 0.0, dx / rr, 0.0)

        # Off-center CCW rotation (v_theta = +0.085) plus uniform DC offset (+0.030, -0.025)
        v_th_true = 0.085
        dc_x, dc_y = 0.030, -0.025
        vx_raw = v_th_true * ux_unit + dc_x
        vy_raw = v_th_true * uy_unit + dc_y
        vx = vx_raw - dc_x
        vy = vy_raw - dc_y

        cx, cy, ratio = find_flow_centre(vx, vy, px, py)
        np.testing.assert_allclose([cx, cy], [cx_true, cy_true], atol=1e-12)
        self.assertLess(ratio, 1e-9)

        vr, vth = polar_components(vx, vy, px, py, (cx, cy))
        away = rr > 0.0
        np.testing.assert_allclose(vr[away], 0.0, atol=1e-12)
        np.testing.assert_allclose(vth[away], +v_th_true, atol=1e-12)
        self.assertTrue(np.all(np.isnan(vr[~away])) and np.all(np.isnan(vth[~away])))

        # Reversing flow flips v_theta sign (CW -> negative), and 2-D (npos, nbin) broadcasts
        vr_2d, vth_2d = polar_components(
            np.column_stack([vx, -vx]), np.column_stack([vy, -vy]), px, py, (cx, cy),
        )
        np.testing.assert_allclose(vr_2d[away], 0.0, atol=1e-12)
        np.testing.assert_allclose(vth_2d[away, 0], +v_th_true, atol=1e-12)
        np.testing.assert_allclose(vth_2d[away, 1], -v_th_true, atol=1e-12)

    def test_fit_exponential_relaxation_recovers_decay_and_offset(self):
        t = np.linspace(5.0, 8.0, 61)
        tau_true = 0.55
        amp_true = -0.22
        off_true = 0.015
        y = amp_true * np.exp(-(t - 5.1) / tau_true) + off_true
        fit = fit_exponential_relaxation(t, y, t_start=5.1, t_end=7.9)
        self.assertAlmostEqual(fit["tau"], tau_true, places=5)
        self.assertAlmostEqual(fit["rate"], 1.0 / tau_true, places=5)
        self.assertAlmostEqual(fit["amp"], amp_true, places=5)
        self.assertAlmostEqual(fit["offset"], off_true, places=5)
        self.assertGreater(fit["r2"], 0.999999)

        # Too few finite points returns NaN gracefully
        fit_empty = fit_exponential_relaxation(t[:2], y[:2], min_points=4)
        self.assertTrue(np.isnan(fit_empty["tau"]))


if __name__ == "__main__":
    unittest.main()
