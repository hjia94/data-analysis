"""Unit tests for Mach probe face-ratio binning and sweep-edge alignment."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

from data_analysis.plasma.flow import (
    antisymmetric_rotation_profile,
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
import Jun2026_mach_line as jml


class TestBinnedFaceRatio(unittest.TestCase):
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
            expected_r = self.plus[:, s0:s1].sum() / self.minus[:, s0:s1].sum()
            expected_a = (self.plus[:, s0:s1].sum() + self.minus[:, s0:s1].sum()) / (5 * (s1 - s0))
            self.assertAlmostEqual(ratio[k], expected_r, places=12)
            self.assertAlmostEqual(amp[k], expected_a, places=12)

    def test_leading_and_trailing_zero_width_bins_are_nan(self):
        # Reproduces Run 08-14 sweep_bin_edges when IV timestamps bracket the Mach record.
        edges = np.array([0, 0, 0, 10, 20, 100, 100, 100])
        ratio, counts, amp = binned_face_ratio(self.plus, self.minus, edges)
        for k in (0, 1, 5, 6):
            self.assertEqual(counts[k], 0)
            self.assertTrue(np.isnan(ratio[k]))
            self.assertTrue(np.isnan(amp[k]))
        self.assertEqual(counts[2], 50)
        self.assertAlmostEqual(
            ratio[2], self.plus[:, 0:10].sum() / self.minus[:, 0:10].sum(), places=12
        )
        self.assertEqual(counts[3], 50)
        self.assertAlmostEqual(
            ratio[3], self.plus[:, 10:20].sum() / self.minus[:, 10:20].sum(), places=12
        )

    def test_final_edge_before_record_end_does_not_bleed_tail(self):
        plus = np.ones((3, 20))
        minus = np.ones((3, 20))
        plus[:, 10:] = 999.0  # spike strictly past edges[-1] = 10
        ratio, counts, amp = binned_face_ratio(plus, minus, np.array([0, 5, 10]))
        np.testing.assert_array_equal(counts, [15, 15])
        np.testing.assert_allclose(ratio, [1.0, 1.0])
        np.testing.assert_allclose(amp, [2.0, 2.0])

    def test_non_positive_samples_are_excluded(self):
        plus_bad = self.plus.copy()
        plus_bad[:, 10:20] = -1.0
        ratio, counts, _ = binned_face_ratio(plus_bad, self.minus, np.array([0, 10, 20, 30]))
        self.assertEqual(counts[0], 50)
        self.assertEqual(counts[1], 0)
        self.assertTrue(np.isnan(ratio[1]))
        self.assertEqual(counts[2], 50)

    def test_sweep_bin_edges_pre_record_bins_are_empty(self):
        tarr = np.linspace(-2e-3, 8e-3, 100001)
        timestamps = np.arange(-11.9, 12.2, 1.0) * 1e-3
        edges, _ = jml.sweep_bin_edges(tarr, timestamps)
        dummy = np.ones((2, tarr.size))
        ratio, counts, _ = binned_face_ratio(dummy, dummy, edges)
        n_pre = int(np.sum(timestamps[1:] <= -2e-3))
        self.assertGreater(n_pre, 0)
        np.testing.assert_array_equal(counts[:n_pre], 0)
        self.assertTrue(np.all(np.isnan(ratio[:n_pre])))
        self.assertTrue(np.all(counts[n_pre:n_pre + 10] > 0))
        np.testing.assert_allclose(ratio[n_pre:n_pre + 10], 1.0)


class TestMachRescalingAndFlowShear(unittest.TestCase):
    def test_mach_lab_axis_sign_convention(self):
        # When the '+' tip faces +axis (FLOW_SIGN_PLUS_FACING_AXIS = -1), plasma
        # flowing along +axis strikes the '-' face so R = j_+/j_- < kappa -> M_lab > 0.
        self.assertEqual(FLOW_SIGN_PLUS_FACING_AXIS, -1)
        self.assertGreater(mach_single(R=0.5, kappa=1.0), 0.0)
        self.assertLess(mach_single(R=2.0, kappa=1.0), 0.0)
        # Reversing orientation swaps R_a and R_b -> M_lab > 0 when R_a < R_b
        self.assertGreater(mach_number(R_a=0.5, R_b=2.0), 0.0)
        self.assertAlmostEqual(
            mach_number(R_a=0.5, R_b=2.0), mach_single(R=0.5, kappa=1.0), places=12
        )
        # Explicit flow_sign=+1 overrides for probes whose '+' tip faces -axis
        self.assertAlmostEqual(
            mach_single(R=2.0, kappa=1.0, flow_sign=+1),
            -mach_single(R=2.0, kappa=1.0),
            places=12,
        )

    def test_hutchinson_ratio_calibration_exact_factor(self):
        # Hutchinson (1987): R = kappa * exp(s * M / M_c) with M_c = K_HUTCHINSON = 0.45.
        # For R = exp(1 / 0.45), kappa = 1: |M| must be 1.0 (not 0.5).
        self.assertAlmostEqual(K_HUTCHINSON, 0.45, places=12)
        self.assertAlmostEqual(
            mach_single(R=np.exp(1.0 / K_HUTCHINSON), kappa=1.0), -1.0, places=12
        )
        self.assertAlmostEqual(
            mach_single(R=np.exp(-1.0 / K_HUTCHINSON), kappa=1.0), +1.0, places=12
        )
        self.assertAlmostEqual(
            mach_single(R=np.exp(1.0 / K_HUTCHINSON), kappa=1.0, flow_sign=+1),
            +1.0,
            places=12,
        )
        # Opposing orientations: R_a = kappa * exp(s * M / K), R_b = kappa * exp(-s * M / K)
        kappa_true, M_true = 1.35, 0.42
        R_a = kappa_true * np.exp(FLOW_SIGN_PLUS_FACING_AXIS * M_true / K_HUTCHINSON)
        R_b = kappa_true * np.exp(-FLOW_SIGN_PLUS_FACING_AXIS * M_true / K_HUTCHINSON)
        self.assertAlmostEqual(area_ratio(R_a, R_b), kappa_true, places=12)
        self.assertAlmostEqual(mach_number(R_a, R_b), M_true, places=12)
        self.assertAlmostEqual(mach_single(R_a, kappa_true), M_true, places=12)
        # Joint calibration design matrix: ln R = ln kappa + (s / K) * sign * M_lab
        k_y, k_z, m1, m2 = 1.18, 0.92, 0.25, -0.15
        design = np.array([
            [1.0, 0.0, FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON, 0.0],
            [1.0, 0.0, -FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON, 0.0],
            [1.0, 0.0, 0.0, FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON],
            [1.0, 0.0, 0.0, -FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON],
            [0.0, 1.0, 0.0, FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON],
            [0.0, 1.0, 0.0, -FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON],
            [0.0, 1.0, FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON, 0.0],
            [0.0, 1.0, -FLOW_SIGN_PLUS_FACING_AXIS / K_HUTCHINSON, 0.0],
        ])
        truth = np.array([np.log(k_y), np.log(k_z), m1, m2])
        ln_R = design @ truth
        params, _, residuals, _ = fit_calibration(ln_R, np.full(8, 0.01), design)
        np.testing.assert_allclose(params, truth, atol=1e-12)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_rescale_mach_velocity_errors_and_invalid_te(self):
        M = np.array([0.0, 0.25, -0.5])
        Te = np.array([4.0, 4.0, -1.0])
        v, v_err = rescale_mach_velocity(M, Te, mu=4.0, te_err_ev=0.4, M_err=0.05)
        cs = flow_velocity(1.0, 4.0, 4.0)
        # M = 0 must not produce 0/0 NaN; sigma_v = cs * M_err
        self.assertAlmostEqual(v[0], 0.0, places=12)
        self.assertAlmostEqual(v_err[0], cs * 0.05, places=12)
        # M = 0.25, Te = 4.0, te_err = 0.4 (10% -> 5% on sqrt(Te))
        self.assertAlmostEqual(v[1], 0.25 * cs, places=12)
        expected_err1 = np.sqrt((cs * 0.05) ** 2 + (0.5 * v[1] * 0.1) ** 2)
        self.assertAlmostEqual(v_err[1], expected_err1, places=12)
        # Non-positive Te -> NaN for both v and v_err
        self.assertTrue(np.isnan(v[2]) and np.isnan(v_err[2]))

    def test_antisymmetric_rotation_cancels_uniform_offset(self):
        x = np.arange(-6.0, 8.1, 1.0)
        xc = 1.0
        # Rigid rotation v_rot = 0.4 * r plus uniform DC offset +0.85 km/s
        v = 0.85 + 0.4 * (x - xc)
        r, v_rot, v_sym = antisymmetric_rotation_profile(x, v, x_center_cm=xc)
        np.testing.assert_allclose(r, np.arange(1.0, 7.1, 1.0))
        np.testing.assert_allclose(v_rot, 0.4 * r, atol=1e-12)
        np.testing.assert_allclose(v_sym, 0.85, atol=1e-12)

    def test_radial_flow_shear_rigid_body_zero_and_flat_profile_exact(self):
        r = np.arange(1.0, 10.1, 1.0)
        # Rigid-body rotation: v_theta = Omega0 * r -> gamma_E = r d(Omega)/dr == 0
        v_rigid = 0.5 * r  # 0.5 km/s per cm -> Omega = 5e4 rad/s
        sh_rigid = radial_flow_shear(r, v_rigid, smooth_window=5, polyorder=2)
        np.testing.assert_allclose(sh_rigid.dv_dr_s, 5e4, rtol=1e-10)
        np.testing.assert_allclose(sh_rigid.omega_rad_s, 5e4, rtol=1e-10)
        np.testing.assert_allclose(sh_rigid.gamma_E_s, 0.0, atol=1e-7)

        # Constant velocity v_theta = v0 = 2.0 km/s -> dv/dr = 0, gamma_E = -v0 / r
        v_flat = np.full_like(r, 2.0)
        sh_flat = radial_flow_shear(r, v_flat, smooth_window=1)
        np.testing.assert_allclose(sh_flat.dv_dr_s, 0.0, atol=1e-12)
        np.testing.assert_allclose(sh_flat.gamma_E_s, -(2.0 / r) * 1e5, rtol=1e-12)

        # Negative coordinates (raw full-chord x_cm) must be rejected explicitly
        with self.assertRaises(ValueError):
            radial_flow_shear(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), np.ones(5))

    def test_x_center_mismatch_raises_and_missing_prebias_warns_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. profiles_summary with x_center_cm=1.0 must fail if caller asks for 2.0
            prof_p = os.path.join(tmp, "profiles_summary_jun2026.npz")
            x = np.arange(-6.0, 8.1, 1.0)
            np.savez(
                prof_p,
                **{
                    "06__x_cm": x,
                    "06__Te": np.full((x.size, 2), 3.0),
                    "06__Te_err": np.full((x.size, 2), 0.1),
                    "06__v_ExB_m_s": np.full((x.size, 2), 1500.0),
                    "06__t_mid_ms": np.array([0.1, 1.1]),
                    "06__ref_sweep_idx": np.int64(1),
                    "06__gas_puff_ms": np.float64(23.0),
                    "06__x_center_cm": np.float64(1.0),
                },
            )
            with self.assertRaises(ValueError):
                jml._load_paired_iv_fields(tmp, "06", profiles_npz=prof_p, x_center_cm=2.0)

            # 2. Mach line with no pre-bias bins (all t_ms >= 0) warns and sets dv_Y_te=NaN
            ifn = os.path.join(tmp, "08-He-800G-bias40V-Mach-p33-line_2026-06-11.hdf5")
            with open(ifn, "wb") as f:
                f.write(b"")
            mach_npz = jml.line_npz_path(ifn)
            np.savez(
                mach_npz,
                iv_run=np.str_("06"),
                pos=x,
                t_ms=np.array([0.6, 1.6]),
                iv_sweep_index=np.array([0, 1], dtype=np.int32),
                axes=np.array(["X", "Y"]),
                ion_mu=np.float64(4.0),
                K_hutchinson=np.float64(0.45),
                flow_sign=np.int64(FLOW_SIGN_PLUS_FACING_AXIS),
                kappa=np.array([1.0, 1.0]),
                kappa_err_sys=np.array([0.05, 0.05]),
                M_X=np.zeros((x.size, 2)),
                M_Y=np.tile(0.2 * np.sign(x - 1.0)[:, None], (1, 2)),
            )
            with self.assertWarns(RuntimeWarning):
                res = jml.rescale_line_with_iv("08", data_dir=tmp, profiles_npz=prof_p)
            self.assertFalse(res["has_pre_bias"])
            self.assertTrue(np.all(np.isnan(res["dv_Y_te"])))
            self.assertTrue(np.all(np.isnan(res["dv_Y_dM_te"])))
            self.assertEqual(res["profiles_file"], "profiles_summary_jun2026.npz")

    def test_batch_rescaled_summary_settings_validation_and_partial_overwrite(self):
        fake = {
            "mach_run": "08", "iv_run": "06", "gas_puff_ms": 23.0,
            "ref_bin_idx": 13, "ref_bin_t_ms": 1.6, "ref_iv_swp_idx": 14,
            "x_cm": np.array([-1.0, 0.0, 1.0]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "mach_rescaled.npz")
            with mock.patch.object(jml, "rescale_line_with_iv", return_value=fake) as m_rs:
                jml.batch_rescaled_summary(
                    data_dir=tmp, runs=["08", "09"], te_mode="upper", out_path=out_npz
                )
                self.assertEqual(m_rs.call_count, 2)

            # Matching settings skip cleanly
            with mock.patch.object(jml, "rescale_line_with_iv") as m_rs:
                jml.batch_rescaled_summary(
                    data_dir=tmp, runs=["08", "09"], te_mode="upper",
                    out_path=out_npz, overwrite=False,
                )
                m_rs.assert_not_called()

            # Mismatched te_mode raises ValueError advising overwrite=True
            with self.assertRaises(ValueError) as ctx:
                jml.batch_rescaled_summary(
                    data_dir=tmp, runs=["08"], te_mode="midpoint",
                    out_path=out_npz, overwrite=False,
                )
            self.assertIn("overwrite=True", str(ctx.exception))

            # Partial overwrite updates only run 08 and keeps run 09 untouched
            with mock.patch.object(jml, "rescale_line_with_iv", return_value=fake) as m_rs:
                jml.batch_rescaled_summary(
                    data_dir=tmp, runs=["08"], te_mode="midpoint",
                    out_path=out_npz, overwrite=True,
                )
                self.assertEqual(m_rs.call_count, 1)
            with np.load(out_npz) as d:
                self.assertEqual(str(d["08__te_mode"]), "midpoint")
                self.assertEqual(str(d["09__te_mode"]), "upper")


if __name__ == "__main__":
    unittest.main()
