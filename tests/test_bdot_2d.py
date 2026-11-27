"""Unit tests for 2-D field divergence, B_L/B_R polarization, reference-locked phasors, and Jun2026_bdot_2d."""

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

from data_analysis.plasma.bdot import BdotCalibration
from data_analysis.plasma.fields import (
    circular_polarization,
    curl,
    divergence_2d,
    reference_locked_phasor,
)
import Jun2026_bdot_2d as jb2


class TestElectromagnetic2DHelpers(unittest.TestCase):
    def test_circular_polarization_handedness_and_parseval(self):
        # Pure left-hand (CW around +z, ion-cyclotron sense):
        # Bx(t) = cos(wt), By(t) = -sin(wt)
        # In exp(-iwt): Bx = 1, By = -i -> B_L = sqrt(2), B_R = 0, ellip = -1.
        # In exp(+iwt): Bx = 1, By = +i -> same B_L = sqrt(2), B_R = 0, ellip = -1.
        bL_m, bR_m, el_m = circular_polarization(1.0 + 0j, -1j, convention="exp_minus_iwt", b_sign_z=+1)
        bL_p, bR_p, el_p = circular_polarization(1.0 + 0j, +1j, convention="exp_plus_iwt", b_sign_z=+1)
        self.assertAlmostEqual(abs(bL_m), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(abs(bR_m), 0.0, places=12)
        self.assertAlmostEqual(float(el_m), -1.0, places=12)
        self.assertAlmostEqual(abs(bL_p), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(abs(bR_p), 0.0, places=12)
        self.assertAlmostEqual(float(el_p), -1.0, places=12)

        # Reversing B0 (b_sign_z = -1) swaps B_L <-> B_R and flips ellipticity to +1
        bL_rev, bR_rev, el_rev = circular_polarization(1.0 + 0j, -1j, convention="exp_minus_iwt", b_sign_z=-1)
        self.assertAlmostEqual(abs(bL_rev), 0.0, places=12)
        self.assertAlmostEqual(abs(bR_rev), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(float(el_rev), +1.0, places=12)

        # Parseval identity on arbitrary complex arrays
        rng = np.random.default_rng(7)
        bx = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        by = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        bL, bR, _ = circular_polarization(bx, by)
        np.testing.assert_allclose(np.abs(bL) ** 2 + np.abs(bR) ** 2, np.abs(bx) ** 2 + np.abs(by) ** 2)

    def test_reference_locked_phasor_recovers_amplitude_and_phase_with_random_shot_phase(self):
        rng = np.random.default_rng(12)
        dt = 1e-5
        t = np.arange(512) * dt
        f0 = 4000.0
        nshots = 40
        phi_rand = rng.uniform(0.0, 2.0 * np.pi, size=(nshots, 1))
        dphi_true = 1.2  # rad (signal leads reference by +1.2 rad in exp(i*(phi - wt)))
        amp_true = 2.5
        ref = 1.3 * np.cos(phi_rand - 2.0 * np.pi * f0 * t[None, :])
        sig = amp_true * np.cos(phi_rand + dphi_true - 2.0 * np.pi * f0 * t[None, :])

        xc, g2 = reference_locked_phasor(sig, ref, dt, f0, convention="exp_minus_iwt")
        self.assertAlmostEqual(abs(xc), amp_true, delta=1e-2)
        self.assertAlmostEqual(float(np.angle(xc)), dphi_true, delta=1e-2)
        self.assertAlmostEqual(float(g2), 1.0, delta=1e-6)

    def test_rotating_current_filament_analytic_curl_div_and_polarization_flip(self):
        # Exact Biot-Savart vector potential of a localized CW (m_J = -1) Gaussian
        # field-aligned current ring J_z(r, theta) = (r/a^3) * exp(-r^2/(2a^2)) * exp(i*(-theta - wt)):
        #   A_z(r, theta) = (a / r) * (1 - exp(-r^2/(2a^2))) * exp(-i*theta)
        # Interior (r << a): A_z ~ (r / 2a) * exp(-i*theta) -> pure LHCP (ellip -> -1).
        # Exterior (r >> a): A_z ~ (a / r) * exp(-i*theta) -> pure RHCP (ellip -> +1).
        # Everywhere: div(B_perp) == 0 identically and curl(B)_z = -nabla^2 A_z = mu_0 J_z.
        x_cm = np.linspace(-8.0, 8.0, 33)
        y_cm = np.linspace(-8.0, 8.0, 33)
        XX, YY = np.meshgrid(x_cm * 1e-2, y_cm * 1e-2)
        RR = np.hypot(XX, YY)
        TH = np.arctan2(YY, XX)
        a = 0.020  # 2.0 cm current-ring peak radius
        with np.errstate(divide="ignore", invalid="ignore"):
            radial = np.where(RR > 0.0, (a / RR) * (1.0 - np.exp(-0.5 * (RR / a) ** 2)), 0.0)
        Az = radial * np.exp(-1j * TH)
        dx = float(x_cm[1] - x_cm[0]) * 1e-2
        Bx = np.gradient(Az, dx, axis=-2, edge_order=2)
        By = -np.gradient(Az, dx, axis=-1, edge_order=2)
        Bz = np.zeros_like(Bx)
        Jz_true = curl(Bx, By, 0, dx, dx, None, axes=(-1, -2, -3))[2]
        LP = -1.0 * Jz_true  # exact 180 deg antiphase dummy LP
        g2_ones = np.ones_like(RR)

        res = jb2.analyze_harmonic_fields(
            x_cm, y_cm, Bx, By, Bz, LP, g2_ones, g2_ones, g2_ones, g2_ones,
            calibrated=False, b_sign_z=+1,
        )
        # Maxwell div/curl check: as-wired << flipped
        self.assertLess(res["div_curl_ratio"], 0.05)
        self.assertGreater(res["div_curl_ratio_flipped"], 0.8)
        self.assertEqual(res["by_sign_used"], +1)
        self.assertTrue(res["coil_handedness_ok"])
        self.assertFalse(res["abs_coil_polarity_verified"])
        # Mode m_J = -1 with near-unit coherence-weighted purity
        self.assertEqual(res["m_J"], -1)
        self.assertGreater(res["m_J_purity"], 0.98)
        # Core is LHCP (< -0.4), outer Biot-Savart exterior is RHCP (> +0.8)
        self.assertLess(res["core_ellip_med"], -0.4)
        self.assertGreater(res["outer_ellip_med"], +0.8)
        # Antiphase J_z vs LP locking
        self.assertAlmostEqual(abs(res["dphi_J_minus_LP_deg"]), 180.0, delta=1e-6)
        self.assertAlmostEqual(res["R_J_LP"], 1.0, delta=1e-6)

        # Inverted By: default on_polarity_mismatch='raise' must fail fast
        with self.assertRaises(ValueError):
            jb2.analyze_harmonic_fields(
                x_cm, y_cm, Bx, -By, Bz, LP, g2_ones, g2_ones, g2_ones, g2_ones,
                calibrated=False, b_sign_z=+1,
            )

        # Inverted By with on_polarity_mismatch='flip' corrects By and recovers m_J = -1
        res_flip = jb2.analyze_harmonic_fields(
            x_cm, y_cm, Bx, -By, Bz, LP, g2_ones, g2_ones, g2_ones, g2_ones,
            calibrated=False, b_sign_z=+1, on_polarity_mismatch="flip",
        )
        self.assertEqual(res_flip["by_sign_used"], -1)
        self.assertTrue(res_flip["coil_handedness_ok"])
        self.assertTrue(res_flip["polarity_auto_flipped"])
        self.assertEqual(res_flip["m_J"], -1)
        self.assertLess(res_flip["div_curl_ratio"], 0.05)
        self.assertAlmostEqual(res_flip["core_ellip_med"], res["core_ellip_med"], places=12)
        self.assertAlmostEqual(res_flip["outer_ellip_med"], res["outer_ellip_med"], places=12)

        # Common-mode (-Bx, -By) inversion leaves div/curl, m_J, and ellipticity
        # invariant while shifting dphi(J - LP) by 180 deg (mod-180 caveat)
        res_cm = jb2.analyze_harmonic_fields(
            x_cm, y_cm, -Bx, -By, Bz, LP, g2_ones, g2_ones, g2_ones, g2_ones,
            calibrated=False, b_sign_z=+1,
        )
        self.assertEqual(res_cm["m_J"], -1)
        self.assertAlmostEqual(res_cm["div_curl_ratio"], res["div_curl_ratio"], places=12)
        self.assertAlmostEqual(res_cm["outer_ellip_med"], res["outer_ellip_med"], places=12)
        self.assertAlmostEqual(res_cm["dphi_J_minus_LP_deg"], 0.0, delta=1e-6)

    def test_batch_all_incremental_settings_and_partial_overwrite(self):
        fake = {
            "b0_g": 800.0, "f0_hz": 4110.0,
            "x_cm": np.array([-1.0, 0.0, 1.0]), "y_cm": np.array([-1.0, 0.0, 1.0]),
            "t0_offset_us": 0.0, "bdot_calibrated": False, "bdot_gain": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "bdot_2d.npz")
            with mock.patch.object(jb2, "analyze_run", return_value=fake) as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26", "31"], out_path=out_npz, bdot_cal=None)
                self.assertEqual(m_ar.call_count, 2)

            # Matching settings skip cleanly
            with mock.patch.object(jb2, "analyze_run") as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26", "31"], out_path=out_npz, bdot_cal=None, overwrite=False)
                m_ar.assert_not_called()

            # Mismatched tmin_ms raises ValueError
            with self.assertRaises(ValueError):
                jb2.batch_all(data_dir=tmp, runs=["26"], tmin_ms=1.5, out_path=out_npz, bdot_cal=None, overwrite=False)

            # Partial overwrite refreshes only Run 26 while preserving Run 31
            with mock.patch.object(jb2, "analyze_run", return_value=fake) as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26"], tmin_ms=1.5, out_path=out_npz, bdot_cal=None, overwrite=True)
                self.assertEqual(m_ar.call_count, 1)
            with np.load(out_npz) as d:
                self.assertAlmostEqual(float(d["26__tmin_ms"]), 1.5)
                self.assertAlmostEqual(float(d["31__tmin_ms"]), jb2.DEFAULT_TMIN_MS)

    def test_apply_bdot_calibration_3x3_real_level_and_channel_signs(self):
        A = np.array([
            [1.15e-4, 0.0, 1.25e-5],
            [1.5e-6, 9.8e-5, 9.4e-7],
            [2.3e-6, 1.01e-5, 8.7e-5],
        ])
        cal = BdotCalibration(
            probe_id="C16",
            area_matrix_m2=A,
            r2_matrix=np.ones((3, 3)),
            phase_deg_matrix=np.full((3, 3), 90.0),
            phase_R_matrix=np.ones((3, 3)),
            diag_band_hz=(150e3, 500e3),
            offdiag_band_hz=(250e3, 1000e3),
        )
        f0 = 4110.0
        w0 = 2.0 * np.pi * f0
        B_true = np.array([
            [[3e-5 + 1e-5j, -1e-5 + 2e-5j]],
            [[-2e-5 + 3e-5j, 4e-5 - 1e-5j]],
            [[4e-6 - 2e-6j, 1e-6 + 3e-6j]],
        ])  # (3, 1, 2)
        gain = 1000.0
        atten_db = 6.0
        eff_gain = gain * 10.0 ** (-atten_db / 20.0)
        ch_signs = (+1, +1, -1)
        # V_coil = -i * w * A @ B; scope sees channel_signs * eff_gain * V_coil
        V_coil = -1j * w0 * np.einsum("ij,j...->i...", A, B_true)
        V_scope = np.asarray(ch_signs)[:, None, None] * eff_gain * V_coil

        Bx, By, Bz, cal_flag = jb2._apply_bdot_calibration(
            V_scope[0], V_scope[1], V_scope[2], f0,
            bdot_cal=cal, bdot_gain=gain, bdot_atten_db=atten_db,
            channel_signs=ch_signs,
        )
        self.assertTrue(cal_flag)
        np.testing.assert_allclose(np.stack([Bx, By, Bz], axis=0), B_true, rtol=1e-12)

    def test_batch_all_calibrated_ndarray_settings_mismatch(self):
        A1 = np.diag([1.1e-4, 1.0e-4, 0.9e-4])
        A2 = np.diag([1.2e-4, 1.0e-4, 0.9e-4])
        fake = {
            "b0_g": 800.0, "f0_hz": 4110.0,
            "x_cm": np.array([-1.0, 0.0, 1.0]), "y_cm": np.array([-1.0, 0.0, 1.0]),
            "t0_offset_us": 0.0, "bdot_calibrated": True, "bdot_gain": 1000.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "bdot_2d.npz")
            with mock.patch.object(jb2, "analyze_run", return_value=fake):
                jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A1, bdot_gain=1000.0)
            # Same 3x3 matrix skips cleanly
            with mock.patch.object(jb2, "analyze_run") as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A1, bdot_gain=1000.0)
                m_ar.assert_not_called()
            # Changed 3x3 matrix fails fast unless overwrite=True
            with self.assertRaises(ValueError):
                jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A2, bdot_gain=1000.0)

    def test_resolve_bdot_gain_auto_from_hdf5_and_fail_fast(self):
        chain = {"header": "P28: Bdot-10T-C16", "chain": "X10 => X100",
                 "gain": 1000.0, "atten_db": 0.0, "effective_gain": 1000.0}
        # Uncalibrated defaults to (1.0, 0.0) even without a recorded chain
        self.assertEqual(
            jb2._resolve_bdot_gain(None, calibrated=False, bdot_gain=None, bdot_atten_db=None, run_num="26"),
            (1.0, 0.0),
        )
        # Calibrated with bdot_gain=None auto-reads recorded chain
        self.assertEqual(
            jb2._resolve_bdot_gain(chain, calibrated=True, bdot_gain=None, bdot_atten_db=None, run_num="26"),
            (1000.0, 0.0),
        )
        # Calibrated with bdot_gain=None and missing chain raises instead of guessing 1.0
        with self.assertRaises(ValueError):
            jb2._resolve_bdot_gain(None, calibrated=True, bdot_gain=None, bdot_atten_db=None, run_num="26")
        # Explicit gain disagreeing with HDF5 emits UserWarning
        with self.assertWarns(UserWarning):
            self.assertEqual(
                jb2._resolve_bdot_gain(chain, calibrated=True, bdot_gain=10.0, bdot_atten_db=None, run_num="26"),
                (10.0, 0.0),
            )

    def test_read_bdot_chain_propagates_ambiguity_and_batch_auto_gain_mismatch(self):
        # Ambiguous multiple active P28 B-dot entries must propagate ValueError
        ambig_desc = mock.Mock(
            raw="- P28: Bdot-10T-C16\n  - Connection: X10\n- P28: Bdot-5T-B2\n  - Connection: X100\n"
        )
        run_ambig = mock.Mock()
        run_ambig.description.return_value = ambig_desc
        with self.assertRaises(ValueError):
            jb2._read_bdot_chain(run_ambig)

        # Unreadable run description still degrades cleanly to None
        run_err = mock.Mock()
        run_err.description.side_effect = KeyError("no description")
        self.assertIsNone(jb2._read_bdot_chain(run_err))

        # batch_all(..., bdot_gain=None) must flag stale stored gain vs HDF5 chain
        A1 = np.diag([1.1e-4, 1.0e-4, 0.9e-4])
        fake_10 = {
            "b0_g": 800.0, "f0_hz": 4110.0,
            "x_cm": np.array([-1.0, 0.0, 1.0]), "y_cm": np.array([-1.0, 0.0, 1.0]),
            "t0_offset_us": 0.0, "bdot_calibrated": True, "bdot_gain": 10.0, "bdot_atten_db": 0.0,
        }
        fake_1000 = dict(fake_10, bdot_gain=1000.0)
        run_1000 = mock.Mock()
        run_1000.description.return_value = mock.Mock(
            raw="- P28: Bdot-10T-C16\n  - Connection: 2.5BLP => X10 => 2.5BLP => X100\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "26-dummy.hdf5"), "wb").close()
            out_npz = os.path.join(tmp, "bdot_2d.npz")
            with mock.patch.object(jb2, "analyze_run", return_value=fake_10):
                jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A1, bdot_gain=10.0)
            with mock.patch.object(jb2, "open_lapd", return_value=run_1000):
                with self.assertRaises(ValueError):
                    jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A1, bdot_gain=None)

            # Matching stored gain (1000.0) with bdot_gain=None skips cleanly
            with mock.patch.object(jb2, "analyze_run", return_value=fake_1000):
                jb2.batch_all(
                    data_dir=tmp, runs=["26"], out_path=out_npz,
                    bdot_area_m2=A1, bdot_gain=1000.0, overwrite=True,
                )
            with mock.patch.object(jb2, "open_lapd", return_value=run_1000), \
                 mock.patch.object(jb2, "analyze_run") as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26"], out_path=out_npz, bdot_area_m2=A1, bdot_gain=None)
                m_ar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
