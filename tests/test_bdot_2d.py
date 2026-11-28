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
            calibrated=False, b_sign_z=+1, abs_coil_polarity_verified=False,
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

    def test_apply_bdot_time_calibration_3x3_and_channel_signs(self):
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
        rng = np.random.default_rng(42)
        bdot_true = rng.normal(scale=10.0, size=(3, 4, 5, 16))
        gain = 1000.0
        atten_db = 6.0
        eff_gain = gain * 10.0 ** (-atten_db / 20.0)
        signs = (-1, -1, +1)
        v_coil = np.einsum("ij,j...->i...", A, bdot_true)
        v_scope = np.asarray(signs, dtype=float)[:, None, None, None] * eff_gain * v_coil

        bdot_out, cal_flag = jb2._apply_bdot_time_calibration(
            v_scope, bdot_cal=cal, bdot_gain=gain, bdot_atten_db=atten_db, channel_signs=signs,
        )
        self.assertTrue(cal_flag)
        np.testing.assert_allclose(bdot_out, bdot_true, rtol=1e-12)

        # Uncalibrated returns probe-level v_coil with calibrated=False
        v_uncal, cal_flag_u = jb2._apply_bdot_time_calibration(
            v_scope, bdot_cal=None, bdot_gain=gain, bdot_atten_db=atten_db, channel_signs=signs,
        )
        self.assertFalse(cal_flag_u)
        np.testing.assert_allclose(v_uncal, v_coil, rtol=1e-12)

    def test_analyze_dc_bias_steps_analytic_current_diamagnetic_and_polarity_checks(self):
        from scipy.constants import mu_0

        x_cm = np.linspace(-8.0, 8.0, 17)
        y_cm = np.linspace(-8.0, 8.0, 17)
        XX, YY = np.meshgrid(x_cm * 1e-2, y_cm * 1e-2)
        RR = np.hypot(XX, YY)

        # Analytic axisymmetric Gaussian current column:
        #   J_z(r) = (I_tot / (2*pi*sigma^2)) * exp(-r^2 / (2*sigma^2))
        #   B_theta(r) = (mu_0 * I_tot / (2*pi*r)) * (1 - exp(-r^2 / (2*sigma^2)))
        I_tot_on = -25.0   # A (electrons into +z Port 35 => conventional J_z < 0)
        I_tot_off = +22.0  # A (bias turn-off step)
        sigma = 0.025      # 2.5 cm column radius (well inside 15x15 cm interior)
        with np.errstate(divide="ignore", invalid="ignore"):
            bth_unit = np.where(
                RR > 0.0,
                (mu_0 / (2.0 * np.pi * RR)) * (1.0 - np.exp(-0.5 * (RR / sigma) ** 2)),
                0.0,
            )
            bx_unit = np.where(RR > 0.0, -bth_unit * (YY / RR), 0.0)
            by_unit = np.where(RR > 0.0, +bth_unit * (XX / RR), 0.0)

        # Diamagnetic core hollowing dB_z > 0 at turn-on, < 0 at turn-off
        bz_core_on_T = +0.28e-4
        bz_core_off_T = -0.28e-4
        bz_profile = np.exp(-0.5 * (RR / 0.045) ** 2)

        # Smooth step pulses in dB/dt at t = 0.05 ms (width 20 us) and t = 5.05 ms (width 20 us)
        dt_s = 1e-6  # 1 us
        t_s = np.arange(-0.5e-3, 6.6e-3, dt_s)
        tau = 20e-6
        g_on = np.exp(-0.5 * ((t_s - 0.05e-3) / tau) ** 2)
        g_on /= np.trapezoid(g_on, dx=dt_s)
        g_off = np.exp(-0.5 * ((t_s - 5.05e-3) / tau) ** 2)
        g_off /= np.trapezoid(g_off, dx=dt_s)

        # Add nonzero DC amplifier offsets on all 3 channels to verify baseline subtraction
        offsets = np.array([+0.7, -0.5, -7.3])[:, None, None, None]  # T/s
        bdot = np.zeros((3, y_cm.size, x_cm.size, t_s.size), dtype=float)
        bdot[0] = (I_tot_on * bx_unit)[..., None] * g_on + (I_tot_off * bx_unit)[..., None] * g_off
        bdot[1] = (I_tot_on * by_unit)[..., None] * g_on + (I_tot_off * by_unit)[..., None] * g_off
        bdot[2] = (bz_core_on_T * bz_profile)[..., None] * g_on + (bz_core_off_T * bz_profile)[..., None] * g_off
        bdot += offsets

        res = jb2.analyze_dc_bias_steps(x_cm, y_cm, t_s, bdot, calibrated=True)

        self.assertEqual(res["dB_on"].shape, (3, 17, 17))
        self.assertEqual(res["dB_off"].shape, (3, 17, 17))
        self.assertEqual(res["Jz_on"].shape, (17, 17))
        self.assertEqual(res["Jz_off"].shape, (17, 17))
        # Maxwell div/curl << flipped
        self.assertLess(res["div_curl_on"], 0.02)
        self.assertGreater(res["div_curl_on_flipped"], 1.0)
        self.assertLess(res["div_curl_off"], 0.02)
        self.assertGreater(res["div_curl_off_flipped"], 1.0)
        # Integrated currents & Stokes check recover injected I_tot accurately
        self.assertAlmostEqual(res["I_int_on"], I_tot_on, delta=0.5)
        self.assertAlmostEqual(res["I_int_off"], I_tot_off, delta=0.5)
        self.assertAlmostEqual(res["I_stokes_on"], I_tot_on, delta=0.5)
        self.assertAlmostEqual(res["I_stokes_off"], I_tot_off, delta=0.5)
        # Disjoint concentric 2"/4"/6" sums stay close to I_int
        self.assertAlmostEqual(
            res["I_2in_on"] + res["I_4in_on"] + res["I_6in_on"], res["I_int_on"], delta=0.25,
        )
        # Core & wall diamagnetic dB_z signs and magnitudes
        self.assertGreater(res["dBz_core_on"], 0.2e-4)
        self.assertLess(res["dBz_core_off"], -0.2e-4)
        self.assertGreater(res["dBz_wall_on"], 0.0)
        self.assertLess(res["dBz_wall_off"], 0.0)
        # All 4 polarity consistency criteria pass
        self.assertTrue(res["polarity_i_on_ok"])
        self.assertTrue(res["polarity_i_off_ok"])
        self.assertTrue(res["polarity_dbz_on_ok"])
        self.assertTrue(res["polarity_dbz_off_ok"])
        self.assertTrue(res["polarity_consistent"])
        # Sensitivity array matches primary windows at 0.40 ms (on) and 0.25 ms (off)
        np.testing.assert_allclose(res["sens_dur_ms"], [0.25, 0.40])
        self.assertAlmostEqual(res["sens_I_int_on"][1], res["I_int_on"], places=10)
        self.assertAlmostEqual(res["sens_I_int_off"][0], res["I_int_off"], places=10)

        # Common-mode (-Bx, -By) inversion must fail I_int polarity checks
        bdot_xy_flip = bdot.copy()
        bdot_xy_flip[0:2] *= -1.0
        res_xy = jb2.analyze_dc_bias_steps(x_cm, y_cm, t_s, bdot_xy_flip, calibrated=True)
        self.assertFalse(res_xy["polarity_i_on_ok"])
        self.assertFalse(res_xy["polarity_i_off_ok"])
        self.assertTrue(res_xy["polarity_dbz_on_ok"])
        self.assertTrue(res_xy["polarity_dbz_off_ok"])
        self.assertFalse(res_xy["polarity_consistent"])

        # Z-coil inversion must fail dBz polarity checks (and reversing b_sign_z flips expectation)
        bdot_z_flip = bdot.copy()
        bdot_z_flip[2] *= -1.0
        res_z = jb2.analyze_dc_bias_steps(x_cm, y_cm, t_s, bdot_z_flip, calibrated=True)
        self.assertTrue(res_z["polarity_i_on_ok"])
        self.assertTrue(res_z["polarity_i_off_ok"])
        self.assertFalse(res_z["polarity_dbz_on_ok"])
        self.assertFalse(res_z["polarity_dbz_off_ok"])
        self.assertFalse(res_z["polarity_consistent"])
        self.assertFalse(res_z["polarity_dbz_sens_consistent"])
        self.assertTrue(res["polarity_dbz_sens_consistent"])
        res_z_rev = jb2.analyze_dc_bias_steps(x_cm, y_cm, t_s, bdot_z_flip, calibrated=True, b_sign_z=-1)
        self.assertTrue(res_z_rev["polarity_dbz_on_ok"])
        self.assertTrue(res_z_rev["polarity_dbz_off_ok"])
        self.assertTrue(res_z_rev["polarity_consistent"])
        self.assertTrue(res_z_rev["polarity_dbz_sens_consistent"])

    def test_batch_dc_bias_steps_merges_and_preserves_harmonic_keys(self):
        fake_dc = {
            "on_win_ms": np.array([-0.02, 0.40]),
            "on_base_ms": np.array([-0.40, -0.05]),
            "off_win_ms": np.array([4.98, 5.25]),
            "off_base_ms": np.array([5.50, 6.50]),
            "dB_on": np.zeros((3, 3, 3)),
            "dB_off": np.zeros((3, 3, 3)),
            "Jz_on": np.zeros((3, 3)),
            "Jz_off": np.zeros((3, 3)),
            "div_curl_on": 0.15,
            "div_curl_on_flipped": 1.30,
            "div_curl_off": 0.19,
            "div_curl_off_flipped": 1.25,
            "I_int_on": -25.7,
            "I_int_off": +22.0,
            "I_2in_on": -7.8,
            "I_2in_off": +6.9,
            "I_4in_on": -11.3,
            "I_4in_off": +10.1,
            "I_6in_on": -6.4,
            "I_6in_off": +4.9,
            "I_stokes_on": -25.8,
            "I_stokes_off": +22.0,
            "dBz_core_on": +0.27e-4,
            "dBz_core_off": -0.30e-4,
            "dBz_wall_on": +0.27e-4,
            "dBz_wall_off": -0.30e-4,
            "polarity_consistent": True,
            "n_shots_per_pos": 5,
            "sens_dur_ms": np.array([0.25, 0.40]),
            "sens_I_int_on": np.array([-25.2, -25.7]),
            "sens_I_int_off": np.array([+22.0, +22.1]),
            "sens_div_curl_on": np.array([0.125, 0.156]),
            "sens_div_curl_off": np.array([0.199, 0.219]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "bdot_2d.npz")
            # Seed with pre-existing harmonic keys
            np.savez(
                out_npz,
                **{
                    "26__f0_hz": np.float64(4110.0),
                    "26__h1__m_J": np.int64(-1),
                    "31__f0_hz": np.float64(3529.0),
                    "31__h1__m_J": np.int64(-1),
                },
            )
            with mock.patch.object(jb2, "analyze_run_dc_bias_steps", return_value=fake_dc) as m_dc:
                jb2.batch_dc_bias_steps(
                    data_dir=tmp, runs=["26", "31"], out_path=out_npz, bdot_cal=None,
                )
                self.assertEqual(m_dc.call_count, 2)

            with np.load(out_npz) as d:
                # Existing harmonic keys preserved intact
                self.assertAlmostEqual(float(d["26__f0_hz"]), 4110.0)
                self.assertEqual(int(d["26__h1__m_J"]), -1)
                self.assertAlmostEqual(float(d["31__f0_hz"]), 3529.0)
                # New dc__* keys merged cleanly
                self.assertAlmostEqual(float(d["26__dc__I_int_on"]), -25.7)
                self.assertAlmostEqual(float(d["31__dc__I_int_off"]), +22.0)
                self.assertTrue(bool(d["26__dc__polarity_consistent"]))

            # Second call with overwrite=False skips cleanly
            with mock.patch.object(jb2, "analyze_run_dc_bias_steps") as m_dc:
                jb2.batch_dc_bias_steps(
                    data_dir=tmp, runs=["26", "31"], out_path=out_npz, bdot_cal=None, overwrite=False,
                )
                m_dc.assert_not_called()

            # Mismatched channel_signs or by_sign with overwrite=False must raise
            with self.assertRaises(ValueError):
                jb2.batch_dc_bias_steps(
                    data_dir=tmp, runs=["26"], out_path=out_npz,
                    bdot_cal=None, channel_signs=(+1, +1, +1), overwrite=False,
                )
            with self.assertRaises(ValueError):
                jb2.batch_dc_bias_steps(
                    data_dir=tmp, runs=["26"], out_path=out_npz,
                    bdot_cal=None, by_sign=-1, overwrite=False,
                )

            # Partial overwrite refreshes only Run 26 dc__* while preserving Run 31 dc__* and all h1 keys
            fake_dc_v2 = dict(fake_dc, I_int_on=-26.1)
            with mock.patch.object(jb2, "analyze_run_dc_bias_steps", return_value=fake_dc_v2) as m_dc:
                jb2.batch_dc_bias_steps(
                    data_dir=tmp, runs=["26"], out_path=out_npz, bdot_cal=None, overwrite=True,
                )
                self.assertEqual(m_dc.call_count, 1)
            with np.load(out_npz) as d:
                self.assertAlmostEqual(float(d["26__dc__I_int_on"]), -26.1)
                self.assertAlmostEqual(float(d["31__dc__I_int_on"]), -25.7)
                self.assertAlmostEqual(float(d["26__f0_hz"]), 4110.0)


if __name__ == "__main__":
    unittest.main()
