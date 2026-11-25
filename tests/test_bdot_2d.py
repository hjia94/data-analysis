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
        # Mode m_J = -1 with near-unit purity
        self.assertEqual(res["m_J"], -1)
        self.assertGreater(res["m_J_purity"], 0.98)
        # Core is LHCP (< -0.4), outer Biot-Savart exterior is RHCP (> +0.8)
        self.assertLess(res["core_ellip_med"], -0.4)
        self.assertGreater(res["outer_ellip_med"], +0.8)
        # Antiphase J_z vs LP locking
        self.assertAlmostEqual(abs(res["dphi_J_minus_LP_deg"]), 180.0, delta=1e-6)
        self.assertAlmostEqual(res["R_J_LP"], 1.0, delta=1e-6)

    def test_batch_all_incremental_settings_and_partial_overwrite(self):
        fake = {
            "b0_g": 800.0, "f0_hz": 4110.0,
            "x_cm": np.array([-1.0, 0.0, 1.0]), "y_cm": np.array([-1.0, 0.0, 1.0]),
            "t0_offset_us": 0.0, "bdot_calibrated": False, "bdot_gain": 1.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "bdot_2d.npz")
            with mock.patch.object(jb2, "analyze_run", return_value=fake) as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26", "31"], out_path=out_npz)
                self.assertEqual(m_ar.call_count, 2)

            # Matching settings skip cleanly
            with mock.patch.object(jb2, "analyze_run") as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26", "31"], out_path=out_npz, overwrite=False)
                m_ar.assert_not_called()

            # Mismatched tmin_ms raises ValueError
            with self.assertRaises(ValueError):
                jb2.batch_all(data_dir=tmp, runs=["26"], tmin_ms=1.5, out_path=out_npz, overwrite=False)

            # Partial overwrite refreshes only Run 26 while preserving Run 31
            with mock.patch.object(jb2, "analyze_run", return_value=fake) as m_ar:
                jb2.batch_all(data_dir=tmp, runs=["26"], tmin_ms=1.5, out_path=out_npz, overwrite=True)
                self.assertEqual(m_ar.call_count, 1)
            with np.load(out_npz) as d:
                self.assertAlmostEqual(float(d["26__tmin_ms"]), 1.5)
                self.assertAlmostEqual(float(d["31__tmin_ms"]), jb2.DEFAULT_TMIN_MS)


if __name__ == "__main__":
    unittest.main()
