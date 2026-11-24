"""Unit tests for plasma/profiles.py and Jun2026_profiles.py."""

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

from data_analysis.plasma.profiles import (
    E_CHARGE_J_PER_EV,
    electron_pressure,
    floating_potential_from_iv,
    local_drift_scales,
    radial_profile_gradients,
    sheath_factor,
)
import Jun2026_profiles as jpr


class TestPlasmaProfiles(unittest.TestCase):
    def test_electron_pressure_units_and_error_propagation(self):
        # 1e12 cm^-3 = 1e18 m^-3, Te = 5 eV -> pe = 1.602176634e-19 * 1e18 * 5 = 0.801088317 Pa
        pe, pe_err = electron_pressure(1e12, 5.0, ne_err_cm3=0.1e12, te_err_ev=0.5)
        expected = E_CHARGE_J_PER_EV * 1e18 * 5.0
        self.assertAlmostEqual(float(pe), expected, places=12)
        # Relative error = sqrt(0.1^2 + 0.1^2) = 0.1 * sqrt(2)
        self.assertAlmostEqual(float(pe_err / pe), 0.1 * np.sqrt(2.0), places=12)
        # Non-positive inputs -> NaN
        self.assertTrue(np.isnan(electron_pressure(-1.0, 5.0)))
        self.assertTrue(np.isnan(electron_pressure(1e12, 0.0)))

    def test_signed_gradients_exact_pressure_identity_and_radial_sign(self):
        # Symmetric hollow profile around x_center = +1.0 cm:
        # ne(r) = n0 * exp(+a * r), Te(r) = T0 * exp(-b * r), Vp(r) = V0 - E0 * r
        x = np.arange(-10.0, 12.1, 1.0)
        xc = 1.0
        r = np.abs(x - xc)
        a, b, E0 = 0.30, 0.50, 2.0  # cm^-1, cm^-1, V/cm
        ne = 1e11 * np.exp(a * r)
        te = 5.0 * np.exp(-b * r)
        vp = 25.0 - E0 * r
        ne_err = 0.05 * ne
        te_err = 0.05 * te
        vp_err = np.full_like(vp, 0.1)

        for sw in (1, 5):
            g = radial_profile_gradients(
                x, ne, te, vp, ne_err=ne_err, te_err=te_err, vp_err=vp_err,
                x_center_cm=xc, smooth_window=sw, polyorder=2,
            )
            # Exact identity: d(ln pe)/dr == d(ln ne)/dr + d(ln Te)/dr everywhere
            np.testing.assert_allclose(g.dln_pe_dx, g.dln_ne_dx + g.dln_te_dx, rtol=1e-13, atol=1e-13)
            # Away from cusp at x = xc and boundaries, BOTH left and right walls give
            # dln_ne/dr = +a, dln_te/dr = -b, dln_pe/dr = a - b = -0.20, Er_r = +E0
            away = (r >= 4.0) & (r <= 8.0)
            np.testing.assert_allclose(g.dln_ne_dx[away], a, atol=1e-10)
            np.testing.assert_allclose(g.dln_te_dx[away], -b, atol=1e-10)
            np.testing.assert_allclose(g.dln_pe_dx[away], a - b, atol=1e-10)
            np.testing.assert_allclose(g.eta_e[away], -b / a, atol=1e-10)
            np.testing.assert_allclose(g.er_v_per_cm[away], E0, atol=1e-10)
            self.assertTrue(np.all(np.isfinite(g.dln_pe_err[away])))
            self.assertTrue(np.all(g.dln_pe_err[away] > 0.0))

    def test_floating_potential_ascending_descending_and_sheath_factor(self):
        v_up = np.linspace(-40.0, 30.0, 1001)
        vf_true = 12.5
        te_true = 3.0
        # Exponential Langmuir model: I(V) = I_sat * (exp((V - Vf)/Te) - 1)
        i_up = 1.0 * (np.exp(np.clip((v_up - vf_true) / te_true, -20, 5)) - 1.0)
        vf_est_up = floating_potential_from_iv(v_up, i_up, smooth_samples=1)
        self.assertAlmostEqual(vf_est_up, vf_true, places=3)

        # Descending voltage sweep gives identical Vf
        vf_est_dn = floating_potential_from_iv(v_up[::-1], i_up[::-1], smooth_samples=1)
        self.assertAlmostEqual(vf_est_dn, vf_true, places=3)

        # Non-crossing current -> NaN
        self.assertTrue(np.isnan(floating_potential_from_iv(v_up, np.ones_like(v_up), smooth_samples=1)))

        # Sheath factor & uncertainty
        lam, lam_err = sheath_factor(23.0, 12.5, 3.0, vp_err=0.3, vf_err=0.4, te_err=0.15)
        self.assertAlmostEqual(float(lam), 3.5, places=12)
        expected_err = np.sqrt((0.3 ** 2 + 0.4 ** 2) / 9.0 + (3.5 * 0.15 / 3.0) ** 2)
        self.assertAlmostEqual(float(lam_err), expected_err, places=12)

    def test_local_drift_scales_physical_values(self):
        # He+ (mu=4), B0=800 G = 0.08 T, Te=4 eV, ne=1e12 cm^-3
        sc = local_drift_scales(
            1e12, 4.0, 800.0, mu=4.0,
            k_perp_cm=0.4, dln_ne_dr_cm=0.2, dln_te_dr_cm=-0.5, er_v_per_cm=1.6,
        )
        # cs = sqrt(e * 4 / (4 * mp)) ~ 9789 m/s; omega_ci = e*0.08/(4*mp) ~ 1.916e6 rad/s
        # rho_s = cs / omega_ci ~ 0.511 cm
        self.assertAlmostEqual(float(sc["cs_m_s"]), 9789.0, delta=30.0)
        self.assertAlmostEqual(float(sc["rho_s_cm"]), 0.511, delta=0.01)
        # v_ExB = (1.6 V/cm = 160 V/m) / 0.08 T = 2000 m/s
        self.assertAlmostEqual(float(sc["v_ExB_m_s"]), 2000.0, places=6)
        # omega_star_p = omega_star_n + omega_star_T
        self.assertAlmostEqual(
            float(sc["omega_star_p_rad_s"]),
            float(sc["omega_star_n_rad_s"] + sc["omega_star_T_rad_s"]),
            places=8,
        )
        self.assertGreater(float(sc["nu_ei_s"]), 1e6)


class TestJun2026ProfilesAdapter(unittest.TestCase):
    def test_classify_sweeps_and_reference_selection(self):
        t0 = np.array([-1.0, 0.02, 1.02, 1.52, 2.02, 5.02])
        t1 = t0 + 0.16
        t_mid = 0.5 * (t0 + t1)
        # Onset IQR q25=1.60, med=1.65 (like Run 02): 1.02..1.18 is pre_onset,
        # 1.52..1.68 straddles, 2.02..2.18 is post_onset
        reg = jpr.classify_sweeps(t0, t1, t_onset_q25_ms=1.60, t_onset_med_ms=1.65)
        self.assertEqual(reg, ["pre_bias", "pre_onset", "pre_onset", "straddling", "post_onset", "post_bias"])
        idx, lbl = jpr.select_reference_pre_onset_sweep(t_mid, reg)
        self.assertEqual((idx, lbl), (2, "pre_onset"))

        # Stable run (t_onset = NaN): selects sweep closest to prefer_t_ms=2.1
        reg_stbl = jpr.classify_sweeps(t0, t1, np.nan, np.nan)
        self.assertEqual(reg_stbl, ["pre_bias", "pre_onset", "pre_onset", "pre_onset", "pre_onset", "post_bias"])
        idx_s, lbl_s = jpr.select_reference_pre_onset_sweep(t_mid, reg_stbl, prefer_t_ms=2.1)
        self.assertEqual((idx_s, lbl_s), (4, "pre_onset"))

    def test_batch_all_per_run_settings_and_partial_overwrite(self):
        fake = {
            "tip": "P29-R", "gas_puff_ms": 24.0, "b0_g": 800.0, "calibrated": True,
            "ref_sweep_idx": 2, "ref_sweep_t_ms": 2.1, "ref_sweep_regime": "pre_onset",
            "x_cm": np.array([0.0, 1.0, 2.0]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "profiles.npz")
            with mock.patch.object(jpr, "analyze_run", return_value=fake) as m_ar:
                jpr.batch_all(data_dir=tmp, runs=["00", "01"], smooth_window=5, out_path=out_npz)
                self.assertEqual(m_ar.call_count, 2)

            # Matching settings skip cleanly
            with mock.patch.object(jpr, "analyze_run") as m_ar:
                jpr.batch_all(data_dir=tmp, runs=["00", "01"], smooth_window=5, out_path=out_npz, overwrite=False)
                m_ar.assert_not_called()

            # Mismatch raises ValueError advising overwrite=True
            with self.assertRaises(ValueError) as ctx:
                jpr.batch_all(data_dir=tmp, runs=["00"], smooth_window=7, out_path=out_npz, overwrite=False)
            self.assertIn("overwrite=True", str(ctx.exception))

            # Partial overwrite updates only run 00 and keeps run 01 untouched
            with mock.patch.object(jpr, "analyze_run", return_value=fake) as m_ar:
                jpr.batch_all(data_dir=tmp, runs=["00"], smooth_window=7, out_path=out_npz, overwrite=True)
                self.assertEqual(m_ar.call_count, 1)
            with np.load(out_npz) as d:
                self.assertEqual(int(d["00__smooth_window"]), 7)
                self.assertEqual(int(d["01__smooth_window"]), 5)


if __name__ == "__main__":
    unittest.main()
