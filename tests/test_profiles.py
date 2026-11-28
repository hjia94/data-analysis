"""Unit tests for data_analysis.plasma.profiles physics & error propagation."""

import unittest

import numpy as np

from data_analysis.plasma.profiles import (
    E_CHARGE_J_PER_EV,
    HollowProfileMetrics,
    electron_pressure,
    floating_potential_from_iv,
    hollow_profile_metrics,
    local_drift_scales,
    orbit_smeared_profile,
    radial_profile_gradients,
    sheath_factor,
)


class TestPlasmaProfiles(unittest.TestCase):
    def test_electron_pressure_units_and_error_propagation(self):
        pe, pe_err = electron_pressure(1e12, 5.0, ne_err_cm3=0.1e12, te_err_ev=0.5)
        expected = E_CHARGE_J_PER_EV * 1e18 * 5.0
        self.assertAlmostEqual(float(pe), expected, places=12)
        self.assertAlmostEqual(float(pe_err / pe), 0.1 * np.sqrt(2.0), places=12)
        self.assertTrue(np.isnan(electron_pressure(-1.0, 5.0)))
        self.assertTrue(np.isnan(electron_pressure(1e12, 0.0)))

    def test_signed_gradients_exact_pressure_identity_and_radial_sign(self):
        x = np.arange(-10.0, 12.1, 1.0)
        xc = 1.0
        r = np.abs(x - xc)
        a, b, E0 = 0.30, 0.50, 2.0
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
            np.testing.assert_allclose(g.dln_pe_dx, g.dln_ne_dx + g.dln_te_dx, rtol=1e-13, atol=1e-13)
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
        vf_true, te_true = 12.5, 3.0
        i_up = 1.0 * (np.exp(np.clip((v_up - vf_true) / te_true, -20, 5)) - 1.0)
        self.assertAlmostEqual(floating_potential_from_iv(v_up, i_up, smooth_samples=1), vf_true, places=3)
        self.assertAlmostEqual(floating_potential_from_iv(v_up[::-1], i_up[::-1], smooth_samples=1), vf_true, places=3)
        self.assertTrue(np.isnan(floating_potential_from_iv(v_up, np.ones_like(v_up), smooth_samples=1)))

        lam, lam_err = sheath_factor(23.0, 12.5, 3.0, vp_err=0.3, vf_err=0.4, te_err=0.15)
        self.assertAlmostEqual(float(lam), 3.5, places=12)
        expected_err = np.sqrt((0.3 ** 2 + 0.4 ** 2) / 9.0 + (3.5 * 0.15 / 3.0) ** 2)
        self.assertAlmostEqual(float(lam_err), expected_err, places=12)

    def test_local_drift_scales_physical_values(self):
        sc = local_drift_scales(
            1e12, 4.0, 800.0, mu=4.0,
            k_perp_cm=0.4, dln_ne_dr_cm=0.2, dln_te_dr_cm=-0.5, er_v_per_cm=1.6,
        )
        self.assertAlmostEqual(float(sc["cs_m_s"]), 9789.0, delta=30.0)
        self.assertAlmostEqual(float(sc["rho_s_cm"]), 0.511, delta=0.01)
        self.assertAlmostEqual(float(sc["v_ExB_m_s"]), 2000.0, places=6)
        self.assertAlmostEqual(
            float(sc["omega_star_p_rad_s"]),
            float(sc["omega_star_n_rad_s"] + sc["omega_star_T_rad_s"]),
            places=8,
        )
        self.assertGreater(float(sc["nu_ei_s"]), 1e6)
        sc_nan = local_drift_scales(np.nan, 4.0, 800.0, er_v_per_cm=1.0)
        self.assertAlmostEqual(float(sc_nan["v_ExB_m_s"]), 1250.0, places=6)

    def test_savgol_boundary_error_matches_monte_carlo(self):
        from scipy.signal import savgol_filter

        x = np.arange(9, dtype=float)
        vp = 2.0 * x
        vp_err = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 1.0, 0.5, 0.2, 0.1])
        g = radial_profile_gradients(
            x, np.full_like(x, 1e12), np.full_like(x, 3.0), vp,
            vp_err=vp_err, smooth_window=5, polyorder=2,
        )
        rng = np.random.default_rng(7)
        trials = vp[None, :] + rng.normal(0.0, vp_err[None, :], size=(20000, x.size))
        mc_std = np.std(savgol_filter(trials, 5, 2, deriv=1, delta=1.0, axis=1), axis=0)
        np.testing.assert_allclose(g.er_err_v_per_cm, mc_std, rtol=0.03)

    def test_1d_analytic_exponential_wall(self):
        r = np.linspace(0.0, 8.0, 81)
        prof = np.where(r <= 2.0, 1.0, np.where(r >= 4.5, 3.0, np.exp(0.4 * (r - 1.5))))
        m_wall = (r >= 1.5) & (r <= 4.5)
        prof[m_wall] = np.exp(0.4 * (r[m_wall] - 1.5))
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

    def test_nonpositive_core_returns_nan_hollowness(self):
        r = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        prof = np.array([-0.1, 0.0, 0.0, 2.0, 2.2])
        m = hollow_profile_metrics(r, prof)
        self.assertTrue(np.isnan(m.hollowness))

    def test_orbit_smeared_profile_identity_and_apparent_filling(self):
        r = np.linspace(0.0, 8.0, 161)
        prof = 1.0 - 0.6 * np.exp(-(r / 2.5) ** 2)
        np.testing.assert_allclose(orbit_smeared_profile(r, prof, 0.0), prof)
        dr = 1.5
        smeared = orbit_smeared_profile(r, prof, dr)
        self.assertAlmostEqual(smeared[0], float(np.interp(dr, r, prof)), places=6)
        h0 = hollow_profile_metrics(r, prof).hollowness
        hs = hollow_profile_metrics(r, smeared).hollowness
        self.assertLess(hs, h0)


if __name__ == "__main__":
    unittest.main()
