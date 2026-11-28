"""Unit tests for data_analysis.plasma.modes (azimuthal/axial mode numbers & dispersion)."""

import unittest

import numpy as np

from data_analysis.plasma.modes import (
    axial_wavenumber,
    bdot_cartesian_dipole_check,
    fit_azimuthal_mode,
    plasma_frame_dispersion,
)
from data_analysis.signal import avg_cross_spectrum


class TestModesAndDispersion(unittest.TestCase):
    def test_fit_azimuthal_mode_both_channel_orders(self):
        dt = 1e-5
        t = np.arange(2048) * dt
        f0 = 4000.0
        cx_true, cy_true = 1.5, -0.5
        thetas = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        px = cx_true + 4.0 * np.cos(thetas)
        py = cy_true + 4.0 * np.sin(thetas)
        ref = np.cos(-2.0 * np.pi * f0 * t)[None, :]

        for m_true in (+1, -1, +2):
            g2_a, ph_a, g2_b, ph_b = [], [], [], []
            for th in thetas:
                sig = np.cos(m_true * th - 2.0 * np.pi * f0 * t)[None, :]
                f, g_ab, p_ab, _ = avg_cross_spectrum(sig, ref, dt, nperseg=1024)
                _, g_ba, p_ba, _ = avg_cross_spectrum(ref, sig, dt, nperseg=1024)
                k = int(np.argmin(np.abs(f - f0)))
                g2_a.append(g_ab[k]); ph_a.append(p_ab[k])
                g2_b.append(g_ba[k]); ph_b.append(p_ba[k])

            res_a = fit_azimuthal_mode(px, py, g2_a, ph_a, center=(cx_true, cy_true), moving_is_a=True)
            res_b = fit_azimuthal_mode(px, py, g2_b, ph_b, center=(cx_true, cy_true), moving_is_a=False)
            self.assertEqual(res_a.m, m_true)
            self.assertEqual(res_b.m, m_true)
            self.assertGreater(res_a.purity, 0.999)
            self.assertGreater(res_b.purity, 0.999)

    def test_bdot_rotating_current_filament_gives_mB_2mJ_and_quadrature(self):
        dt = 1e-5
        t = np.arange(2048) * dt
        f0 = 4000.0
        thetas = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        px = 4.0 * np.cos(thetas)
        py = 4.0 * np.sin(thetas)
        ref = np.cos(-2.0 * np.pi * f0 * t)[None, :]

        for m_j in (+1, -1):
            g_bx, p_bx, g_by, p_by = [], [], [], []
            for th in thetas:
                arg = m_j * th - 2.0 * np.pi * f0 * t
                br, bth = -m_j * np.sin(arg), np.cos(arg)
                bx = (br * np.cos(th) - bth * np.sin(th))[None, :]
                by = (br * np.sin(th) + bth * np.cos(th))[None, :]
                f, gx, px_ab, _ = avg_cross_spectrum(ref, bx, dt, nperseg=1024)
                _, gy, py_ab, _ = avg_cross_spectrum(ref, by, dt, nperseg=1024)
                k = int(np.argmin(np.abs(f - f0)))
                g_bx.append(gx[k]); p_bx.append(px_ab[k])
                g_by.append(gy[k]); p_by.append(py_ab[k])

            res_bx = fit_azimuthal_mode(px, py, g_bx, p_bx, center=(0.0, 0.0), moving_is_a=False)
            res_by = fit_azimuthal_mode(px, py, g_by, p_by, center=(0.0, 0.0), moving_is_a=False)
            self.assertEqual((res_bx.m, res_by.m), (2 * m_j, 2 * m_j))
            self.assertGreater(res_bx.purity, 0.999)

            dip = bdot_cartesian_dipole_check(p_bx, p_by, g_bx, g_by, moving_is_a=False)
            self.assertEqual(dip.inferred_m_j, m_j)
            self.assertAlmostEqual(dip.dphi_by_minus_bx_deg, -90.0 * m_j, delta=1.0)
            self.assertGreater(dip.resultant_R, 0.999)

    def test_axial_wavenumber_wraps_across_pi(self):
        dphi = np.deg2rad([175.0, -175.0, 178.0, -178.0])
        res = axial_wavenumber(dphi, dz_m=1.28)
        self.assertAlmostEqual(abs(res.mean_dphi_deg), 180.0, delta=0.5)
        self.assertGreater(res.resultant_R, 0.99)
        self.assertAlmostEqual(abs(res.k_par_rad_m), np.pi / 1.28, delta=0.05)
        self.assertAlmostEqual(res.lambda_par_m, 2.56, delta=0.05)

    def test_plasma_frame_dispersion_doppler_and_alfven_regime(self):
        disp = plasma_frame_dispersion(
            f_lab_hz=4000.0, m=-1, r_cm=3.0, v_rot_m_s=-1500.0,
            b0_g=800.0, ne_cm3=2e12, te_ev=2.0,
            dln_ne_dr_cm=+0.3, dln_te_dr_cm=-0.4,
            k_par_m=0.8, mu=4.0, b_sign_z=+1,
        )
        f_rot_exp = -1500.0 / (2.0 * np.pi * 0.03)
        self.assertAlmostEqual(disp["f_rot_hz"], f_rot_exp, places=4)
        self.assertAlmostEqual(disp["f_plasma_hz"], 4000.0 - (-1) * f_rot_exp, places=4)
        self.assertGreater(disp["f_star_n_hz"], 0.0)
        self.assertLess(disp["f_star_T_hz"], 0.0)
        self.assertAlmostEqual(disp["f_star_p_hz"], disp["f_star_n_hz"] + disp["f_star_T_hz"], places=6)
        self.assertGreater(disp["beta_e_over_me_mi"], 1.0)
        self.assertGreater(disp["f_alfven_hz"], 1e4)


if __name__ == "__main__":
    unittest.main()
