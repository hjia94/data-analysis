"""Unit tests for azimuthal/axial mode decomposition and plasma-frame dispersion."""

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

from data_analysis.plasma.modes import (
    axial_wavenumber,
    bdot_cartesian_dipole_check,
    fit_azimuthal_mode,
    plasma_frame_dispersion,
)
from data_analysis.signal import avg_cross_spectrum
import Jun2026_modes as jmo
import Jun2026_plot as jpl


class TestModesAndDispersion(unittest.TestCase):
    def test_fit_azimuthal_mode_both_channel_orders(self):
        # Verify physical m sign directly through avg_cross_spectrum (no sign assumptions).
        dt = 1e-5
        t = np.arange(2048) * dt
        f0 = 4000.0
        cx_true, cy_true = 1.5, -0.5
        thetas = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        r0 = 4.0
        px = cx_true + r0 * np.cos(thetas)
        py = cy_true + r0 * np.sin(thetas)
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

            res_a = fit_azimuthal_mode(
                px, py, g2_a, ph_a, center=(cx_true, cy_true), moving_is_a=True
            )
            res_b = fit_azimuthal_mode(
                px, py, g2_b, ph_b, center=(cx_true, cy_true), moving_is_a=False
            )
            self.assertEqual(res_a.m, m_true)
            self.assertEqual(res_b.m, m_true)
            self.assertGreater(res_a.purity, 0.999)
            self.assertGreater(res_b.purity, 0.999)

    def test_bdot_rotating_current_filament_gives_mB_2mJ_and_quadrature(self):
        # Rotating axial current filament J_z ~ cos(m_J * theta - w*t), |m_J| = 1:
        # Outside the filament: A_z ~ r^-1 cos(m_J*theta - w*t)
        # B_r = (1/r) dA_z/dtheta ~ -m_J sin(m_J*theta - w*t), B_theta = -dA_z/dr ~ +cos(m_J*theta - w*t)
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
                br = -m_j * np.sin(arg)
                bth = np.cos(arg)
                bx = (br * np.cos(th) - bth * np.sin(th))[None, :]
                by = (br * np.sin(th) + bth * np.cos(th))[None, :]
                f, gx, px_ab, _ = avg_cross_spectrum(ref, bx, dt, nperseg=1024)
                _, gy, py_ab, _ = avg_cross_spectrum(ref, by, dt, nperseg=1024)
                k = int(np.argmin(np.abs(f - f0)))
                g_bx.append(gx[k]); p_bx.append(px_ab[k])
                g_by.append(gy[k]); p_by.append(py_ab[k])

            # Cartesian components have m_B = 2 * m_J (with moving_is_a=False)
            res_bx = fit_azimuthal_mode(px, py, g_bx, p_bx, center=(0.0, 0.0), moving_is_a=False)
            res_by = fit_azimuthal_mode(px, py, g_by, p_by, center=(0.0, 0.0), moving_is_a=False)
            self.assertEqual(res_bx.m, 2 * m_j)
            self.assertEqual(res_by.m, 2 * m_j)
            self.assertGreater(res_bx.purity, 0.999)

            dip = bdot_cartesian_dipole_check(p_bx, p_by, g_bx, g_by, moving_is_a=False)
            self.assertEqual(dip.inferred_m_j, m_j)
            self.assertAlmostEqual(dip.dphi_by_minus_bx_deg, -90.0 * m_j, delta=1.0)
            self.assertGreater(dip.resultant_R, 0.999)

    def test_axial_wavenumber_wraps_across_pi(self):
        # Angles near +-175 deg have circular mean 180 deg, NOT 0 deg.
        dphi = np.deg2rad([175.0, -175.0, 178.0, -178.0])
        res = axial_wavenumber(dphi, dz_m=1.28)
        self.assertAlmostEqual(abs(res.mean_dphi_deg), 180.0, delta=0.5)
        self.assertGreater(res.resultant_R, 0.99)
        self.assertAlmostEqual(abs(res.k_par_rad_m), np.pi / 1.28, delta=0.05)
        self.assertAlmostEqual(res.lambda_par_m, 2.56, delta=0.05)

    def test_plasma_frame_dispersion_doppler_and_alfven_regime(self):
        # Mode rotating CW (m = -1) at f_lab = 4.0 kHz in a plasma rotating CW
        # at v_rot = -1500 m/s at r = 3.0 cm:
        # f_rot = -1500 / (2*pi*0.03) = -7957.7 Hz -> f_doppler = (-1)*(-7957.7) = +7957.7 Hz
        # -> f_plasma = 4000 - 7957.7 = -3957.7 Hz (propagates opposite to rotation in plasma frame).
        disp = plasma_frame_dispersion(
            f_lab_hz=4000.0, m=-1, r_cm=3.0, v_rot_m_s=-1500.0,
            b0_g=800.0, ne_cm3=2e12, te_ev=2.0,
            dln_ne_dr_cm=+0.3, dln_te_dr_cm=-0.4,
            k_par_m=0.8, mu=4.0, b_sign_z=+1,
        )
        f_rot_exp = -1500.0 / (2.0 * np.pi * 0.03)
        self.assertAlmostEqual(disp["f_rot_hz"], f_rot_exp, places=4)
        self.assertAlmostEqual(disp["f_plasma_hz"], 4000.0 - (-1) * f_rot_exp, places=4)
        # With B along +z, E_r > 0 gives CW ExB (m=-1). Inverted dn/dr > 0 has
        # v_*n in the SAME CW direction (f_*n > 0), while dT/dr < 0 has v_*T OPPOSITE (f_*T < 0).
        self.assertGreater(disp["f_star_n_hz"], 0.0)
        self.assertLess(disp["f_star_T_hz"], 0.0)
        self.assertAlmostEqual(disp["f_star_p_hz"], disp["f_star_n_hz"] + disp["f_star_T_hz"], places=6)
        self.assertGreater(disp["beta_e_over_me_mi"], 1.0)  # kinetic Alfven regime (beta_e > me/mi)
        self.assertGreater(disp["f_alfven_hz"], 1e4)

    def test_blockmean_to_grid_suppresses_aliasing(self):
        # 5 MHz source with a 4 kHz signal + huge 104 kHz tone (aliases to 4 kHz under
        # unfiltered 100 kHz point-sampling, but integrates out over 50-sample block mean!)
        dt_src = 2e-7  # 5 MHz
        t_src = np.arange(15000) * dt_src  # 3.0 ms
        t_grid = np.arange(0.0, 2.95e-3, 1e-5)  # 100 kHz
        sig_4k = 1.0 * np.sin(2.0 * np.pi * 4000.0 * t_src)
        hf_104k = 10.0 * np.sin(2.0 * np.pi * 104000.0 * t_src)
        stack = (sig_4k + hf_104k)[None, :]

        out_aa = jmo._blockmean_to_grid(t_src, stack, t_grid)[0]
        out_raw = np.interp(t_grid, t_src, stack[0])
        ref_4k = np.sin(2.0 * np.pi * 4000.0 * t_grid)

        # Unfiltered interp folds 100% of the 10x 104 kHz tone (RMS ~ 7.07);
        # 10 us block-mean has exact sinc(1.04) = 0.038 attenuation -> >20x suppression.
        err_aa = float(np.std(out_aa - ref_4k))
        err_raw = float(np.std(out_raw - ref_4k))
        self.assertLess(err_aa, 0.35)
        self.assertGreater(err_raw, 5.0)
        self.assertLess(err_aa, 0.05 * err_raw)

    def test_build_dispersion_table_preserves_measured_velocity_signs(self):
        # Flipping the sign of v_ExB_r or Mach dv_rot must flip the Doppler shift
        # rather than forcing clockwise via abs().
        with tempfile.TemporaryDirectory() as tmp:
            x = np.arange(-6.0, 7.0, 1.0)
            r = np.arange(1.0, 7.0, 1.0)
            fake_onset = {
                ir: {"f_peak_hz": 4000.0, "gamma_med_s": 3000.0, "t_onset_med_ms": 2.0}
                for ir in ("06", "00", "01", "02", "03", "04", "05")
            }
            for sign_v in (+1.0, -1.0):
                prof_d = {}
                mach_d = {}
                for ir in ("06", "00", "01", "02", "03", "04", "05"):
                    prof_d[f"{ir}__ref_sweep_idx"] = np.int64(0)
                    prof_d[f"{ir}__x_cm"] = x
                    prof_d[f"{ir}__x_center_cm"] = np.float64(0.0)
                    prof_d[f"{ir}__ne"] = np.full((x.size, 1), 5e12)
                    prof_d[f"{ir}__Te"] = np.full((x.size, 1), 2.0)
                    prof_d[f"{ir}__dln_ne_dr"] = np.full((x.size, 1), +0.3)
                    prof_d[f"{ir}__dln_te_dr"] = np.full((x.size, 1), -0.3)
                for mr in ("08", "09", "10", "11", "12", "13", "14"):
                    mach_d[f"{mr}__ref_bin_idx"] = np.int64(0)
                    mach_d[f"{mr}__r_cm"] = r
                    # sign_v > 0 -> outward Er > 0 (CW v_theta < 0) and lab-frame Mach dv_rot < 0 (CW)
                    mach_d[f"{mr}__v_ExB_r_km_s"] = np.full((r.size, 1), sign_v * 1.5)
                    mach_d[f"{mr}__dv_rot_km_s"] = np.full((r.size, 1), -sign_v * 1.2)

                np.savez(os.path.join(tmp, "profiles_summary_jun2026.npz"), **prof_d)
                np.savez(os.path.join(tmp, "mach_line_rescaled_jun2026.npz"), **mach_d)

                with mock.patch.object(jmo.jon, "load_onset_table", return_value=fake_onset):
                    rows = jmo.build_dispersion_table(data_dir=tmp, m=-1, b_sign_z=+1)
                # sign_v > 0 -> CW rotation -> m=-1 has positive Doppler shift;
                # sign_v < 0 -> CCW rotation -> m=-1 has negative Doppler shift!
                self.assertGreater(rows["02"]["f_doppler_exb_hz"] * sign_v, 0.0)
                self.assertGreater(rows["02"]["f_doppler_mach_hz"] * sign_v, 0.0)

            # Stable run (t_onset = NaN) must yield NaN for f_plasma_* and nu_ei_over_omega
            # instead of fabricating values from a placeholder f_lab, while still returning
            # frequency-independent Doppler, diamagnetic, and Alfven scales.
            fake_onset["06"] = {"f_peak_hz": np.nan, "gamma_med_s": np.nan, "t_onset_med_ms": np.nan}
            with mock.patch.object(jmo.jon, "load_onset_table", return_value=fake_onset):
                rows_stbl = jmo.build_dispersion_table(data_dir=tmp, m=-1, b_sign_z=+1)
            r06 = rows_stbl["06"]
            self.assertFalse(r06["unstable"])
            self.assertTrue(np.isnan(r06["f_plasma_exb_hz"]))
            self.assertTrue(np.isnan(r06["f_plasma_mach_hz"]))
            self.assertTrue(np.isnan(r06["nu_ei_over_omega"]))
            self.assertTrue(np.isnan(r06["gamma_over_omega"]))
            self.assertTrue(np.isfinite(r06["f_doppler_exb_hz"]))
            self.assertTrue(np.isfinite(r06["f_star_n_hz"]))
            self.assertTrue(np.isfinite(r06["f_alfven_hz"]))

    def test_require_xcorr_raises_keyerror_and_batch_all_skips_by_default(self):
        with mock.patch.object(jpl, "_load_xcorr_run", return_value=None):
            with self.assertRaises(KeyError):
                jmo._require_xcorr("26.hdf5", ("s", "C1"), ("s", "C2"), "missing.npz")

        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "modes_dispersion_jun2026.npz")
            np.savez(out_npz, sentinel=np.int64(42))
            res = jmo.batch_all(data_dir=tmp, out_path=out_npz, overwrite=False)
            self.assertEqual(res, out_npz)
            with np.load(out_npz) as d:
                self.assertEqual(int(d["sentinel"]), 42)


if __name__ == "__main__":
    unittest.main()
