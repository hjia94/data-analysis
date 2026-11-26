"""Unit tests for broadband spectrum/intermittency/Reynolds helpers and Jun2026_turbulence."""
import os
import sys
import tempfile
import unittest

import numpy as np

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "experiments", "ucla-lapd", "Jun-2026"))

from data_analysis.signal import (
    amplitude_normalized_cross_phase,
    fit_broadband_spectrum,
    fluctuation_moments,
    reynolds_shear_production,
)
import Jun2026_turbulence as jtb


class TestTurbulenceHelpers(unittest.TestCase):
    def test_fit_broadband_spectrum_distinguishes_exponential_vs_power_law(self):
        freqs = np.linspace(500.0, 80000.0, 320)
        bg = np.full_like(freqs, 1e-12)

        # Pure exponential continuum + strong 4 kHz coherent peak
        f_e = 8000.0
        psd_exp = 1e-4 * np.exp(-freqs / f_e)
        psd_exp[np.abs(freqs - 4000.0) <= 500.0] += 1.0
        fit_e = fit_broadband_spectrum(freqs, psd_exp, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_e.preferred_model, "exponential")
        self.assertGreater(fit_e.r2_exp, 0.999)
        self.assertAlmostEqual(fit_e.f_e_hz, f_e, delta=1e-3 * f_e)
        self.assertAlmostEqual(fit_e.tau_lorentz_s, 1.0 / (4.0 * np.pi * f_e), delta=1e-8)
        self.assertGreater(fit_e.coh_harmonic_frac, 0.95)

        # Pure power-law continuum f^-3.5
        psd_pow = 1e6 * (freqs ** -3.5)
        fit_p = fit_broadband_spectrum(freqs, psd_pow, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_p.preferred_model, "power_law")
        self.assertGreater(fit_p.r2_pow, 0.999)
        self.assertAlmostEqual(fit_p.alpha, 3.5, places=3)

        # Below SNR floor -> insufficient_data
        fit_low = fit_broadband_spectrum(freqs, bg * 2.0, bg_psd=bg, snr_min=5.0)
        self.assertEqual(fit_low.preferred_model, "insufficient_data")
        self.assertTrue(np.isnan(fit_low.r2_exp))

    def test_fluctuation_moments_sine_wave_and_gaussian(self):
        t = np.linspace(0.0, 10.0, 4000, endpoint=False)
        # Pure sinusoid superposed on linear ramp -> exact excess kurtosis = -1.5, skew = 0
        sine_stack = 10.0 + 0.5 * t[None, :] + 1.0 * np.sin(2.0 * np.pi * 5.0 * t)[None, :]
        m_sin = fluctuation_moments(sine_stack, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_sin.skewness_med, 0.0, delta=1e-2)
        self.assertAlmostEqual(m_sin.excess_kurtosis_med, -1.5, delta=0.05)

        # Gaussian noise -> skew ~ 0, excess kurtosis ~ 0
        rng = np.random.default_rng(123)
        gauss = 5.0 + 0.5 * rng.standard_normal((20, 4000))
        m_g = fluctuation_moments(gauss, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_g.skewness_med, 0.0, delta=0.1)
        self.assertAlmostEqual(m_g.excess_kurtosis_med, 0.0, delta=0.15)
        self.assertAlmostEqual(m_g.rel_rms_med, 0.1, delta=0.01)

    def test_reynolds_shear_production_signs_and_rigid_rotation_zero(self):
        r = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        rs = np.full_like(r, +2e-4)

        # Rigid-body rotation v_theta = Omega * r -> shear = 0 -> P_shear = 0
        prod_rigid = reynolds_shear_production(r, rs, -0.05 * r)
        np.testing.assert_allclose(prod_rigid.shear_per_cm, 0.0, atol=1e-12)
        self.assertAlmostEqual(prod_rigid.p_shear_integrated, 0.0, places=12)

        # Decaying clockwise flow M_theta = -0.25 (constant) -> Omega = -0.25/r -> dOmega/dr = +0.25/r^2 > 0
        # Positive Reynolds stress * positive shear -> P_shear < 0 (wave feeds mean flow)
        prod_feed = reynolds_shear_production(r, rs, np.full_like(r, -0.25))
        self.assertTrue(np.all(prod_feed.shear_per_cm > 0.0))
        self.assertTrue(np.all(prod_feed.p_shear_local < 0.0))
        self.assertLess(prod_feed.p_shear_integrated, 0.0)

        # Opposite-sign Reynolds stress -> P_shear > 0 (KH extraction from mean flow)
        prod_kh = reynolds_shear_production(r, -rs, np.full_like(r, -0.25))
        self.assertGreater(prod_kh.p_shear_integrated, 0.0)

    def test_compute_mach_transport_recovers_quadrature_and_reynolds_stress(self):
        # Synthetic 4 kHz wave at position (dx, dy) = (3.0, 0.0) -> r_hat = x_hat, theta_hat = y_hat
        fs = 100e3
        t = np.arange(0, 300) / fs
        w = 2.0 * np.pi * 4000.0
        # Construct X+, X-, Y+, Y- with known dn/n (10%), dM_r in quadrature (+90 deg to dn),
        # and dM_theta in phase with dM_r (so c_nr ~ 0, c_rt ~ +1)!
        K = 1.34
        dn = 0.10 * np.cos(w * t)
        dmr = 0.05 * np.sin(w * t)
        dmt = 0.05 * np.sin(w * t)
        # mach_single uses FLOW_SIGN = -1 -> I_+/I_- = kappa * exp(-K * M)
        xp = (1.0 + dn) * np.exp(-0.5 * K * dmr)
        xm = (1.0 + dn) * np.exp(+0.5 * K * dmr)
        yp = (1.0 + dn) * np.exp(-0.5 * K * dmt)
        ym = (1.0 + dn) * np.exp(+0.5 * K * dmt)
        ch_d = {k: np.tile(v, (4, 1)) for k, v in (("X+", xp), ("X-", xm), ("Y+", yp), ("Y-", ym))}
        rec = jtb._compute_mach_transport_at_position(
            ch_d, {"X": 1.0, "Y": 1.0}, dx_cm=3.0, dy_cm=0.0, fs_hz=fs, ref_ds=np.tile(1.0 + dn, (4, 1))
        )
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec["c_nr"], 0.0, delta=0.05)
        self.assertAlmostEqual(rec["c_rt"], +1.0, delta=0.05)
        self.assertGreater(rec["rs_f0"], 0.0)

    def test_amplitude_normalized_cross_phase_separates_growth_from_saturation(self):
        # Exact synthetic test: exponential ramp (gamma = 10 ms^-1) with 0 deg cross-phase during
        # growth (t < 1.5 ms) and +60 deg cross-phase in saturation (t >= 1.8 ms).
        fs = 100e3
        t_s = np.arange(0, 450) / fs  # 0..4.5 ms
        t_ms = t_s * 1e3
        w = 2.0 * np.pi * 4000.0
        amp = np.minimum(0.01 * np.exp(10.0 * (t_ms - 1.0)), 1.0)
        # Smooth phase switch from 0 deg on [1.0, 1.45] ms to +60 deg for t >= 1.8 ms
        phi_rad = np.where(t_ms < 1.50, 0.0, np.where(t_ms >= 1.80, np.radians(60.0),
                           np.radians(60.0) * (t_ms - 1.50) / 0.30))
        sig_a = np.tile(amp * np.cos(w * t_s), (4, 1))
        sig_b = np.tile(amp * np.cos(w * t_s + phi_rad), (4, 1))

        m_grow = (t_ms >= 1.05) & (t_ms <= 1.45)
        m_sat = (t_ms >= 2.20) & (t_ms <= 4.00)
        plv_g, phi_g, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=m_grow)
        plv_s, phi_s, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=m_sat)

        self.assertGreater(plv_g, 0.98)
        self.assertGreater(plv_s, 0.98)
        self.assertAlmostEqual(phi_g, 0.0, delta=3.0)
        self.assertAlmostEqual(phi_s, +60.0, delta=3.0)

    def test_per_shot_ramp_masks_early_sat_does_not_overlap_fast_ramp(self):
        # Fast exponential ramp (gamma = 10 ms^-1) reaching saturation around t = 1.6 ms:
        # 0 deg during growth (amp < 0.85 * peak), +60 deg once saturated (amp >= 0.85 * peak).
        fs = 100e3
        t_s = np.arange(0, 490) / fs
        t_ms = t_s * 1e3
        w = 2.0 * np.pi * 4000.0
        # DC baseline = 1.0, relative AC amplitude grows 0.002 -> 0.20 (100x) at gamma = 10 ms^-1
        rel_amp = np.minimum(0.002 * np.exp(10.0 * (t_ms - 1.15)), 0.20)
        phi_rad = np.where(rel_amp < 0.17, 0.0, np.radians(60.0))
        sig_a = np.tile(1.0 + rel_amp * np.cos(w * t_s), (6, 1))
        sig_b = np.tile(1.0 + rel_amp * np.cos(w * t_s + phi_rad), (6, 1))
        f_band = (2500.0, 5500.0)

        m_grow, m_early, m_late, t_ons, ramp_durs, t_90s = jtb._per_shot_ramp_masks(t_s, sig_a, f_band)
        self.assertTrue(np.all(np.isfinite(t_ons)))
        # Hilbert t_90 must land at/after saturation (~1.60 ms) so early_sat sees pure +60 deg!
        self.assertGreater(float(np.median(t_90s)), 1.55)
        _, phi_g, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_grow)
        _, phi_e, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_early)
        _, phi_s, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_late)
        self.assertAlmostEqual(phi_g, 0.0, delta=5.0)
        self.assertAlmostEqual(phi_e, +60.0, delta=5.0)
        self.assertAlmostEqual(phi_s, +60.0, delta=5.0)

    def test_batch_all_respects_overwrite_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, jtb.OUT_NAME)
            np.savez(p, sentinel=np.array(99))
            loaded = jtb.batch_all(data_dir=tmp, overwrite=False)
            self.assertEqual(int(loaded["sentinel"]), 99)


if __name__ == "__main__":
    unittest.main()
