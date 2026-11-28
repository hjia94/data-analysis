"""Unit tests for data_analysis.signal.turbulence (spectra, moments, Reynolds production, steepening)."""

import unittest

import numpy as np

from data_analysis.signal import (
    amplitude_normalized_cross_phase,
    fit_broadband_spectrum,
    fluctuation_moments,
    harmonic_steepening,
    reynolds_shear_production,
)


class TestTurbulenceHelpers(unittest.TestCase):
    def test_fit_broadband_spectrum_distinguishes_exponential_vs_power_law(self):
        freqs = np.linspace(500.0, 80000.0, 320)
        bg = np.full_like(freqs, 1e-12)

        f_e = 8000.0
        psd_exp = 1e-4 * np.exp(-freqs / f_e)
        psd_exp[np.abs(freqs - 4000.0) <= 500.0] += 1.0
        fit_e = fit_broadband_spectrum(freqs, psd_exp, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_e.preferred_model, "exponential")
        self.assertGreater(fit_e.r2_exp, 0.999)
        self.assertAlmostEqual(fit_e.f_e_hz, f_e, delta=1e-3 * f_e)
        self.assertAlmostEqual(fit_e.tau_lorentz_s, 1.0 / (4.0 * np.pi * f_e), delta=1e-8)
        self.assertGreater(fit_e.coh_harmonic_frac, 0.95)

        psd_pow = 1e6 * (freqs ** -3.5)
        fit_p = fit_broadband_spectrum(freqs, psd_pow, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_p.preferred_model, "power_law")
        self.assertGreater(fit_p.r2_pow, 0.999)
        self.assertAlmostEqual(fit_p.alpha, 3.5, places=3)

        fit_low = fit_broadband_spectrum(freqs, bg * 2.0, bg_psd=bg, snr_min=5.0)
        self.assertEqual(fit_low.preferred_model, "insufficient_data")
        self.assertTrue(np.isnan(fit_low.r2_exp))

    def test_fluctuation_moments_sine_wave_and_gaussian(self):
        t = np.linspace(0.0, 10.0, 4000, endpoint=False)
        sine_stack = 10.0 + 0.5 * t[None, :] + 1.0 * np.sin(2.0 * np.pi * 5.0 * t)[None, :]
        m_sin = fluctuation_moments(sine_stack, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_sin.skewness_med, 0.0, delta=1e-2)
        self.assertAlmostEqual(m_sin.excess_kurtosis_med, -1.5, delta=0.05)

        rng = np.random.default_rng(123)
        gauss = 5.0 + 0.5 * rng.standard_normal((20, 4000))
        m_g = fluctuation_moments(gauss, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_g.skewness_med, 0.0, delta=0.1)
        self.assertAlmostEqual(m_g.excess_kurtosis_med, 0.0, delta=0.15)
        self.assertAlmostEqual(m_g.rel_rms_med, 0.1, delta=0.01)

    def test_reynolds_shear_production_signs_and_rigid_rotation_zero(self):
        r = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        rs = np.full_like(r, +2e-4)

        prod_rigid = reynolds_shear_production(r, rs, -0.05 * r)
        np.testing.assert_allclose(prod_rigid.shear_per_cm, 0.0, atol=1e-12)
        self.assertAlmostEqual(prod_rigid.p_shear_integrated, 0.0, places=12)

        prod_feed = reynolds_shear_production(r, rs, np.full_like(r, -0.25))
        self.assertTrue(np.all(prod_feed.shear_per_cm > 0.0))
        self.assertTrue(np.all(prod_feed.p_shear_local < 0.0))
        self.assertLess(prod_feed.p_shear_integrated, 0.0)

        prod_kh = reynolds_shear_production(r, -rs, np.full_like(r, -0.25))
        self.assertGreater(prod_kh.p_shear_integrated, 0.0)

    def test_amplitude_normalized_cross_phase_separates_growth_from_saturation(self):
        fs = 100e3
        t_s = np.arange(0, 450) / fs
        t_ms = t_s * 1e3
        w = 2.0 * np.pi * 4000.0
        amp = np.minimum(0.01 * np.exp(10.0 * (t_ms - 1.0)), 1.0)
        phi_rad = np.where(
            t_ms < 1.50, 0.0,
            np.where(t_ms >= 1.80, np.radians(60.0), np.radians(60.0) * (t_ms - 1.50) / 0.30),
        )
        sig_a = np.tile(amp * np.cos(w * t_s), (4, 1))
        sig_b = np.tile(amp * np.cos(w * t_s + phi_rad), (4, 1))

        plv_g, phi_g, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=(t_ms >= 1.05) & (t_ms <= 1.45))
        plv_s, phi_s, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=(t_ms >= 2.20) & (t_ms <= 4.00))
        self.assertGreater(plv_g, 0.98)
        self.assertGreater(plv_s, 0.98)
        self.assertAlmostEqual(phi_g, 0.0, delta=3.0)
        self.assertAlmostEqual(phi_s, +60.0, delta=3.0)

    def test_harmonic_steepening_separates_curvature_vs_sawtooth_and_folds_waveform(self):
        fs = 100e3
        t = np.arange(0, 2000) / fs
        f0 = 4000.0
        rng = np.random.default_rng(7)
        phases = rng.uniform(-np.pi, np.pi, size=8)
        f_jit = f0 + rng.uniform(-120.0, 120.0, size=8)

        def _make_stack(a2_cos, a2_sin):
            return np.vstack([
                10.0 + 0.2 * t + np.cos(2.0 * np.pi * fj * t + ph)
                + a2_cos * np.cos(4.0 * np.pi * fj * t + 2.0 * ph)
                + a2_sin * np.sin(4.0 * np.pi * fj * t + 2.0 * ph)
                for ph, fj in zip(phases, f_jit)
            ])

        st_sym = harmonic_steepening(_make_stack(+0.35, 0.0), fs, f0)
        self.assertGreater(st_sym.biphase_lock, 0.99)
        self.assertAlmostEqual(st_sym.a2_a1_med, 0.35, delta=0.02)
        self.assertAlmostEqual(st_sym.biphase_deg, 0.0, delta=2.0)
        self.assertAlmostEqual(st_sym.curvature_comp, +0.35, delta=0.02)
        self.assertAlmostEqual(st_sym.steepening_comp, 0.0, delta=0.02)
        np.testing.assert_allclose(st_sym.folded_wave, np.cos(st_sym.phase_rad) + 0.35 * np.cos(2.0 * st_sym.phase_rad), atol=0.08)

        st_rise = harmonic_steepening(_make_stack(0.0, -0.35), fs, f0)
        self.assertAlmostEqual(st_rise.biphase_deg, +90.0, delta=2.0)
        self.assertAlmostEqual(st_rise.steepening_comp, +0.35, delta=0.02)
        self.assertGreater(st_rise.slope_ratio_med, 1.7)
        self.assertLess(st_rise.rise_frac_med, 0.40)

        st_fall = harmonic_steepening(_make_stack(0.0, +0.35), fs, f0)
        self.assertAlmostEqual(st_fall.biphase_deg, -90.0, delta=2.0)
        self.assertAlmostEqual(st_fall.steepening_comp, -0.35, delta=0.02)
        self.assertLess(st_fall.slope_ratio_med, 0.6)
        self.assertGreater(st_fall.rise_frac_med, 0.60)

        st_1d = harmonic_steepening(np.cos(2.0 * np.pi * f0 * t), fs, f0)
        self.assertEqual(st_1d.n_shots, 1)
        self.assertFalse(st_1d.biphase_locked)
        st_bad = harmonic_steepening(np.cos(2.0 * np.pi * f0 * t), fs, 0.3 * fs)
        self.assertEqual(st_bad.n_shots, 0)
        self.assertTrue(np.isnan(st_bad.biphase_deg))

    def test_harmonic_steepening_gates_unlocked_random_biphase_to_nan(self):
        fs = 100e3
        t = np.arange(0, 2000) / fs
        f0 = 4000.0
        rng = np.random.default_rng(42)
        th = 2.0 * np.pi * f0 * t
        rows = [
            10.0 + np.cos(th + rng.uniform(-np.pi, np.pi))
            + 0.30 * np.cos(2.0 * th + rng.uniform(-np.pi, np.pi))
            for _ in range(25)
        ]
        st = harmonic_steepening(np.vstack(rows), fs, f0)
        self.assertFalse(st.biphase_locked)
        self.assertTrue(np.isnan(st.biphase_deg))
        self.assertTrue(np.isnan(st.curvature_comp))
        self.assertTrue(np.isnan(st.steepening_comp))
        st_raw = harmonic_steepening(np.vstack(rows), fs, f0, gate_unlocked=False)
        self.assertTrue(np.isfinite(st_raw.biphase_deg))


if __name__ == "__main__":
    unittest.main()
