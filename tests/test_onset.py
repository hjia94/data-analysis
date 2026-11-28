"""Unit tests for data_analysis.signal onset, exponential growth, and frequency helpers."""

import unittest

import numpy as np

from data_analysis.signal import (
    detect_onset_time,
    dominant_frequency,
    fit_exponential_growth,
    sliding_detrended_rms,
)


class TestSignalOnsetHelpers(unittest.TestCase):
    def test_sliding_detrended_rms_ignores_linear_dc_ramp(self):
        t = np.arange(500) * 1e-5
        ramp = 1.0 + 200.0 * t
        t_env, env = sliding_detrended_rms(ramp, t, win_s=0.25e-3, step_s=0.05e-3)
        self.assertGreater(t_env.size, 50)
        np.testing.assert_allclose(env, 0.0, atol=1e-13)

        amp = 0.2
        sig = 2.0 + amp * np.sin(2 * np.pi * 4000.0 * t)
        _, env_s = sliding_detrended_rms(sig, t, win_s=0.50e-3, step_s=0.05e-3)
        _, env_2d = sliding_detrended_rms(np.vstack([sig, sig]), t, win_s=0.50e-3, step_s=0.05e-3)
        np.testing.assert_allclose(env_2d[0], env_s)
        np.testing.assert_allclose(env_2d[1], env_s)
        expected_rms = amp / (np.sqrt(2.0) * 2.0)
        self.assertAlmostEqual(float(np.median(env_s)), expected_rms, delta=0.005)
        self.assertLess(float(np.std(env_s) / np.mean(env_s)), 0.03)

    def test_detect_onset_time_sustain_and_t_min(self):
        t_env = np.arange(20) * 0.1
        s0 = np.zeros(20)
        s0[1] = 0.05
        s0[6] = 0.05
        s0[10:14] = 0.03
        s1 = np.zeros(20)
        s1[0:3] = 0.05
        onset = detect_onset_time(t_env, np.vstack([s0, s1]), threshold=0.015, t_min=0.4, sustain=2)
        self.assertAlmostEqual(onset[0], 1.0)
        self.assertTrue(np.isnan(onset[1]))

    def test_fit_exponential_growth_recovers_true_gamma(self):
        t = np.arange(0.0, 5.0e-3, 0.05e-3)
        gamma_true = 4000.0
        grow = 0.002 * np.exp(gamma_true * (t - 1.0e-3))
        env = 0.003 + 0.15 * grow / (0.15 + grow)
        fit = fit_exponential_growth(t, env, t_onset=1.2e-3, noise_floor=0.003)
        self.assertGreaterEqual(fit.n_points, 5)
        self.assertGreater(fit.r2, 0.98)

        pure_t = np.arange(0.0, 2.0e-3, 0.05e-3)
        pure_env = np.minimum(0.001 * np.exp(gamma_true * pure_t), 0.20)
        pure_fit = fit_exponential_growth(pure_t, pure_env)
        self.assertAlmostEqual(pure_fit.gamma, gamma_true, delta=1e-6)
        self.assertAlmostEqual(pure_fit.r2, 1.0, places=10)

    def test_dominant_frequency_sub_bin_accuracy(self):
        dt = 1e-5
        t = np.arange(250) * dt
        f_true = 3870.0
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, 2 * np.pi, size=(12, 1))
        stack = 1.0 + 0.15 * np.sin(2 * np.pi * f_true * t[None, :] + phases)
        stack += rng.normal(0, 0.005, size=stack.shape)
        res = dominant_frequency(stack, dt, fband=(1500.0, 12000.0), pad_factor=4)
        self.assertEqual(res.n_used, 12)
        self.assertAlmostEqual(res.f_peak_hz, f_true, delta=35.0)
        self.assertGreater(res.peak_snr, 100.0)
        self.assertLess(res.f_fwhm_hz, 700.0)


if __name__ == "__main__":
    unittest.main()
