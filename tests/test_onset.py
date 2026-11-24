"""Unit tests for onset detection, exponential growth-rate fit, and dominant frequency."""

import os
import sys
import unittest

import numpy as np

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

from data_analysis.signal import (
    detect_onset_time,
    dominant_frequency,
    fit_exponential_growth,
    sliding_detrended_rms,
)
import Jun2026_onset as jon


class TestSignalOnsetHelpers(unittest.TestCase):
    def test_sliding_detrended_rms_ignores_linear_dc_ramp(self):
        t = np.arange(500) * 1e-5  # 5.0 ms at 100 kHz
        # Pure linear ramp with zero oscillation -> detrended RMS must be ~0
        ramp = 1.0 + 200.0 * t
        t_env, env = sliding_detrended_rms(ramp, t, win_s=0.25e-3, step_s=0.05e-3)
        self.assertGreater(t_env.size, 50)
        np.testing.assert_allclose(env, 0.0, atol=1e-13)

        # Known 4 kHz sinusoid on DC=2.0 with 0.50 ms Hann window (~2 cycles, FWHM=0.25 ms):
        # true relative RMS = A / (sqrt(2) * DC) = 0.07071 with minimal 2f phase ripple.
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
        # Shot 0: transient at t=0.1, single spike at t=0.6, sustained crossing at t=1.0
        s0 = np.zeros(20)
        s0[1] = 0.05
        s0[6] = 0.05
        s0[10:14] = 0.03
        # Shot 1: never crosses after t_min
        s1 = np.zeros(20)
        s1[0:3] = 0.05
        onset = detect_onset_time(t_env, np.vstack([s0, s1]), threshold=0.015, t_min=0.4, sustain=2)
        self.assertAlmostEqual(onset[0], 1.0)
        self.assertTrue(np.isnan(onset[1]))

    def test_fit_exponential_growth_recovers_true_gamma(self):
        t = np.arange(0.0, 5.0e-3, 0.05e-3)  # 5 ms, 0.05 ms step
        gamma_true = 4000.0  # 4.0 ms^-1 (e-folding time 0.25 ms)
        # Smooth logistic-like saturation: grows as exp(gamma * t), saturates at 0.15
        grow = 0.002 * np.exp(gamma_true * (t - 1.0e-3))
        env = 0.003 + 0.15 * grow / (0.15 + grow)  # noise floor 0.003, peak ~0.153
        fit = fit_exponential_growth(t, env, t_onset=1.2e-3, noise_floor=0.003)
        self.assertGreaterEqual(fit.n_points, 5)
        self.assertGreater(fit.r2, 0.98)
        # Logistic bending slightly softens the effective slope over [0.15, 0.65] peak,
        # check pure exponential recovers exact gamma:
        pure_t = np.arange(0.0, 2.0e-3, 0.05e-3)
        pure_env = np.minimum(0.001 * np.exp(gamma_true * pure_t), 0.20)
        pure_fit = fit_exponential_growth(pure_t, pure_env)
        self.assertAlmostEqual(pure_fit.gamma, gamma_true, delta=1e-6)
        self.assertAlmostEqual(pure_fit.r2, 1.0, places=10)

    def test_dominant_frequency_sub_bin_accuracy(self):
        dt = 1e-5  # 100 kHz
        t = np.arange(250) * dt  # 2.5 ms -> raw df = 400 Hz
        f_true = 3870.0  # deliberately off the 400 Hz bin grid
        rng = np.random.default_rng(7)
        phases = rng.uniform(0, 2 * np.pi, size=(12, 1))
        stack = 1.0 + 0.15 * np.sin(2 * np.pi * f_true * t[None, :] + phases)
        stack += rng.normal(0, 0.005, size=stack.shape)
        res = dominant_frequency(stack, dt, fband=(1500.0, 12000.0), pad_factor=4)
        self.assertEqual(res.n_used, 12)
        self.assertAlmostEqual(res.f_peak_hz, f_true, delta=35.0)
        self.assertGreater(res.peak_snr, 100.0)
        self.assertLess(res.f_fwhm_hz, 700.0)

    def test_analyze_envelope_stack_end_to_end_with_shot_jitter(self):
        dt = 1e-5
        t = np.arange(500) * dt  # 0..5 ms
        gamma_true = 3500.0
        f_true = 4120.0
        rng = np.random.default_rng(123)
        nshot = 25
        # Shots have different onset times (1.4..2.0 ms) to verify onset-aligned growth rate
        t0_shots = rng.uniform(1.4e-3, 2.0e-3, size=nshot)
        rows = []
        for t0 in t0_shots:
            # Landau-Stuart saturation: dA/dt = gamma*A - beta*A^3
            # Stays purely exponential (d ln A / dt = gamma * (1 - (A/A_max)^2))
            # through the early rising ramp before smoothly saturating at 0.20.
            g = 0.005 * np.exp(gamma_true * (t - t0))
            amp = 0.20 / np.sqrt(1.0 + (0.20 / g) ** 2)
            phase = rng.uniform(0, 2 * np.pi)
            sig = 1.0 + amp * np.sin(2 * np.pi * f_true * t + phase) + rng.normal(0, 0.002, size=t.size)
            rows.append(sig)
        res = jon.analyze_envelope_stack(t, np.vstack(rows))
        self.assertEqual(res["n_shots"], nshot)
        self.assertAlmostEqual(res["unstable_frac"], 1.0)
        self.assertAlmostEqual(res["f_peak_hz"], f_true, delta=60.0)
        self.assertTrue(np.isfinite(res["gamma_med_s"]))
        self.assertTrue(np.isfinite(res["gamma_aligned_s"]))
        self.assertGreater(res["gamma_aligned_r2"], 0.95)
        # Over [0.15, 0.65] of saturation, Landau-Stuart d(ln A)/dt = gamma*(1-(A/A_max)^2)
        # averages ~0.75-0.85 * gamma_true; onset alignment must recover it closely
        # and stay strictly steeper than an unaligned ensemble median smeared by jitter.
        self.assertGreater(res["gamma_aligned_s"], 0.70 * gamma_true)
        self.assertLess(res["gamma_aligned_s"], 1.10 * gamma_true)
        unaligned_fit = fit_exponential_growth(res["t_env_ms"] * 1e-3, res["env_med"])
        self.assertGreater(res["gamma_aligned_s"], unaligned_fit.gamma)


if __name__ == "__main__":
    unittest.main()
