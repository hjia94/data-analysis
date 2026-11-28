"""Unit tests for the math/DSP helpers in data_analysis.viz."""

import unittest

import numpy as np

from data_analysis.viz import prepare_bdot_2d_fields, windowed_fft_amp
from data_analysis.viz.animations import _resolve_loop_frequencies


class TestVizMathHelpers(unittest.TestCase):
    def test_windowed_fft_amp_recovers_tone_frequency_and_amplitude(self):
        dt_s = 1e-5
        t = np.arange(1000) * dt_s
        sig = 0.5 + 1.75 * np.cos(2.0 * np.pi * 4000.0 * t + 0.4)
        mask = (t >= 0.002) & (t <= 0.008)
        f_khz, amp = windowed_fft_amp(sig, dt_s, mask=mask, pad_factor=8)
        i_pk = int(np.argmax(amp))
        self.assertAlmostEqual(float(f_khz[i_pk]), 4.0, delta=0.05)
        self.assertAlmostEqual(float(amp[i_pk]), 1.75, delta=0.05)

    def test_prepare_bdot_2d_fields_units_and_phase_alignment(self):
        xs = np.linspace(-6.0, 6.0, 9)
        XX, YY = np.meshgrid(xs, xs)
        env = np.exp(-0.5 * ((np.hypot(XX, YY) - 2.5) / 1.4) ** 2)
        ph = np.exp(-1j * np.arctan2(YY, XX))
        Jz = 1500.0 * env * ph * np.exp(0.6j)
        Bx = 2.5e-5 * env * ph
        By = -2.5e-5j * env * ph

        prep_ka = prepare_bdot_2d_fields(Bx, By, Jz, calibrated=True, phase_align=True)
        self.assertAlmostEqual(prep_ka["j_scale"], 1e-3)
        self.assertAlmostEqual(float(np.nanmax(np.real(prep_ka["Jz_plot"]))), prep_ka["j_vlim"], places=6)

        prep_a = prepare_bdot_2d_fields(Bx, By, Jz * 0.1, calibrated=True, phase_align=True)
        self.assertAlmostEqual(prep_a["j_scale"], 1.0)

        prep_u = prepare_bdot_2d_fields(Bx, By, Jz, calibrated=False, phase_align=True)
        self.assertAlmostEqual(prep_u["j_vlim"], 1.0)

    def test_resolve_loop_frequencies_snaps_within_tol_and_warns_otherwise(self):
        f_eff, t_loop = _resolve_loop_frequencies(4.11, 6.15, seamless_loop=True)
        self.assertAlmostEqual(f_eff, 1.5 * 4.11, places=9)
        self.assertAlmostEqual(t_loop, 2.0 / 4.11, places=9)

        with self.assertWarns(UserWarning):
            f_eff_11, _ = _resolve_loop_frequencies(4.0, 4.4, seamless_loop=True)
        self.assertAlmostEqual(f_eff_11, 4.4, places=9)


if __name__ == "__main__":
    unittest.main()
