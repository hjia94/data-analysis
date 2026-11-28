"""Unit tests for the math/DSP helpers and headless smoke coverage in data_analysis.viz."""

import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.viz import (
    animate_lab_vs_plasma_frame,
    draw_bdot_2d_map,
    draw_ne_te_profile,
    draw_psd_overlay,
    draw_pulse_and_zoom,
    draw_stacked_traces,
    prepare_bdot_2d_fields,
    windowed_fft_amp,
)
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
            f_eff_11, t_loop_11 = _resolve_loop_frequencies(4.0, 4.4, seamless_loop=True)
        self.assertAlmostEqual(f_eff_11, 4.4, places=9)
        self.assertLessEqual(t_loop_11 * 4.0, 4.0 + 1e-9)

        # Near-resonant ratio must cap t_loop_ms to <= max_loop_turns lab periods
        with self.assertWarns(UserWarning):
            f_eff_102, t_loop_102 = _resolve_loop_frequencies(4.0, 4.08, seamless_loop=True, max_loop_turns=4)
        self.assertAlmostEqual(f_eff_102, 4.08, places=9)
        self.assertAlmostEqual(t_loop_102, 4.0 / 4.0, places=9)

    def test_headless_smoke_all_painters_and_animation(self):
        t = np.linspace(0.0, 5.0, 200)
        y1 = 10.0 + np.sin(2.0 * np.pi * 2.0 * t)
        y2 = 5.0 + 0.5 * np.cos(2.0 * np.pi * 2.0 * t)

        # draw_stacked_traces (single-axes & multi-axes), draw_psd_overlay, draw_ne_te_profile
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        draw_stacked_traces(axes[0, 0], [(t, y1, "A"), (t, y2, "B")], title="stack")
        draw_psd_overlay(
            axes[0, 1],
            [{"freq": np.linspace(0.5, 20.0, 64), "power": np.linspace(1.0, 0.05, 64), "label": "S", "peak_freq": 4.0}],
            harmonics=4.0, title="psd",
        )
        x_cm = np.linspace(-6.0, 6.0, 13)
        draw_ne_te_profile(axes[1, 0], x_cm, 1.0 + 0.05 * x_cm ** 2, 3.0 - 0.03 * x_cm ** 2, eta_e=-0.8)
        draw_pulse_and_zoom(
            axes[1, 1], axes[0, 0], t,
            [{"y": y1, "label": "I1"}, {"y": y2, "label": "I2"}],
            zoom_window_ms=(2.0, 3.0),
        )
        plt.close(fig)

        fig2, (ax_s1, ax_s2) = plt.subplots(2, 1)
        draw_stacked_traces([ax_s1, ax_s2], [(t, y1, "A"), (t, y2, "B")], ylabel="V")
        plt.close(fig2)

        # draw_bdot_2d_map + animate_lab_vs_plasma_frame
        xs = np.linspace(-6.0, 6.0, 7)
        XX, YY = np.meshgrid(xs, xs)
        env = np.exp(-0.5 * ((np.hypot(XX, YY) - 2.5) / 1.5) ** 2)
        ph = np.exp(-1j * np.arctan2(YY, XX))
        Jz = 1200.0 * env * ph
        Bx = 2e-5 * env * ph
        By = -2e-5j * env * ph

        fig3, ax3 = plt.subplots()
        res_map = draw_bdot_2d_map(
            ax3, xs, xs, Bx, By, Jz,
            circles=[{"radius": 2.5, "label": "2 in"}],
            markers=[{"x": 0.0, "y": 0.0, "label": "center"}],
            note_lines=["test"],
        )
        self.assertIn("im", res_map)
        plt.close(fig3)

        with tempfile.TemporaryDirectory() as tmp:
            gif_path = os.path.join(tmp, "mode.gif")
            prev_path = os.path.join(tmp, "preview.png")
            out = animate_lab_vs_plasma_frame(
                xs, xs, Bx, By, Jz,
                f_lab_khz=1.0, f_exb_khz=2.0,
                n_frames=16, fine_n=17, dpi=40,
                out_gif=gif_path, preview_png=prev_path,
            )
            self.assertEqual(out, gif_path)
            self.assertTrue(os.path.isfile(gif_path) and os.path.getsize(gif_path) > 0)
            self.assertTrue(os.path.isfile(prev_path) and os.path.getsize(prev_path) > 0)


if __name__ == "__main__":
    unittest.main()
