"""Headless unit tests for data_analysis.viz.panels, animations, and Jun2026_figures.

All tests run on synthetic NumPy arrays under the Agg backend with zero
dependence on ``D:\\data\\`` or external calibration files.
"""

from __future__ import annotations

import inspect
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

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
import Jun2026_figures as jfig
import Jun2026_plot as jpl


def _synthetic_dipole_grid(n=9):
    """Return a clean synthetic m=-1 complex 2-D (xs, ys, Bx, By, Jz) grid."""
    xs = np.linspace(-6.0, 6.0, n)
    ys = np.linspace(-6.0, 6.0, n)
    XX, YY = np.meshgrid(xs, ys)
    RR = np.hypot(XX, YY)
    TH = np.arctan2(YY, XX)
    env = np.exp(-0.5 * ((RR - 2.5) / 1.4) ** 2)
    Jz = 1500.0 * env * np.exp(-1j * TH) * np.exp(1j * 0.6)  # A/m^2
    Bx = 2.5e-5 * env * np.exp(-1j * TH)                      # T (~0.25 G)
    By = -2.5e-5j * env * np.exp(-1j * TH)                    # T
    return xs, ys, Bx, By, Jz


class TestVizPanels(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_windowed_fft_amp_recovers_tone_frequency_and_amplitude(self):
        dt_s = 1e-5  # 100 kHz
        t = np.arange(1000) * dt_s
        f0_hz = 4000.0
        amp_true = 1.75
        sig = 0.5 + amp_true * np.cos(2.0 * np.pi * f0_hz * t + 0.4)
        mask = (t >= 0.002) & (t <= 0.008)
        f_khz, amp = windowed_fft_amp(sig, dt_s, mask=mask, pad_factor=8)
        i_pk = int(np.argmax(amp))
        self.assertAlmostEqual(float(f_khz[i_pk]), 4.0, delta=0.05)
        self.assertAlmostEqual(float(amp[i_pk]), amp_true, delta=0.05)

        # Mask=None works identically on pre-sliced window
        f2, a2 = windowed_fft_amp(sig[mask], dt_s, mask=None, pad_factor=4)
        self.assertAlmostEqual(float(f2[int(np.argmax(a2))]), 4.0, delta=0.05)

    def test_prepare_bdot_2d_fields_units_and_phase_alignment(self):
        _, _, Bx, By, Jz = _synthetic_dipole_grid()

        # Calibrated kA/m^2 (>= 1e3 A/m^2)
        prep_ka = prepare_bdot_2d_fields(Bx, By, Jz, calibrated=True, phase_align=True)
        self.assertIn("kA/m", prep_ka["cb_label"])
        self.assertAlmostEqual(prep_ka["j_scale"], 1e-3)
        # Peak Re[J_par] after phase alignment equals +max(|J_par|)
        self.assertAlmostEqual(
            float(np.nanmax(np.real(prep_ka["Jz_plot"]))),
            prep_ka["j_vlim"],
            places=6,
        )

        # Calibrated A/m^2 (< 1e3 A/m^2)
        prep_a = prepare_bdot_2d_fields(Bx, By, Jz * 0.1, calibrated=True, phase_align=True)
        self.assertIn("A/m", prep_a["cb_label"])
        self.assertNotIn("kA", prep_a["cb_label"])

        # Uncalibrated normalizes to [-1, 1]
        prep_u = prepare_bdot_2d_fields(Bx, By, Jz, calibrated=False, phase_align=True)
        self.assertAlmostEqual(prep_u["j_vlim"], 1.0)
        self.assertIn("J_{\\max}", prep_u["cb_label"])

    def test_draw_stacked_traces_single_and_multi_axes(self):
        t = np.linspace(-0.5, 5.5, 200)
        traces = [
            {"t": t, "y": 0.5 + 0.05 * np.sin(2.0 * np.pi * t), "label": "Run A", "color": "#2563eb"},
            {"t": t, "y": 0.8 + 0.10 * np.cos(2.0 * np.pi * t), "label": "Run B", "color": "#dc2626"},
            (t, 0.6 + 0.08 * np.sin(4.0 * np.pi * t), "Run C", "#0d9488"),
        ]

        # 1. Single-Axes vertically offset mode
        fig1, ax1 = plt.subplots()
        lines1 = draw_stacked_traces(ax1, traces, xlim=(-0.5, 5.5), ylabel=r"$I_{\mathrm{sat}}$ [V]", title="Single Ax")
        self.assertEqual(len(lines1), 3)
        self.assertEqual(ax1.get_title(), "Single Ax")
        self.assertEqual(len(ax1.texts), 3)
        # Traces should be vertically separated (top trace median > bottom trace median)
        self.assertGreater(float(np.median(lines1[0].get_ydata())), float(np.median(lines1[-1].get_ydata())))
        fig1.canvas.draw()

        # 2. Multi-Axes sub-panel mode
        fig2, axs2 = plt.subplots(3, 1, sharex=True)
        lines2 = draw_stacked_traces(
            axs2, traces, xlim=(-0.5, 5.5), ylim=(0.2, 1.2), yticks=[0.4, 0.8],
            xlabel="Time [ms]", ylabel=r"$I_{\mathrm{sat}}$ [V]", title="Multi Ax",
        )
        self.assertEqual(len(lines2), 3)
        self.assertEqual(axs2[0].get_title(), "Multi Ax")
        self.assertEqual(axs2[1].get_ylabel(), r"$I_{\mathrm{sat}}$ [V]")
        self.assertEqual(axs2[2].get_xlabel(), "Time [ms]")
        fig2.canvas.draw()

        # Mismatched axes count raises ValueError
        with self.assertRaises(ValueError):
            draw_stacked_traces(axs2[:2], traces)

    def test_draw_psd_overlay_log_peaks_and_linear_harmonics(self):
        f = np.linspace(0.5, 20.0, 400)
        p1 = 1e-9 + 1e-5 * np.exp(-0.5 * ((f - 4.0) / 0.25) ** 2) + 3e-6 * np.exp(-0.5 * ((f - 8.0) / 0.25) ** 2)
        p2 = 1e-9 + 8e-6 * np.exp(-0.5 * ((f - 5.2) / 0.25) ** 2)

        # Log-y PSD with peak markers
        fig1, ax1 = plt.subplots()
        lines1 = draw_psd_overlay(
            ax1,
            [
                {"freq": f, "power": p1, "label": "800 G", "color": "#dc2626", "peak_freq": 4.0},
                {"freq": f, "power": p2, "label": "1000 G", "color": "#7c3aed", "peak_freq": 5.2},
            ],
            xlim=(0.5, 15.0), ylim=(1e-10, 5e-5), logy=True, title="Log PSD",
        )
        self.assertEqual(len(lines1), 2)
        self.assertEqual(ax1.get_yscale(), "log")
        # 2 spectrum curves + 2 peak marker dots = 4 Line2D objects
        self.assertEqual(len(ax1.lines), 4)
        fig1.canvas.draw()

        # Linear-y normalized FFT with harmonic annotations
        fig2, ax2 = plt.subplots()
        lines2 = draw_psd_overlay(
            ax2,
            [
                {"freq": f, "power": p1 / p1.max(), "label": r"$I_{\mathrm{sat}}$", "color": "#1d4ed8"},
                {"freq": f, "power": p2 / p2.max(), "label": "B-dot", "color": "#dc2626", "ls": "--"},
            ],
            xlim=(0.0, 20.0), ylim=(0.0, 1.28), logy=False,
            harmonics={"f0": 4.0, "orders": (1, 2, 3, 4), "half_width": 0.5, "show_f0_value": True},
            title="Harmonics",
        )
        self.assertEqual(len(lines2), 2)
        self.assertEqual(ax2.get_yscale(), "linear")
        self.assertEqual(len(ax2.texts), 4)
        self.assertIn("4.0 kHz", ax2.texts[0].get_text())
        fig2.canvas.draw()

    def test_draw_ne_te_profile(self):
        x = np.linspace(-10.0, 10.0, 21)
        ne = 12.0 - 6.0 * np.exp(-(x / 3.0) ** 2)
        te = 2.5 - 1.0 * np.exp(-(x / 4.0) ** 2)
        fig, ax = plt.subplots()
        ax2, l1, l2 = draw_ne_te_profile(
            ax, x, ne, te, xlim=(-9.5, 9.5), ne_ylim_min=17.0, te_ylim_min=3.1,
            eta_e=0.45, title="Hollow Profile",
        )
        self.assertIsNot(ax2, ax)
        self.assertEqual(ax.get_title(), "Hollow Profile")
        self.assertGreaterEqual(ax.get_ylim()[1], 17.0)
        self.assertGreaterEqual(ax2.get_ylim()[1], 3.1)
        self.assertTrue(any("0.45" in t.get_text() for t in ax.texts))
        fig.canvas.draw()

    def test_draw_bdot_2d_map_calibrated_and_uncalibrated(self):
        xs, ys, Bx, By, Jz = _synthetic_dipole_grid()

        # Calibrated map with quiverkey, circles, markers, notes
        fig1, ax1 = plt.subplots()
        out1 = draw_bdot_2d_map(
            ax1, xs, ys, Bx, By, Jz, calibrated=True, phase_align=True,
            circles=[{"radius": 2.54, "label": '2" plate'}, {"radius": 5.08, "ls": "--"}],
            markers=[{"x": 1.0, "y": -2.0, "marker": "*", "label": "B-dot"}],
            note_lines=[r"$|\delta B_z|/|\delta B_\perp| = 0.08$"],
            title="Calibrated 2D",
        )
        self.assertIsNotNone(out1["im"])
        self.assertIsNotNone(out1["quiver"])
        self.assertIsNotNone(out1["colorbar"])
        self.assertIn("kA/m", out1["cb_label"])
        self.assertGreaterEqual(len(ax1.patches), 3)  # 2 circles + quiverkey box
        fig1.canvas.draw()

        # Uncalibrated map
        fig2, ax2 = plt.subplots()
        out2 = draw_bdot_2d_map(ax2, xs, ys, Bx, By, Jz, calibrated=False, add_colorbar=False)
        self.assertIsNone(out2["colorbar"])
        self.assertAlmostEqual(out2["j_vlim"], 1.0)
        fig2.canvas.draw()

    def test_draw_pulse_and_zoom(self):
        t_ms = np.linspace(-0.5, 5.5, 601)
        i2 = np.where(t_ms > 0.0, 12.0 + 3.0 * np.sin(2.0 * np.pi * 4.0 * t_ms), 0.0)
        i4 = np.where(t_ms > 0.0, 18.0 - 3.0 * np.sin(2.0 * np.pi * 4.0 * t_ms), 0.0)
        fig, (ax1, ax2) = plt.subplots(2, 1)
        res = draw_pulse_and_zoom(
            ax1, ax2, t_ms,
            [
                {"y": i2, "color": "#7c3aed", "full_label": "I2", "zoom_label": "dI2"},
                {"y": i4, "color": "#ea580c", "full_label": "I4", "zoom_label": "dI4", "zoom_ls": "--"},
            ],
            zoom_window_ms=(2.8, 3.8),
            t_window_ms=(-0.5, 5.5),
            zoom_note=r"$\gamma^2 = 0.99$",
            full_title="Full",
            zoom_title="Zoom",
        )
        self.assertEqual(len(res["full_lines"]), 2)
        self.assertEqual(len(res["zoom_lines"]), 2)
        # Zoomed traces must be zero-mean
        for line in res["zoom_lines"]:
            self.assertAlmostEqual(float(np.mean(line.get_ydata())), 0.0, places=6)
        fig.canvas.draw()

    def test_animate_lab_vs_plasma_frame_writes_gif_and_preview(self):
        xs, ys, Bx, By, Jz = _synthetic_dipole_grid(n=7)
        with tempfile.TemporaryDirectory() as tmp:
            gif_p = os.path.join(tmp, "mode.gif")
            prev_p = os.path.join(tmp, "preview.png")
            out = animate_lab_vs_plasma_frame(
                xs, ys, Bx, By, Jz,
                f_lab_khz=4.1, f_exb_khz=6.1, center=(0.0, 0.0),
                calibrated=True, seamless_loop=True,
                n_frames=4, fps=10, dpi=60, fine_n=33,
                title_prefix="Synth (800 G)",
                out_gif=gif_p, preview_png=prev_p,
            )
            self.assertEqual(out, gif_p)
            self.assertTrue(os.path.exists(gif_p))
            self.assertTrue(os.path.exists(prev_p))
            with Image.open(gif_p) as im:
                self.assertEqual(im.n_frames, 4)
            with Image.open(prev_p) as pim:
                self.assertGreater(pim.size[1], pim.size[0])

    def test_animation_loop_snapping_warns_instead_of_distorting_and_title_is_sign_aware(self):
        # Run-26-like ratio (~1.496) snaps cleanly to 3/2 in 2 lab turns with no warning
        f_eff, t_loop = _resolve_loop_frequencies(4.11, 6.15, seamless_loop=True)
        self.assertAlmostEqual(f_eff, 1.5 * 4.11, places=9)
        self.assertAlmostEqual(t_loop, 2.0 / 4.11, places=9)

        # Ratio 1.1 cannot snap within 5% in <=4 turns -> must warn and keep exact f_exb
        with self.assertWarns(UserWarning):
            f_eff_11, _ = _resolve_loop_frequencies(4.0, 4.4, seamless_loop=True)
        self.assertAlmostEqual(f_eff_11, 4.4, places=9)

        # Negative f_exb - f_lab must label plasma-frame rotation as CW (and never hardcode grad(Te))
        xs, ys, Bx, By, Jz = _synthetic_dipole_grid(n=7)
        captured_titles = []
        orig_set_title = plt.Axes.set_title

        def _spy_title(ax_self, label, *args, **kwargs):
            captured_titles.append(label)
            return orig_set_title(ax_self, label, *args, **kwargs)

        with mock.patch.object(plt.Axes, "set_title", _spy_title):
            animate_lab_vs_plasma_frame(
                xs, ys, Bx, By, Jz,
                f_lab_khz=4.0, f_exb_khz=2.0,
                seamless_loop=True, n_frames=2, dpi=50, fine_n=17, out_gif=None,
            )
        self.assertTrue(any("CW" in t and "2.0 kHz" in t for t in captured_titles), captured_titles)
        self.assertFalse(any("nabla T_e" in t for t in captured_titles), captured_titles)


class _FakeLapdRun:
    """Minimal in-memory mock of open_lapd() for synthetic figure tests."""

    def __init__(self, scopes):
        self._scopes = scopes

    def time_array(self, scope_name=None):
        return self._scopes[scope_name]["t_s"]

    def channel(self, chan, scope_name=None, shots=None):
        arr = self._scopes[scope_name]["channels"][chan]
        return np.asarray(arr)[None, :], self._scopes[scope_name]["t_s"]


class TestJun2026FiguresRegistryAndRecipes(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_registry_keys_and_cli_list_and_only(self):
        expected = {
            "fig1a_profiles",
            "fig1_em_instability",
            "fig2_sheath_closure",
            "fig3_mode_propagation",
            "fig2_1_puff_scaling",
            "fig2_2_b0_scaling",
            "fig2_3_shear_gating",
            "turbulence_steepening",
        }
        self.assertTrue(expected.issubset(set(jfig.FIGURES.keys())))

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = jfig.main(["--list"])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        for key in expected:
            self.assertIn(key, out)
            self.assertIn(jfig.FIGURES[key].output, out)

        # CLI --only dispatches the selected recipe with --out-dir
        called = []
        fake_spec = jfig.FigureSpec(
            lambda **kw: called.append(kw), "dummy.png", "Test", "Dummy test figure"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(jfig.FIGURES, {"dummy_fig": fake_spec}, clear=False):
                rc2 = jfig.main(["--only", "dummy_fig", "--out-dir", tmp])
            self.assertEqual(rc2, 0)
            self.assertEqual(len(called), 1)
            self.assertEqual(called[0]["save_fig"], os.path.join(tmp, "dummy.png"))

        # Unknown key raises KeyError in run_figures
        with self.assertRaises(KeyError):
            jfig.run_figures(["nonexistent_figure_key"])

    def test_backward_compatible_wrappers_preserve_signatures(self):
        for fname in (
            "plot_hollow_profiles",
            "plot_em_instability_overview",
            "plot_sheath_closure",
            "animate_mode_propagation",
            "plot_turbulence_steepening",
        ):
            sig_plot = inspect.signature(getattr(jpl, fname))
            sig_fig = inspect.signature(getattr(jfig, fname))
            self.assertEqual(sig_plot, sig_fig, f"Signature mismatch for {fname}")

    def test_synthetic_finding1_and_finding2_recipes_headless(self):
        import Jun2026_bdot_2d as jbd
        import Jun2026_IV as jiv
        import Jun2026_mach_line as jml
        import Jun2026_modes as jmd
        import Jun2026_onset as jon
        import Jun2026_profiles as jpr

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(jbd, "DEFAULT_BDOT_CAL", os.path.join(tmp, "nonexistent_cal.h5")):
            # 1. Synthetic profiles_summary_jun2026.npz
            x_cm = np.linspace(-10.0, 10.0, 21)
            r_abs = np.abs(x_cm - 1.0)
            ne_prof = (12.0e12 - 5.0e12 * np.exp(-(r_abs / 3.0) ** 2))[:, None]
            te_prof = (2.4 - 0.8 * np.exp(-(r_abs / 3.5) ** 2))[:, None]
            vp_prof = (8.0 + 4.0 * np.exp(-(r_abs / 3.0) ** 2))[:, None]
            prof_d = {}
            for ir, _, puff in jfig.SHEAR_GATING_PAIRS:
                p = f"{ir}__"
                prof_d[p + "x_cm"] = x_cm
                prof_d[p + "x_center_cm"] = np.float64(1.0)
                prof_d[p + "ref_sweep_idx"] = np.int64(0)
                prof_d[p + "ne"] = ne_prof
                prof_d[p + "Te"] = te_prof
                prof_d[p + "Vp"] = vp_prof
                prof_d[p + "dln_ne_dr"] = np.full((x_cm.size, 1), +0.35)
                prof_d[p + "dln_te_dr"] = np.full((x_cm.size, 1), +0.20)
                prof_d[p + "er_r_v_cm"] = np.full((x_cm.size, 1), +0.80)
                prof_d[p + "lambda"] = np.full((x_cm.size, 1), 3.5)
                prof_d[p + "ref_eta_e_steep"] = np.float64(0.42)
                prof_d[p + "gas_puff_ms"] = np.float64(puff)
            prof_p = os.path.join(tmp, jpr.OUT_NAME)
            np.savez(prof_p, **prof_d)

            # Test plot_hollow_profiles (via Jun2026_plot wrapper too)
            out_prof = os.path.join(tmp, "fig1a_profiles.png")
            jpl.plot_hollow_profiles(run_num="02", profiles_npz=prof_p, save_fig=out_prof, show=False)
            self.assertGreater(os.path.getsize(out_prof), 1000)

            # 2. Synthetic bdot_2d_jun2026.npz & modes_dispersion_jun2026.npz
            xs, ys, Bx, By, Jz = _synthetic_dipole_grid(n=9)
            bdot_p = os.path.join(tmp, jbd.OUT_NAME)
            np.savez(
                bdot_p,
                **{
                    "26__x_cm": xs, "26__y_cm": ys,
                    "26__h1__Bx": Bx, "26__h1__By": By, "26__h1__Jz": Jz,
                    "26__b0_g": np.float64(800.0), "26__f0_hz": np.float64(4100.0),
                    "26__h1__bz_over_bperp": np.float64(0.08),
                    "26__h1__center_x_cm": np.float64(0.0),
                    "26__h1__center_y_cm": np.float64(0.0),
                    "26__bdot_calibrated": np.bool_(True),
                },
            )
            modes_p = os.path.join(tmp, jmd.OUT_NAME)
            np.savez(
                modes_p,
                **{
                    "disp__03__beta_e_over_me_mi": np.float64(2.1),
                    "disp__03__f_doppler_exb_hz": np.float64(6150.0),
                    "bias__02__g2_I2_I4": np.float64(0.99),
                    "bias__02__dphi_I4_minus_I2_deg": np.float64(-179.0),
                },
            )

            # Test plot_em_instability_overview + verify dB_perp/B0 comes from calibrated 2-D h1 map
            t_s = np.linspace(-0.5e-3, 5.5e-3, 2400)
            lp_sig = 0.8 + 0.15 * np.cos(2.0 * np.pi * 4100.0 * t_s)
            bd_sig = 0.25 * np.sin(2.0 * np.pi * 4100.0 * t_s)
            fake_run_26 = _FakeLapdRun({
                "lpscope": {"t_s": t_s, "channels": {"C2": lp_sig}},
                "bdotscope": {"t_s": t_s, "channels": {"C1": bd_sig}},
            })
            fake_pos = np.array([(716, 1.0, -2.0)], dtype=[("shot_num", int), ("x", float), ("y", float)])
            out_em = os.path.join(tmp, "fig1_em_instability.png")
            with mock.patch.object(jfig, "open_lapd", return_value=fake_run_26), \
                 mock.patch.object(jiv, "read_lp_positions", return_value=fake_pos):
                jfig.plot_em_instability_overview(
                    run_num="26", shot=716, ifn="dummy26.hdf5",
                    bdot_npz=bdot_p, modes_npz=modes_p, save_fig=out_em, show=False,
                )
            self.assertGreater(os.path.getsize(out_em), 1000)

            # Test plot_sheath_closure
            tb_s = np.linspace((jmd.BIASSCOPE_OFFSET_MS - 0.5) * 1e-3, (jmd.BIASSCOPE_OFFSET_MS + 5.5) * 1e-3, 2500)
            t_rel = tb_s * 1e3 - jmd.BIASSCOPE_OFFSET_MS
            i2_v = np.where(t_rel > 0.0, 0.12 + 0.03 * np.sin(2.0 * np.pi * 4.0 * t_rel), 0.0)
            i4_v = np.where(t_rel > 0.0, 0.16 - 0.03 * np.sin(2.0 * np.pi * 4.0 * t_rel), 0.0)
            fake_run_02 = _FakeLapdRun({"biasscope": {"t_s": tb_s, "channels": {"C1": i2_v, "C2": i4_v}}})
            out_sh = os.path.join(tmp, "fig2_sheath_closure.png")
            with mock.patch.object(jfig, "open_lapd", return_value=fake_run_02):
                jfig.plot_sheath_closure(
                    run_num="02", shot=100, ifn="dummy02.hdf5",
                    modes_npz=modes_p, save_fig=out_sh, show=False,
                )
            self.assertGreater(os.path.getsize(out_sh), 1000)

            # Test animate_mode_propagation
            out_gif = os.path.join(tmp, "fig3_mode_propagation.gif")
            res_gif = jfig.animate_mode_propagation(
                run_num="26", bdot_npz=bdot_p, modes_npz=modes_p,
                n_frames=4, fps=10, dpi=60, save_fig=out_gif,
            )
            self.assertEqual(res_gif, out_gif)
            self.assertGreater(os.path.getsize(out_gif), 1000)

            # 3. Synthetic onset_summary_jun2026.npz + mach_line_rescaled_jun2026.npz for Finding #2
            psd_f = np.linspace(500.0, 16000.0, 200)
            on_d = {}
            all_onset_runs = [r for r, _ in jfig.PUFF_ALL_RUNS] + [r for r, _, _ in jfig.B0_SCAN_RUNS]
            for rnum in all_onset_runs:
                is_stable = rnum in ("06", "05")
                f_pk = 3500.0 if rnum == "28" else (4800.0 if rnum == "30" else 4100.0)
                on_d[f"{rnum}__shot_nums"] = np.array([10, 20, 30])
                on_d[f"{rnum}__t_onset_ms"] = np.array([np.nan, np.nan, np.nan] if is_stable else [1.8, 2.0, 2.2])
                on_d[f"{rnum}__peak_rel_rms"] = np.array([0.005, 0.006, 0.007] if is_stable else [0.09, 0.10, 0.11])
                on_d[f"{rnum}__unstable_frac"] = np.float64(0.0 if is_stable else 1.0)
                on_d[f"{rnum}__t_onset_med_ms"] = np.float64(np.nan if is_stable else 2.0)
                on_d[f"{rnum}__peak_rel_rms_med"] = np.float64(0.006 if is_stable else 0.10)
                on_d[f"{rnum}__psd_freq_hz"] = psd_f
                on_d[f"{rnum}__psd_mean"] = 1e-9 + (0.0 if is_stable else 5e-6) * np.exp(-0.5 * ((psd_f - f_pk) / 300.0) ** 2)
                on_d[f"{rnum}__gamma_med_s"] = np.float64(np.nan if is_stable else 4500.0)
                on_d[f"{rnum}__gamma_q25_s"] = np.float64(np.nan if is_stable else 3800.0)
                on_d[f"{rnum}__gamma_q75_s"] = np.float64(np.nan if is_stable else 5200.0)
                on_d[f"{rnum}__f_peak_hz"] = np.float64(f_pk)
                on_d[f"{rnum}__scope"] = np.array("scope" if rnum in ("28", "27", "30") else "machscope")
                on_d[f"{rnum}__chan"] = np.array("C5" if rnum in ("28", "27", "30") else "C2")
                open(os.path.join(tmp, f"{rnum}-dummy.hdf5"), "wb").close()
            onset_p = os.path.join(tmp, jon.OUT_NAME)
            np.savez(onset_p, **on_d)

            fake_isat_run = _FakeLapdRun({
                "machscope": {"t_s": t_s, "channels": {"C2": lp_sig}},
                "scope": {"t_s": t_s, "channels": {"C5": lp_sig}},
            })
            out_puff = os.path.join(tmp, "fig2_1_puff_scaling.png")
            out_b0 = os.path.join(tmp, "fig2_2_b0_scaling.png")
            with mock.patch.object(jfig, "open_lapd", return_value=fake_isat_run):
                jfig.plot_puff_scaling(data_dir=tmp, onset_npz=onset_p, save_fig=out_puff, show=False)
                jfig.plot_b0_scaling(data_dir=tmp, onset_npz=onset_p, save_fig=out_b0, show=False)
            self.assertGreater(os.path.getsize(out_puff), 1000)
            self.assertGreater(os.path.getsize(out_b0), 1000)

            # Synthetic mach_line_rescaled_jun2026.npz for plot_shear_gating
            # Note: stability_linear_jun2026.npz does NOT exist in tmp, proving plot_shear_gating
            # computes strictly pre-onset CWM from ref_sweep_idx without ever reading near_swp!
            r_cm = np.linspace(1.0, 12.0, 12)
            mach_p = os.path.join(tmp, jml.RESCALED_OUT_NAME)
            np.savez(
                mach_p,
                **{
                    "11__r_cm": r_cm,
                    "11__ref_bin_idx": np.int64(0),
                    "11__dv_rot_km_s": (-1.5 * np.exp(-((r_cm - 4.0) / 2.5) ** 2))[:, None],
                    "11__v_ExB_r_km_s": (1.6 * np.exp(-((r_cm - 4.0) / 2.5) ** 2))[:, None],
                },
            )
            out_sg = os.path.join(tmp, "fig2_3_shear_gating.png")
            jfig.plot_shear_gating(
                data_dir=tmp, mach_npz=mach_p, profiles_npz=prof_p, onset_npz=onset_p,
                save_fig=out_sg, show=False,
            )
            self.assertGreater(os.path.getsize(out_sg), 1000)

    def test_resolve_onset_channel_and_dynamic_shear_gating_title(self):
        # 1. _resolve_onset_channel reads from onset npz, else falls back to HDF5 channel_wiring
        with tempfile.TemporaryDirectory() as tmp:
            p_has = os.path.join(tmp, "onset_has.npz")
            np.savez(p_has, **{"99__scope": np.array("customscope"), "99__chan": np.array("C7")})
            with np.load(p_has, allow_pickle=True) as onf:
                self.assertEqual(jfig._resolve_onset_channel(onf, "x.hdf5", "99"), ("customscope", "C7"))

            p_empty = os.path.join(tmp, "onset_empty.npz")
            np.savez(p_empty, dummy=np.int64(0))
            fake_wiring = {("altscope", "C4"): "Isat, Mach@P33-X+"}
            with np.load(p_empty, allow_pickle=True) as onf, \
                 mock.patch("data_analysis.io.probe_map.channel_wiring", return_value=fake_wiring):
                self.assertEqual(jfig._resolve_onset_channel(onf, "x.hdf5", "99"), ("altscope", "C4"))

        # 2. plot_shear_gating panel (a) title tracks profile_run (e.g. Run 12 -> 35 ms)
        with tempfile.TemporaryDirectory() as tmp:
            r_cm = np.linspace(1.0, 12.0, 12)
            mach_p = os.path.join(tmp, "mach.npz")
            np.savez(
                mach_p,
                **{
                    "12__r_cm": r_cm,
                    "12__ref_bin_idx": np.int64(0),
                    "12__dv_rot_km_s": (-1.2 * np.exp(-((r_cm - 4.0) / 2.5) ** 2))[:, None],
                    "12__v_ExB_r_km_s": (1.3 * np.exp(-((r_cm - 4.0) / 2.5) ** 2))[:, None],
                },
            )
            x_cm = np.linspace(-10.0, 10.0, 21)
            r_abs = np.abs(x_cm - 1.0)
            prof_p = os.path.join(tmp, "prof.npz")
            np.savez(
                prof_p,
                **{
                    "03__x_cm": x_cm, "03__x_center_cm": np.float64(1.0), "03__ref_sweep_idx": np.int64(0),
                    "03__ne": (12e12 - 5e12 * np.exp(-(r_abs / 3.0) ** 2))[:, None],
                    "03__Te": (2.4 - 0.8 * np.exp(-(r_abs / 3.5) ** 2))[:, None],
                    "03__Vp": (8.0 + 4.0 * np.exp(-(r_abs / 3.0) ** 2))[:, None],
                    "03__dln_ne_dr": np.full((x_cm.size, 1), +0.35),
                    "03__dln_te_dr": np.full((x_cm.size, 1), +0.20),
                    "03__er_r_v_cm": np.full((x_cm.size, 1), +0.80),
                    "03__lambda": np.full((x_cm.size, 1), 3.5),
                },
            )
            onset_p = os.path.join(tmp, "onset.npz")
            np.savez(onset_p, **{"03__unstable_frac": np.float64(1.0), "03__gamma_med_s": np.float64(3500.0)})
            captured_titles = []
            orig_set_title = plt.Axes.set_title

            def _spy_title(ax_self, label, *args, **kwargs):
                captured_titles.append(label)
                return orig_set_title(ax_self, label, *args, **kwargs)

            with mock.patch.object(plt.Axes, "set_title", _spy_title):
                jfig.plot_shear_gating(
                    data_dir=tmp, mach_npz=mach_p, profiles_npz=prof_p, onset_npz=onset_p,
                    pairs=(("03", "12", 35),), profile_run="12", save_fig=False, show=False,
                )
            self.assertIn("(a) Azimuthal Flow Profile (35 ms)", captured_titles)

    def test_stability_print_summary_formats_stable_nan_as_dashes(self):
        import Jun2026_stability as jst

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "stab.npz")
            np.savez(
                p,
                l_par_m=np.float64(14.0), wall_width_cm=np.float64(4.0),
                **{
                    "puff__06__gas_puff_ms": np.float64(23.0),
                    "puff__06__exp_unstable": np.bool_(False),
                    "puff__06__exp_gamma_s": np.float64(np.nan),
                    "puff__06__exp_f_plasma_exb_hz": np.float64(np.nan),
                    "puff__06__exp_f_lab_hz": np.float64(np.nan),
                    "puff__06__near_sweep_t_ms": np.float64(1.0),
                    "puff__06__near_sweep_regime": np.array("pre_onset"),
                    "puff__06__rad_kh_sens_max_gamma_s": np.float64(0.0),
                    "puff__06__ax_ins_top_f_hz": np.float64(0.0),
                    "puff__06__ax_ins_top_gamma_s": np.float64(-100.0),
                    "puff__06__ax_cwm_ref_f_hz": np.float64(np.nan),
                    "puff__06__ax_cwm_ref_gamma_s": np.float64(np.nan),
                    "puff__06__ax_cwm_ref_dphi_ab_deg": np.float64(np.nan),
                    "puff__06__ax_cwm_near_f_hz": np.float64(np.nan),
                    "puff__06__ax_cwm_near_gamma_s": np.float64(np.nan),
                    "puff__06__ax_cwm_near_dphi_ab_deg": np.float64(np.nan),
                    "puff__06__ax_cwm_near_gsh1_gamma_s": np.float64(np.nan),
                    "puff__06__ax_cwm_near_nu15_gamma_s": np.float64(np.nan),
                    "puff__06__rad_cwm_nosh_f_lab_hz": np.float64(4000.0),
                    "puff__06__rad_cwm_nosh_gamma_s": np.float64(8000.0),
                    "puff__06__rad_cwm_sh_f_lab_hz": np.float64(4000.0),
                    "puff__06__rad_cwm_sh_gamma_s": np.float64(1000.0),
                    "puff__06__rad_cwm_sh_cent_f_lab_hz": np.float64(4000.0),
                    "puff__06__rad_cwm_sh_cent_gamma_s": np.float64(1300.0),
                },
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                jst.print_summary(p)
            out = buf.getvalue()
            self.assertNotIn("nan", out.lower())
            self.assertIn("---", out)

    def test_label_hygiene_no_fixed_bias_isat(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        paths = [
            os.path.join(repo_root, "src", "data_analysis", "viz", "panels.py"),
            os.path.join(repo_root, "src", "data_analysis", "viz", "animations.py"),
            os.path.join(repo_root, "experiments", "ucla-lapd", "Jun-2026", "Jun2026_figures.py"),
        ]
        for p in paths:
            with open(p, "r", encoding="utf-8") as fh:
                text = fh.read().lower()
            self.assertNotIn("fixed-bias", text, f"Forbidden 'fixed-bias' text found in {p}")


if __name__ == "__main__":
    unittest.main()
