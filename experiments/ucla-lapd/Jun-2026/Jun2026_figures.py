"""Registry-driven publication and presentation figures for the Jun-2026 LAPD campaign.

Each figure recipe loads saved campaign ``.npz`` (and, where needed, single
raw HDF5 shots via :func:`data_analysis.io.open_lapd`), composes campaign-agnostic
painters from :mod:`data_analysis.viz.panels` and :mod:`data_analysis.viz.animations`,
and saves into ``<OUTPUT_ROOT>/figures/Jun2026/`` via
:func:`data_analysis.viz.plot_utils.finalize_figure`.

Usage::

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_figures.py --list
    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_figures.py --only fig1_em_instability fig2_1_puff_scaling
    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_figures.py --all --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from data_analysis.io import open_lapd
from data_analysis.plasma.stability import (
    fit_smooth_radial_profile,
    solve_radial_cylindrical_modes,
)
from data_analysis.signal.core import downsample_blockmean
from data_analysis.viz import (
    animate_lab_vs_plasma_frame,
    configure_publication_style,
    draw_bdot_2d_map,
    draw_ne_te_profile,
    draw_psd_overlay,
    draw_pulse_and_zoom,
    draw_stacked_traces,
    fig_path,
    finalize_figure,
    windowed_fft_amp,
)

FIG_SUBDIR = "Jun2026"

# Default run tables for Finding #2 scaling figures
PUFF_SHOW_RUNS = (
    ("06", 23, "#64748b"),  # slate gray (stable)
    ("00", 24, "#2563eb"),  # blue (threshold)
    ("02", 30, "#dc2626"),  # red (peak growth)
    ("03", 35, "#d97706"),  # amber (earliest onset, highest f0)
    ("05", 45, "#0d9488"),  # teal (quenched)
)
PUFF_ALL_RUNS = (
    ("06", 23), ("00", 24), ("01", 25), ("02", 30),
    ("03", 35), ("04", 40), ("05", 45),
)
B0_SCAN_RUNS = (
    ("28", 600, "#2563eb"),   # blue
    ("27", 800, "#dc2626"),   # red
    ("30", 1000, "#7c3aed"),  # purple
)
SHEAR_GATING_PAIRS = (
    ("06", "08", 23),
    ("00", "09", 24),
    ("01", "10", 25),
    ("02", "11", 30),
    ("03", "12", 35),
    ("04", "13", 40),
    ("05", "14", 45),
)


# =========================================================================== #
#  Campaign I/O helpers
# =========================================================================== #

def _resolve_save(save_fig, name, *, ext=".png"):
    """Resolve ``save_fig`` (True -> centralized Jun2026 path, str -> explicit, False -> None)."""
    path = fig_path(name, FIG_SUBDIR, ext=ext) if save_fig is True else (save_fig or None)
    if path:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
    return path


def _run_hdf5(data_dir, run_num):
    """Locate the single HDF5 file for ``run_num`` inside ``data_dir``."""
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    if len(hits) > 1:
        raise ValueError(f"multiple HDF5 files for run {run_num}: {hits}")
    return hits[0]


def _shot_xy(pos, shot):
    """Return ``(x_cm, y_cm)`` for integer ``shot`` from a ``ProbePositions`` or array."""
    arr = pos.pos_array if hasattr(pos, "pos_array") else pos
    shots = np.asarray(arr["shot_num"], dtype=int)
    hits = np.flatnonzero(shots == int(shot))
    if hits.size == 0:
        raise ValueError(f"shot {shot} not found in motion positions array")
    i = int(hits[0])
    return float(arr["x"][i]), float(arr["y"][i])


def _load_bdot_2d_snapshot(bdot_npz_path, run_num):
    """Load complex ``h1`` 2-D ``(xs, ys, Bx, By, Jz, meta)`` from ``bdot_2d_jun2026.npz``."""
    with np.load(bdot_npz_path, allow_pickle=True) as bd:
        p = f"{run_num}__"
        xs = np.asarray(bd[p + "x_cm"], dtype=float)
        ys = np.asarray(bd[p + "y_cm"], dtype=float)
        Bx = np.asarray(bd[p + "h1__Bx"], dtype=complex)
        By = np.asarray(bd[p + "h1__By"], dtype=complex)
        Jz = np.asarray(bd[p + "h1__Jz"], dtype=complex)
        if Bx.shape == (xs.size, ys.size) and xs.size != ys.size:
            Bx, By, Jz = Bx.T, By.T, Jz.T
        meta = {
            "b0_g": float(bd[p + "b0_g"]),
            "f0_hz": float(bd[p + "f0_hz"]),
            "bz_over_bperp": float(bd[p + "h1__bz_over_bperp"]),
            "center_x_cm": float(bd[p + "h1__center_x_cm"]),
            "center_y_cm": float(bd[p + "h1__center_y_cm"]),
            "bdot_calibrated": bool(bd[p + "bdot_calibrated"]),
        }
    return xs, ys, Bx, By, Jz, meta


def _pick_representative_shot(onf, run_num):
    """Pick the shot closest to the ensemble median onset time and peak RMS."""
    sh = onf[f"{run_num}__shot_nums"]
    t_on = onf[f"{run_num}__t_onset_ms"]
    pk = onf[f"{run_num}__peak_rel_rms"]
    un_frac = float(onf[f"{run_num}__unstable_frac"])
    med_t = float(onf[f"{run_num}__t_onset_med_ms"])
    med_pk = float(onf[f"{run_num}__peak_rel_rms_med"])
    if un_frac >= 0.5 and np.isfinite(med_t):
        ok = np.isfinite(t_on) & np.isfinite(pk)
        score = np.abs(t_on[ok] - med_t) / 0.25 + np.abs(pk[ok] - med_pk) / max(med_pk, 1e-3)
        idx = int(np.where(ok)[0][np.argmin(score)])
    else:
        ok = np.isfinite(pk)
        idx = int(np.where(ok)[0][np.argmin(np.abs(pk[ok] - med_pk))])
    return int(sh[idx]), float(t_on[idx]), float(pk[idx])


def _read_scope_shot(hdf5_path, scope, chan, shot_num, *, t_lim_ms=(-0.5, 5.5), target_dt_s=5e-6):
    """Read one downsampled voltage trace ``(t_ms, v_ds)`` from an LAPD HDF5 file."""
    run = open_lapd(hdf5_path)
    t_s = np.asarray(run.time_array(scope_name=scope), dtype=float)
    stack, _ = run.channel(chan, scope_name=scope, shots=[int(shot_num)])
    if stack is None or stack.size == 0:
        raise ValueError(f"Could not read {scope}/{chan} shot {shot_num} from {hdf5_path}")
    v_raw = np.asarray(stack[0], dtype=float)
    m = (t_s >= t_lim_ms[0] * 1e-3) & (t_s <= t_lim_ms[1] * 1e-3)
    t_sub = t_s[m]
    v_sub = v_raw[m]
    dt_raw = float(t_sub[1] - t_sub[0])
    q = max(1, int(round(float(target_dt_s) / dt_raw)))
    nblk = t_sub.size // q
    t_ms = t_sub[:nblk * q].reshape(nblk, q).mean(axis=1) * 1e3
    v_ds = v_sub[:nblk * q].reshape(nblk, q).mean(axis=1)
    return t_ms, v_ds


# =========================================================================== #
#  Finding #1 figures (hollow profiles, EM instability, sheath closure, GIF)
# =========================================================================== #

def plot_hollow_profiles(
    run_num="02",
    *,
    data_dir=None,
    profiles_npz=None,
    x_lim=(-9.5, 9.5),
    save_fig=True,
    show=False,
):
    """Clean dual-axis hollow-column ``n_e(x)`` and ``T_e(x)`` radial profiles."""
    import Jun2026_profiles as jpr

    ddir = data_dir or jpr.DATA_DIR
    prof_path = profiles_npz or os.path.join(ddir, jpr.OUT_NAME)
    with np.load(prof_path, allow_pickle=True) as pf:
        p = f"{run_num}__"
        x_cm = np.asarray(pf[p + "x_cm"], dtype=float)
        xc_cm = float(pf[p + "x_center_cm"])
        ir = int(pf[p + "ref_sweep_idx"])
        ne = np.asarray(pf[p + "ne"][:, ir], dtype=float) * 1e-12  # 10^18 m^-3
        te = np.asarray(pf[p + "Te"][:, ir], dtype=float)
        eta_e = float(pf[p + "ref_eta_e_steep"]) if (p + "ref_eta_e_steep") in pf.files else np.nan
        puff_ms = float(pf[p + "gas_puff_ms"]) if (p + "gas_puff_ms") in pf.files else np.nan

    puff_str = f", {puff_ms:g} ms" if np.isfinite(puff_ms) else ""
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    draw_ne_te_profile(
        ax, x_cm - xc_cm, ne, te,
        xlim=x_lim, ne_ylim_min=17.0, te_ylim_min=3.1,
        eta_e=eta_e, title=f"Profiles (Run {run_num}{puff_str})",
    )
    save_name = "fig1a_profiles" if str(run_num) == "02" else f"fig1a_profiles_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show)


def plot_em_instability_overview(
    run_num="26",
    shot=716,
    *,
    bdot_ch="C1",
    lp_ch="C2",
    matched_puff_run="03",
    data_dir=None,
    ifn=None,
    bdot_npz=None,
    modes_npz=None,
    t_window_ms=(-0.5, 5.5),
    fft_window_ms=(2.0, 4.8),
    fmax_khz=20.0,
    pad_factor=8,
    save_fig=True,
    show=False,
):
    """4-panel electromagnetic instability overview (default: Run 26, shot 716).

    Panel (d) computes ``delta B_perp / B_0`` directly from the plotted
    calibrated ``f_0`` 2-D map (``max`` time-RMS = ``2.9e-4`` on Run 26).
    """
    import Jun2026_IV as jiv
    import Jun2026_bdot_2d as jbd
    import Jun2026_modes as jmd

    ddir = data_dir or jbd.DATA_DIR
    hdf5_path = ifn or _run_hdf5(ddir, run_num)
    bdot_path = bdot_npz or os.path.join(ddir, jbd.OUT_NAME)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    run = open_lapd(hdf5_path)
    tl = np.asarray(run.time_array(scope_name="lpscope"), dtype=float) * 1e3
    tb = np.asarray(run.time_array(scope_name="bdotscope"), dtype=float) * 1e3
    dt_l, dt_b = float(tl[1] - tl[0]) * 1e-3, float(tb[1] - tb[0]) * 1e-3

    st_lp, _ = run.channel(lp_ch, scope_name="lpscope", shots=[int(shot)])
    st_bd, _ = run.channel(bdot_ch, scope_name="bdotscope", shots=[int(shot)])
    lp_raw = np.asarray(st_lp[0], dtype=float)
    bd_raw = np.asarray(st_bd[0], dtype=float)

    lp_x, lp_y = _shot_xy(jiv.read_lp_positions(hdf5_path, motion_group_name=jbd.LP_MOTION_GROUP), shot)
    bd_x, bd_y = _shot_xy(jiv.read_lp_positions(hdf5_path, motion_group_name=jbd.BDOT_MOTION_GROUP), shot)

    t_lo, t_hi = float(t_window_ms[0]), float(t_window_ms[1])
    w_lo, w_hi = float(fft_window_ms[0]), float(fft_window_ms[1])
    m_pre_b = (tb >= t_lo) & (tb <= -0.05)
    bd_zero = bd_raw - (float(np.mean(bd_raw[m_pre_b])) if np.any(m_pre_b) else 0.0)
    m_tl, m_tb = (tl >= t_lo) & (tl <= t_hi), (tb >= t_lo) & (tb <= t_hi)
    m_wl, m_wb = (tl >= w_lo) & (tl <= w_hi), (tb >= w_lo) & (tb <= w_hi)

    fl, sl = windowed_fft_amp(lp_raw, dt_l, m_wl, pad_factor=pad_factor)
    fb, sb = windowed_fft_amp(bd_zero, dt_b, m_wb, pad_factor=pad_factor)
    m_f0_l, m_f0_b = (fl >= 2.5) & (fl <= 5.5), (fb >= 2.5) & (fb <= 5.5)
    f0_khz = float(fl[m_f0_l][int(np.argmax(sl[m_f0_l]))])

    xs, ys, Bx, By, Jz, meta = _load_bdot_2d_snapshot(bdot_path, run_num)
    is_cal = bool(meta["bdot_calibrated"])
    db_over_b0 = np.nan
    if is_cal and np.isfinite(meta.get("b0_g", np.nan)) and meta["b0_g"] > 0.0:
        # Compute f0 time-RMS peak |dB_perp| / B0 directly from the plotted calibrated h1 map
        bp_rms_t = np.sqrt(0.5 * (np.abs(Bx) ** 2 + np.abs(By) ** 2))
        db_over_b0 = float(np.nanmax(bp_rms_t)) / (float(meta["b0_g"]) * 1e-4)
    beta_rat = np.nan
    if os.path.exists(modes_path):
        with np.load(modes_path, allow_pickle=True) as md:
            bkey = f"disp__{matched_puff_run}__beta_e_over_me_mi"
            if bkey in md.files:
                beta_rat = float(md[bkey])

    b_lbl = {"C1": r"$\dot{B}_x$", "C2": r"$\dot{B}_y$", "C3": r"$\dot{B}_z$"}.get(bdot_ch, bdot_ch)

    fig = plt.figure(figsize=(14.4, 7.6))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[1.18, 1.0], wspace=0.22, hspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[:, 1])

    # (a) Raw Langmuir Isat
    ax_a.plot(tl[m_tl], lp_raw[m_tl], color="#1d4ed8", lw=0.9)
    ax_a.axvspan(w_lo, w_hi, color="0.91", alpha=0.65, zorder=0)
    ax_a.set_ylabel(r"Langmuir $I_{\mathrm{sat}}$ [V]")
    ax_a.set_xlim(t_lo, t_hi)
    ax_a.grid(alpha=0.22)
    ax_a.set_title(f"(a) Raw Langmuir Probe (Run {run_num}, shot {shot})", fontweight="bold")

    # (b) Raw B-dot
    ylim_b = max(1.5 * float(np.nanpercentile(np.abs(bd_zero[m_wb] * 1e3), 99.5)), 100.0)
    ax_b.plot(tb[m_tb], bd_zero[m_tb] * 1e3, color="#dc2626", lw=0.8)
    ax_b.axvspan(w_lo, w_hi, color="0.91", alpha=0.65, zorder=0)
    ax_b.set_xlabel("Time relative to bias turn-on [ms]")
    ax_b.set_ylabel(f"B-dot {b_lbl} [mV]")
    ax_b.set_xlim(t_lo, t_hi)
    ax_b.set_ylim(-ylim_b, ylim_b)
    ax_b.grid(alpha=0.22)
    ax_b.set_title("(b) Raw Magnetic Probe (P28 B-dot, same shot)", fontweight="bold")

    # (c) Normalized FFT magnitude with harmonic annotations
    draw_psd_overlay(
        ax_c,
        [
            {"freq": fl, "power": sl / float(sl[m_f0_l].max()), "color": "#1d4ed8", "lw": 1.9,
             "label": r"Langmuir $I_{\mathrm{sat}}$"},
            {"freq": fb, "power": sb / float(sb[m_f0_b].max()), "color": "#dc2626", "lw": 1.6,
             "ls": "--", "label": f"B-dot {b_lbl}"},
        ],
        xlim=(0.0, float(fmax_khz)), ylim=(0.0, 1.28), logy=False,
        ylabel="Normalized FFT", grid_alpha=0.22,
        harmonics={"f0": f0_khz, "orders": (1, 2, 3, 4), "half_width": 0.5, "show_f0_value": True},
        title=f"(c) FFT Magnitude ({w_lo:.1f}–{w_hi:.1f} ms shaded window)",
    )

    # (d) Calibrated 2-D J_par & B_perp map with electrode rings and probe markers
    note_lines = [f"$|\\delta B_z| / |\\delta B_\\perp| = {meta['bz_over_bperp']:.2f}$"]
    if np.isfinite(db_over_b0) and db_over_b0 > 0.0:
        exp = int(np.floor(np.log10(db_over_b0)))
        mant = db_over_b0 / (10.0 ** exp)
        note_lines.append(rf"$\delta B_\perp / B_0 = {mant:.1f}\times 10^{{{exp}}}$")
    if np.isfinite(beta_rat):
        note_lines.append(f"$\\beta_e / (m_e/m_i) = {beta_rat:.1f}$")

    draw_bdot_2d_map(
        ax_d, xs, ys, Bx, By, Jz, calibrated=is_cal, phase_align=True,
        circles=[
            {"radius": 2.54, "ec": "#16a34a", "ls": "-", "lw": 1.7, "label": '2" / 4" plates'},
            {"radius": 5.08, "ec": "#16a34a", "ls": "--", "lw": 1.5},
        ],
        markers=[
            {"x": bd_x, "y": bd_y, "marker": "*", "ms": 13, "color": "#facc15",
             "label": f"Shot {shot} B-dot ({bd_x:+.0f}, {bd_y:+.0f})".replace("+0", "0").replace("-0", "0")},
            {"x": lp_x, "y": lp_y, "marker": "D", "ms": 7.5, "color": "#38bdf8",
             "label": f"Shot {shot} LP ({lp_x:+.0f}, {lp_y:+.0f})".replace("+0", "0").replace("-0", "0")},
        ],
        note_lines=note_lines,
        title=rf"(d) 2-D $J_\parallel$ & $\mathbf{{B}}_\perp$ (Run {run_num}, $f_0 = {meta['f0_hz']*1e-3:.1f}$ kHz)",
    )

    fig.subplots_adjust(left=0.065, right=0.945, top=0.94, bottom=0.08)
    save_name = "fig1_em_instability" if str(run_num) == "26" else f"fig1_em_instability_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show, tight_layout=False)


def plot_sheath_closure(
    run_num="02",
    shot=100,
    *,
    data_dir=None,
    ifn=None,
    modes_npz=None,
    t_window_ms=(-0.5, 5.5),
    zoom_window_ms=(2.8, 3.8),
    shunt_a_per_v=100.0,
    downsample_q=25,
    save_fig=True,
    show=False,
):
    """2-panel Port-35 conducting-sheath current closure figure (default: Run 02)."""
    import Jun2026_modes as jmd

    ddir = data_dir or jmd.DATA_DIR
    hdf5_path = ifn or _run_hdf5(ddir, run_num)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    run = open_lapd(hdf5_path)
    tb_ms = np.asarray(run.time_array(scope_name="biasscope"), dtype=float) * 1e3 - jmd.BIASSCOPE_OFFSET_MS
    st_i2, _ = run.channel("C1", scope_name="biasscope", shots=[int(shot)])
    st_i4, _ = run.channel("C2", scope_name="biasscope", shots=[int(shot)])

    t_lo, t_hi = float(t_window_ms[0]), float(t_window_ms[1])
    z_lo, z_hi = float(zoom_window_ms[0]), float(zoom_window_ms[1])
    m_full = (tb_ms >= t_lo) & (tb_ms <= t_hi)
    t_raw = tb_ms[m_full]
    i2_raw = np.asarray(st_i2[0][m_full], dtype=float) * float(shunt_a_per_v)
    i4_raw = np.asarray(st_i4[0][m_full], dtype=float) * float(shunt_a_per_v)
    m_pre = (t_raw >= t_lo) & (t_raw <= -0.05)
    if np.any(m_pre):
        i2_raw = i2_raw - float(np.mean(i2_raw[m_pre]))
        i4_raw = i4_raw - float(np.mean(i4_raw[m_pre]))

    q = max(int(downsample_q), 1)
    t_ms, i2 = downsample_blockmean(t_raw, i2_raw, q)
    _, i4 = downsample_blockmean(t_raw, i4_raw, q)

    g2_24 = dphi_24 = np.nan
    if os.path.exists(modes_path):
        with np.load(modes_path, allow_pickle=True) as md:
            if f"bias__{run_num}__g2_I2_I4" in md.files:
                g2_24 = float(md[f"bias__{run_num}__g2_I2_I4"])
            if f"bias__{run_num}__dphi_I4_minus_I2_deg" in md.files:
                dphi_24 = abs(float(md[f"bias__{run_num}__dphi_I4_minus_I2_deg"]))
    note = (
        f"$\\gamma^2 = {g2_24:.2f},\\; \\Delta\\phi = {dphi_24:.0f}^\\circ$"
        if (np.isfinite(g2_24) and np.isfinite(dphi_24)) else None
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.2), gridspec_kw={"height_ratios": [1.15, 1.0]})
    draw_pulse_and_zoom(
        ax1, ax2, t_ms,
        [
            {"y": i2, "color": "#7c3aed", "full_ls": "-", "zoom_ls": "-",
             "full_label": r'Port-35 2" disc ($I_{2^{\prime\prime}}$)',
             "zoom_label": r'$\delta I_{2^{\prime\prime}}$'},
            {"y": i4, "color": "#ea580c", "full_ls": "-", "zoom_ls": "--",
             "full_label": r'Port-35 4" ring ($I_{4^{\prime\prime}}$)',
             "zoom_label": r'$\delta I_{4^{\prime\prime}}$'},
        ],
        zoom_window_ms=(z_lo, z_hi),
        t_window_ms=(t_lo, t_hi),
        full_ylabel="End-plate current $I$ [A]",
        zoom_ylabel=r"Oscillating current $\delta I$ [A]",
        full_title=f"(a) Port-35 End-Plate Currents Across Full Bias Pulse (Run {run_num}, shot {shot})",
        zoom_title=f"(b) Zoomed Shaded Window ({z_lo:.1f}–{z_hi:.1f} ms): $180^\\circ$ Out-of-Phase Closure",
        zoom_note=note,
    )
    fig.tight_layout(pad=1.2)
    save_name = "fig2_sheath_closure" if str(run_num) == "02" else f"fig2_sheath_closure_run{run_num}"
    finalize_figure(fig, save_fig=_resolve_save(save_fig, save_name), show=show)


def animate_mode_propagation(
    run_num="26",
    *,
    matched_puff_run="03",
    f_exb_khz=None,
    seamless_loop=True,
    data_dir=None,
    bdot_npz=None,
    modes_npz=None,
    n_frames=60,
    fps=15,
    dpi=100,
    r_orbit_cm=3.2,
    save_fig=True,
    preview_png=None,
):
    """Side-by-side GIF animating 2-D ``J_par`` & ``B_perp`` in lab vs. plasma frame."""
    import Jun2026_bdot_2d as jbd
    import Jun2026_modes as jmd

    ddir = data_dir or jbd.DATA_DIR
    bdot_path = bdot_npz or os.path.join(ddir, jbd.OUT_NAME)
    modes_path = modes_npz or os.path.join(ddir, jmd.OUT_NAME)

    xs, ys, Bx, By, Jz, meta = _load_bdot_2d_snapshot(bdot_path, run_num)
    if f_exb_khz is None:
        with np.load(modes_path, allow_pickle=True) as md:
            f_exb_meas = float(md[f"disp__{matched_puff_run}__f_doppler_exb_hz"]) * 1e-3
    else:
        f_exb_meas = float(f_exb_khz)

    save_name = "fig3_mode_propagation" if str(run_num) == "26" else f"fig3_mode_propagation_run{run_num}"
    out_gif = _resolve_save(save_fig, save_name, ext=".gif")
    return animate_lab_vs_plasma_frame(
        xs, ys, Bx, By, Jz,
        f_lab_khz=meta["f0_hz"] * 1e-3,
        f_exb_khz=f_exb_meas,
        center=(meta["center_x_cm"], meta["center_y_cm"]),
        calibrated=bool(meta["bdot_calibrated"]),
        seamless_loop=seamless_loop,
        n_frames=n_frames,
        fps=fps,
        dpi=dpi,
        r_orbit_cm=r_orbit_cm,
        title_prefix=f"Run {run_num} ({meta['b0_g']:.0f} G)",
        out_gif=out_gif,
        preview_png=preview_png,
    )


# =========================================================================== #
#  Finding #2 figures (puff scaling, B0 scaling, strictly pre-onset shear gating)
# =========================================================================== #

def plot_puff_scaling(
    *,
    data_dir=None,
    onset_npz=None,
    show_runs=PUFF_SHOW_RUNS,
    all_puffs=PUFF_ALL_RUNS,
    scope="machscope",
    chan="C2",
    t_lim_ms=(-0.5, 5.5),
    save_fig=True,
    show=False,
    dpi=180,
):
    """3-panel gas-puff scaling figure: stacked P33 ``Isat``, PSD overlay, and growth-rate window."""
    import Jun2026_onset as jon

    ddir = data_dir or jon.DATA_DIR
    onset_path = onset_npz or os.path.join(ddir, jon.OUT_NAME)

    with np.load(onset_path, allow_pickle=True) as onf:
        traces = []
        spectra = []
        for rnum, puff, col in show_runs:
            shot, _, _ = _pick_representative_shot(onf, rnum)
            t_ms, v = _read_scope_shot(_run_hdf5(ddir, rnum), scope, chan, shot, t_lim_ms=t_lim_ms)
            m_bias = (t_ms >= 0.5) & (t_ms <= 4.5)
            if np.any(m_bias) and float(np.mean(v[m_bias])) < 0.0:
                v = -v
            traces.append({"t": t_ms, "y": v, "label": f"{puff} ms", "color": col})
            is_edge = puff in (23, 45)
            spectra.append({
                "freq": np.asarray(onf[f"{rnum}__psd_freq_hz"], dtype=float) * 1e-3,
                "power": np.asarray(onf[f"{rnum}__psd_mean"], dtype=float),
                "label": f"{puff} ms",
                "color": col,
                "ls": "--" if is_edge else "-",
                "lw": 1.4 if is_edge else 2.0,
            })

        puffs = np.array([p for _, p in all_puffs], dtype=float)
        gam = np.zeros_like(puffs)
        q25 = np.zeros_like(puffs)
        q75 = np.zeros_like(puffs)
        stable = np.zeros(len(all_puffs), dtype=bool)
        for i, (rnum, _) in enumerate(all_puffs):
            if float(onf[f"{rnum}__unstable_frac"]) == 0.0:
                stable[i] = True
            else:
                gam[i] = float(onf[f"{rnum}__gamma_med_s"]) * 1e-3
                q25[i] = float(onf[f"{rnum}__gamma_q25_s"]) * 1e-3
                q75[i] = float(onf[f"{rnum}__gamma_q75_s"]) * 1e-3

    fig = plt.figure(figsize=(13.2, 7.4))
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.18, 1.0], wspace=0.26)
    gs_left = gs_outer[0].subgridspec(len(traces), 1, hspace=0.14)
    gs_right = gs_outer[1].subgridspec(2, 1, hspace=0.34, height_ratios=[1.12, 1.0])

    left_axes = []
    for i in range(len(traces)):
        left_axes.append(fig.add_subplot(gs_left[i, 0], sharex=left_axes[0] if left_axes else None))
    draw_stacked_traces(
        left_axes, traces,
        xlim=t_lim_ms, ylim=(0.15, 1.38), yticks=[0.4, 0.8, 1.2],
        xlabel="Time relative to bias turn-on [ms]",
        ylabel=r"$I_{\mathrm{sat}}$ [V]", ylabel_index=len(traces) // 2,
        title=r"(a) Raw P33 $I_{\mathrm{sat}}$ Across Gas Puff (800 G)",
        grid_alpha=0.20,
    )

    ax_b = fig.add_subplot(gs_right[0, 0])
    draw_psd_overlay(
        ax_b, spectra,
        xlim=(0.5, 12.0), ylim=(3e-10, 2e-5), logy=True,
        ylabel=r"Relative $I_{\mathrm{sat}}$ PSD [Hz$^{-1}$]",
        legend_kw={"loc": "upper right", "ncol": 2},
        title="(b) Ensemble Fluctuation Spectrum",
    )

    ax_c = fig.add_subplot(gs_right[1, 0])
    ax_c.plot(puffs, gam, "-", color="#1e293b", lw=1.6, zorder=2)
    u = ~stable
    if np.any(u):
        yerr = np.vstack([np.maximum(gam[u] - q25[u], 0.0), np.maximum(q75[u] - gam[u], 0.0)])
        ax_c.errorbar(
            puffs[u], gam[u], yerr=yerr, fmt="o", color="#dc2626", ecolor="#94a3b8",
            elinewidth=1.6, capsize=4, ms=7.2, zorder=3, label="Median (IQR)",
        )
    if np.any(stable):
        ax_c.plot(
            puffs[stable], gam[stable], "o", mfc="white", mec="#64748b", mew=2.0,
            ms=7.5, zorder=4, label="Stable",
        )
    ax_c.set_xlim(21.5, 46.5)
    ax_c.set_ylim(-0.4, 9.2)
    ax_c.set_xticks([23, 25, 30, 35, 40, 45])
    ax_c.set_xlabel("Gas puff duration [ms]")
    ax_c.set_ylabel(r"Growth rate $\gamma$ [ms$^{-1}$]")
    ax_c.grid(alpha=0.25)
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    ax_c.legend(handles_c[::-1], labels_c[::-1], loc="upper right", frameon=True, framealpha=0.92)
    ax_c.set_title("(c) Instability Growth Rate Window", fontweight="bold")

    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.09)
    finalize_figure(fig, save_fig=_resolve_save(save_fig, "fig2_1_puff_scaling"), show=show, dpi=dpi, tight_layout=False)


def plot_b0_scaling(
    *,
    data_dir=None,
    onset_npz=None,
    b0_runs=B0_SCAN_RUNS,
    scope="scope",
    chan="C5",
    t_lim_ms=(-0.5, 5.5),
    save_fig=True,
    show=False,
    dpi=180,
):
    """2-panel ``B_0`` scaling figure: stacked P33 ``Isat`` traces and harmonic PSD march."""
    import Jun2026_onset as jon

    ddir = data_dir or jon.DATA_DIR
    onset_path = onset_npz or os.path.join(ddir, jon.OUT_NAME)

    with np.load(onset_path, allow_pickle=True) as onf:
        traces = []
        spectra = []
        for rnum, b0, col in b0_runs:
            shot, _, _ = _pick_representative_shot(onf, rnum)
            t_ms, v = _read_scope_shot(_run_hdf5(ddir, rnum), scope, chan, shot, t_lim_ms=t_lim_ms)
            m_bias = (t_ms >= 0.5) & (t_ms <= 4.5)
            if np.any(m_bias) and float(np.mean(v[m_bias])) < 0.0:
                v = -v
            f0_khz = float(onf[f"{rnum}__f_peak_hz"]) * 1e-3
            traces.append({"t": t_ms, "y": v, "label": f"{b0} G", "color": col})
            spectra.append({
                "freq": np.asarray(onf[f"{rnum}__psd_freq_hz"], dtype=float) * 1e-3,
                "power": np.asarray(onf[f"{rnum}__psd_mean"], dtype=float),
                "label": f"{b0} G ($f_0 = {f0_khz:.2f}$ kHz)",
                "color": col,
                "lw": 2.0,
                "peak_freq": f0_khz,
            })

    fig = plt.figure(figsize=(12.8, 5.2))
    gs_outer = GridSpec(1, 2, figure=fig, width_ratios=[1.12, 1.0], wspace=0.25)
    gs_left = gs_outer[0].subgridspec(len(traces), 1, hspace=0.14)

    left_axes = []
    for i in range(len(traces)):
        left_axes.append(fig.add_subplot(gs_left[i, 0], sharex=left_axes[0] if left_axes else None))
    draw_stacked_traces(
        left_axes, traces,
        xlim=t_lim_ms, ylim=None,
        xlabel="Time relative to bias turn-on [ms]",
        ylabel=r"$I_{\mathrm{sat}}$ [V]", ylabel_index=len(traces) // 2,
        title=r"(a) Raw P33 $I_{\mathrm{sat}}$ Across $B_0$ (35 ms puff)",
        label_fontsize=10.5,
    )

    ax_b = fig.add_subplot(gs_outer[1])
    draw_psd_overlay(
        ax_b, spectra,
        xlim=(0.5, 15.0), ylim=(2e-10, 4e-5), logy=True,
        ylabel=r"Relative $I_{\mathrm{sat}}$ PSD [Hz$^{-1}$]",
        title=r"(b) Ensemble Fluctuation Spectrum vs. $B_0$",
    )

    fig.subplots_adjust(left=0.075, right=0.97, top=0.91, bottom=0.12)
    finalize_figure(fig, save_fig=_resolve_save(save_fig, "fig2_2_b0_scaling"), show=show, dpi=dpi, tight_layout=False)


def plot_shear_gating(
    *,
    data_dir=None,
    mach_npz=None,
    profiles_npz=None,
    onset_npz=None,
    pairs=SHEAR_GATING_PAIRS,
    profile_run="11",
    b0_g=800.0,
    save_fig=True,
    show=False,
    dpi=180,
):
    """2-panel strictly pre-onset shear-gating figure (``ref_sweep_idx`` only).

    Computes 1-D radial cylindrical CWM growth rates directly from each run's
    strictly pre-onset ``ref_sweep_idx`` profiles (never ``near_swp``, which
    straddles onset on 25 ms and 30 ms).
    """
    import Jun2026_mach_line as jml
    import Jun2026_onset as jon
    import Jun2026_profiles as jpr
    import Jun2026_stability as jst

    ddir = data_dir or jst.DATA_DIR
    mach_path = mach_npz or os.path.join(ddir, jml.RESCALED_OUT_NAME)
    prof_path = profiles_npz or os.path.join(ddir, jpr.OUT_NAME)
    onset_path = onset_npz or os.path.join(ddir, jon.OUT_NAME)

    with np.load(mach_path, allow_pickle=True) as mf, \
         np.load(prof_path, allow_pickle=True) as pf, \
         np.load(onset_path, allow_pickle=True) as onf:

        # (a) Strictly pre-onset radial rotation profile (default: Run 11, 30 ms)
        r_cm = np.asarray(mf[f"{profile_run}__r_cm"], dtype=float)
        ib = int(mf[f"{profile_run}__ref_bin_idx"])
        v_mach_cw = -np.asarray(mf[f"{profile_run}__dv_rot_km_s"][:, ib], dtype=float)
        v_exb_cw = np.asarray(mf[f"{profile_run}__v_ExB_r_km_s"][:, ib], dtype=float)
        m_r = r_cm <= 12.0

        # (b) Strictly pre-onset 1-D radial CWM theory vs. measured growth rate
        puffs = np.array([p for _, _, p in pairs], dtype=float)
        gam_exp = np.zeros(len(pairs), dtype=float)
        stable = np.zeros(len(pairs), dtype=bool)
        cwm_nosh = np.zeros(len(pairs), dtype=float)
        cwm_shc = np.zeros(len(pairs), dtype=float)

        r_grid = jst.DEFAULT_R_GRID_CM
        for i, (ir, _, _) in enumerate(pairs):
            if float(onf[f"{ir}__unstable_frac"]) == 0.0:
                stable[i] = True
                gam_exp[i] = 0.0
            else:
                gam_exp[i] = float(onf[f"{ir}__gamma_med_s"]) * 1e-3

            rs = int(pf[f"{ir}__ref_sweep_idx"])
            x = pf[f"{ir}__x_cm"]
            xc = float(pf[f"{ir}__x_center_cm"])
            _, lam_ref = jst._extract_wall_state(pf, ir, rs, b0_g=float(b0_g))
            ne_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{ir}__ne"][:, rs], r_grid, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            te_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{ir}__Te"][:, rs], r_grid, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            vp_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{ir}__Vp"][:, rs], r_grid, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            ne_g = np.maximum(ne_g, 1e10)
            te_g = np.maximum(te_g, 0.1)

            r_no = solve_radial_cylindrical_modes(
                r_grid, ne_g, te_g, vp_g, b0_g=float(b0_g), m=-1, model="cwm_sheath",
                l_par_m=jst.DEFAULT_L_PAR_M, sheath_lambda=lam_ref, k_par_m=jst.DEFAULT_K_PAR_M,
                include_centrifugal=False, include_exb_shear=False,
            )
            r_sc = solve_radial_cylindrical_modes(
                r_grid, ne_g, te_g, vp_g, b0_g=float(b0_g), m=-1, model="cwm_sheath",
                l_par_m=jst.DEFAULT_L_PAR_M, sheath_lambda=lam_ref, k_par_m=jst.DEFAULT_K_PAR_M,
                include_centrifugal=True, include_exb_shear=True,
            )
            cwm_nosh[i] = float(r_no["gamma_s"]) * 1e-3
            cwm_shc[i] = float(r_sc["gamma_s"]) * 1e-3

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    ax_a.plot(r_cm[m_r], v_mach_cw[m_r], "o-", color="#1d4ed8", lw=2.0, ms=5.8, label="P33 Mach")
    ax_a.plot(r_cm[m_r], v_exb_cw[m_r], "s--", color="#dc2626", lw=1.9, ms=5.4, label=r"P29 $E\times B$")
    ax_a.axhline(0.0, color="0.55", lw=0.8)
    ax_a.set_xlim(0.5, 12.2)
    ax_a.set_ylim(-0.25, 2.40)
    ax_a.set_xlabel(r"Radius $r$ [cm]")
    ax_a.set_ylabel(r"CW rotation speed $v_\theta$ [km/s]")
    ax_a.grid(alpha=0.25)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax_a.set_title("(a) Azimuthal Flow Profile (30 ms)", fontweight="bold")

    u = ~stable
    ax_b.plot(puffs, cwm_nosh, "^:", color="#64748b", lw=1.8, ms=5.8, label="Theory: CWM (no shear)")
    ax_b.plot(puffs, cwm_shc, "s-", color="#1d4ed8", lw=2.0, ms=5.8, label="Theory: CWM (+shear, cent.)")
    if np.any(u):
        ax_b.plot(puffs[u], gam_exp[u], "o-", color="#dc2626", lw=2.0, ms=6.4, label=r"Experiment: measured $\gamma$")
    if np.any(stable):
        ax_b.plot(puffs[stable], gam_exp[stable], "o", mfc="white", mec="#dc2626", mew=1.8, ms=6.6, zorder=5)
    ax_b.set_xlim(21.5, 46.5)
    ax_b.set_ylim(-0.4, 10.2)
    ax_b.set_xticks([23, 25, 30, 35, 40, 45])
    ax_b.set_xlabel("Gas puff duration [ms]")
    ax_b.set_ylabel(r"Growth rate $\gamma$ [ms$^{-1}$]")
    ax_b.grid(alpha=0.25)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.92)
    ax_b.set_title("(b) 1-D Radial CWM Theory vs. Experiment", fontweight="bold")

    finalize_figure(fig, save_fig=_resolve_save(save_fig, "fig2_3_shear_gating"), show=show, dpi=dpi)


# =========================================================================== #
#  Turbulence steepening figure (migrated from Jun2026_plot.py)
# =========================================================================== #

def plot_turbulence_steepening(npz_path=None, *, data_dir=None, save_fig=True, show=False):
    """4-panel nonlinear steepening & phase-folded sawtooth waveform figure."""
    import Jun2026_turbulence as jtb

    path = npz_path or os.path.join(data_dir or jtb.DATA_DIR, jtb.OUT_NAME)
    with np.load(path, allow_pickle=True) as d:
        res = {k: d[k] for k in d.files}

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.2))
    th_ref = np.linspace(-np.pi, np.pi, 256)

    # (a) Phase-folded stationary Isat waveforms vs pure cos(theta)
    ax = axs[0, 0]
    ax.plot(th_ref / (2.0 * np.pi), np.cos(th_ref), "k--", lw=1.6, alpha=0.65, label=r"Pure sine $\cos\theta$")
    for rnum, lbl, col in (
        ("02", "Run 02 (30 ms, inner wall)", "#d62728"),
        ("03", "Run 03 (35 ms, inner wall)", "#ff7f0e"),
        ("04", "Run 04 (40 ms, broadband)", "#7f7f7f"),
        ("27", "Run 27 (35 ms, outer probe)", "#1f77b4"),
    ):
        p = f"spec_{rnum}__"
        if p + "folded_wave" not in res:
            continue
        ph = np.asarray(res[p + "folded_phase_rad"], dtype=float) / (2.0 * np.pi)
        w = np.asarray(res[p + "folded_wave"], dtype=float)
        ws = np.asarray(res[p + "folded_std"], dtype=float)
        ax.plot(
            ph, w, lw=2.2, color=col,
            label=rf"{lbl}: $\beta={float(res[p+'biphase_deg']):+.0f}^\circ$, "
                  rf"slp={float(res[p+'slope_ratio_med']):.2f}$\times$, $f_r={float(res[p+'rise_frac_med']):.2f}$",
        )
        ax.fill_between(ph, w - ws, w + ws, color=col, alpha=0.12)
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"Fundamental cycle phase $\theta / 2\pi$")
    ax.set_ylabel(r"Normalized waveform $\tilde{I}_{\mathrm{sat}} / A_1$")
    ax.set_title(r"(a) Phase-Folded Stationary $I_{\mathrm{sat}}$ vs. Pure Sinusoid")
    ax.legend(fontsize=8.2, loc="lower center", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (b) Run 32 wall [3.5, 4.5) cm velocity (M_r, M_theta) vs density
    ax = axs[0, 1]
    ax.plot(th_ref / (2.0 * np.pi), np.cos(th_ref), "k--", lw=1.6, alpha=0.65, label=r"Pure sine $\cos\theta$")
    if "r32__steep_phase_rad" in res:
        ph32 = np.asarray(res["r32__steep_phase_rad"], dtype=float) / (2.0 * np.pi)
        bi = 2
        for tag, lbl, col in (
            ("mr", r"Radial velocity $\tilde{M}_r$", "#d62728"),
            ("mt", r"Azimuthal velocity $\tilde{M}_\theta$", "#2ca02c"),
            ("isat0", r"Flow-cancelled $I_{\mathrm{sat},0}$", "#1f77b4"),
        ):
            w = np.asarray(res[f"r32__steep_{tag}_folded_waves"], dtype=float)[bi]
            bkey = f"r32__steep_linear_tip_{tag}_steepening_comp"
            bstr = f" (lin-tip {float(res[bkey][bi]):+.2f})" if (bkey in res and np.isfinite(res[bkey][bi])) else ""
            ax.plot(
                ph32, w, lw=2.3, color=col,
                label=rf"{lbl}: $\beta={float(res[f'r32__steep_{tag}_biphase_deg'][bi]):+.0f}^\circ$, "
                      rf"sin={float(res[f'r32__steep_{tag}_steepening_comp'][bi]):+.2f}{bstr}, "
                      rf"slp={float(res[f'r32__steep_{tag}_slope_ratio'][bi]):.2f}$\times$",
            )
    ax.axhline(0.0, color="0.75", lw=0.8)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel(r"Fundamental cycle phase $\theta / 2\pi$")
    ax.set_ylabel(r"Normalized waveform $\tilde{x} / A_1$")
    ax.set_title(r"(b) Run 32 Wall $[3.5, 4.5)$ cm: Intrinsic Velocity vs. Density Steepening")
    ax.legend(fontsize=8.3, loc="lower center", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (c) Radial decomposition across hollow wall (Runs 11 & 12)
    ax = axs[1, 0]
    for rnum, puff, col, mk in (("11", 30, "#d62728", "o"), ("12", 35, "#1f77b4", "s")):
        pr = f"radst_{rnum}__"
        if pr + "r_cm" not in res:
            continue
        r = np.asarray(res[pr + "r_cm"], dtype=float)
        ax.plot(r, np.asarray(res[pr + "steepening_comp"], dtype=float), color=col, marker=mk, lw=2.2, ms=6,
                label=rf"Run {rnum} ({puff} ms) steepening $(A_2/A_1)\sin\beta$")
        ax.plot(r, np.asarray(res[pr + "curvature_comp"], dtype=float), color=col, marker=mk, lw=1.4, ls="--",
                ms=4.5, alpha=0.65, label=rf"Run {rnum} ({puff} ms) curvature $(A_2/A_1)\cos\beta$")
    ax.axhline(0.0, color="0.5", lw=0.9)
    ax.axvspan(2.5, 3.5, color="#ffcccc", alpha=0.25, label="Mid-wall (steep rise)")
    ax.axvspan(5.0, 6.2, color="#cce5ff", alpha=0.25, label="Outer shear (steep fall)")
    ax.set_xlabel(r"Radius $r = x - x_0$ [cm]")
    ax.set_ylabel(r"Harmonic component amplitude")
    ax.set_title(r"(c) Radial Decomposition Across Hollow Wall (Runs 11–12 $I_{\mathrm{sat},0}$)")
    ax.legend(fontsize=7.8, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.25)

    # (d) Campaign summary: Puff scan (00-04) & B0 scan (28, 27, 30)
    ax = axs[1, 1]
    run_order = (
        ("00", "24ms\n800G"), ("01", "25ms\n800G"), ("02", "30ms\n800G"),
        ("03", "35ms\n800G"), ("04", "40ms\n800G"),
        ("28", "35ms\n600G*"), ("27", "35ms\n800G*"), ("30", "35ms\n1000G*"),
    )
    xs = np.arange(len(run_order))
    stps = [float(res[f"spec_{r}__steepening_comp"]) for r, _ in run_order]
    crvs = [float(res[f"spec_{r}__curvature_comp"]) for r, _ in run_order]
    slps = [float(res[f"spec_{r}__slope_ratio_med"]) for r, _ in run_order]
    w_bar = 0.36
    ax.bar(xs - w_bar / 2, stps, width=w_bar, color="#d62728", alpha=0.88,
           label=r"Sawtooth steepening $(A_2/A_1)\sin\beta$")
    ax.bar(xs + w_bar / 2, crvs, width=w_bar, color="#1f77b4", alpha=0.72,
           label=r"Symmetric curvature $(A_2/A_1)\cos\beta$")
    ax.axhline(0.0, color="0.4", lw=0.9)
    ax.axvline(4.5, color="0.4", lw=1.0, ls=":")
    for xi, sv, slp in zip(xs, stps, slps):
        if not (np.isfinite(sv) and np.isfinite(slp) and slp > 0.0):
            continue
        va, off = ("bottom", 0.02) if sv >= 0 else ("top", -0.02)
        txt = f"{slp:.1f}x↑" if slp >= 1.0 else f"{1.0/slp:.1f}x↓"
        ax.text(xi - w_bar / 2, sv + off, txt, ha="center", va=va, fontsize=7.8,
                fontweight="bold", color="#8b0000")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in run_order], fontsize=8.5)
    ax.set_ylabel(r"2nd-harmonic component")
    ax.set_title(r"(d) Puff Scan & $B_0$ Scan (*outer probe; slope ratio annotated)")
    ax.set_ylim(-0.42, 0.56)
    ax.legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Jun-2026 Nonlinear Wave Steepening vs. Symmetric Wall Curvature",
                 fontsize=13.5, fontweight="bold")
    finalize_figure(fig, save_fig=_resolve_save(save_fig, "turbulence_steepening"), show=show)


# =========================================================================== #
#  Figure Registry & CLI
# =========================================================================== #

class FigureSpec(NamedTuple):
    func: Callable
    output: str
    group: str
    description: str


FIGURES: dict[str, FigureSpec] = {
    "fig1a_profiles": FigureSpec(
        plot_hollow_profiles,
        "fig1a_profiles.png",
        "Finding #1",
        "Dual-axis hollow-column n_e(x) and T_e(x) radial profiles (Run 02)",
    ),
    "fig1_em_instability": FigureSpec(
        plot_em_instability_overview,
        "fig1_em_instability.png",
        "Finding #1",
        "4-panel electromagnetic instability overview: Isat, B-dot, FFT, 2-D J_par & B_perp (Run 26)",
    ),
    "fig2_sheath_closure": FigureSpec(
        plot_sheath_closure,
        "fig2_sheath_closure.png",
        "Finding #1",
        "2-panel Port-35 conducting-sheath 180-deg current closure (Run 02)",
    ),
    "fig3_mode_propagation": FigureSpec(
        animate_mode_propagation,
        "fig3_mode_propagation.gif",
        "Finding #1",
        "Lab-frame vs. co-rotating E x B plasma-rest-frame 2-D propagation GIF (Run 26)",
    ),
    "fig2_1_puff_scaling": FigureSpec(
        plot_puff_scaling,
        "fig2_1_puff_scaling.png",
        "Finding #2",
        "Gas-puff scan (23-45 ms): stacked P33 Isat traces, ensemble PSD, and growth-rate window",
    ),
    "fig2_2_b0_scaling": FigureSpec(
        plot_b0_scaling,
        "fig2_2_b0_scaling.png",
        "Finding #2",
        "B0 scan (600/800/1000 G): stacked P33 Isat traces and harmonic PSD march",
    ),
    "fig2_3_shear_gating": FigureSpec(
        plot_shear_gating,
        "fig2_3_shear_gating.png",
        "Finding #2",
        "Strictly pre-onset azimuthal flow profile and 1-D radial CWM theory vs. experiment",
    ),
    "turbulence_steepening": FigureSpec(
        plot_turbulence_steepening,
        "turbulence_steepening.png",
        "Turbulence",
        "4-panel nonlinear wave steepening vs. symmetric hollow-wall curvature",
    ),
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate Jun-2026 LAPD publication and presentation figures."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List registered figure keys and output filenames.")
    mode.add_argument("--only", nargs="+", metavar="NAME", help="Generate one or more registered figures.")
    mode.add_argument("--all", action="store_true", help="Generate all registered figures.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--data-dir", default=None, help="Optional campaign data directory override.")
    return parser


def run_figures(names, *, out_dir=None, data_dir=None):
    """Execute each named figure recipe in ``FIGURES`` and return output paths."""
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        raise KeyError(f"Unknown figure key(s): {unknown}. Valid keys: {list(FIGURES)}")

    outputs = {}
    for name in names:
        spec = FIGURES[name]
        save_arg = os.path.join(out_dir, spec.output) if out_dir else True
        kwargs = {"save_fig": save_arg}
        if data_dir is not None:
            kwargs["data_dir"] = data_dir
        print(f"Generating {name} -> {spec.output} ...")
        spec.func(**kwargs)
        outputs[name] = save_arg if out_dir else fig_path(spec.output, FIG_SUBDIR)
    return outputs


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        w_key = max(len(k) for k in FIGURES)
        w_out = max(len(s.output) for s in FIGURES.values())
        for key, spec in FIGURES.items():
            print(f"  {key:<{w_key}}  {spec.output:<{w_out}}  [{spec.group}] {spec.description}")
        return 0

    configure_publication_style()
    selected = list(FIGURES.keys()) if args.all else list(args.only)
    try:
        run_figures(selected, out_dir=args.out_dir, data_dir=args.data_dir)
    except KeyError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
