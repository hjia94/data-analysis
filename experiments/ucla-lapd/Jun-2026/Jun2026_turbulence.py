"""Jun-2026 broadband turbulence spectrum, intermittency, cross-phase & Reynolds-stress transport.

Tests four turbulence and energy-transfer diagnostics from the LAPD biasing and
drift-wave literature (Pace et al. 2008; Maggs & Morales 2011; Friedman et al.
2013; Schaffner et al. 2013) across the Jun-2026 campaign:

1. **Broadband PSD & fluctuation PDF moments (Runs 06, 00–05, 08–14, 26–28, 30–31):**
   Quantifies the coherent harmonic power fraction (``1..4 f_0``), compares
   exponential (Lorentzian-pulse ``P(f) ~ exp(-f/f_e)``) vs. power-law
   (``P(f) ~ f^-alpha``) continuum fits strictly above the per-run pre-bias
   **ambient plasma** PSD (``[-1.5, -0.2] ms`` while the main discharge is on,
   not digitizer dark noise; block-mean downsampling provides ~`-10 dB`
   aliasing rejection at ``1.5x`` Nyquist, adequate for steep ``alpha ~ 3–4.6``
   tails), and measures per-shot skewness and excess kurtosis.
2. **Radial intermittency polarity across the hollow wall (Runs 11–14):**
   Tracks flow-cancelled ``Isat_0`` skewness and excess kurtosis across
   ``core (r=0)``, ``inner_wall (r=2.5 cm)``, ``outer_wall (r=4.0 cm)``, and
   ``ring (r=6.0 cm)`` — confirming inverted hollow-column skewness polarity
   (``S > 0`` on the low-density inner wall, ``S < 0`` on the outer wall) with
   strongly negative wave-like kurtosis at peak coherent saturation.
3. **Per-shot amplitude-normalized linear-growth vs. early- & late-saturation
   cross-phase evolution (Runs 01–04):**
   Aligns each shot to its own onset ``t_on_k`` and 50%-of-peak ramp end
   ``t_50_k`` and uses amplitude-normalized analytic-signal unit phasors
   (:func:`amplitude_normalized_cross_phase`) so the saturated tail cannot
   dominate the ramp by ``|A(t)|^2``, comparing ``Isat(P33)`` vs. Port-35
   ``I_2"``, ``I_4"``, ``V_2"`` (plus same-scope ``I_2"`` vs. ``I_4"``).
4. **Direct Reynolds stress, spiral-sign verification & finite-tip flux-artifact
   benchmark (Run 32, 6-tip Mach plane at P33, 800 G, 35 ms):**
   Across five radial annuli ``[1.5, 7.0) cm`` about the fitted flow center,
   computes simultaneous ``<dn/n * dM_r>``, ``<dM_r * dM_theta>`` (including a
   4-quadrant check on ``[2.5, 4.5) cm`` ruling out Cartesian cross-talk), the
   direct ``Isat_0`` radial phase spiral ``m * dPhi/dr`` (proving
   ``<dM_r * dM_theta> > 0`` has the opposite sign to finite-probe-size
   ``grad(n)`` contamination and is 10–12x larger on the outer shear layer),
   cylindrical shear-flow energy production
   ``P_shear = -<dM_r * dM_theta> * r * d(M_theta / r) / dr``, AND an explicit
   benchmark of the finite-tip ``grad(n)`` artifact
   ``-(K_H * d / 2) * d<(dn/n)^2>/dr`` against measured ``<dn/n * dM_r>`` (which
   shows raw Mach-probe radial flux is gradient-confounded at ``0.3–1.9x``).
5. **Nonlinear wave steepening vs. symmetric wall-curvature decomposition:**
   Computes the bispectral biphase ``beta = phi_2 - 2*phi_1`` (separating
   symmetric wall curvature ``(A_2/A_1)*cos(beta)`` from true sawtooth
   steepening ``(A_2/A_1)*sin(beta)``), Elgar-Guza Hilbert asymmetry,
   rise/fall slope ratio, rise-time fraction, and fundamental-phase-folded
   cycle waveforms across stationary ``Isat`` (puff & ``B_0`` scans), radially
   across ``r = 0..7 cm`` on Runs 11–14, and directly on Run 32's radial and
   azimuthal velocity fluctuations (``M_r``, ``M_theta``) vs. ``Isat_0``.

Note: No run in Jun-2026 carries a simultaneous floating-potential ``Vf``
channel alongside fixed-bias ``Isat``; turbulent particle flux and Reynolds
stress are therefore measured directly via Run 32's simultaneous 6-tip Mach
probe velocity and flow-cancelled density channels.

Saves ``turbulence_transport_jun2026.npz`` in ``DATA_DIR`` and prints summary
tables.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_turbulence.py
"""

import datetime
import glob
import os

import numpy as np
from scipy import constants as const
from scipy import signal

trapezoid = getattr(np, "trapezoid", None) or np.trapz

from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.plasma.flow import find_flow_centre, polar_components
from data_analysis.plasma.mach import K_HUTCHINSON, mach_single
from data_analysis.signal import (
    amplitude_normalized_cross_phase,
    clip_time_window,
    detect_onset_time,
    fit_broadband_spectrum,
    fluctuation_moments,
    harmonic_steepening,
    reynolds_shear_production,
    sliding_detrended_rms,
)
from data_analysis.utils import merge_save_npz

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
import Jun2026_mach_line as jml
import Jun2026_modes as jmo
import Jun2026_onset as jon

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "turbulence_transport_jun2026.npz"

STATIONARY_RUNS = (
    "06", "00", "01", "02", "03", "04", "05",
    "08", "09", "10", "11", "12", "13", "14",
    "28", "31", "26", "27", "30",
)
RADIAL_MOMENT_RUNS = (("11", 30.0), ("12", 35.0), ("13", 40.0), ("14", 45.0))
CROSSPHASE_RUNS = (("01", 25.0), ("02", 30.0), ("03", 35.0), ("04", 40.0))

TARGET_FS_PSD_HZ = 200e3         # 200 kHz -> Nyquist 100 kHz (resolves 1..70 kHz)
TARGET_FS_TRANSPORT_HZ = 100e3   # 100 kHz (10 us block-mean) for transport & cross-phase
DEFAULT_MAX_SHOTS = 80
RUN32_RADIAL_BANDS = ((1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 7.0))
RUN32_F0_BAND_HZ = (2000.0, 6500.0)
RUN32_BB_BAND_HZ = (1000.0, 25000.0)
RUN32_MAX_POS_PER_BAND = 24
DEFAULT_X_CENTER_CM = 1.0


def _time_seconds(tarr):
    t = np.asarray(tarr, dtype=float)
    return t * 1e-3 if np.ptp(t) > 1.0 else t


def _blockmean_2d(arr, step):
    """Anti-aliased block-mean along axis 1 of a 2-D ``(nshot, nt)`` array."""
    step = max(1, int(step))
    n = arr.shape[1] // step
    if n < 1:
        return np.empty((arr.shape[0], 0), dtype=float)
    return arr[:, :n * step].reshape(arr.shape[0], n, step).mean(axis=2)


def _blockmean_1d(arr, step):
    step = max(1, int(step))
    n = arr.size // step
    if n < 1:
        return np.empty(0, dtype=float)
    return arr[:n * step].reshape(n, step).mean(axis=1)


def _detrend_relative(seg):
    """Linearly detrend each shot and divide by ``|shot_mean|``."""
    nsh, nt = seg.shape
    t_lin = np.linspace(-1.0, 1.0, nt)
    denom_t = float(np.dot(t_lin, t_lin))
    out = np.full_like(seg, np.nan, dtype=float)
    for k in range(nsh):
        row = seg[k]
        if not np.all(np.isfinite(row)):
            continue
        mu = float(np.mean(row))
        if abs(mu) <= 0.0:
            continue
        sl = float(np.dot(t_lin, row - mu) / denom_t)
        out[k] = (row - (mu + sl * t_lin)) / abs(mu)
    return out


def _nanmean_rows(arr):
    cnt = np.sum(np.isfinite(arr), axis=1)
    s = np.nansum(arr, axis=1)
    return np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1), np.nan)


def _wrap_deg(angle_deg):
    return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)


# ============================================================================
# 1. Stationary Isat broadband PSD & PDF moments across puff and B0 scans
# ============================================================================
def analyze_stationary_spectra_and_moments(data_dir=DATA_DIR, *, max_shots=DEFAULT_MAX_SHOTS):
    """Broadband exponential vs. power-law PSD and (skewness, kurtosis) on stationary Isat."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}

    for rnum in STATIONARY_RUNS:
        spec = jon.ONSET_RUN_MAP[rnum]
        ifn = jon.run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        t_s = _time_seconds(run.time_array(scope_name=spec.scope))
        dt_raw = float(t_s[1] - t_s[0])
        step = max(1, int(round((1.0 / TARGET_FS_PSD_HZ) / dt_raw)))
        fs = 1.0 / (dt_raw * step)

        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        f0 = float(on[f"{rnum}__f_peak_hz"])
        t_lo = max(t_on + 0.5, 1.8) if (np.isfinite(t_on) and t_on < 3.8) else 2.0
        t_hi = 4.8
        i0, i1 = clip_time_window(t_s, t_lo, t_hi)
        ib0, ib1 = clip_time_window(t_s, -1.5, -0.2)

        all_shots = np.asarray(run.shots(scope_name=spec.scope), dtype=int)
        idx = np.unique(np.linspace(0, all_shots.size - 1, min(int(max_shots), all_shots.size), dtype=int))
        shots = all_shots[idx].tolist()
        raw, _ = run.channel(spec.chan, scope_name=spec.scope, shots=shots)
        raw = np.asarray(raw, dtype=float)

        seg_sat = _blockmean_2d(raw[:, i0:i1], step)
        seg_pre = _blockmean_2d(raw[:, ib0:ib1], step)
        sgn = -1.0 if float(np.nanmedian(seg_sat)) < 0.0 else 1.0
        seg_sat = sgn * seg_sat
        seg_pre = sgn * seg_pre

        mom = fluctuation_moments(seg_sat, normalize_by_mean=True, detrend=True)
        fluc_sat = _detrend_relative(seg_sat)
        fluc_pre = _detrend_relative(seg_pre)
        valid_sat = np.all(np.isfinite(fluc_sat), axis=1)
        valid_pre = np.all(np.isfinite(fluc_pre), axis=1)

        nper = min(256, fluc_sat.shape[1], fluc_pre.shape[1])
        freqs, pxx = signal.welch(fluc_sat[valid_sat], fs=fs, nperseg=nper, axis=1)
        _, pxx_bg = signal.welch(fluc_pre[valid_pre], fs=fs, nperseg=nper, axis=1)
        psd = np.mean(pxx, axis=0)
        bg_psd = np.mean(pxx_bg, axis=0)

        fit = fit_broadband_spectrum(
            freqs, psd, bg_psd=bg_psd,
            harmonic_f0_hz=f0 if np.isfinite(f0) else None,
        )
        steep = harmonic_steepening(seg_sat, fs, f0 if np.isfinite(f0) else np.nan)

        puff = float(on[f"{rnum}__gas_puff_ms"]) if f"{rnum}__gas_puff_ms" in on.files else (
            float(spec.gas_puff_ms) if spec.gas_puff_ms is not None else np.nan
        )
        p = f"spec_{rnum}__"
        out[p + "gas_puff_ms"] = puff
        out[p + "b0_g"] = float(spec.b0_g)
        out[p + "group"] = str(spec.group)
        out[p + "t_onset_med_ms"] = t_on
        out[p + "f_peak_hz"] = f0
        out[p + "fs_hz"] = float(fs)
        out[p + "freqs_hz"] = freqs
        out[p + "psd"] = psd
        out[p + "bg_psd"] = bg_psd
        out[p + "coh_harmonic_frac"] = fit.coh_harmonic_frac
        out[p + "n_fit_bins"] = int(fit.n_fit_bins)
        out[p + "f_fit_lo_hz"] = fit.f_fit_lo_hz
        out[p + "f_fit_hi_hz"] = fit.f_fit_hi_hz
        out[p + "r2_exp"] = fit.r2_exp
        out[p + "f_e_hz"] = fit.f_e_hz
        out[p + "tau_lorentz_s"] = fit.tau_lorentz_s
        out[p + "r2_pow"] = fit.r2_pow
        out[p + "alpha"] = fit.alpha
        out[p + "preferred_model"] = fit.preferred_model
        out[p + "rel_rms_med"] = mom.rel_rms_med
        out[p + "skewness_med"] = mom.skewness_med
        out[p + "skewness_iqr"] = np.array(mom.skewness_iqr, dtype=float)
        out[p + "excess_kurtosis_med"] = mom.excess_kurtosis_med
        out[p + "excess_kurtosis_iqr"] = np.array(mom.excess_kurtosis_iqr, dtype=float)
        out[p + "n_shots"] = int(mom.n_shots)
        out[p + "a2_a1_med"] = steep.a2_a1_med
        out[p + "a3_a1_med"] = steep.a3_a1_med
        out[p + "biphase_lock"] = steep.biphase_lock
        out[p + "biphase_lock_crit"] = steep.biphase_lock_crit
        out[p + "biphase_locked"] = bool(steep.biphase_locked)
        out[p + "biphase_deg"] = steep.biphase_deg
        out[p + "curvature_comp"] = steep.curvature_comp
        out[p + "steepening_comp"] = steep.steepening_comp
        out[p + "hilbert_asymmetry_med"] = steep.hilbert_asymmetry_med
        out[p + "slope_ratio_med"] = steep.slope_ratio_med
        out[p + "rise_frac_med"] = steep.rise_frac_med
        out[p + "folded_phase_rad"] = steep.phase_rad
        out[p + "folded_wave"] = steep.folded_wave
        out[p + "folded_std"] = steep.folded_std
    return out


# ============================================================================
# 2. Radial (Skewness, Excess Kurtosis) across hollow core / wall / ring
# ============================================================================
def analyze_radial_moments(data_dir=DATA_DIR, *, x_center_cm=DEFAULT_X_CENTER_CM):
    """Flow-cancelled ``Isat_0`` radial skewness, kurtosis, and harmonic steepening on Runs 11-14."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}
    zones = (
        ("core", x_center_cm),
        ("inner_wall", x_center_cm + 2.5),
        ("outer_wall", x_center_cm + 4.0),
        ("ring", x_center_cm + 6.0),
    )
    for rnum, puff in RADIAL_MOMENT_RUNS:
        ifn = jon.run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        chans = jml.tip_channels(ifn)
        shunts = parse_shunts(run.description().raw)
        pos = jiv.read_lp_positions(ifn)
        x_pos, _ = jml.scanned_axis(pos)
        sc, ch_yp = chans["Y+"]
        _, ch_ym = chans["Y-"]
        t_s = _time_seconds(run.time_array(scope_name=sc))
        dt = float(t_s[1] - t_s[0])
        step = max(1, int(round((1.0 / TARGET_FS_TRANSPORT_HZ) / dt)))
        fs = 1.0 / (dt * step)
        i0, i1 = clip_time_window(t_s, 2.5, 4.8)
        f0 = float(on[f"{rnum}__f_peak_hz"]) if f"{rnum}__f_peak_hz" in on.files else np.nan

        def _read_isat0(p_idx):
            sh = position_shots(pos.shot_nums, int(p_idx), pos.nshot)
            yp, _ = run.channel(ch_yp, scope_name=sc, shots=sh)
            ym, _ = run.channel(ch_ym, scope_name=sc, shots=sh)
            if yp is None or ym is None:
                return None
            yp_d = _blockmean_2d(np.asarray(yp, dtype=float)[:, i0:i1], step) / float(shunts["Y+"])
            ym_d = _blockmean_2d(np.asarray(ym, dtype=float)[:, i0:i1], step) / float(shunts["Y-"])
            if float(np.nanmedian(yp_d)) < 0.0:
                yp_d = -yp_d
            if float(np.nanmedian(ym_d)) < 0.0:
                ym_d = -ym_d
            ok = (yp_d > 0.0) & (ym_d > 0.0)
            return np.where(ok, np.sqrt(np.where(ok, yp_d * ym_d, 1.0)), np.nan)

        for zname, x_t in zones:
            p_idx = int(np.argmin(np.abs(x_pos - float(x_t))))
            isat0 = _read_isat0(p_idx)
            pz = f"radmom_{rnum}_{zname}__"
            out[pz + "gas_puff_ms"] = float(puff)
            out[pz + "x_cm"] = float(x_pos[p_idx])
            if isat0 is None:
                out[pz + "rel_rms_med"] = np.nan
                out[pz + "skewness_med"] = np.nan
                out[pz + "excess_kurtosis_med"] = np.nan
                continue
            mom = fluctuation_moments(isat0, normalize_by_mean=True, detrend=True)
            out[pz + "rel_rms_med"] = mom.rel_rms_med
            out[pz + "skewness_med"] = mom.skewness_med
            out[pz + "excess_kurtosis_med"] = mom.excess_kurtosis_med

        # Full radial steepening profile across unique grid positions r in [0, 7] cm
        r_all = x_pos - float(x_center_cm)
        sel = np.flatnonzero((r_all >= -0.2) & (r_all <= 7.2))
        sel = sel[np.argsort(r_all[sel])]
        nr = sel.size
        pr = f"radst_{rnum}__"
        out[pr + "gas_puff_ms"] = float(puff)
        out[pr + "f_peak_hz"] = f0
        out[pr + "r_cm"] = r_all[sel].astype(float)
        n0_v = np.full(nr, np.nan, dtype=float)
        a21_v = np.full(nr, np.nan, dtype=float)
        a31_v = np.full(nr, np.nan, dtype=float)
        blk_v = np.full(nr, np.nan, dtype=float)
        bcr_v = np.full(nr, np.nan, dtype=float)
        blc_v = np.zeros(nr, dtype=bool)
        bet_v = np.full(nr, np.nan, dtype=float)
        crv_v = np.full(nr, np.nan, dtype=float)
        stp_v = np.full(nr, np.nan, dtype=float)
        skw_v = np.full(nr, np.nan, dtype=float)
        ahb_v = np.full(nr, np.nan, dtype=float)
        slp_v = np.full(nr, np.nan, dtype=float)
        rfr_v = np.full(nr, np.nan, dtype=float)
        fld_v = np.full((nr, 64), np.nan, dtype=float)
        phase_rad = np.linspace(-np.pi, np.pi, 65)
        phase_rad = 0.5 * (phase_rad[:-1] + phase_rad[1:])
        for ii, p_idx in enumerate(sel):
            isat0 = _read_isat0(p_idx)
            if isat0 is None:
                continue
            n0_v[ii] = float(np.nanmean(isat0))
            st = harmonic_steepening(isat0, fs, f0)
            a21_v[ii] = st.a2_a1_med
            a31_v[ii] = st.a3_a1_med
            blk_v[ii] = st.biphase_lock
            bcr_v[ii] = st.biphase_lock_crit
            blc_v[ii] = bool(st.biphase_locked)
            bet_v[ii] = st.biphase_deg
            crv_v[ii] = st.curvature_comp
            stp_v[ii] = st.steepening_comp
            skw_v[ii] = st.skewness_med
            ahb_v[ii] = st.hilbert_asymmetry_med
            slp_v[ii] = st.slope_ratio_med
            rfr_v[ii] = st.rise_frac_med
            fld_v[ii] = st.folded_wave
            phase_rad = st.phase_rad
        out[pr + "n0_mean"] = n0_v
        out[pr + "a2_a1"] = a21_v
        out[pr + "a3_a1"] = a31_v
        out[pr + "biphase_lock"] = blk_v
        out[pr + "biphase_lock_crit"] = bcr_v
        out[pr + "biphase_locked"] = blc_v
        out[pr + "biphase_deg"] = bet_v
        out[pr + "curvature_comp"] = crv_v
        out[pr + "steepening_comp"] = stp_v
        out[pr + "skewness"] = skw_v
        out[pr + "hilbert_asymmetry"] = ahb_v
        out[pr + "slope_ratio"] = slp_v
        out[pr + "rise_frac"] = rfr_v
        out[pr + "folded_phase_rad"] = phase_rad
        out[pr + "folded_waves"] = fld_v
    return out


# ============================================================================
# 3. Per-shot amplitude-normalized growth -> early-sat -> late-sat cross-phase
# ============================================================================
def _resample_pair_common_grid(t_a_s, sig_a, t_b_s, sig_b, t_lo_ms=0.0, t_hi_ms=4.9,
                               *, target_fs_hz=TARGET_FS_TRANSPORT_HZ):
    """Block-mean downsample two channels onto a common uniform time grid [s]."""
    pad = 0.05
    ia0, ia1 = clip_time_window(t_a_s, t_lo_ms - pad, t_hi_ms + pad)
    ib0, ib1 = clip_time_window(t_b_s, t_lo_ms - pad, t_hi_ms + pad)
    ta_seg = t_a_s[ia0:ia1]
    tb_seg = t_b_s[ib0:ib1]
    step_a = max(1, int(round((1.0 / target_fs_hz) / float(ta_seg[1] - ta_seg[0]))))
    step_b = max(1, int(round((1.0 / target_fs_hz) / float(tb_seg[1] - tb_seg[0]))))
    ta_ds = _blockmean_1d(ta_seg, step_a)
    tb_ds = _blockmean_1d(tb_seg, step_b)
    nsh = min(sig_a.shape[0], sig_b.shape[0])
    sa_ds = _blockmean_2d(np.asarray(sig_a[:nsh, ia0:ia1], dtype=float), step_a)
    sb_ds = _blockmean_2d(np.asarray(sig_b[:nsh, ib0:ib1], dtype=float), step_b)

    t_grid = np.arange(t_lo_ms * 1e-3, t_hi_ms * 1e-3, 1.0 / float(target_fs_hz))
    a_arr = np.empty((nsh, t_grid.size), dtype=float)
    b_arr = np.empty((nsh, t_grid.size), dtype=float)
    for k in range(nsh):
        a_arr[k] = np.interp(t_grid, ta_ds, sa_ds[k])
        b_arr[k] = np.interp(t_grid, tb_ds, sb_ds[k])
    return t_grid, a_arr, b_arr


def _per_shot_ramp_masks(t_grid_s, isat_arr, f_band_hz):
    """Detect per-shot onset ``t_on_k`` plus Hilbert ``t_50_k`` & ``t_90_k`` -> non-overlapping masks.

    Uses the narrowband ``f_band_hz`` Hilbert amplitude envelope (lightly
    smoothed over 50 us) instead of the broad 0.5 ms Hann window to locate the
    50% (``t_50_k``) and 90% (``t_90_k``) crossings of each shot's post-onset
    peak so fast ramps (``gamma ~ 5-10 ms^-1``) are resolved cleanly:

    - ``mask_grow`` covers ``[t_on_k - 0.10, t_50_k] ms`` (strictly rising ramp
      below 50% of peak amplitude),
    - ``mask_early`` covers ``[t_90_k + 0.05, min(4.8, t_90_k + 0.75)] ms``
      (strictly after saturation is reached at 90% of peak, preventing ramp
      leakage into ``early_sat``),
    - ``mask_late`` covers ``[max(t_90_k + 0.80, 3.0), 4.8] ms``.

    Returns ``(mask_grow, mask_early, mask_late, t_on_ms, ramp_dur_ms, t_90_ms)``.
    """
    nsh, nt = isat_arr.shape
    t_ms = t_grid_s * 1e3
    fs_hz = 1.0 / float(t_grid_s[1] - t_grid_s[0])
    m_grow = np.zeros((nsh, nt), dtype=bool)
    m_early = np.zeros((nsh, nt), dtype=bool)
    m_late = np.zeros((nsh, nt), dtype=bool)
    t_ons = np.full(nsh, np.nan, dtype=float)
    ramp_durs = np.full(nsh, np.nan, dtype=float)
    t_90s = np.full(nsh, np.nan, dtype=float)

    t_env, env = sliding_detrended_rms(isat_arr, t_grid_s, win_s=jon.WIN_S, step_s=jon.STEP_S)
    t_on_s = detect_onset_time(
        t_env, env, threshold=jon.ONSET_THRESH,
        t_min=jon.T_MIN_ONSET_MS * 1e-3, sustain=jon.SUSTAIN_BINS,
    )

    # High-time-resolution Hilbert amplitude envelope in f_band_hz (50 us light smoothing)
    sos = signal.butter(2, [float(f_band_hz[0]), float(f_band_hz[1])],
                        btype="bandpass", fs=fs_hz, output="sos")
    det = isat_arr - np.mean(isat_arr, axis=1, keepdims=True)
    env_h = np.abs(signal.hilbert(signal.sosfiltfilt(sos, det, axis=1), axis=1))
    k_sm = max(1, int(round(5e-5 * fs_hz)))
    if k_sm > 1:
        kern = np.ones(k_sm, dtype=float) / float(k_sm)
        env_h = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, env_h)

    for k in range(nsh):
        if not np.isfinite(t_on_s[k]):
            continue
        ton_ms = float(t_on_s[k]) * 1e3
        t_ons[k] = ton_ms
        m_after = (t_ms >= ton_ms) & (t_ms <= 4.8) & np.isfinite(env_h[k])
        if not np.any(m_after):
            continue
        idx_a = np.flatnonzero(m_after)
        env_pk = float(np.max(env_h[k, idx_a]))
        if env_pk <= 0.0:
            continue

        hit50 = idx_a[env_h[k, idx_a] >= 0.50 * env_pk]
        hit90 = idx_a[env_h[k, idx_a] >= 0.90 * env_pk]
        if hit50.size == 0 or hit90.size == 0:
            continue
        t50_ms = float(t_ms[hit50[0]])
        t90_ms = float(t_ms[hit90[0]])
        if t90_ms < t50_ms:
            t90_ms = t50_ms
        ramp_durs[k] = t50_ms - ton_ms
        t_90s[k] = t90_ms

        grow_lo = max(0.2, ton_ms - 0.10)
        if t50_ms - grow_lo >= 0.08:
            m_grow[k] = (t_ms >= grow_lo) & (t_ms <= t50_ms)
        early_lo = t90_ms + 0.05
        early_hi = min(4.8, t90_ms + 0.75)
        if early_hi - early_lo >= 0.20:
            m_early[k] = (t_ms >= early_lo) & (t_ms <= early_hi)
        late_lo = max(t90_ms + 0.80, 3.0)
        if 4.8 - late_lo >= 0.30:
            m_late[k] = (t_ms >= late_lo) & (t_ms <= 4.8)

    return m_grow, m_early, m_late, t_ons, ramp_durs, t_90s


def analyze_growth_vs_sat_crossphase(data_dir=DATA_DIR, *, max_shots=60,
                                     biasscope_offset_ms=jmo.BIASSCOPE_OFFSET_MS):
    """Per-shot amplitude-normalized ``Isat`` vs. Port-35 cross-phase: linear ramp -> early sat -> late sat."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}

    for rnum, puff in CROSSPHASE_RUNS:
        ifn = jon.run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        f0 = float(on[f"{rnum}__f_peak_hz"])
        f_band = (max(1500.0, f0 - 1500.0), f0 + 1500.0)
        tm_s = _time_seconds(run.time_array(scope_name="machscope"))
        tb_rel_s = _time_seconds(run.time_array(scope_name="biasscope")) - float(biasscope_offset_ms) * 1e-3

        all_sh = np.asarray(run.shots(scope_name="machscope"), dtype=int)
        idx = np.unique(np.linspace(0, all_sh.size - 1, min(int(max_shots), all_sh.size), dtype=int))
        sh = all_sh[idx].tolist()

        isat_raw, _ = run.channel("C2", scope_name="machscope", shots=sh)
        i2_raw, _ = run.channel("C1", scope_name="biasscope", shots=sh)
        i4_raw, _ = run.channel("C2", scope_name="biasscope", shots=sh)
        v2_raw, _ = run.channel("C4", scope_name="biasscope", shots=sh)

        t_grid, isat_g, i2_g = _resample_pair_common_grid(tm_s, isat_raw, tb_rel_s, i2_raw)
        _, _, i4_g = _resample_pair_common_grid(tm_s, isat_raw, tb_rel_s, i4_raw)
        _, _, v2_g = _resample_pair_common_grid(tm_s, isat_raw, tb_rel_s, v2_raw)

        m_grow, m_early, m_late, t_ons, ramp_durs, t_90s = _per_shot_ramp_masks(t_grid, isat_g, f_band)
        n_unstable = int(np.sum(np.isfinite(t_ons)))
        t_on_ps_med = float(np.nanmedian(t_ons)) if n_unstable else np.nan
        ramp_med = float(np.nanmedian(ramp_durs)) if np.any(np.isfinite(ramp_durs)) else np.nan
        t90_med = float(np.nanmedian(t_90s)) if np.any(np.isfinite(t_90s)) else np.nan

        pairs = (
            ("isat_i2", isat_g, i2_g),
            ("isat_i4", isat_g, i4_g),
            ("isat_v2", isat_g, v2_g),
            ("i2_i4", i2_g, i4_g),  # same-scope control!
        )
        for tag, sa, sb in pairs:
            plv_g, phi_g, _ = amplitude_normalized_cross_phase(sa, sb, TARGET_FS_TRANSPORT_HZ, f_band, mask=m_grow)
            plv_e, phi_e, _ = amplitude_normalized_cross_phase(sa, sb, TARGET_FS_TRANSPORT_HZ, f_band, mask=m_early)
            plv_s, phi_s, _ = amplitude_normalized_cross_phase(sa, sb, TARGET_FS_TRANSPORT_HZ, f_band, mask=m_late)

            shift_gs = _wrap_deg(phi_s - phi_g) if (np.isfinite(phi_s) and np.isfinite(phi_g)) else np.nan
            shift_es = _wrap_deg(phi_s - phi_e) if (np.isfinite(phi_s) and np.isfinite(phi_e)) else np.nan
            p = f"xph_{rnum}_{tag}__"
            out[p + "gas_puff_ms"] = float(puff)
            out[p + "t_onset_med_ms"] = t_on
            out[p + "t_on_per_shot_med_ms"] = t_on_ps_med
            out[p + "t_90_med_ms"] = t90_med
            out[p + "n_unstable_shots"] = n_unstable
            out[p + "ramp_dur_med_ms"] = ramp_med
            out[p + "plv_grow"] = plv_g
            out[p + "phi_grow_deg"] = phi_g
            out[p + "plv_early_sat"] = plv_e
            out[p + "phi_early_sat_deg"] = phi_e
            out[p + "plv_sat"] = plv_s
            out[p + "phi_sat_deg"] = phi_s
            out[p + "phi_shift_grow_to_sat_deg"] = shift_gs
            out[p + "phi_shift_early_to_sat_deg"] = shift_es
    return out


# ============================================================================
# 4. Run 32 direct radial particle flux, Reynolds stress & spiral sign check
# ============================================================================
def _compute_mach_transport_at_position(ch_data, kappa, dx_cm, dy_cm, fs_hz,
                                        *, ref_ds=None,
                                        f0_band_hz=RUN32_F0_BAND_HZ,
                                        bb_band_hz=RUN32_BB_BAND_HZ):
    """Pure helper: radial flux, Reynolds stress, and ref-locked ``Isat_0`` phasor at one position."""
    xp, xm, yp, ym = ch_data["X+"], ch_data["X-"], ch_data["Y+"], ch_data["Y-"]
    if np.any(xp <= 0.0) or np.any(xm <= 0.0) or np.any(yp <= 0.0) or np.any(ym <= 0.0):
        ok = (xp > 0.0) & (xm > 0.0) & (yp > 0.0) & (ym > 0.0)
        if not np.all(np.any(ok, axis=1)):
            return None
        xp = np.where(ok, xp, np.nan)
        xm = np.where(ok, xm, np.nan)
        yp = np.where(ok, yp, np.nan)
        ym = np.where(ok, ym, np.nan)
        if not np.all(np.isfinite(xp)):
            return None

    mx = mach_single(xp / xm, kappa["X"])
    my = mach_single(yp / ym, kappa["Y"])
    isat0 = (xp * xm * yp * ym) ** 0.25

    r_p = float(np.hypot(dx_cm, dy_cm))
    if r_p <= 0.0:
        return None
    ur, ut = dx_cm / r_p, dy_cm / r_p
    mr = mx * ur + my * ut
    mt = -mx * ut + my * ur

    sos_f0 = signal.butter(4, list(f0_band_hz), btype="bandpass", fs=fs_hz, output="sos")
    sos_bb = signal.butter(4, list(bb_band_hz), btype="bandpass", fs=fs_hz, output="sos")

    n_mean = np.mean(isat0, axis=1, keepdims=True)
    dn_raw = (isat0 - n_mean) / np.maximum(n_mean, 1e-12)
    dmr_raw = mr - np.mean(mr, axis=1, keepdims=True)
    dmt_raw = mt - np.mean(mt, axis=1, keepdims=True)

    dn_f0 = signal.sosfiltfilt(sos_f0, dn_raw, axis=1)
    mr_f0 = signal.sosfiltfilt(sos_f0, dmr_raw, axis=1)
    mt_f0 = signal.sosfiltfilt(sos_f0, dmt_raw, axis=1)

    dn_bb = signal.sosfiltfilt(sos_bb, dn_raw, axis=1)
    mr_bb = signal.sosfiltfilt(sos_bb, dmr_raw, axis=1)
    mt_bb = signal.sosfiltfilt(sos_bb, dmt_raw, axis=1)

    sig_n = float(np.std(dn_f0))
    sig_r = float(np.std(mr_f0))
    sig_t = float(np.std(mt_f0))
    et_f0 = 0.5 * (float(np.mean(mr_f0 ** 2)) + float(np.mean(mt_f0 ** 2)))
    et_bb = 0.5 * (float(np.mean(mr_bb ** 2)) + float(np.mean(mt_bb ** 2)))
    flux_f0 = float(np.mean(dn_f0 * mr_f0))
    rs_f0 = float(np.mean(mr_f0 * mt_f0))

    phasor = np.nan + 0j
    if ref_ds is not None and np.all(np.isfinite(ref_ds)):
        rmu = np.mean(ref_ds, axis=1, keepdims=True)
        dref = signal.sosfiltfilt(sos_f0, (ref_ds - rmu) / np.maximum(np.abs(rmu), 1e-12), axis=1)
        an = signal.hilbert(dn_f0, axis=1)
        ar = signal.hilbert(dref, axis=1)
        denom = float(np.sqrt(np.mean(np.abs(an) ** 2) * np.mean(np.abs(ar) ** 2)))
        if denom > 0.0:
            phasor = complex(np.mean(an * np.conj(ar)) / denom)

    return {
        "dn_rms": sig_n,
        "mr_rms": sig_r,
        "mt_rms": sig_t,
        "et_f0": et_f0,
        "et_bb": et_bb,
        "flux_f0": flux_f0,
        "flux_bb": float(np.mean(dn_bb * mr_bb)),
        "rs_f0": rs_f0,
        "rs_bb": float(np.mean(mr_bb * mt_bb)),
        "c_nr": float(flux_f0 / max(sig_n * sig_r, 1e-30)),
        "c_rt": float(rs_f0 / max(sig_r * sig_t, 1e-30)),
        "phasor": phasor,
        "isat0_raw": isat0,
        "mr_raw": mr,
        "mt_raw": mt,
        "tips_raw": {"X+": xp, "X-": xm, "Y+": yp, "Y-": ym},
        "ur": ur,
        "ut": ut,
    }


def _eich_manz_shear_transfer(
    r_mids_cm,
    mth_band,
    rs_f0_band,
    rs_bb_band,
    et_f0_mean,
    et_bb_mean,
    te_band_ev,
    ne_band_cm3,
    er_band_v_cm,
    *,
    b0_g=800.0,
    gamma_exp_s=np.nan,
    gamma_cwm_s=np.nan,
    mu=4.0,
    b_sign_z=+1,
    wall_r_max_cm=4.5,
):
    """Physical-velocity Reynolds transfer & Eich-Manz (NF 2021, Eq. H.2) ratio ``P_EM / (gamma E_t)``.

    Uses Cauchy-Schwarz-consistent band-mean fluctuation kinetic energy
    ``E_t = 0.5 * (<dM_r^2> + <dM_theta^2>) * c_s(r)^2`` and mass-weighted
    cylindrical integration ``int n_0(r) * P_EM(r) * r dr / int n_0(r) * E_t(r) * r dr``
    with mean-flow-budget sign ``P_EM = +<dv_r dv_theta> * r * d(v_theta / r) / dr``
    (positive = turbulence feeds mean shear flow, opposite sign to ``P_shear``).

    Evaluates both radial-``Te(r)``-rescaled Mach velocity (``gamma = 1`` isothermal
    and ``gamma = 5/3`` adiabatic) and independent Langmuir-probe ``v_ExB = -Er / B0``
    (where both ``c_s`` and Hutchinson's ``K_H`` cancel identically in the local ratio).
    """
    r_cm = np.asarray(r_mids_cm, dtype=float)
    r_m = r_cm * 1e-2
    mth = np.asarray(mth_band, dtype=float)
    rs_f0 = np.asarray(rs_f0_band, dtype=float)
    rs_bb = np.asarray(rs_bb_band, dtype=float)
    et_f0 = np.asarray(et_f0_mean, dtype=float)
    et_bb = np.asarray(et_bb_mean, dtype=float)
    te = np.asarray(te_band_ev, dtype=float)
    ne = np.asarray(ne_band_cm3, dtype=float)
    er = np.asarray(er_band_v_cm, dtype=float)

    mi_kg = float(mu) * const.proton_mass
    b0_t = float(b0_g) * 1e-4
    sgn = int(np.sign(b_sign_z or 1))
    valid_te = np.isfinite(te) & (te > 0.0)
    cs1 = np.where(valid_te, np.sqrt(const.e * np.where(valid_te, te, 1.0) / mi_kg), np.nan)
    cs53 = cs1 * np.sqrt(5.0 / 3.0)

    v_mach1 = mth * cs1
    v_mach53 = mth * cs53
    v_exb = -sgn * (er * 1e2) / b0_t

    def _cyl_shear(v_arr):
        ok = np.isfinite(v_arr) & np.isfinite(r_m) & (r_m > 0.0)
        if int(np.sum(ok)) < 2:
            return np.full_like(r_m, np.nan)
        out_g = np.full_like(r_m, np.nan)
        rv = r_m[ok]
        out_g[ok] = rv * np.gradient(v_arr[ok] / rv, rv)
        return out_g

    gE_mach1 = _cyl_shear(v_mach1)
    gE_mach53 = _cyl_shear(v_mach53)
    gE_exb = _cyl_shear(v_exb)

    w_mass = np.where(np.isfinite(ne) & (ne > 0.0), ne, np.nan)
    wall_m = r_cm < float(wall_r_max_cm)

    def _rate(rs_arr, et_arr, cs_arr, gE_arr):
        p_em = rs_arr * (cs_arr ** 2) * gE_arr
        et_p = et_arr * (cs_arr ** 2)
        ok = np.isfinite(p_em) & np.isfinite(et_p) & np.isfinite(w_mass) & (et_p > 0.0)
        if int(np.sum(ok)) >= 2:
            num = float(trapezoid(w_mass[ok] * p_em[ok] * r_m[ok], r_m[ok]))
            den = float(trapezoid(w_mass[ok] * et_p[ok] * r_m[ok], r_m[ok]))
            g_full = num / den if den > 0.0 else np.nan
        else:
            g_full = np.nan
        ok_w = ok & wall_m
        if int(np.sum(ok_w)) >= 2:
            num_w = float(trapezoid(w_mass[ok_w] * p_em[ok_w] * r_m[ok_w], r_m[ok_w]))
            den_w = float(trapezoid(w_mass[ok_w] * et_p[ok_w] * r_m[ok_w], r_m[ok_w]))
            g_wall = num_w / den_w if den_w > 0.0 else np.nan
        else:
            g_wall = np.nan
        return p_em, g_full, g_wall

    p_em_m1_f0, g_m1_f0, g_m1_f0_w = _rate(rs_f0, et_f0, cs1, gE_mach1)
    _, g_m53_f0, g_m53_f0_w = _rate(rs_f0, et_f0, cs53, gE_mach53)
    p_em_exb_f0, g_exb_f0, g_exb_f0_w = _rate(rs_f0, et_f0, cs1, gE_exb)
    _, g_m1_bb, g_m1_bb_w = _rate(rs_bb, et_bb, cs1, gE_mach1)
    _, g_exb_bb, g_exb_bb_w = _rate(rs_bb, et_bb, cs1, gE_exb)

    def _ratio(g_sh, g_ref):
        return float(g_sh / g_ref) if (np.isfinite(g_sh) and np.isfinite(g_ref) and g_ref > 0.0) else np.nan

    return {
        "te_band_ev": te,
        "ne_band_cm3": ne,
        "er_band_v_cm": er,
        "cs1_band_m_s": cs1,
        "v_theta_mach_m_s": v_mach1,
        "v_theta_exb_m_s": v_exb,
        "gamma_E_mach_s": gE_mach1,
        "gamma_E_exb_s": gE_exb,
        "p_em_mach_f0_m2_s3": p_em_m1_f0,
        "p_em_exb_f0_m2_s3": p_em_exb_f0,
        "gamma_sh_mach_f0_s": g_m1_f0,
        "gamma_sh_mach_f0_wall_s": g_m1_f0_w,
        "gamma_sh_mach_f0_gam53_s": g_m53_f0,
        "gamma_sh_mach_f0_gam53_wall_s": g_m53_f0_w,
        "gamma_sh_exb_f0_s": g_exb_f0,
        "gamma_sh_exb_f0_wall_s": g_exb_f0_w,
        "gamma_sh_mach_bb_s": g_m1_bb,
        "gamma_sh_mach_bb_wall_s": g_m1_bb_w,
        "gamma_sh_exb_bb_s": g_exb_bb,
        "gamma_sh_exb_bb_wall_s": g_exb_bb_w,
        "gamma_exp_ref_s": float(gamma_exp_s),
        "gamma_cwm_ref_s": float(gamma_cwm_s),
        "R_em_mach_f0_vs_exp": _ratio(g_m1_f0, gamma_exp_s),
        "R_em_mach_f0_wall_vs_exp": _ratio(g_m1_f0_w, gamma_exp_s),
        "R_em_mach_f0_vs_cwm": _ratio(g_m1_f0, gamma_cwm_s),
        "R_em_mach_f0_gam53_vs_exp": _ratio(g_m53_f0, gamma_exp_s),
        "R_em_exb_f0_vs_exp": _ratio(g_exb_f0, gamma_exp_s),
        "R_em_exb_f0_wall_vs_exp": _ratio(g_exb_f0_w, gamma_exp_s),
        "R_em_exb_f0_vs_cwm": _ratio(g_exb_f0, gamma_cwm_s),
        "R_em_mach_bb_vs_exp": _ratio(g_m1_bb, gamma_exp_s),
        "R_em_exb_bb_vs_exp": _ratio(g_exb_bb, gamma_exp_s),
    }


def _mach_linear_tip_bracket(recs, kappa, fs_hz, f0_hz):
    """Model-dependence bracket: apparent M steepening if tip currents, not M, were linear sinusoids.

    Under Hutchinson's exponential Mach law ``I_+- ~ n * exp(-+ M / (2 K_H))``, the
    log-ratio ``M = -K_H ln(I_+ / (kappa I_-))`` cancels density fluctuations
    exactly and returns ``0`` steepening for a pure-sinusoid ``M`` even at large
    ``dn/n``.  Conversely, if the probe tip response at ``~4 kHz`` were linear
    (purely sinusoidal tip currents rather than exponential), passing those
    sinusoidal tips through ``ln(I_+/I_-)`` would produce a spurious ``sin(beta)``
    second harmonic.

    This helper replaces each measured tip trace with its DC mean plus narrowband
    ``1 f_0`` fundamental only, applies ``mach_single`` and polar projection, and
    returns the resulting ``(isat0, mr, mt)`` stacks.  **Do not subtract this
    bracket from calibrated ``M`` steepening** (subtracting it when Hutchinson
    holds invents fake negative steepening); instead, compare raw ``M`` steepening
    against it — values well outside the bracket are robust under either model.
    """
    if not recs or not np.isfinite(f0_hz) or f0_hz <= 0.0:
        return None
    f_lo = max(500.0, 0.65 * float(f0_hz))
    f_hi = min(1.35 * float(f0_hz), 0.45 * float(fs_hz))
    if not (0.0 < f_lo < f_hi < 0.5 * float(fs_hz)):
        return None
    sos_f1 = signal.butter(4, [f_lo, f_hi], btype="bandpass", fs=float(fs_hz), output="sos")

    out = {"isat0": [], "mr": [], "mt": []}
    for r in recs:
        tips = r["tips_raw"]
        sin_d = {}
        ok = True
        for tip in ("X+", "X-", "Y+", "Y-"):
            arr = tips[tip]
            mu = np.mean(arr, axis=1, keepdims=True)
            tsin = mu + signal.sosfiltfilt(sos_f1, arr - mu, axis=1)
            if not np.all(np.isfinite(tsin)) or np.any(tsin <= 0.0):
                ok = False
                break
            sin_d[tip] = tsin
        if not ok:
            continue
        mx_s = mach_single(sin_d["X+"] / sin_d["X-"], kappa["X"])
        my_s = mach_single(sin_d["Y+"] / sin_d["Y-"], kappa["Y"])
        ur, ut = r["ur"], r["ut"]
        out["isat0"].append((sin_d["X+"] * sin_d["X-"] * sin_d["Y+"] * sin_d["Y-"]) ** 0.25)
        out["mr"].append(mx_s * ur + my_s * ut)
        out["mt"].append(-mx_s * ut + my_s * ur)
    if not out["mr"]:
        return None
    return {k: np.vstack(v) for k, v in out.items()}


def _resolve_run32_channels(ifn):
    """Resolve Run 32 ``(X+, X-, Y+, Y-, REF)`` scope channels from HDF5 wiring descriptions.

    Uses :func:`Jun2026_mach_flow.tip_channels` for the Mach probe faces and
    locates the stationary ``LP@P29-R`` reference probe from
    :func:`Jun2026_mach_flow.channel_wiring`, avoiding any hardcoded ``C1=X+`` map.
    """
    tips = jmf.tip_channels(ifn)
    ch_map = {tip: tips[tip] for tip in ("X+", "X-", "Y+", "Y-")}
    ref_hits = [
        chan for (scope, chan), desc in jmf.channel_wiring(ifn).items()
        if scope == "scope" and "p29-r" in desc.lower()
    ]
    if len(ref_hits) != 1:
        raise ValueError(
            f"{ifn}: expected exactly 1 'LP@P29-R' reference channel on 'scope', found {ref_hits}"
        )
    ch_map["REF"] = ref_hits[0]
    return ch_map


def analyze_run32_transport_and_reynolds(data_dir=DATA_DIR, *, cal_npz=None,
                                         bands=RUN32_RADIAL_BANDS,
                                         max_pos_per_band=RUN32_MAX_POS_PER_BAND):
    """Run 32 direct radial particle flux, Reynolds stress, 4-quadrant audit & spiral check."""
    cal_path = cal_npz or os.path.join(data_dir, os.path.basename(jmf.CAL_NPZ))
    kappa, _, _ = jmf.load_calibration(cal_path)

    # 1. Exact flow center and mean M_theta(r) from cached 32-mach-flow-data.npz
    npz32 = os.path.join(data_dir, "32-mach-flow-data.npz")
    if not os.path.exists(npz32):
        raise FileNotFoundError(f"missing {npz32}; run Jun2026_mach_flow.py first")
    d32 = np.load(npz32, allow_pickle=True)
    t_npz = np.asarray(d32["t_ms"], dtype=float)
    px_n, py_n = np.asarray(d32["pos_x"], dtype=float), np.asarray(d32["pos_y"], dtype=float)
    mx_n, my_n = np.asarray(d32["M_X"], dtype=float), np.asarray(d32["M_Y"], dtype=float)
    m_on = (t_npz >= 2.0) & (t_npz <= 4.8)
    m_pre = t_npz < -0.5
    mx_m = _nanmean_rows(mx_n[:, m_on])
    my_m = _nanmean_rows(my_n[:, m_on])
    dmx_m = mx_m - _nanmean_rows(mx_n[:, m_pre]) if np.any(m_pre) else mx_m
    dmy_m = my_m - _nanmean_rows(my_n[:, m_pre]) if np.any(m_pre) else my_m
    # Velocities FIRST, positions SECOND! Fit center on pre-bias-subtracted dM so
    # uniform kappa/background offsets do not shift the zero-rotation point.
    cx, cy, ratio = find_flow_centre(dmx_m, dmy_m, px_n, py_n)
    _, mth_mean = polar_components(mx_m, my_m, px_n, py_n, (cx, cy))
    rr_n = np.hypot(px_n - cx, py_n - cy)

    # 2. Read raw Run 32 6-tip Mach fluctuations + stationary P29-R ref (from HDF5 wiring)
    hits = sorted(glob.glob(os.path.join(data_dir, "32-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no Run 32 HDF5 in {data_dir}")
    ifn32 = hits[0]
    ch_map32 = _resolve_run32_channels(ifn32)
    run32 = open_lapd(ifn32)
    sh32 = parse_shunts(run32.description().raw)
    pos32 = jiv.read_lp_positions(ifn32)
    pa32 = pos32.pos_array
    px = np.asarray(pa32["x"][::pos32.nshot], dtype=float)
    py = np.asarray(pa32["y"][::pos32.nshot], dtype=float)
    rr = np.hypot(px - cx, py - cy)
    th = np.arctan2(py - cy, px - cx)

    t_s = _time_seconds(run32.time_array(scope_name="scope"))
    dt_raw = float(t_s[1] - t_s[0])
    step = max(1, int(round((1.0 / TARGET_FS_TRANSPORT_HZ) / dt_raw)))
    fs = 1.0 / (dt_raw * step)
    i0, i1 = clip_time_window(t_s, 2.0, 4.8)

    nb = len(bands)
    r_mids = np.array([0.5 * (a + b) for a, b in bands], dtype=float)
    mth_band = np.full(nb, np.nan, dtype=float)
    dn_band = np.full(nb, np.nan, dtype=float)
    mr_band = np.full(nb, np.nan, dtype=float)
    mt_band = np.full(nb, np.nan, dtype=float)
    et_f0_band = np.full(nb, np.nan, dtype=float)
    et_bb_band = np.full(nb, np.nan, dtype=float)
    flux_f0_band = np.full(nb, np.nan, dtype=float)
    flux_bb_band = np.full(nb, np.nan, dtype=float)
    c_nr_band = np.full(nb, np.nan, dtype=float)
    rs_f0_band = np.full(nb, np.nan, dtype=float)
    rs_bb_band = np.full(nb, np.nan, dtype=float)
    c_rt_band = np.full(nb, np.nan, dtype=float)
    m_mode_z_band = np.full(nb, np.nan, dtype=float)
    phi_spiral_deg = np.full(nb, np.nan, dtype=float)
    n_pos_band = np.zeros(nb, dtype=int)

    # Collect wall [2.5, 4.5) cm positions for 4-quadrant Cartesian-artifact audit
    quad_th_deg, quad_crt, quad_cnr = [], [], []
    # Band-level harmonic steepening on density (isat0) and velocity (mr, mt),
    # plus an unsubtracted linear-tip model-dependence bracket for M = -K_H ln(I_+/I_-).
    steep_tags = ("isat0", "mr", "mt")
    steep_metrics = ("a2_a1", "biphase_lock", "biphase_lock_crit", "biphase_deg",
                     "curvature_comp", "steepening_comp", "skewness",
                     "hilbert_asymmetry", "slope_ratio", "rise_frac")
    r32_steep = {f"{t}_{m}": np.full(nb, np.nan, dtype=float) for t in steep_tags for m in steep_metrics}
    r32_steep_locked = {t: np.zeros(nb, dtype=bool) for t in steep_tags}
    bracket_metrics = ("a2_a1", "biphase_lock", "biphase_deg", "curvature_comp",
                       "steepening_comp", "slope_ratio")
    r32_lintip = {f"{t}_{m}": np.full(nb, np.nan, dtype=float) for t in steep_tags for m in bracket_metrics}
    r32_folded = {t: np.full((nb, 64), np.nan, dtype=float) for t in steep_tags}
    r32_phase_rad = np.linspace(-np.pi, np.pi, 65)
    r32_phase_rad = 0.5 * (r32_phase_rad[:-1] + r32_phase_rad[1:])
    ref_f0_ests = []
    used_f0_ests = []

    for bi, (r0, r1) in enumerate(bands):
        mb_mean = (rr_n >= r0) & (rr_n < r1) & np.isfinite(mth_mean)
        if np.any(mb_mean):
            mth_band[bi] = float(np.median(mth_mean[mb_mean]))

        idx = np.flatnonzero((rr >= r0) & (rr < r1))
        idx = idx[np.argsort(th[idx])]
        if idx.size > int(max_pos_per_band):
            idx = idx[np.linspace(0, idx.size - 1, int(max_pos_per_band), dtype=int)]

        recs = []
        thetas = []
        for p in idx:
            sh = position_shots(pos32.shot_nums, int(p), pos32.nshot)
            ch_d = {}
            ok = True
            for tip in ("X+", "X-", "Y+", "Y-", "REF"):
                ch = ch_map32[tip]
                d, _ = run32.channel(ch, scope_name="scope", shots=sh)
                if d is None:
                    ok = False
                    break
                ds = _blockmean_2d(np.asarray(d, dtype=float)[:, i0:i1], step)
                if tip != "REF":
                    ds = ds / float(sh32[tip])
                if float(np.nanmedian(ds)) < 0.0:
                    ds = -ds
                ch_d[tip] = ds
            if not ok:
                continue
            ref_ds = ch_d.pop("REF")
            if len(ref_f0_ests) < 12 and np.all(np.isfinite(ref_ds)):
                rf_f, rf_p = signal.welch(ref_ds - np.mean(ref_ds, axis=1, keepdims=True),
                                          fs=fs, nperseg=min(256, ref_ds.shape[1]), axis=1)
                rf_pm = np.mean(rf_p, axis=0)
                m_f = (rf_f >= RUN32_F0_BAND_HZ[0]) & (rf_f <= RUN32_F0_BAND_HZ[1])
                if np.any(m_f):
                    ref_f0_ests.append(float(rf_f[m_f][int(np.argmax(rf_pm[m_f]))]))
            rec = _compute_mach_transport_at_position(
                ch_d, kappa, float(px[p] - cx), float(py[p] - cy), fs, ref_ds=ref_ds
            )
            if rec is None:
                continue
            recs.append(rec)
            thetas.append(float(th[p]))
            if 2.5 <= rr[p] < 4.5:
                quad_th_deg.append(float(np.degrees(th[p])))
                quad_crt.append(rec["c_rt"])
                quad_cnr.append(rec["c_nr"])

        n_pos_band[bi] = len(recs)
        if not recs:
            continue
        dn_band[bi] = float(np.median([r["dn_rms"] for r in recs]))
        mr_band[bi] = float(np.median([r["mr_rms"] for r in recs]))
        mt_band[bi] = float(np.median([r["mt_rms"] for r in recs]))
        et_f0_band[bi] = float(np.mean([r["et_f0"] for r in recs]))
        et_bb_band[bi] = float(np.mean([r["et_bb"] for r in recs]))
        flux_f0_band[bi] = float(np.mean([r["flux_f0"] for r in recs]))
        flux_bb_band[bi] = float(np.mean([r["flux_bb"] for r in recs]))
        c_nr_band[bi] = float(np.median([r["c_nr"] for r in recs]))
        rs_f0_band[bi] = float(np.mean([r["rs_f0"] for r in recs]))
        rs_bb_band[bi] = float(np.mean([r["rs_bb"] for r in recs]))
        c_rt_band[bi] = float(np.median([r["c_rt"] for r in recs]))

        if ref_f0_ests:
            f0_est = float(np.median(ref_f0_ests))
        else:
            # Estimate f0 directly from band Isat_0 Welch PSD rather than a hardcoded fallback.
            is_stack = np.vstack([r["isat0_raw"] for r in recs])
            bf, bp = signal.welch(is_stack - np.mean(is_stack, axis=1, keepdims=True),
                                  fs=fs, nperseg=min(256, is_stack.shape[1]), axis=1)
            bpm = np.mean(bp, axis=0)
            mf = (bf >= RUN32_F0_BAND_HZ[0]) & (bf <= RUN32_F0_BAND_HZ[1])
            f0_est = float(bf[mf][int(np.argmax(bpm[mf]))]) if np.any(mf) else np.nan
        if np.isfinite(f0_est):
            used_f0_ests.append(f0_est)
        lintip_stacks = _mach_linear_tip_bracket(recs, kappa, fs, f0_est)
        for tag, key in (("isat0", "isat0_raw"), ("mr", "mr_raw"), ("mt", "mt_raw")):
            st_b = harmonic_steepening(np.vstack([r[key] for r in recs]), fs, f0_est, f0_search_hz=800.0)
            r32_steep[f"{tag}_a2_a1"][bi] = st_b.a2_a1_med
            r32_steep[f"{tag}_biphase_lock"][bi] = st_b.biphase_lock
            r32_steep[f"{tag}_biphase_lock_crit"][bi] = st_b.biphase_lock_crit
            r32_steep_locked[tag][bi] = bool(st_b.biphase_locked)
            r32_steep[f"{tag}_biphase_deg"][bi] = st_b.biphase_deg
            r32_steep[f"{tag}_curvature_comp"][bi] = st_b.curvature_comp
            r32_steep[f"{tag}_steepening_comp"][bi] = st_b.steepening_comp
            r32_steep[f"{tag}_skewness"][bi] = st_b.skewness_med
            r32_steep[f"{tag}_hilbert_asymmetry"][bi] = st_b.hilbert_asymmetry_med
            r32_steep[f"{tag}_slope_ratio"][bi] = st_b.slope_ratio_med
            r32_steep[f"{tag}_rise_frac"][bi] = st_b.rise_frac_med
            r32_folded[tag][bi] = st_b.folded_wave
            r32_phase_rad = st_b.phase_rad
            if lintip_stacks is not None:
                sb = harmonic_steepening(lintip_stacks[tag], fs, f0_est, f0_search_hz=800.0, gate_unlocked=False)
                r32_lintip[f"{tag}_a2_a1"][bi] = sb.a2_a1_med
                r32_lintip[f"{tag}_biphase_lock"][bi] = sb.biphase_lock
                r32_lintip[f"{tag}_biphase_deg"][bi] = sb.biphase_deg
                r32_lintip[f"{tag}_curvature_comp"][bi] = sb.curvature_comp
                r32_lintip[f"{tag}_steepening_comp"][bi] = sb.steepening_comp
                r32_lintip[f"{tag}_slope_ratio"][bi] = sb.slope_ratio_med

        # Physical m=-1 phase Phi(r) in cos(-theta + Phi(r) - omega*t):
        # hilbert(cos(omega*t - psi)) = exp(+i*(omega*t - psi)) -> an*conj(ar) = exp(-i*(psi - psi_ref))
        # For psi = -theta + Phi(r), exp(-i*psi) = exp(+i*theta - i*Phi(r)), so
        # mean(phasor * exp(-i*theta)) ~ exp(-i*Phi(r)) -> Phi(r) = -angle(...).
        phasors = np.array([r["phasor"] for r in recs], dtype=complex)
        th_arr = np.array(thetas, dtype=float)
        ok_ph = np.isfinite(phasors.real) & np.isfinite(phasors.imag)
        if np.any(ok_ph):
            z = complex(np.mean(phasors[ok_ph] * np.exp(-1j * th_arr[ok_ph])))
            m_mode_z_band[bi] = float(abs(z))
            phi_spiral_deg[bi] = float(-np.degrees(np.angle(z)))

    # Unwrapped radial spiral slope dPhi/dr [rad/cm] and invariant product m * dPhi/dr (m = -1)
    ok_sp = np.isfinite(phi_spiral_deg)
    dphi_dr_rad_cm = np.nan
    m_times_dphi_dr = np.nan
    if int(np.sum(ok_sp)) >= 2:
        phi_unw = np.unwrap(np.radians(phi_spiral_deg[ok_sp]))
        dphi_dr_rad_cm = float(np.polyfit(r_mids[ok_sp], phi_unw, 1)[0])
        m_times_dphi_dr = float(-1.0 * dphi_dr_rad_cm)

    # Cylindrical shear-flow energy transfer P_shear = -<dM_r dM_theta> * r*d(M_theta/r)/dr
    prod = reynolds_shear_production(r_mids, rs_f0_band, mth_band)

    # Physical-velocity Reynolds transfer & Eich-Manz (NF 2021, Eq. H.2) ratio using matched
    # Run 03 (35 ms, 800 G) radial Te(r), ne(r), Er(r) and linear growth rates when available.
    te_band = np.full(nb, np.nan, dtype=float)
    ne_band = np.full(nb, np.nan, dtype=float)
    er_band = np.full(nb, np.nan, dtype=float)
    gamma_exp_ref = np.nan
    gamma_cwm_ref = np.nan
    prof_p = os.path.join(data_dir, "profiles_summary_jun2026.npz")
    if os.path.isfile(prof_p):
        with np.load(prof_p, allow_pickle=True) as pf:
            if "03__Te" in pf.files and "03__ref_sweep_idx" in pf.files:
                x03 = np.asarray(pf["03__x_cm"], dtype=float)
                xc03 = float(pf["03__x_center_cm"])
                ir03 = int(pf["03__ref_sweep_idx"])
                rr03 = np.abs(x03 - xc03)
                for bi, (r0, r1) in enumerate(bands):
                    mb = (rr03 >= r0) & (rr03 < r1)
                    if np.any(mb):
                        te_band[bi] = float(np.nanmedian(pf["03__Te"][mb, ir03]))
                        ne_band[bi] = float(np.nanmedian(pf["03__ne"][mb, ir03]))
                        er_band[bi] = float(np.nanmedian(pf["03__er_r_v_cm"][mb, ir03]))
    onset_p = os.path.join(data_dir, jon.OUT_NAME)
    if os.path.isfile(onset_p):
        with np.load(onset_p, allow_pickle=True) as of:
            if "03__gamma_med_s" in of.files:
                gamma_exp_ref = float(of["03__gamma_med_s"])
    stab_p = os.path.join(data_dir, "stability_linear_jun2026.npz")
    if os.path.isfile(stab_p):
        with np.load(stab_p, allow_pickle=True) as sf:
            if "puff__03__rad_cwm_sh_cent_gamma_s" in sf.files:
                gamma_cwm_ref = float(sf["puff__03__rad_cwm_sh_cent_gamma_s"])
    em_res = _eich_manz_shear_transfer(
        r_mids, mth_band, rs_f0_band, rs_bb_band, et_f0_band, et_bb_band,
        te_band, ne_band, er_band,
        b0_g=800.0, gamma_exp_s=gamma_exp_ref, gamma_cwm_s=gamma_cwm_ref,
    )

    # Finite-probe-size grad(n) artifact benchmark for typical Mach tip separations d = 0.3, 0.5 cm.
    # Under Hutchinson's law M = -K_H * ln(I_+ / (kappa I_-)) with K_H = M_c = 0.45:
    #   dM_grad,r = -K_H * d * d(dn/n)/dr
    #   <dn/n * dM_grad,r>   = -(K_H / 2) * d * d<(dn/n)^2>/dr
    #   <dM_grad,r * dM_g,t> = +(K_H * d)^2 * (m * dPhi/dr / r) * <(dn/n)^2>
    dn2 = dn_band ** 2
    ddn2_dr = np.gradient(dn2, r_mids)
    flux_grad_d0p3 = -(K_HUTCHINSON / 2.0) * 0.3 * ddn2_dr
    flux_grad_d0p5 = -(K_HUTCHINSON / 2.0) * 0.5 * ddn2_dr
    rs_grad_d0p3 = ((K_HUTCHINSON * 0.3) ** 2) * (m_times_dphi_dr / r_mids) * dn2
    rs_grad_d0p5 = ((K_HUTCHINSON * 0.5) ** 2) * (m_times_dphi_dr / r_mids) * dn2
    flux_confounded = bool(np.any(np.abs(flux_grad_d0p3) >= 0.5 * np.abs(flux_f0_band)))

    # 4-quadrant wall audit on [2.5, 4.5) cm
    q_th = np.array(quad_th_deg, dtype=float)
    q_crt = np.array(quad_crt, dtype=float)
    q_cnr = np.array(quad_cnr, dtype=float)
    q_bounds = ((0.0, 90.0), (90.0, 180.0), (-180.0, -90.0), (-90.0, 0.0))
    quad_crt_med = np.array([
        float(np.median(q_crt[(q_th >= lo) & (q_th < hi)])) if np.any((q_th >= lo) & (q_th < hi)) else np.nan
        for lo, hi in q_bounds
    ], dtype=float)
    quad_cnr_med = np.array([
        float(np.median(q_cnr[(q_th >= lo) & (q_th < hi)])) if np.any((q_th >= lo) & (q_th < hi)) else np.nan
        for lo, hi in q_bounds
    ], dtype=float)

    out32 = {
        "r32__center_xy_cm": np.array([cx, cy], dtype=float),
        "r32__center_ratio": float(ratio),
        "r32__fs_hz": float(fs),
        "r32__f0_hz": float(np.median(used_f0_ests)) if used_f0_ests else np.nan,
        "r32__band_edges_cm": np.array(bands, dtype=float),
        "r32__r_mids_cm": r_mids,
        "r32__n_pos": n_pos_band,
        "r32__M_theta_mean": mth_band,
        "r32__omega_per_cm": prod.omega_per_cm,
        "r32__shear_per_cm": prod.shear_per_cm,
        "r32__dn_rms": dn_band,
        "r32__mr_rms": mr_band,
        "r32__mt_rms": mt_band,
        "r32__et_f0_mean": et_f0_band,
        "r32__et_bb_mean": et_bb_band,
        "r32__rs_over_et_f0": np.where(et_f0_band > 0.0, rs_f0_band / et_f0_band, np.nan),
        "r32__rs_over_et_bb": np.where(et_bb_band > 0.0, rs_bb_band / et_bb_band, np.nan),
        "r32__flux_nr_f0": flux_f0_band,
        "r32__flux_nr_bb": flux_bb_band,
        "r32__flux_grad_art_d0p3": flux_grad_d0p3,
        "r32__flux_grad_art_d0p5": flux_grad_d0p5,
        "r32__flux_grad_confounded": flux_confounded,
        "r32__c_nr_med": c_nr_band,
        "r32__rs_rt_f0": rs_f0_band,
        "r32__rs_rt_bb": rs_bb_band,
        "r32__rs_grad_art_d0p3": rs_grad_d0p3,
        "r32__rs_grad_art_d0p5": rs_grad_d0p5,
        "r32__c_rt_med": c_rt_band,
        "r32__p_shear_local": prod.p_shear_local,
        "r32__p_shear_integrated": prod.p_shear_integrated,
        "r32__m1_coherence": m_mode_z_band,
        "r32__phi_spiral_deg": phi_spiral_deg,
        "r32__dphi_dr_rad_cm": dphi_dr_rad_cm,
        "r32__m_times_dphi_dr_rad_cm": m_times_dphi_dr,
        "r32__wall_quad_c_rt_med": quad_crt_med,
        "r32__wall_quad_c_nr_med": quad_cnr_med,
        "r32__steep_phase_rad": r32_phase_rad,
    }
    for k_em, v_em in em_res.items():
        out32[f"r32__{k_em}"] = v_em
    for k_st, v_st in r32_steep.items():
        out32[f"r32__steep_{k_st}"] = v_st
    for t_st, v_lk in r32_steep_locked.items():
        out32[f"r32__steep_{t_st}_biphase_locked"] = v_lk
    for k_lt, v_lt in r32_lintip.items():
        out32[f"r32__steep_linear_tip_{k_lt}"] = v_lt
    for t_st, v_fld in r32_folded.items():
        out32[f"r32__steep_{t_st}_folded_waves"] = v_fld
    return out32


# ============================================================================
# Main orchestration & reporting
# ============================================================================
def run_turbulence_analysis(data_dir=DATA_DIR, out_name=OUT_NAME, overwrite=True):
    """Run all four turbulence/transport checks and save summary npz."""
    out_path = os.path.join(data_dir, out_name)
    if not overwrite and os.path.exists(out_path):
        print(f"  {out_name} exists, skipping (pass overwrite=True to regenerate)")
        with np.load(out_path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}

    res = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "simultaneous_vf_available": False,
    }
    res.update(analyze_stationary_spectra_and_moments(data_dir))
    res.update(analyze_radial_moments(data_dir))
    res.update(analyze_growth_vs_sat_crossphase(data_dir))
    res.update(analyze_run32_transport_and_reynolds(data_dir))

    merge_save_npz(out_path, res)
    print_summary(res)
    print(f"\nSaved {out_path} ({len(res)} keys)")
    return res


def batch_all(data_dir=DATA_DIR, overwrite=False):
    """Campaign entry point matching sibling Jun-2026 batch modules."""
    return run_turbulence_analysis(data_dir=data_dir, overwrite=overwrite)


def print_summary(res):
    """Print concise summary tables for spectra, radial moments, cross-phase, and Run 32."""
    print("\n=== 1. Stationary Isat Coherent Fraction, Continuum Fit & PDF Moments ===")
    print("Run Puff  B0   f0[k] | coh%  |   S     K   | Model       R2_exp(fe[k],tauL[us]) vs R2_pow(alpha)")
    for rnum in STATIONARY_RUNS:
        p = f"spec_{rnum}__"
        print(
            f" {rnum} {res[p+'gas_puff_ms']:4.0f} {res[p+'b0_g']:4.0f} {res[p+'f_peak_hz']*1e-3:5.2f} | "
            f"{res[p+'coh_harmonic_frac']*100:5.1f} | {res[p+'skewness_med']:+5.2f} {res[p+'excess_kurtosis_med']:+5.2f} | "
            f"{str(res[p+'preferred_model']):11s} "
            f"{res[p+'r2_exp']:5.3f}({res[p+'f_e_hz']*1e-3:4.1f},{res[p+'tau_lorentz_s']*1e6:4.1f}) vs "
            f"{res[p+'r2_pow']:5.3f}({res[p+'alpha']:4.2f})"
        )

    print("\n=== 2. Radial Hollow-Wall Skewness & Excess Kurtosis (Runs 11-14, P33 Isat_0) ===")
    for rnum, puff in RADIAL_MOMENT_RUNS:
        parts = []
        for z in ("core", "inner_wall", "outer_wall", "ring"):
            pz = f"radmom_{rnum}_{z}__"
            parts.append(f"{z}: S={res[pz+'skewness_med']:+.2f}, K={res[pz+'excess_kurtosis_med']:+.2f}")
        print(f"  Run {rnum} ({puff:.0f}ms): " + " | ".join(parts))

    print("\n=== 3. Per-Shot Amplitude-Normalized Cross-Phase: Linear Ramp -> Early Sat -> Late Sat (Runs 01-04) ===")
    for rnum, puff in CROSSPHASE_RUNS:
        p0 = f"xph_{rnum}_isat_i2__"
        print(
            f"  Run {rnum} ({puff:.0f}ms, n_unst={int(res[p0+'n_unstable_shots'])}, "
            f"t_on={res[p0+'t_on_per_shot_med_ms']:.2f}ms [cat {res[p0+'t_onset_med_ms']:.2f}ms], "
            f"t50-ton={res[p0+'ramp_dur_med_ms']:.2f}ms, t90={res[p0+'t_90_med_ms']:.2f}ms):"
        )
        for tag in ("isat_i2", "isat_i4", "isat_v2", "i2_i4"):
            p = f"xph_{rnum}_{tag}__"
            print(
                f"    {tag:7s}: grow(PLV={res[p+'plv_grow']:.2f}, phi={res[p+'phi_grow_deg']:+6.1f}d) -> "
                f"early(PLV={res[p+'plv_early_sat']:.2f}, phi={res[p+'phi_early_sat_deg']:+6.1f}d) -> "
                f"sat(PLV={res[p+'plv_sat']:.2f}, phi={res[p+'phi_sat_deg']:+6.1f}d) | "
                f"dphi(g->s)={res[p+'phi_shift_grow_to_sat_deg']:+5.1f}d, dphi(e->s)={res[p+'phi_shift_early_to_sat_deg']:+5.1f}d"
            )

    print("\n=== 4. Run 32 Reynolds Stress, Shear Production & Finite-Tip Flux-Artifact Benchmark ===")
    c32 = res["r32__center_xy_cm"]
    print(
        f"  Flow center=({c32[0]:+.2f},{c32[1]:+.2f}) cm (ratio={res['r32__center_ratio']:.2f}) | "
        f"Spiral dPhi/dr={res['r32__dphi_dr_rad_cm']:+.2f} rad/cm -> m*dPhi/dr={res['r32__m_times_dphi_dr_rad_cm']:+.2f} rad/cm\n"
        f"  Wall [2.5,4.5) cm 4-quadrant c_rt medians: {np.round(res['r32__wall_quad_c_rt_med'], 2)} | "
        f"Integrated P_shear = {res['r32__p_shear_integrated']:+.3e} (negative => wave feeds mean flow) | "
        f"flux_grad_confounded={bool(res['r32__flux_grad_confounded'])}"
    )
    bands = res["r32__band_edges_cm"]
    for i in range(len(bands)):
        print(
            f"  [{bands[i,0]:.1f},{bands[i,1]:.1f}) cm (n={int(res['r32__n_pos'][i]):2d}): "
            f"dn={res['r32__dn_rms'][i]*100:4.1f}% | <dn*dMr>={res['r32__flux_nr_f0'][i]:+.2e} "
            f"(grad_art@0.3cm={res['r32__flux_grad_art_d0p3'][i]:+.2e}) | "
            f"<dMr*dMt>={res['r32__rs_rt_f0'][i]:+.2e} (art={res['r32__rs_grad_art_d0p3'][i]:+.2e}, c={res['r32__c_rt_med'][i]:+.2f}) | "
            f"shear={res['r32__shear_per_cm'][i]:+.4f} -> P_sh={res['r32__p_shear_local'][i]:+.2e}"
        )
    if "r32__gamma_sh_mach_f0_s" in res and np.isfinite(float(res["r32__gamma_sh_mach_f0_s"])):
        print(
            f"  Eich-Manz Eq.(H.2) physical-velocity P_EM/(gamma*E_t) [mass-w, vs EXP Run 03 = {float(res['r32__gamma_exp_ref_s'])*1e-3:.2f} ms^-1]:\n"
            f"    Mach v_th(r) (gam=1):   f0 full={float(res['r32__gamma_sh_mach_f0_s'])*1e-3:+.2f} ms^-1 "
            f"(R_exp={float(res['r32__R_em_mach_f0_vs_exp']):+.2f}, wall={float(res['r32__R_em_mach_f0_wall_vs_exp']):+.2f}, "
            f"R_cwm={float(res['r32__R_em_mach_f0_vs_cwm']):+.2f}) | BB(1-25kHz) R_exp={float(res['r32__R_em_mach_bb_vs_exp']):+.2f}\n"
            f"    LP v_ExB (cs&K_H inv):  f0 full={float(res['r32__gamma_sh_exb_f0_s'])*1e-3:+.2f} ms^-1 "
            f"(R_exp={float(res['r32__R_em_exb_f0_vs_exp']):+.2f}, wall={float(res['r32__R_em_exb_f0_wall_vs_exp']):+.2f}, "
            f"R_cwm={float(res['r32__R_em_exb_f0_vs_cwm']):+.2f}) | BB(1-25kHz) R_exp={float(res['r32__R_em_exb_bb_vs_exp']):+.2f}"
        )

    if "spec_02__biphase_deg" in res:
        print("\n=== 5. Nonlinear Steepening vs. Wall Curvature (Biphase beta = phi_2 - 2*phi_1) ===")
        print("Run Puff  B0  | A2/A1 b_lk   beta   | curv(cos) steep(sin) | A_hlb  slp_r/f rise_f")
        for rnum in ("00", "01", "02", "03", "04", "11", "12", "28", "26", "27", "30"):
            p = f"spec_{rnum}__"
            if p + "biphase_deg" not in res:
                continue
            print(
                f" {rnum} {res[p+'gas_puff_ms']:4.0f} {res[p+'b0_g']:4.0f} | "
                f"{res[p+'a2_a1_med']:5.2f} {res[p+'biphase_lock']:4.2f} {res[p+'biphase_deg']:+6.1f}d | "
                f"  {res[p+'curvature_comp']:+.3f}     {res[p+'steepening_comp']:+.3f}    | "
                f"{res[p+'hilbert_asymmetry_med']:+5.2f}   {res[p+'slope_ratio_med']:5.2f}   {res[p+'rise_frac_med']:.3f}"
            )
    if "r32__steep_mr_biphase_deg" in res:
        print("  Run 32 wall steepening (raw calibrated M; lin-tip = bracket if tip currents, not M, were linear):")
        for bi in range(len(bands)):
            b0, b1 = bands[bi, 0], bands[bi, 1]
            parts = []
            for tag, lbl in (("isat0", "Isat0"), ("mr", "M_r"), ("mt", "M_th")):
                stp = float(res[f"r32__steep_{tag}_steepening_comp"][bi])
                lkey = f"r32__steep_linear_tip_{tag}_steepening_comp"
                lst = float(res[lkey][bi]) if lkey in res else np.nan
                parts.append(
                    f"{lbl}: beta={res[f'r32__steep_{tag}_biphase_deg'][bi]:+5.1f}d, "
                    f"steep={stp:+.2f}(lin-tip {lst:+.2f}), "
                    f"slp={res[f'r32__steep_{tag}_slope_ratio'][bi]:.2f}"
                )
            print(f"    [{b0:.1f},{b1:.1f}) cm: " + " | ".join(parts))


if __name__ == "__main__":
    run_turbulence_analysis()
