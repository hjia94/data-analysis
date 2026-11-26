"""Jun-2026 broadband turbulence spectrum, intermittency, cross-phase & Reynolds-stress transport.

Tests four turbulence and energy-transfer diagnostics from the LAPD biasing and
drift-wave literature (Pace et al. 2008; Maggs & Morales 2011; Friedman et al.
2013; Schaffner et al. 2013) across the Jun-2026 campaign:

1. **Broadband PSD & fluctuation PDF moments (Runs 06, 00–05, 08–14, 26–28, 30–31):**
   Quantifies the coherent harmonic power fraction (``1..4 f_0``), compares
   exponential (Lorentzian-pulse ``P(f) ~ exp(-f/f_e)``) vs. power-law
   (``P(f) ~ f^-alpha``) continuum fits strictly above the per-run pre-bias
   background noise floor, and measures per-shot skewness and excess kurtosis.
2. **Radial intermittency polarity across the hollow wall (Runs 11–14):**
   Tracks flow-cancelled ``Isat_0`` skewness and excess kurtosis across
   ``core (r=0)``, ``inner_wall (r=2.5 cm)``, ``outer_wall (r=4.0 cm)``, and
   ``ring (r=6.0 cm)`` — confirming inverted hollow-column skewness polarity
   (``S > 0`` on the low-density inner wall, ``S < 0`` on the outer wall) with
   strongly negative wave-like kurtosis at peak coherent saturation.
3. **Linear-growth vs. saturated cross-phase evolution (Runs 01–04):**
   Compares ``Isat(P33)`` vs. Port-35 ``I_2"``, ``I_4"``, and ``V_2"``
   (``biasscope``, ``t_offset = 12.0 ms``) plus same-scope ``I_2"`` vs.
   ``I_4"`` between linear growth and nonlinear saturation using block-mean
   anti-aliased downsampling onto a common 100 kHz grid.
4. **Direct radial particle flux, Reynolds stress & spiral-sign verification
   (Run 32, 6-tip Mach plane at P33, 800 G, 35 ms):**
   Across five radial annuli ``[1.5, 7.0) cm`` about the fitted flow center,
   computes simultaneous ``<dn/n * dM_r>``, ``<dM_r * dM_theta>`` (including a
   4-quadrant check on ``[2.5, 4.5) cm`` ruling out Cartesian cross-talk), the
   direct ``Isat_0`` radial phase spiral ``m * dPhi/dr`` (proving
   ``<dM_r * dM_theta> > 0`` has the opposite sign to finite-probe-size
   ``grad(n)`` contamination), and cylindrical shear-flow energy production
   ``P_shear = -<dM_r * dM_theta> * r * d(M_theta / r) / dr``.

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
from scipy import signal

from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.plasma.flow import find_flow_centre, polar_components
from data_analysis.plasma.mach import mach_single
from data_analysis.signal import (
    clip_time_window,
    fit_broadband_spectrum,
    fluctuation_moments,
    reynolds_shear_production,
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
    return out


# ============================================================================
# 2. Radial (Skewness, Excess Kurtosis) across hollow core / wall / ring
# ============================================================================
def analyze_radial_moments(data_dir=DATA_DIR, *, x_center_cm=DEFAULT_X_CENTER_CM):
    """Flow-cancelled ``Isat_0`` radial skewness & excess kurtosis on Runs 11-14."""
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
        i0, i1 = clip_time_window(t_s, 2.5, 4.8)

        for zname, x_t in zones:
            p_idx = int(np.argmin(np.abs(x_pos - float(x_t))))
            sh = position_shots(pos.shot_nums, p_idx, pos.nshot)
            yp, _ = run.channel(ch_yp, scope_name=sc, shots=sh)
            ym, _ = run.channel(ch_ym, scope_name=sc, shots=sh)
            pz = f"radmom_{rnum}_{zname}__"
            out[pz + "gas_puff_ms"] = float(puff)
            out[pz + "x_cm"] = float(x_pos[p_idx])
            if yp is None or ym is None:
                out[pz + "rel_rms_med"] = np.nan
                out[pz + "skewness_med"] = np.nan
                out[pz + "excess_kurtosis_med"] = np.nan
                continue
            yp_d = _blockmean_2d(np.asarray(yp, dtype=float)[:, i0:i1], step) / float(shunts["Y+"])
            ym_d = _blockmean_2d(np.asarray(ym, dtype=float)[:, i0:i1], step) / float(shunts["Y-"])
            if float(np.nanmedian(yp_d)) < 0.0:
                yp_d = -yp_d
            if float(np.nanmedian(ym_d)) < 0.0:
                ym_d = -ym_d
            ok = (yp_d > 0.0) & (ym_d > 0.0)
            isat0 = np.where(ok, np.sqrt(np.where(ok, yp_d * ym_d, 1.0)), np.nan)
            mom = fluctuation_moments(isat0, normalize_by_mean=True, detrend=True)
            out[pz + "rel_rms_med"] = mom.rel_rms_med
            out[pz + "skewness_med"] = mom.skewness_med
            out[pz + "excess_kurtosis_med"] = mom.excess_kurtosis_med
    return out


# ============================================================================
# 3. Linear-growth vs. saturated cross-phase (Isat vs Port-35 discs)
# ============================================================================
def _aligned_window_csd(t_a_s, sig_a, t_b_s, sig_b, t_lo_ms, t_hi_ms, f_target_hz,
                        *, target_fs_hz=TARGET_FS_TRANSPORT_HZ):
    """Anti-aliased block-mean resample + Welch coherence, cross-phase & local peak freq."""
    if t_hi_ms - t_lo_ms < 0.25:
        return np.nan, np.nan, np.nan
    pad = 0.05
    ia0, ia1 = clip_time_window(t_a_s, t_lo_ms - pad, t_hi_ms + pad)
    ib0, ib1 = clip_time_window(t_b_s, t_lo_ms - pad, t_hi_ms + pad)
    ta_seg = t_a_s[ia0:ia1]
    tb_seg = t_b_s[ib0:ib1]
    if ta_seg.size < 4 or tb_seg.size < 4:
        return np.nan, np.nan, np.nan

    step_a = max(1, int(round((1.0 / target_fs_hz) / float(ta_seg[1] - ta_seg[0]))))
    step_b = max(1, int(round((1.0 / target_fs_hz) / float(tb_seg[1] - tb_seg[0]))))
    ta_ds = _blockmean_1d(ta_seg, step_a)
    tb_ds = _blockmean_1d(tb_seg, step_b)
    nsh = min(sig_a.shape[0], sig_b.shape[0])
    sa_ds = _blockmean_2d(np.asarray(sig_a[:nsh, ia0:ia1], dtype=float), step_a)
    sb_ds = _blockmean_2d(np.asarray(sig_b[:nsh, ib0:ib1], dtype=float), step_b)

    dt_out = 1.0 / float(target_fs_hz)
    t_grid = np.arange(t_lo_ms * 1e-3, t_hi_ms * 1e-3, dt_out)
    if t_grid.size < 16:
        return np.nan, np.nan, np.nan

    a_arr = np.empty((nsh, t_grid.size), dtype=float)
    b_arr = np.empty((nsh, t_grid.size), dtype=float)
    for k in range(nsh):
        a_arr[k] = np.interp(t_grid, ta_ds, sa_ds[k])
        b_arr[k] = np.interp(t_grid, tb_ds, sb_ds[k])
    a_arr -= np.mean(a_arr, axis=1, keepdims=True)
    b_arr -= np.mean(b_arr, axis=1, keepdims=True)

    nper = min(128, t_grid.size)
    ff, paa = signal.welch(a_arr, fs=target_fs_hz, nperseg=nper, axis=1)
    _, pbb = signal.welch(b_arr, fs=target_fs_hz, nperseg=nper, axis=1)
    _, pab = signal.csd(a_arr, b_arr, fs=target_fs_hz, nperseg=nper, axis=1)
    Paa, Pbb, Pab = np.mean(paa, axis=0), np.mean(pbb, axis=0), np.mean(pab, axis=0)

    # Pick local peak within +-1.5 kHz of f_target_hz so small frequency chirps don't miss the mode
    m_loc = (ff >= max(1500.0, f_target_hz - 1500.0)) & (ff <= f_target_hz + 1500.0)
    if not np.any(m_loc):
        return np.nan, np.nan, np.nan
    idx_loc = np.flatnonzero(m_loc)
    kf = int(idx_loc[int(np.argmax(Paa[idx_loc]))])
    coh = float(np.abs(Pab[kf]) ** 2 / max(float(Paa[kf] * Pbb[kf]), 1e-30))
    phi_deg = float(np.degrees(np.angle(Pab[kf])))
    return coh, phi_deg, float(ff[kf])


def analyze_growth_vs_sat_crossphase(data_dir=DATA_DIR, *, max_shots=60,
                                     biasscope_offset_ms=jmo.BIASSCOPE_OFFSET_MS):
    """Compare ``Isat(P33)`` vs. Port-35 disc currents/voltage across growth -> saturation."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}

    for rnum, puff in CROSSPHASE_RUNS:
        ifn = jon.run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        f0 = float(on[f"{rnum}__f_peak_hz"])
        tm_s = _time_seconds(run.time_array(scope_name="machscope"))
        tb_rel_s = _time_seconds(run.time_array(scope_name="biasscope")) - float(biasscope_offset_ms) * 1e-3

        all_sh = np.asarray(run.shots(scope_name="machscope"), dtype=int)
        idx = np.unique(np.linspace(0, all_sh.size - 1, min(int(max_shots), all_sh.size), dtype=int))
        sh = all_sh[idx].tolist()

        isat_raw, _ = run.channel("C2", scope_name="machscope", shots=sh)
        i2_raw, _ = run.channel("C1", scope_name="biasscope", shots=sh)
        i4_raw, _ = run.channel("C2", scope_name="biasscope", shots=sh)
        v2_raw, _ = run.channel("C4", scope_name="biasscope", shots=sh)

        w_grow = (max(0.3, t_on - 0.05), min(4.8, t_on + 0.75))
        w_sat = (max(t_on + 1.0, 3.0), 4.8) if t_on < 3.6 else (np.nan, np.nan)

        pairs = (
            ("isat_i2", tm_s, isat_raw, tb_rel_s, i2_raw),
            ("isat_i4", tm_s, isat_raw, tb_rel_s, i4_raw),
            ("isat_v2", tm_s, isat_raw, tb_rel_s, v2_raw),
            ("i2_i4", tb_rel_s, i2_raw, tb_rel_s, i4_raw),  # same-scope control!
        )
        for tag, ta, sa, tb, sb in pairs:
            cg, pg, fg = _aligned_window_csd(ta, sa, tb, sb, *w_grow, f0)
            if np.all(np.isfinite(w_sat)):
                cs, ps, fs_pk = _aligned_window_csd(ta, sa, tb, sb, *w_sat, f0)
            else:
                cs, ps, fs_pk = np.nan, np.nan, np.nan
            shift = _wrap_deg(ps - pg) if (np.isfinite(ps) and np.isfinite(pg)) else np.nan
            p = f"xph_{rnum}_{tag}__"
            out[p + "gas_puff_ms"] = float(puff)
            out[p + "t_onset_med_ms"] = t_on
            out[p + "coh_grow"] = cg
            out[p + "phi_grow_deg"] = pg
            out[p + "f_grow_hz"] = fg
            out[p + "coh_sat"] = cs
            out[p + "phi_sat_deg"] = ps
            out[p + "f_sat_hz"] = fs_pk
            out[p + "phi_shift_deg"] = shift
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
        "flux_f0": flux_f0,
        "flux_bb": float(np.mean(dn_bb * mr_bb)),
        "rs_f0": rs_f0,
        "rs_bb": float(np.mean(mr_bb * mt_bb)),
        "c_nr": float(flux_f0 / max(sig_n * sig_r, 1e-30)),
        "c_rt": float(rs_f0 / max(sig_r * sig_t, 1e-30)),
        "phasor": phasor,
    }


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
    mx_m = _nanmean_rows(mx_n[:, m_on])
    my_m = _nanmean_rows(my_n[:, m_on])
    # Velocities FIRST, positions SECOND!
    cx, cy, ratio = find_flow_centre(mx_m, my_m, px_n, py_n)
    _, mth_mean = polar_components(mx_m, my_m, px_n, py_n, (cx, cy))
    rr_n = np.hypot(px_n - cx, py_n - cy)

    # 2. Read raw Run 32 6-tip Mach fluctuations + stationary P29-R ref (C8)
    hits = sorted(glob.glob(os.path.join(data_dir, "32-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no Run 32 HDF5 in {data_dir}")
    ifn32 = hits[0]
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
            for tip, ch in (("X+", "C1"), ("X-", "C2"), ("Y+", "C3"), ("Y-", "C4"), ("REF", "C8")):
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
        flux_f0_band[bi] = float(np.mean([r["flux_f0"] for r in recs]))
        flux_bb_band[bi] = float(np.mean([r["flux_bb"] for r in recs]))
        c_nr_band[bi] = float(np.median([r["c_nr"] for r in recs]))
        rs_f0_band[bi] = float(np.mean([r["rs_f0"] for r in recs]))
        rs_bb_band[bi] = float(np.mean([r["rs_bb"] for r in recs]))
        c_rt_band[bi] = float(np.median([r["c_rt"] for r in recs]))

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

    return {
        "r32__center_xy_cm": np.array([cx, cy], dtype=float),
        "r32__center_ratio": float(ratio),
        "r32__fs_hz": float(fs),
        "r32__band_edges_cm": np.array(bands, dtype=float),
        "r32__r_mids_cm": r_mids,
        "r32__n_pos": n_pos_band,
        "r32__M_theta_mean": mth_band,
        "r32__omega_per_cm": prod.omega_per_cm,
        "r32__shear_per_cm": prod.shear_per_cm,
        "r32__dn_rms": dn_band,
        "r32__mr_rms": mr_band,
        "r32__mt_rms": mt_band,
        "r32__flux_nr_f0": flux_f0_band,
        "r32__flux_nr_bb": flux_bb_band,
        "r32__c_nr_med": c_nr_band,
        "r32__rs_rt_f0": rs_f0_band,
        "r32__rs_rt_bb": rs_bb_band,
        "r32__c_rt_med": c_rt_band,
        "r32__p_shear_local": prod.p_shear_local,
        "r32__p_shear_integrated": prod.p_shear_integrated,
        "r32__m1_coherence": m_mode_z_band,
        "r32__phi_spiral_deg": phi_spiral_deg,
        "r32__dphi_dr_rad_cm": dphi_dr_rad_cm,
        "r32__m_times_dphi_dr_rad_cm": m_times_dphi_dr,
        "r32__wall_quad_c_rt_med": quad_crt_med,
        "r32__wall_quad_c_nr_med": quad_cnr_med,
    }


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

    print("\n=== 3. Linear-Growth -> Saturated Cross-Phase Evolution (Runs 01-04) ===")
    for rnum, puff in CROSSPHASE_RUNS:
        print(f"  Run {rnum} ({puff:.0f}ms):")
        for tag in ("isat_i2", "isat_i4", "isat_v2", "i2_i4"):
            p = f"xph_{rnum}_{tag}__"
            print(
                f"    {tag:7s}: grow(coh={res[p+'coh_grow']:.2f}, phi={res[p+'phi_grow_deg']:+6.1f}d, f={res[p+'f_grow_hz']*1e-3:.2f}k) -> "
                f"sat(coh={res[p+'coh_sat']:.2f}, phi={res[p+'phi_sat_deg']:+6.1f}d, f={res[p+'f_sat_hz']*1e-3:.2f}k) | "
                f"shift={res[p+'phi_shift_deg']:+5.1f}d"
            )

    print("\n=== 4. Run 32 Direct Radial Particle Flux, Reynolds Stress & Shear Production ===")
    c32 = res["r32__center_xy_cm"]
    print(
        f"  Flow center=({c32[0]:+.2f},{c32[1]:+.2f}) cm (ratio={res['r32__center_ratio']:.2f}) | "
        f"Spiral dPhi/dr={res['r32__dphi_dr_rad_cm']:+.2f} rad/cm -> m*dPhi/dr={res['r32__m_times_dphi_dr_rad_cm']:+.2f} rad/cm\n"
        f"  Wall [2.5,4.5) cm 4-quadrant c_rt medians: {np.round(res['r32__wall_quad_c_rt_med'], 2)} | "
        f"Integrated P_shear = {res['r32__p_shear_integrated']:+.3e} (negative => wave feeds mean flow)"
    )
    bands = res["r32__band_edges_cm"]
    for i in range(len(bands)):
        print(
            f"  [{bands[i,0]:.1f},{bands[i,1]:.1f}) cm (n={int(res['r32__n_pos'][i]):2d}): "
            f"dn={res['r32__dn_rms'][i]*100:4.1f}% | <dn*dMr>={res['r32__flux_nr_f0'][i]:+.2e} (c={res['r32__c_nr_med'][i]:+.2f}) | "
            f"<dMr*dMt>={res['r32__rs_rt_f0'][i]:+.2e} (c={res['r32__c_rt_med'][i]:+.2f}) | "
            f"shear={res['r32__shear_per_cm'][i]:+.4f} -> P_sh={res['r32__p_shear_local'][i]:+.2e} | "
            f"Phi={res['r32__phi_spiral_deg'][i]:+6.1f}d"
        )


if __name__ == "__main__":
    run_turbulence_analysis()
