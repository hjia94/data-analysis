"""Jun-2026 pre- vs. post-onset hollow-core profile relaxation & collapse test.

Directly compares the Jun-2026 multi-electrode biased hollow column against the
rapid (~10–15 us) hollow-core collapse / density-filling disruptions reported
for hollow filaments in Van Compernolle, Poulos & Morales (2022, *Phys. Plasmas*
29, 042104), across four complementary diagnostics:

1. **Swept P29-R IV (Runs 06, 00–05, from ``profiles_summary_jun2026.npz``):**
   Separates true density hollowness ``H_ne = ne_ring / ne_core`` from the
   ion-saturation proxy ``H_Isat = (ne * sqrt(Te))_ring / (ne * sqrt(Te))_core``
   and core electron heating ``Te_core / Te_ring`` across all available bias-on
   sweeps. Because the hardware sweeper terminates before deep post-onset on
   Runs 01–05, this quantifies the ``Isat ~ ne * sqrt(Te)`` conversion factor
   (``H_ne / H_Isat = 1.14–2.29x``) and confirms that Run 00's post-onset sweep
   at ``3.10 ms`` preserves ``H_ne = 3.46x`` (vs ``3.38x`` pre-onset).
2. **Continuous P33 Mach-line flow-cancelled ``Isat_0(r, t)`` & ``M_rot(t)``
   (Runs 08–14, 23–45 ms puff, 800 G):**
   Uses opposite azimuthal Mach faces ``Isat_0 = sqrt(I_Y+ * I_Y-)`` (exact
   first-order flow cancellation under Hutchinson's ``exp(+- K M / 2)`` law;
   ``machscope/C1`` = ``X+`` is undigitized on Runs 08–14) over the full
   ``[-1.0, +7.0] ms`` record at ``50 us`` resolution, plus ``10 us``
   single-shot onset-aligned core / wall / ring traces (Runs 11–14) to test
   for any abrupt ``10–50 us`` collapse events vs. coherent ``m = -1`` wave
   saturation.
3. **Magnetic-field & axial-localization lines (Runs 28=600 G, 27=800 G,
   30=1000 G at P29-R & P25-R):**
   Measures how post-onset hollowness retention ``H_late / H_preOn`` scales
   with ``B0`` and how hollowness decays axially away from Port 35.
4. **Shift-immune 2-D planes (Runs 26=800 G & 31=600 G P29-R ``17x17`` Isat
   planes; Run 32=800 G P33 ``41x41`` 3-D Mach flow plane):**
   Fits the 2-D hollow-core center ``(x_c, y_c)`` pre-onset and late-bias,
   computes azimuthally averaged ``<Isat(r, t)>_theta`` across ``0–8 cm``, and
   verifies that both azimuthal rotation ``M_theta`` and axial core pumping
   ``M_z`` remain intact throughout the saturated post-onset phase.

Saves ``profile_relaxation_jun2026.npz`` in ``DATA_DIR`` and prints a concise
summary table.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_relaxation.py
"""

import datetime
import glob
import os

import numpy as np

from data_analysis.io import open_lapd, parse_gas_puff, parse_shunts, position_shots
from data_analysis.plasma.flow import find_flow_centre, polar_components
from data_analysis.plasma.mach import mach_single, time_bin_edges
from data_analysis.plasma.profiles import hollow_profile_metrics, orbit_smeared_profile
from data_analysis.signal import (
    clip_time_window,
    detect_onset_time,
    sliding_detrended_rms,
)
from data_analysis.utils import merge_save_npz

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
import Jun2026_mach_line as jml
import Jun2026_onset as jon
import Jun2026_profiles as jpr

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "profile_relaxation_jun2026.npz"

MACH_LINE_RUNS = (
    ("08", 23.0),
    ("09", 24.0),
    ("10", 25.0),
    ("11", 30.0),
    ("12", 35.0),
    ("13", 40.0),
    ("14", 45.0),
)
SINGLE_SHOT_RUNS = ("11", "12", "13", "14")
B0_LINE_RUNS = (
    ("28", 600.0),
    ("27", 800.0),
    ("30", 1000.0),
)
ISAT_2D_PLANES = (
    ("26", "<Hermes>    p29_LP", "lpscope", "C2", "P29-R", 800.0),
    ("31", "<Hermes>    p29_LP", "lpscope", "C2", "P29-R", 600.0),
)

CORE_R_MAX_CM = 2.0
RING_R_CM = (4.5, 7.0)
WALL_R_CM = (1.5, 4.5)
ROT_R_CM = (2.5, 5.5)
RADIAL_EDGES_CM = np.arange(0.0, 8.5, 1.0)
DEFAULT_X_CENTER_CM = 1.0
BIN_MS = 0.05                    # 50 us continuous Isat bins
SINGLE_SHOT_DT_S = 1e-5          # 10 us resolution for onset-aligned collapse test
SINGLE_SHOT_CYCLE_S = 2.5e-4     # 0.25 ms boxcar (~1 wave period) to separate mean from oscillation
TAU_ALIGN_MS = np.arange(-0.6, 1.51, 0.02)
BIAS_END_MS = 5.0


def _time_seconds(tarr):
    """Return ``tarr`` in seconds (handles scopes returning ms vs. s)."""
    t = np.asarray(tarr, dtype=float)
    return t * 1e-3 if np.ptp(t) > 1.0 else t


def _bin_nanmean(sig, edges):
    """NaN-safe bin average of a 1-D trace ``sig`` over ``edges``."""
    out = np.full(len(edges) - 1, np.nan, dtype=float)
    for j in range(len(edges) - 1):
        seg = sig[edges[j]:edges[j + 1]]
        fin = seg[np.isfinite(seg)]
        if fin.size:
            out[j] = float(np.mean(fin))
    return out


def _window_bounds(t_on_ms):
    """Standard pre-bias / pre-onset / early-post / late-saturated / post-off windows [ms].

    For stable runs (``t_on_ms`` is NaN), uses matched mid/late bias reference
    windows ``[1.5, 2.5]``, ``[2.5, 3.5]``, and ``[3.5, 4.8] ms``. For runs
    with very late onset where ``t_on + 0.8 >= 4.7 ms`` (e.g. Run 10 at
    ``4.20 ms``), ``late`` is set to ``None`` rather than silently reusing the
    ``early`` window.
    """
    if np.isfinite(t_on_ms):
        pre_on = (max(0.3, float(t_on_ms) - 0.50), max(0.4, float(t_on_ms) - 0.10))
        early = (float(t_on_ms) + 0.10, min(4.80, float(t_on_ms) + 0.60))
        late_lo = max(float(t_on_ms) + 0.80, 3.00)
        late = (late_lo, 4.80) if late_lo <= 4.65 else None
    else:
        pre_on = (1.50, 2.50)
        early = (2.50, 3.50)
        late = (3.50, 4.80)
    return {
        "pre_bias": (-0.80, -0.20),
        "pre_onset": pre_on,
        "early_post": early,
        "late_sat": late,
        "post_off": (5.15, 5.35),
    }


def _window_median(t_ms, arr, bounds):
    """Median of ``arr`` inside ``bounds = (lo, hi)``, or ``NaN`` if ``bounds`` is None/empty."""
    if bounds is None:
        return np.nan
    lo, hi = bounds
    m = (t_ms >= lo) & (t_ms <= hi) & np.isfinite(arr)
    return float(np.median(arr[m])) if np.any(m) else np.nan


def _pos_xy_per_position(pos):
    """Return unique per-position ``(x, y)`` arrays of length ``pos.npos``."""
    pa = pos.pos_array
    return (
        np.asarray(pa["x"][::pos.nshot], dtype=float),
        np.asarray(pa["y"][::pos.nshot], dtype=float),
    )


def _find_2d_core_center(x_u, y_u, vals, r_search_cm=3.5, k_lowest=5):
    """Centroid of the ``k_lowest`` finite samples inside ``r <= r_search_cm``."""
    m = (np.hypot(x_u, y_u) <= float(r_search_cm)) & np.isfinite(vals)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return np.nan, np.nan
    k = min(int(k_lowest), idx.size)
    order = idx[np.argsort(vals[idx])[:k]]
    return float(np.mean(x_u[order])), float(np.mean(y_u[order]))


def _run_ifn(data_dir, rnum):
    hits = sorted(glob.glob(os.path.join(data_dir, f"{rnum}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {rnum} in {data_dir}")
    return hits[0]


def _nanmean_rows(arr):
    """Row-wise nanmean returning NaN on all-NaN rows without RuntimeWarning."""
    cnt = np.sum(np.isfinite(arr), axis=1)
    s = np.nansum(arr, axis=1)
    return np.where(cnt > 0, s / np.where(cnt > 0, cnt, 1), np.nan)


# ============================================================================
# 1. Swept P29-R IV hollowness & Te heating factor (Runs 06, 00-05)
# ============================================================================
def analyze_swept_iv_hollowness(data_dir=DATA_DIR):
    """Compare true ``ne`` hollowness vs. ``Isat ~ ne*sqrt(Te)`` across swept IV runs."""
    pf_path = os.path.join(data_dir, jpr.OUT_NAME)
    if not os.path.exists(pf_path):
        raise FileNotFoundError(f"missing {pf_path}; run Jun2026_profiles.py first")
    pf = np.load(pf_path, allow_pickle=True)
    out = {}

    for spec in jpr.PROFILE_RUNS:
        r = spec.run_num
        t_mid = np.asarray(pf[f"{r}__t_mid_ms"], dtype=float)
        regs = np.asarray(pf[f"{r}__regimes"], dtype=str)
        x = np.asarray(pf[f"{r}__x_cm"], dtype=float)
        xc = float(pf[f"{r}__x_center_cm"])
        rr = np.abs(x - xc)
        ne = np.asarray(pf[f"{r}__ne"], dtype=float)
        te = np.asarray(pf[f"{r}__Te"], dtype=float)
        isat_proxy = np.where((ne > 0.0) & (te > 0.0), ne * np.sqrt(np.maximum(te, 0.0)), np.nan)

        m_ne = hollow_profile_metrics(
            rr, ne, core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM
        )
        m_is = hollow_profile_metrics(
            rr, isat_proxy, core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM
        )
        m_te = hollow_profile_metrics(
            rr, te, core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM
        )
        te_c_over_r = np.where(
            np.isfinite(m_te.core_val) & np.isfinite(m_te.ring_val) & (m_te.ring_val > 0.0),
            m_te.core_val / np.where(m_te.ring_val > 0.0, m_te.ring_val, 1.0),
            np.nan,
        )
        h_ratio = np.where(
            np.isfinite(m_ne.hollowness) & np.isfinite(m_is.hollowness) & (m_is.hollowness > 0.0),
            m_ne.hollowness / np.where(m_is.hollowness > 0.0, m_is.hollowness, 1.0),
            np.nan,
        )

        out[f"iv_{r}__gas_puff_ms"] = float(spec.gas_puff_ms)
        out[f"iv_{r}__t_onset_med_ms"] = float(pf[f"{r}__t_onset_med_ms"])
        out[f"iv_{r}__t_mid_ms"] = t_mid
        out[f"iv_{r}__regimes"] = regs
        out[f"iv_{r}__H_ne"] = m_ne.hollowness
        out[f"iv_{r}__H_isat_proxy"] = m_is.hollowness
        out[f"iv_{r}__H_ne_over_H_isat"] = h_ratio
        out[f"iv_{r}__Te_core_over_ring"] = te_c_over_r
        out[f"iv_{r}__wall_dln_ne_cm"] = m_ne.wall_inv_scale_cm
    return out


# ============================================================================
# 2. Continuous P33 Mach-line Isat_0(r, t), M_rot(t), & 10 us collapse check
# ============================================================================
def analyze_mach_line_relaxation(data_dir=DATA_DIR, *, cal_npz=None,
                                 x_center_cm=DEFAULT_X_CENTER_CM, bin_ms=BIN_MS):
    """Extract flow-cancelled ``Isat_0(r, t)``, ``H(t)``, ``M_rot(t)``, & 10 us onset-aligned traces."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    cal_path = cal_npz or os.path.join(data_dir, os.path.basename(jmf.CAL_NPZ))
    kappa, _, _ = jmf.load_calibration(cal_path)
    out = {}

    for rnum, puff in MACH_LINE_RUNS:
        ifn = _run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        chans = jml.tip_channels(ifn)
        shunts = parse_shunts(run.description().raw)
        scope = chans["Y+"][0]
        pos = jiv.read_lp_positions(ifn)
        x_pos, _ = jml.scanned_axis(pos)
        tarr = run.time_array(scope_name=scope)
        edges, t_ms = time_bin_edges(tarr, (-1.0, 7.0), bin_ms)
        npos, nbins = x_pos.size, t_ms.size

        tip_I = {}
        for tip in ("Y+", "Y-"):
            sc, ch = chans[tip]
            r_sh = float(shunts[tip])
            arr = np.full((npos, nbins), np.nan, dtype=float)
            for p in range(npos):
                sh = position_shots(pos.shot_nums, p, pos.nshot)
                d, _ = run.channel(ch, scope_name=sc, shots=sh)
                if d is None:
                    continue
                sig = np.nanmean(np.asarray(d, dtype=float), axis=0)
                arr[p] = _bin_nanmean(sig, edges) / r_sh
            if np.isfinite(np.nanmedian(arr)) and float(np.nanmedian(arr)) < 0.0:
                arr = -arr
            tip_I[tip] = arr

        # Geometric mean cancels first-order Mach flow asymmetry exp(+- K*M/2).
        valid_pair = (tip_I["Y+"] > 0.0) & (tip_I["Y-"] > 0.0)
        isat0 = np.where(
            valid_pair,
            np.sqrt(np.where(valid_pair, tip_I["Y+"] * tip_I["Y-"], 1.0)),
            np.nan,
        )
        m_y = np.where(
            valid_pair,
            mach_single(
                np.where(valid_pair, tip_I["Y+"] / np.where(valid_pair, tip_I["Y-"], 1.0), 1.0),
                kappa["Y"],
            ),
            np.nan,
        )

        rr = np.abs(x_pos - float(x_center_cm))
        hm = hollow_profile_metrics(
            rr, isat0, core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM
        )

        # Antisymmetric azimuthal Mach speed across r in ROT_R_CM
        r0, r1 = ROT_R_CM
        m_r = (x_pos - float(x_center_cm) >= r0) & (x_pos - float(x_center_cm) <= r1)
        m_l = (float(x_center_cm) - x_pos >= r0) & (float(x_center_cm) - x_pos <= r1)
        m_rot = 0.5 * (np.nanmedian(m_y[m_r], axis=0) - np.nanmedian(m_y[m_l], axis=0))

        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        wins = _window_bounds(t_on)

        # Peak hollowness in [0.2, 4.9] ms
        mb = (t_ms >= 0.2) & (t_ms <= 4.9) & np.isfinite(hm.hollowness)
        if np.any(mb):
            idx_mb = np.flatnonzero(mb)
            kpk = int(idx_mb[int(np.argmax(hm.hollowness[idx_mb]))])
            t_hpk, h_pk = float(t_ms[kpk]), float(hm.hollowness[kpk])
        else:
            t_hpk, h_pk = np.nan, np.nan

        p = f"mach_{rnum}__"
        out[p + "gas_puff_ms"] = float(puff)
        out[p + "t_onset_med_ms"] = t_on
        out[p + "t_ms"] = t_ms
        out[p + "i_core_a"] = hm.core_val
        out[p + "i_ring_a"] = hm.ring_val
        out[p + "H_t"] = hm.hollowness
        out[p + "wall_dln_isat_cm"] = hm.wall_inv_scale_cm
        out[p + "m_rot_t"] = m_rot
        out[p + "H_max"] = h_pk
        out[p + "t_H_max_ms"] = t_hpk

        for wname, bounds in wins.items():
            out[p + f"H_{wname}"] = _window_median(t_ms, hm.hollowness, bounds)
            out[p + f"ic_{wname}_a"] = _window_median(t_ms, hm.core_val, bounds)
            out[p + f"ir_{wname}_a"] = _window_median(t_ms, hm.ring_val, bounds)
            out[p + f"wall_{wname}_cm"] = _window_median(t_ms, hm.wall_inv_scale_cm, bounds)
            out[p + f"m_rot_{wname}"] = _window_median(t_ms, m_rot, bounds)

        # Convenience retention ratios (late / pre_onset)
        h_pre = out[p + "H_pre_onset"]
        h_late = out[p + "H_late_sat"]
        ic_pre = out[p + "ic_pre_onset_a"]
        ic_late = out[p + "ic_late_sat_a"]
        out[p + "H_late_over_pre"] = (
            float(h_late / h_pre) if np.isfinite(h_late) and np.isfinite(h_pre) and h_pre > 0 else np.nan
        )
        out[p + "ic_late_over_pre"] = (
            float(ic_late / ic_pre) if np.isfinite(ic_late) and np.isfinite(ic_pre) and ic_pre > 0 else np.nan
        )

        # Single-shot 10 us onset-aligned traces & rigid-orbit smearing test on Runs 11-14
        if rnum in SINGLE_SHOT_RUNS:
            ss_out = _single_shot_onset_aligned(run, rnum, chans, shunts, pos, x_pos, x_center_cm)
            out.update(ss_out)
            # Estimate rigid m=-1 orbital displacement delta_r = sqrt(2)*(dI_rms/I)/|d(ln I)/dr|
            # and smear the measured pre-onset radial profile to separate orbit smearing from real filling.
            rms_w = float(ss_out.get(f"ss_{rnum}_wall__post_rms_norm_med", np.nan))
            slope_pre = float(out[p + "wall_pre_onset_cm"])
            dr_est = (
                float(np.sqrt(2.0) * rms_w / abs(slope_pre))
                if np.isfinite(rms_w) and np.isfinite(slope_pre) and abs(slope_pre) > 0.0
                else np.nan
            )
            h_sm = np.nan
            smear_factor = np.nan
            if np.isfinite(dr_est) and wins["pre_onset"] is not None:
                lo_p, hi_p = wins["pre_onset"]
                m_pre_t = (t_ms >= lo_p) & (t_ms <= hi_p)
                if np.any(m_pre_t):
                    prof_pre_pos = np.nanmedian(isat0[:, m_pre_t], axis=1)
                    r_fine = np.linspace(0.0, 8.0, 161)
                    # Build symmetric radial profile from both sides of x_center_cm
                    r_bins = np.arange(0.0, 8.5, 0.5)
                    hm_r = hollow_profile_metrics(rr, prof_pre_pos, radial_edges_cm=r_bins)
                    r_mid = 0.5 * (r_bins[:-1] + r_bins[1:])
                    ok_r = np.isfinite(hm_r.radial_profile)
                    if int(np.sum(ok_r)) >= 4:
                        p_fine = np.interp(r_fine, r_mid[ok_r], hm_r.radial_profile[ok_r])
                        hm_0 = hollow_profile_metrics(
                            r_fine, p_fine,
                            core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM,
                        )
                        p_sm = orbit_smeared_profile(r_fine, p_fine, dr_est)
                        hm_sm = hollow_profile_metrics(
                            r_fine, p_sm,
                            core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM,
                        )
                        if np.isfinite(hm_0.hollowness) and hm_0.hollowness > 0.0 and np.isfinite(hm_sm.hollowness):
                            smear_factor = float(hm_sm.hollowness / hm_0.hollowness)
                            h_sm = float(h_pre * smear_factor) if np.isfinite(h_pre) else np.nan
            out[p + "delta_r_orbit_est_cm"] = dr_est
            out[p + "H_orbit_smeared"] = h_sm
            out[p + "H_orbit_smeared_over_pre"] = smear_factor
            out[p + "H_late_over_orbit_smeared"] = _safe_ratio(h_late, h_sm)

    return out


def _single_shot_onset_aligned(run, rnum, chans, shunts, pos, x_pos, x_center_cm):
    """Per-shot 10 us onset-aligned cycle-averaged ``Isat_0`` and fluctuation RMS."""
    sc_m, ch_yp = chans["Y+"]
    _, ch_ym = chans["Y-"]
    tm_s = _time_seconds(run.time_array(scope_name=sc_m))
    i0, i1 = clip_time_window(tm_s, 0.0, BIAS_END_MS)
    tm_c = tm_s[i0:i1]
    dt_m = float(tm_c[1] - tm_c[0])
    step_m = max(1, int(round(SINGLE_SHOT_DT_S / dt_m)))
    n_ds = tm_c.size // step_m
    t_ds_ms = tm_c[:n_ds * step_m].reshape(n_ds, step_m).mean(axis=1) * 1e3
    dt_ds_s = float((t_ds_ms[1] - t_ds_ms[0]) * 1e-3)
    kern_n = max(3, int(round(SINGLE_SHOT_CYCLE_S / dt_ds_s)))
    kern = np.ones(kern_n, dtype=float) / float(kern_n)

    tr_s = _time_seconds(run.time_array(scope_name="lpscope"))
    j0, j1 = clip_time_window(tr_s, 0.0, BIAS_END_MS)
    tr_c = tr_s[j0:j1]
    dt_r = float(tr_c[1] - tr_c[0])
    step_r = max(1, int(round(SINGLE_SHOT_DT_S / dt_r)))
    n_ds_r = tr_c.size // step_r
    t_ds_r_s = tr_c[:n_ds_r * step_r].reshape(n_ds_r, step_r).mean(axis=1)

    out = {f"ss_{rnum}__tau_ms": TAU_ALIGN_MS}
    for zone, x_target in (("core", x_center_cm), ("wall", x_center_cm + 3.0), ("ring", x_center_cm + 6.0)):
        p_idx = int(np.argmin(np.abs(x_pos - float(x_target))))
        sh = position_shots(pos.shot_nums, p_idx, pos.nshot)
        yp, _ = run.channel(ch_yp, scope_name=sc_m, shots=sh)
        ym, _ = run.channel(ch_ym, scope_name=sc_m, shots=sh)
        ref, _ = run.channel("C3", scope_name="lpscope", shots=sh)
        if yp is None or ym is None or ref is None:
            continue

        yp_ds = np.asarray(yp, dtype=float)[:, i0:i0 + n_ds * step_m].reshape(-1, n_ds, step_m).mean(axis=2) / float(shunts["Y+"])
        ym_ds = np.asarray(ym, dtype=float)[:, i0:i0 + n_ds * step_m].reshape(-1, n_ds, step_m).mean(axis=2) / float(shunts["Y-"])
        if float(np.nanmedian(yp_ds)) < 0.0:
            yp_ds = -yp_ds
        if float(np.nanmedian(ym_ds)) < 0.0:
            ym_ds = -ym_ds
        isat0 = np.sqrt(np.maximum(yp_ds, 1e-12) * np.maximum(ym_ds, 1e-12))
        ref_ds = np.asarray(ref, dtype=float)[:, j0:j0 + n_ds_r * step_r].reshape(-1, n_ds_r, step_r).mean(axis=2)

        aligned_mean, aligned_rms, step_jumps = [], [], []
        for s_i in range(isat0.shape[0]):
            t_env, env = sliding_detrended_rms(
                ref_ds[s_i], t_ds_r_s, win_s=jon.WIN_S, step_s=jon.STEP_S
            )
            t_on_s = detect_onset_time(
                t_env, env, threshold=jon.ONSET_THRESH,
                t_min=jon.T_MIN_ONSET_MS * 1e-3, sustain=jon.SUSTAIN_BINS,
            )
            if not np.isfinite(t_on_s):
                continue
            t_on_ms = float(t_on_s) * 1e3
            i_sm = np.convolve(isat0[s_i], kern, mode="same")
            i_fluc = isat0[s_i] - i_sm
            m_pre = (t_ds_ms >= t_on_ms - 0.40) & (t_ds_ms <= t_on_ms - 0.10)
            if not np.any(m_pre):
                continue
            base = float(np.mean(i_sm[m_pre]))
            if not (np.isfinite(base) and base > 0.0):
                continue
            rms_fluc = np.sqrt(np.convolve(i_fluc ** 2, kern, mode="same")) / base
            aligned_mean.append(np.interp(t_on_ms + TAU_ALIGN_MS, t_ds_ms, i_sm / base, left=np.nan, right=np.nan))
            aligned_rms.append(np.interp(t_on_ms + TAU_ALIGN_MS, t_ds_ms, rms_fluc, left=np.nan, right=np.nan))
            m_win = (t_ds_ms >= t_on_ms - 0.10) & (t_ds_ms <= t_on_ms + 0.50)
            seg = isat0[s_i, m_win]
            if seg.size > 5:
                step_jumps.append(float(np.max(np.abs(seg[5:] - seg[:-5])) / base))

        pz = f"ss_{rnum}_{zone}__"
        out[pz + "x_cm"] = float(x_pos[p_idx])
        out[pz + "n_shots"] = int(len(aligned_mean))
        if aligned_mean:
            med_mean = np.nanmedian(aligned_mean, axis=0)
            med_rms = np.nanmedian(aligned_rms, axis=0)
            out[pz + "i_mean_norm"] = med_mean
            out[pz + "di_rms_norm"] = med_rms
            out[pz + "max_50us_jump_med"] = float(np.median(step_jumps)) if step_jumps else np.nan
            # Compact post-onset mean excursion over tau in [+0.2, +1.0] ms
            m_post = (TAU_ALIGN_MS >= 0.2) & (TAU_ALIGN_MS <= 1.0) & np.isfinite(med_mean)
            out[pz + "post_mean_norm_med"] = float(np.median(med_mean[m_post])) if np.any(m_post) else np.nan
            out[pz + "post_rms_norm_med"] = float(np.median(med_rms[m_post])) if np.any(m_post) else np.nan
        else:
            out[pz + "i_mean_norm"] = np.full_like(TAU_ALIGN_MS, np.nan)
            out[pz + "di_rms_norm"] = np.full_like(TAU_ALIGN_MS, np.nan)
            out[pz + "max_50us_jump_med"] = np.nan
            out[pz + "post_mean_norm_med"] = np.nan
            out[pz + "post_rms_norm_med"] = np.nan
    return out


# ============================================================================
# 3. Magnetic-field & axial-localization lines (Runs 28, 27, 30)
# ============================================================================
def analyze_b0_and_axial_lines(data_dir=DATA_DIR, *, bin_ms=BIN_MS):
    """Fixed-bias P29-R & P25-R Isat lines across B0 = 600, 800, 1000 G."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}

    for rnum, b0_g in B0_LINE_RUNS:
        ifn = _run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        puff = parse_gas_puff(run.description().raw)
        puff_ms = float(puff[1]) if isinstance(puff, tuple) else float(puff)
        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        wins = _window_bounds(t_on)
        tarr = run.time_array(scope_name="scope")
        edges, t_ms = time_bin_edges(tarr, (-1.0, 7.0), bin_ms)
        nbins = t_ms.size

        for mg, ch, port_tag in (
            ("<Hermes>    p29_LP", "C2", "p29"),
            ("<Hades>    p25_LP", "C4", "p25"),
        ):
            pos = jiv.read_lp_positions(ifn, motion_group_name=mg)
            x_pos, _ = jml.scanned_axis(pos)
            npos = x_pos.size
            arr = np.full((npos, nbins), np.nan, dtype=float)
            for p in range(npos):
                sh = position_shots(pos.shot_nums, p, pos.nshot)
                d, _ = run.channel(ch, scope_name="scope", shots=sh)
                if d is None:
                    continue
                sig = np.nanmean(np.asarray(d, dtype=float), axis=0)
                arr[p] = _bin_nanmean(sig, edges)
            if float(np.nanmedian(arr)) < 0.0:
                arr = -arr

            # Core center from early-bias hollow minimum inside |x| <= 3 cm
            m_t0 = (t_ms >= 0.5) & (t_ms <= 0.9)
            prof0 = np.nanmedian(arr[:, m_t0], axis=1)
            m_mid = np.abs(x_pos) <= 3.0
            xc = float(x_pos[m_mid][int(np.nanargmin(prof0[m_mid]))])
            rr = np.abs(x_pos - xc)
            hm = hollow_profile_metrics(
                rr, arr, core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM
            )

            pfx = f"b0_{rnum}_{port_tag}__"
            out[pfx + "b0_g"] = float(b0_g)
            out[pfx + "gas_puff_ms"] = puff_ms
            out[pfx + "t_onset_med_ms"] = t_on
            out[pfx + "x_center_cm"] = xc
            out[pfx + "t_ms"] = t_ms
            out[pfx + "H_t"] = hm.hollowness
            for wname, bounds in wins.items():
                out[pfx + f"H_{wname}"] = _window_median(t_ms, hm.hollowness, bounds)
                out[pfx + f"ic_{wname}"] = _window_median(t_ms, hm.core_val, bounds)
                out[pfx + f"ir_{wname}"] = _window_median(t_ms, hm.ring_val, bounds)
            h_pre, h_late = out[pfx + "H_pre_onset"], out[pfx + "H_late_sat"]
            ic_pre, ic_late = out[pfx + "ic_pre_onset"], out[pfx + "ic_late_sat"]
            out[pfx + "H_late_over_pre"] = (
                float(h_late / h_pre) if np.isfinite(h_late) and np.isfinite(h_pre) and h_pre > 0 else np.nan
            )
            out[pfx + "ic_late_over_pre"] = (
                float(ic_late / ic_pre) if np.isfinite(ic_late) and np.isfinite(ic_pre) and ic_pre > 0 else np.nan
            )
    return out


# ============================================================================
# 4. Shift-immune 2-D planes (Runs 26, 31 P29-R Isat & Run 32 P33 3-D Mach)
# ============================================================================
def _safe_ratio(num, den):
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0.0 else np.nan


def _summarize_2d_plane_windows(x_u, y_u, vals, t_on_ms, *,
                                rel_rms_late=None, p10_late=None):
    """Per-window centered 2-D radial hollowness, crash-safe when any window is ``None``.

    Each bias-on / post-off window fits its own 2-D core center ``(xc_k, yc_k)``
    so slow drifts of the hollow core never bias radial hollowness; ``pre_bias``
    (flat background) reuses the ``pre_onset`` center. Also evaluates the rigid
    ``m = -1`` orbit-smearing prediction from ``rel_rms_late`` (if provided) and
    the instantaneous cycle-trough (10th-percentile) hollowness from ``p10_late``.
    """
    nbins = RADIAL_EDGES_CM.size - 1
    out = {
        "t_onset_med_ms": float(t_on_ms),
        "radial_edges_cm": RADIAL_EDGES_CM,
    }

    # Fit per-window centers for non-flat windows; pre_bias reuses pre_onset center.
    centers = {}
    for k in ("pre_onset", "early_post", "late_sat", "post_off"):
        if k in vals:
            centers[k] = _find_2d_core_center(x_u, y_u, vals[k])
        else:
            centers[k] = (np.nan, np.nan)
    c_pre = centers["pre_onset"]
    centers["pre_bias"] = c_pre if np.all(np.isfinite(c_pre)) else (0.0, 0.0)

    out["center_pre_xy_cm"] = np.array(centers["pre_onset"], dtype=float)
    out["center_late_xy_cm"] = np.array(centers["late_sat"], dtype=float)

    for k in ("pre_bias", "pre_onset", "early_post", "late_sat", "post_off"):
        xc_k, yc_k = centers[k]
        out[f"center_{k}_xy_cm"] = np.array([xc_k, yc_k], dtype=float)
        if k not in vals or not (np.isfinite(xc_k) and np.isfinite(yc_k)):
            out[f"H_{k}"] = np.nan
            out[f"ic_{k}"] = np.nan
            out[f"ir_{k}"] = np.nan
            out[f"wall_{k}_cm"] = np.nan
            out[f"r_prof_norm_{k}"] = np.full(nbins, np.nan, dtype=float)
            continue
        rr_k = np.hypot(x_u - xc_k, y_u - yc_k)
        hm = hollow_profile_metrics(
            rr_k, vals[k],
            core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM,
            wall_r_cm=WALL_R_CM, radial_edges_cm=RADIAL_EDGES_CM,
        )
        out[f"H_{k}"] = hm.hollowness
        out[f"ic_{k}"] = hm.core_val
        out[f"ir_{k}"] = hm.ring_val
        out[f"wall_{k}_cm"] = hm.wall_inv_scale_cm
        out[f"r_prof_norm_{k}"] = (
            hm.radial_profile / hm.ring_val
            if np.isfinite(hm.ring_val) and hm.ring_val > 0.0
            else np.full(nbins, np.nan, dtype=float)
        )

    out["H_late_over_pre"] = _safe_ratio(out["H_late_sat"], out["H_pre_onset"])
    out["ic_late_over_pre"] = _safe_ratio(out["ic_late_sat"], out["ic_pre_onset"])

    # Rigid m=-1 orbit-smearing forward model (both median and 90th-percentile wall RMS)
    wall_rms_med = np.nan
    wall_rms_p90 = np.nan
    dr_est = np.nan
    dr_est_p90 = np.nan
    h_sm = np.nan
    h_sm_p90 = np.nan
    smear_factor = np.nan
    smear_factor_p90 = np.nan
    if rel_rms_late is not None and "late_sat" in vals and np.all(np.isfinite(centers["late_sat"])):
        xc_l, yc_l = centers["late_sat"]
        rr_l = np.hypot(x_u - xc_l, y_u - yc_l)
        m_wall_l = (rr_l >= WALL_R_CM[0]) & (rr_l <= WALL_R_CM[1]) & np.isfinite(rel_rms_late)
        if np.any(m_wall_l):
            wall_rms_med = float(np.median(rel_rms_late[m_wall_l]))
            wall_rms_p90 = float(np.percentile(rel_rms_late[m_wall_l], 90))
        slope_pre = float(out["wall_pre_onset_cm"])
        if np.isfinite(wall_rms_med) and np.isfinite(slope_pre) and abs(slope_pre) > 0.0:
            dr_est = float(np.sqrt(2.0) * wall_rms_med / abs(slope_pre))
            dr_est_p90 = float(np.sqrt(2.0) * wall_rms_p90 / abs(slope_pre))
            r_mid = 0.5 * (RADIAL_EDGES_CM[:-1] + RADIAL_EDGES_CM[1:])
            p_pre = out["r_prof_norm_pre_onset"]
            ok_r = np.isfinite(p_pre)
            if int(np.sum(ok_r)) >= 4:
                r_fine = np.linspace(0.0, float(RADIAL_EDGES_CM[-1]), 161)
                p_fine = np.interp(r_fine, r_mid[ok_r], p_pre[ok_r])
                hm_0 = hollow_profile_metrics(
                    r_fine, p_fine,
                    core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM,
                )
                if np.isfinite(hm_0.hollowness) and hm_0.hollowness > 0.0:
                    hm_sm = hollow_profile_metrics(
                        r_fine, orbit_smeared_profile(r_fine, p_fine, dr_est),
                        core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM,
                    )
                    hm_sm90 = hollow_profile_metrics(
                        r_fine, orbit_smeared_profile(r_fine, p_fine, dr_est_p90),
                        core_r_max_cm=CORE_R_MAX_CM, ring_r_cm=RING_R_CM, wall_r_cm=WALL_R_CM,
                    )
                    smear_factor = _safe_ratio(hm_sm.hollowness, hm_0.hollowness)
                    smear_factor_p90 = _safe_ratio(hm_sm90.hollowness, hm_0.hollowness)
                    if np.isfinite(out["H_pre_onset"]):
                        h_sm = float(out["H_pre_onset"] * smear_factor) if np.isfinite(smear_factor) else np.nan
                        h_sm_p90 = float(out["H_pre_onset"] * smear_factor_p90) if np.isfinite(smear_factor_p90) else np.nan

    out["wall_rel_rms_late"] = wall_rms_med
    out["wall_rel_rms_p90_late"] = wall_rms_p90
    out["delta_r_orbit_est_cm"] = dr_est
    out["delta_r_orbit_p90_cm"] = dr_est_p90
    out["H_orbit_smeared"] = h_sm
    out["H_orbit_smeared_p90"] = h_sm_p90
    out["H_orbit_smeared_over_pre"] = smear_factor
    out["H_orbit_smeared_p90_over_pre"] = smear_factor_p90
    out["H_late_over_orbit_smeared"] = _safe_ratio(out["H_late_sat"], h_sm)
    out["H_late_over_orbit_smeared_p90"] = _safe_ratio(out["H_late_sat"], h_sm_p90)

    # Model-free instantaneous cycle-trough core floor inside r <= 3.5 cm (p10 over time)
    ic_trough_late = np.nan
    h_trough_late = np.nan
    if p10_late is not None and "late_sat" in vals and np.all(np.isfinite(centers["late_sat"])):
        xc_l, yc_l = centers["late_sat"]
        rr_l = np.hypot(x_u - xc_l, y_u - yc_l)
        m_in = (rr_l <= 3.5) & np.isfinite(p10_late)
        if np.any(m_in):
            # Median of the 5 lowest cycle-trough points inside r <= 3.5 cm
            ii = np.flatnonzero(m_in)
            k_low = min(5, ii.size)
            ic_trough_late = float(np.median(np.sort(p10_late[ii])[:k_low]))
            h_trough_late = _safe_ratio(out["ir_late_sat"], ic_trough_late)
    out["ic_trough_late"] = ic_trough_late
    out["H_trough_late"] = h_trough_late
    return out


def _analyze_mach32_arrays(t32, px, py, mx, my, mz, t_on_ref_ms, *,
                           t_on_source="matched_run12_35ms"):
    """Compute centered Run 32 3-D Mach ``M_theta`` and ``M_z`` across onset windows.

    Note: :func:`find_flow_centre` and :func:`polar_components` both take
    velocity components ``(vx, vy)`` FIRST and positions ``(pos_x, pos_y)``
    SECOND.
    """
    m_fit = (t32 >= 2.5) & (t32 <= 4.5)
    mx_m = _nanmean_rows(mx[:, m_fit])
    my_m = _nanmean_rows(my[:, m_fit])
    xc, yc, ratio = find_flow_centre(mx_m, my_m, px, py)
    rr = np.hypot(px - xc, py - yc)
    m_wall = (rr >= ROT_R_CM[0]) & (rr <= ROT_R_CM[1])
    m_core = rr <= CORE_R_MAX_CM
    m_ring = (rr >= RING_R_CM[0]) & (rr <= RING_R_CM[1])

    wins32 = _window_bounds(t_on_ref_ms)
    out = {
        "mach32__center_xy_cm": np.array([xc, yc], dtype=float),
        "mach32__center_ratio": float(ratio),
        "mach32__t_onset_ref_ms": float(t_on_ref_ms),
        "mach32__t_onset_source": str(t_on_source),
    }
    for wname, bounds in wins32.items():
        if bounds is None:
            out[f"mach32__M_theta_wall_{wname}"] = np.nan
            out[f"mach32__M_z_core_{wname}"] = np.nan
            out[f"mach32__M_z_ring_{wname}"] = np.nan
            continue
        m = (t32 >= bounds[0]) & (t32 <= bounds[1])
        ux = _nanmean_rows(mx[:, m])
        uy = _nanmean_rows(my[:, m])
        uz = _nanmean_rows(mz[:, m])
        _, uth = polar_components(ux, uy, px, py, (xc, yc))
        out[f"mach32__M_theta_wall_{wname}"] = float(np.nanmedian(uth[m_wall]))
        out[f"mach32__M_z_core_{wname}"] = float(np.nanmedian(uz[m_core]))
        out[f"mach32__M_z_ring_{wname}"] = float(np.nanmedian(uz[m_ring]))

    m_pre = out["mach32__M_theta_wall_pre_onset"]
    m_late = out["mach32__M_theta_wall_late_sat"]
    out["mach32__M_theta_late_over_pre"] = (
        float(m_late / m_pre)
        if np.isfinite(m_pre) and np.isfinite(m_late) and abs(m_pre) > 0.0
        else np.nan
    )
    return out


def analyze_2d_planes_relaxation(data_dir=DATA_DIR):
    """2-D azimuthally averaged ``<Isat(r, t)>_theta`` (Runs 26, 31) & 3-D Mach flow (Run 32)."""
    on = np.load(os.path.join(data_dir, jon.OUT_NAME), allow_pickle=True)
    out = {}

    for rnum, mg, scope, ch, label, b0_g in ISAT_2D_PLANES:
        ifn = _run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        pos = jiv.read_lp_positions(ifn, motion_group_name=mg)
        x_u, y_u = _pos_xy_per_position(pos)
        tarr_s = _time_seconds(run.time_array(scope_name=scope))
        dt_s = float(tarr_s[1] - tarr_s[0])
        step_10us = max(1, int(round(SINGLE_SHOT_DT_S / dt_s)))
        t_on = float(on[f"{rnum}__t_onset_med_ms"])
        wins = _window_bounds(t_on)
        w_idx = {k: clip_time_window(tarr_s, *b) for k, b in wins.items() if b is not None}

        npos = x_u.size
        vals = {k: np.full(npos, np.nan, dtype=float) for k in w_idx}
        rel_rms_late = np.full(npos, np.nan, dtype=float)
        p10_late = np.full(npos, np.nan, dtype=float)
        late_bounds = w_idx.get("late_sat")

        for p in range(npos):
            sh = position_shots(pos.shot_nums, p, pos.nshot)
            d, _ = run.channel(ch, scope_name=scope, shots=sh)
            if d is None:
                continue
            arr_sh = np.asarray(d, dtype=float)
            sig = np.nanmean(arr_sh, axis=0)
            for k, (a, b) in w_idx.items():
                vals[k][p] = float(np.nanmean(sig[a:b]))
            if late_bounds is not None:
                a, b = late_bounds
                seg = arr_sh[:, a:b]
                n_ds = seg.shape[1] // step_10us
                if n_ds >= 10:
                    # 10 us block-mean suppresses digitizer white noise while keeping ~4 kHz m=-1 fully resolved
                    ds = seg[:, :n_ds * step_10us].reshape(seg.shape[0], n_ds, step_10us).mean(axis=2)
                    # Determine positive magnitude sign per position using late mean
                    sgn = -1.0 if float(np.nanmedian(ds)) < 0.0 else 1.0
                    ds = sgn * ds
                    t_lin = np.linspace(-1.0, 1.0, n_ds)
                    shot_rms, shot_p10 = [], []
                    for s_i in range(ds.shape[0]):
                        row = ds[s_i]
                        if not np.all(np.isfinite(row)):
                            continue
                        mu = float(np.mean(row))
                        if mu <= 0.0:
                            continue
                        slope = float(np.dot(t_lin, row - mu) / np.dot(t_lin, t_lin))
                        resid = row - (mu + slope * t_lin)
                        shot_rms.append(float(np.std(resid) / mu))
                        shot_p10.append(float(np.percentile(row, 10)))
                    if shot_rms:
                        rel_rms_late[p] = float(np.median(shot_rms))
                        p10_late[p] = float(np.median(shot_p10))

        for k in vals:
            if np.isfinite(np.nanmedian(vals[k])) and float(np.nanmedian(vals[k])) < 0.0:
                vals[k] = -vals[k]

        plane_res = _summarize_2d_plane_windows(
            x_u, y_u, vals, t_on,
            rel_rms_late=rel_rms_late if late_bounds is not None else None,
            p10_late=p10_late if late_bounds is not None else None,
        )
        pfx = f"plane_{rnum}__"
        out[pfx + "b0_g"] = float(b0_g)
        out[pfx + "probe_label"] = label
        for k, v in plane_res.items():
            out[pfx + k] = v

    # Run 32 3-D Mach plane (cached 32-mach-flow-data.npz, 800 G, 35 ms puff)
    npz32 = os.path.join(data_dir, "32-mach-flow-data.npz")
    if os.path.exists(npz32):
        d32 = np.load(npz32, allow_pickle=True)
        t32 = np.asarray(d32["t_ms"], dtype=float)
        px, py = np.asarray(d32["pos_x"], dtype=float), np.asarray(d32["pos_y"], dtype=float)
        mx, my, mz = (
            np.asarray(d32["M_X"], dtype=float),
            np.asarray(d32["M_Y"], dtype=float),
            np.asarray(d32["M_Z"], dtype=float),
        )
        t_on_32 = float(on["12__t_onset_med_ms"])
        out.update(_analyze_mach32_arrays(t32, px, py, mx, my, mz, t_on_32))

    return out


# ============================================================================
# Main orchestration & reporting
# ============================================================================
def run_relaxation_analysis(data_dir=DATA_DIR, out_name=OUT_NAME, overwrite=True):
    """Run all four hollow-core relaxation checks and save summary npz."""
    out_path = os.path.join(data_dir, out_name)
    if not overwrite and os.path.exists(out_path):
        print(f"  {out_name} exists, skipping (pass overwrite=True to regenerate)")
        with np.load(out_path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}

    res = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "core_r_max_cm": float(CORE_R_MAX_CM),
        "ring_r_cm": np.array(RING_R_CM, dtype=float),
        "wall_r_cm": np.array(WALL_R_CM, dtype=float),
    }
    res.update(analyze_swept_iv_hollowness(data_dir))
    res.update(analyze_mach_line_relaxation(data_dir))
    res.update(analyze_b0_and_axial_lines(data_dir))
    res.update(analyze_2d_planes_relaxation(data_dir))

    merge_save_npz(out_path, res)
    print_summary(res)
    print(f"\nSaved {out_path} ({len(res)} keys)")
    return res


def batch_all(data_dir=DATA_DIR, overwrite=False):
    """Campaign entry point matching sibling Jun-2026 batch modules."""
    return run_relaxation_analysis(data_dir=data_dir, overwrite=overwrite)


def print_summary(res):
    """Print concise tables of pre- vs. post-onset hollowness and collapse checks."""
    print("\n=== 1. P33 Mach-Line Flow-Cancelled Isat_0 Hollowness (Runs 08-14, 800 G) ===")
    print("Run Puff t_on  | H_preB H_preOn H_early H_late | H_late/pre Ic_late/pre | dlnI_pre->late | M_pre->late | H_postOff")
    for rnum, puff in MACH_LINE_RUNS:
        p = f"mach_{rnum}__"
        print(
            f" {rnum} {puff:2.0f}ms {res[p+'t_onset_med_ms']:4.2f}  | "
            f" {res[p+'H_pre_bias']:5.2f}   {res[p+'H_pre_onset']:5.2f}   "
            f"{res[p+'H_early_post']:5.2f}  {res[p+'H_late_sat']:5.2f} | "
            f"   {res[p+'H_late_over_pre']:5.2f}       {res[p+'ic_late_over_pre']:5.2f}    | "
            f"{res[p+'wall_pre_onset_cm']:+.2f}->{res[p+'wall_late_sat_cm']:+.2f}  | "
            f"{res[p+'m_rot_pre_onset']:+.2f}->{res[p+'m_rot_late_sat']:+.2f} | "
            f"  {res[p+'H_post_off']:5.2f}"
        )

    print("\n=== 2. Single-Shot 10 us Onset-Aligned Core / Wall / Ring & Orbit-Smearing (Runs 11-14) ===")
    for rnum in SINGLE_SHOT_RUNS:
        parts = []
        for z in ("core", "wall", "ring"):
            pz = f"ss_{rnum}_{z}__"
            parts.append(
                f"{z}: mean={res[pz+'post_mean_norm_med']:.3f}, "
                f"rms={res[pz+'post_rms_norm_med']*100:.1f}%, "
                f"max50us={res[pz+'max_50us_jump_med']*100:.1f}%"
            )
        pm = f"mach_{rnum}__"
        print(
            f"  Run {rnum}: " + " | ".join(parts) + "\n"
            f"          orbit-smear check: dr_est={res[pm+'delta_r_orbit_est_cm']:.2f}cm -> "
            f"H_sm={res[pm+'H_orbit_smeared']:.2f} (H_sm/pre={res[pm+'H_orbit_smeared_over_pre']:.2f}, "
            f"H_late/H_sm={res[pm+'H_late_over_orbit_smeared']:.2f})"
        )

    print("\n=== 3. B0-Scan & Axial Localization Lines (P29-R vs P25-R, 35 ms puff) ===")
    for rnum, b0 in B0_LINE_RUNS:
        for pt in ("p29", "p25"):
            pfx = f"b0_{rnum}_{pt}__"
            print(
                f"  Run {rnum} ({b0:4.0f}G) {pt.upper()}: t_on={res[pfx+'t_onset_med_ms']:.2f}ms | "
                f"H_preOn={res[pfx+'H_pre_onset']:.2f} -> H_early={res[pfx+'H_early_post']:.2f} -> "
                f"H_late={res[pfx+'H_late_sat']:.2f} (H_late/pre={res[pfx+'H_late_over_pre']:.2f}, "
                f"Ic_late/pre={res[pfx+'ic_late_over_pre']:.2f})"
            )

    print("\n=== 4. Per-Window Centered 2-D Planes (P29-R Isat & P33 3-D Mach) ===")
    for rnum, _, _, _, label, b0 in ISAT_2D_PLANES:
        pfx = f"plane_{rnum}__"
        c0, c1 = res[pfx + "center_pre_xy_cm"], res[pfx + "center_late_xy_cm"]
        print(
            f"  Run {rnum} ({b0:.0f}G) {label}: ctr ({c0[0]:+.2f},{c0[1]:+.2f})->({c1[0]:+.2f},{c1[1]:+.2f}) cm | "
            f"H: preB={res[pfx+'H_pre_bias']:.2f}, preOn={res[pfx+'H_pre_onset']:.2f}, "
            f"early={res[pfx+'H_early_post']:.2f}, late={res[pfx+'H_late_sat']:.2f} "
            f"(H_late/pre={res[pfx+'H_late_over_pre']:.2f}, Ic_late/pre={res[pfx+'ic_late_over_pre']:.2f})\n"
            f"          orbit-smear: wall_rms={res[pfx+'wall_rel_rms_late']*100:.1f}% (p90={res[pfx+'wall_rel_rms_p90_late']*100:.1f}%), "
            f"dr_est={res[pfx+'delta_r_orbit_est_cm']:.2f} ({res[pfx+'delta_r_orbit_p90_cm']:.2f}) cm -> "
            f"H_sm={res[pfx+'H_orbit_smeared']:.2f} ({res[pfx+'H_orbit_smeared_p90']:.2f}) "
            f"[H_sm/pre={res[pfx+'H_orbit_smeared_over_pre']:.2f} ({res[pfx+'H_orbit_smeared_p90_over_pre']:.2f}), "
            f"H_late/H_sm={res[pfx+'H_late_over_orbit_smeared']:.2f} ({res[pfx+'H_late_over_orbit_smeared_p90']:.2f})]"
        )
    if "mach32__M_theta_wall_pre_onset" in res:
        c32 = res["mach32__center_xy_cm"]
        print(
            f"  Run 32 (800G) P33 Mach plane: ctr=({c32[0]:+.2f},{c32[1]:+.2f})cm (ratio={res['mach32__center_ratio']:.2f}) | "
            f"M_theta_wall preOn={res['mach32__M_theta_wall_pre_onset']:+.3f} -> "
            f"early={res['mach32__M_theta_wall_early_post']:+.3f} -> late={res['mach32__M_theta_wall_late_sat']:+.3f} "
            f"(ratio={res['mach32__M_theta_late_over_pre']:.2f}) | "
            f"M_z_core={res['mach32__M_z_core_pre_onset']:+.3f}->{res['mach32__M_z_core_late_sat']:+.3f}"
        )


if __name__ == "__main__":
    run_relaxation_analysis()
