"""Jun-2026 pre-onset IV profile, electron pressure, signed gradient & Vf analysis.

Combines the calibrated Day-1 Langmuir-probe line scans (Runs 06, 00-05;
gas puff 23-45 ms at 800 G) with the instability onset timings from
:mod:`Jun2026_onset` (``onset_summary_jun2026.npz``) to:

1. Classify every Langmuir sweep rigorously using ``sweep_t_start`` and
   ``sweep_t_stop`` against each run's ``t_onset_q25_ms`` and ``t_onset_med_ms``:
   ``"pre_bias"``, ``"pre_onset"``, ``"straddling"``, ``"post_onset"``, or
   ``"post_bias"`` -- strictly separating stationary pre-onset measurements
   from slow sweeps averaged over an unstable state.
2. Extract per-shot floating potential ``Vf(x, t)`` directly from
   ``*-sweep-data.npz`` and compute the sheath factor
   ``Lambda = (Vp - Vf) / Te`` to test whether ``Vp`` invariance across gas
   puffs is physical (``Vf`` dropping with heating ``Te``) and flag primary-
   electron contamination (``Lambda > 6``).
3. Compute electron pressure ``pe [Pa]``, signed outward radial inverse scale
   lengths ``d(ln ne)/dr``, ``d(ln Te)/dr``, ``d(ln pe)/dr``, ``eta_e``, and
   ``Er`` (both raw central differences and Savitzky-Golay ``window=5``) with
   propagated uncertainties, plus local plasma scales (``rho_s``, ``nu_ei``,
   ``beta_e``, ``v_ExB``, and ``m=1`` diamagnetic frequencies).

Saves ``profiles_summary_jun2026.npz`` in ``DATA_DIR`` and prints a summary
table comparing the last strictly pre-onset sweep across the 23-45 ms puff scan.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_profiles.py
"""

import datetime
import os
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from data_analysis.plasma.langmuir import (
    discover_channels,
    load_plasma_data,
    load_sweep_data,
    ne_is_calibrated,
    sweep_npz_paths,
)
from data_analysis.plasma.profiles import (
    floating_potential_from_iv,
    local_drift_scales,
    radial_profile_gradients,
    sheath_factor,
)
from data_analysis.utils import merge_save_npz

import Jun2026_onset as jon

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "profiles_summary_jun2026.npz"

B0_G = 800.0
ION_MU = 4.0                     # He+
X_CENTER_CM = 1.0                # hollow-core symmetry axis [cm]
INNER_WALL_R_CM = (1.5, 4.0)     # radial band of the inner hollowing boundary [cm]
CORE_R_MAX_CM = 1.5              # hollow-core radius [cm]
RING_R_CM = (4.5, 7.0)           # surrounding hot/dense ring [cm]
SMOOTH_WINDOW = 5                # Savitzky-Golay 5-cm window (1 cm grid spacing)
SMOOTH_POLY = 2
VF_SMOOTH_SAMPLES = 51           # ~0.5 V boxcar on 7902-sample sweep near I=0
PRIMARY_LAMBDA_THRESH = 6.0      # Lambda > 6 flags non-thermal / primary electrons
BIAS_END_MS = 5.0
MODE_M = 1                       # azimuthal mode number for diamagnetic estimate


class ProfileRunSpec(NamedTuple):
    run_num: str
    gas_puff_ms: float
    b0_g: float = B0_G
    tip: str = "P29-R"


# Day-1 gas-puff scan in ascending puff order (all 800 G, calibrated P29-R).
PROFILE_RUNS = (
    ProfileRunSpec("06", 23.0),
    ProfileRunSpec("00", 24.0),
    ProfileRunSpec("01", 25.0),
    ProfileRunSpec("02", 30.0),
    ProfileRunSpec("03", 35.0),
    ProfileRunSpec("04", 40.0),
    ProfileRunSpec("05", 45.0),
)
PROFILE_RUN_MAP = {s.run_num: s for s in PROFILE_RUNS}


def classify_sweeps(t_start_ms, t_stop_ms, t_onset_q25_ms=np.nan,
                    t_onset_med_ms=np.nan, bias_end_ms=BIAS_END_MS):
    """Classify each sweep interval ``[t_start, t_stop]`` relative to bias & onset.

    Returns a list of strings (one per sweep):
    - ``"pre_bias"``: ``t_stop < 0.0``
    - ``"post_bias"``: ``t_start >= bias_end_ms``
    - ``"pre_onset"``: bias-on sweep finishing before ``t_onset_q25_ms`` (or any
      bias-on sweep on a stable run with ``t_onset = NaN``)
    - ``"straddling"``: starts before ``t_onset_med_ms`` but extends past
      ``t_onset_q25_ms`` (e.g. Run 02 at 1.6 ms)
    - ``"post_onset"``: starts at or after ``t_onset_med_ms`` (slow sweep over
      an already-unstable plasma)
    """
    t0 = np.asarray(t_start_ms, dtype=float)
    t1 = np.asarray(t_stop_ms, dtype=float)
    has_onset = np.isfinite(t_onset_med_ms)
    q25 = float(t_onset_q25_ms) if np.isfinite(t_onset_q25_ms) else float(t_onset_med_ms)
    med = float(t_onset_med_ms)

    labels = []
    for a, b in zip(t0, t1):
        if b <= 0.0 or 0.5 * (a + b) < 0.0:
            labels.append("pre_bias")
        elif a >= bias_end_ms:
            labels.append("post_bias")
        elif not has_onset:
            labels.append("pre_onset")
        elif b <= q25:
            labels.append("pre_onset")
        elif a < med:
            labels.append("straddling")
        else:
            labels.append("post_onset")
    return labels


def select_reference_pre_onset_sweep(t_mid_ms, regimes, prefer_t_ms=2.1):
    """Pick the best bias-on pre-onset sweep index (and whether it straddles).

    Prefers the latest ``"pre_onset"`` sweep (for stable runs with sweeps all the
    way to 5 ms, prefers the sweep closest to ``prefer_t_ms`` so 23 ms is
    compared at a matched bias duration alongside 24-25 ms). Returns ``(idx, regime)``
    or ``(None, None)`` if no bias-on sweep exists.
    """
    pre = [i for i, r in enumerate(regimes) if r == "pre_onset"]
    if pre:
        # On a stable run (e.g. 06, sweeps up to 4.1 ms), pick the sweep closest
        # to prefer_t_ms if pre-onset sweeps extend past prefer_t_ms + 0.8 ms;
        # on unstable runs (where pre-onset sweeps terminate at t_onset), take
        # the last strictly pre-onset sweep right before instability onset.
        if t_mid_ms[pre[-1]] > prefer_t_ms + 0.8:
            best = min(pre, key=lambda i: abs(float(t_mid_ms[i]) - prefer_t_ms))
            return best, "pre_onset"
        return pre[-1], "pre_onset"
    strad = [i for i, r in enumerate(regimes) if r == "straddling"]
    if strad:
        return strad[0], "straddling"
    return None, None


def extract_vf_profile(Vswp_arr_rs, Iswp_arr_rs, smooth_samples=VF_SMOOTH_SAMPLES):
    """Compute per-position, per-sweep ``(Vf_mean, Vf_err)`` across shots.

    ``Vswp_arr_rs`` has shape ``(npos, nswp, nt)`` and ``Iswp_arr_rs`` has shape
    ``(npos, nshot, nswp, nt)``. ``Vf_err`` is the standard error of the mean
    across finite shots (``ddof=1`` when ``n >= 2``).
    """
    # Broadcast V across the shot axis: (npos, 1, nswp, nt) vs (npos, nshot, nswp, nt)
    vf_shots = floating_potential_from_iv(
        Vswp_arr_rs[:, None, :, :], Iswp_arr_rs, smooth_samples=smooth_samples
    )
    finite = np.isfinite(vf_shots)
    n_good = np.sum(finite, axis=1)
    vf_mean = np.full(n_good.shape, np.nan, dtype=float)
    vf_err = np.full(n_good.shape, np.nan, dtype=float)

    has_any = n_good >= 1
    if np.any(has_any):
        # Avoid RuntimeWarning on all-NaN slices
        vf_sum = np.where(finite, vf_shots, 0.0).sum(axis=1)
        vf_mean[has_any] = vf_sum[has_any] / n_good[has_any]
    has_two = n_good >= 2
    if np.any(has_two):
        diff = np.where(finite, vf_shots - vf_mean[:, None, :], 0.0)
        var = np.sum(diff ** 2, axis=1) / np.maximum(n_good - 1, 1)
        vf_err[has_two] = np.sqrt(var[has_two] / n_good[has_two])
    return vf_mean, vf_err


def _inner_wall_summary(x_cm, ne, te, vp, vf, lam, grads, scales, x_center_cm):
    """Scalar summary on the inner wall (``r in INNER_WALL_R_CM``) + core/ring."""
    r = np.abs(x_cm - x_center_cm)
    wall = (r >= INNER_WALL_R_CM[0]) & (r <= INNER_WALL_R_CM[1])
    core = r <= CORE_R_MAX_CM
    ring = (r >= RING_R_CM[0]) & (r <= RING_R_CM[1])

    def _med(arr, mask):
        vals = arr[mask]
        fin = vals[np.isfinite(vals)]
        return float(np.median(fin)) if fin.size > 0 else np.nan

    def _peak_signed(arr, mask):
        vals = arr[mask]
        fin = vals[np.isfinite(vals)]
        if fin.size == 0:
            return np.nan
        return float(fin[int(np.argmax(np.abs(fin)))])

    # Location of steepest |d ln Te / dr| on the inner wall: sample local signed
    # gradients & scales right where the thermal gradient peaks.
    wall_idx = np.flatnonzero(wall & np.isfinite(grads.dln_te_dx))
    if wall_idx.size > 0:
        i_steep = int(wall_idx[int(np.argmax(np.abs(grads.dln_te_dx[wall_idx])))])
        dln_ne_at_steep = float(grads.dln_ne_dx[i_steep])
        dln_te_at_steep = float(grads.dln_te_dx[i_steep])
        dln_pe_at_steep = float(grads.dln_pe_dx[i_steep])
        eta_e_at_steep = float(grads.eta_e[i_steep])
        x_steep = float(x_cm[i_steep])
    else:
        dln_ne_at_steep = dln_te_at_steep = dln_pe_at_steep = eta_e_at_steep = x_steep = np.nan

    ne_core = _med(ne, core)
    ne_ring = _med(ne, ring)
    te_core = _med(te, core)
    te_ring = _med(te, ring)

    return {
        "x_steep_cm": x_steep,
        "ne_core_cm3": ne_core,
        "ne_ring_cm3": ne_ring,
        "hollow_ratio": ne_ring / ne_core if (np.isfinite(ne_core) and ne_core > 0) else np.nan,
        "te_core_ev": te_core,
        "te_ring_ev": te_ring,
        "te_ratio": te_core / te_ring if (np.isfinite(te_ring) and te_ring > 0) else np.nan,
        "vp_core_v": _med(vp, core),
        "vp_ring_v": _med(vp, ring),
        "vf_core_v": _med(vf, core),
        "vf_ring_v": _med(vf, ring),
        "lambda_core": _med(lam, core),
        "lambda_wall": _med(lam, wall),
        "pe_wall_pa": _med(grads.pe_pa, wall),
        "dln_ne_wall_cm": _med(grads.dln_ne_dx, wall),
        "dln_te_wall_cm": _med(grads.dln_te_dx, wall),
        "dln_pe_wall_cm": _med(grads.dln_pe_dx, wall),
        "dln_ne_steep_cm": dln_ne_at_steep,
        "dln_te_steep_cm": dln_te_at_steep,
        "dln_pe_steep_cm": dln_pe_at_steep,
        "eta_e_steep": eta_e_at_steep,
        "er_wall_v_cm": _peak_signed(grads.er_v_per_cm, wall),
        "v_ExB_wall_m_s": _peak_signed(scales["v_ExB_m_s"], wall),
        "rho_s_wall_cm": _med(scales["rho_s_cm"], wall),
        "nu_ei_wall_s": _med(scales["nu_ei_s"], wall),
        "beta_e_wall": _med(scales["beta_e"], wall),
        "f_star_n_wall_hz": _med(scales["omega_star_n_rad_s"], wall) / (2.0 * np.pi),
        "f_star_T_wall_hz": _med(scales["omega_star_T_rad_s"], wall) / (2.0 * np.pi),
        "f_star_p_wall_hz": _med(scales["omega_star_p_rad_s"], wall) / (2.0 * np.pi),
        "primary_flag_frac": float(np.nanmean(lam[core | wall] > PRIMARY_LAMBDA_THRESH))
        if np.any(np.isfinite(lam[core | wall])) else np.nan,
    }


def analyze_run(spec, data_dir=DATA_DIR, onset_table=None, *,
                x_center_cm=X_CENTER_CM, smooth_window=SMOOTH_WINDOW,
                smooth_poly=SMOOTH_POLY, vf_smooth_samples=VF_SMOOTH_SAMPLES):
    """Analyze one IV line run: pre-onset classification, Vf/Lambda, gradients & scales."""
    Vp, Te, ne, Vp_err, Te_err, ne_err, t_ls = load_plasma_data(data_dir, spec.run_num, spec.tip)
    Vswp, Iswp, _, xpos, _, _, _ = load_sweep_data(data_dir, spec.run_num, spec.tip)
    calibrated = ne_is_calibrated(data_dir, spec.run_num, spec.tip)

    sweep_path, _ = sweep_npz_paths(data_dir, spec.run_num)
    with np.load(sweep_path) as sw:
        t_start_ms = np.asarray(sw[f"{spec.tip}/sweep_t_start"], dtype=float) * 1e3
        t_stop_ms = np.asarray(sw[f"{spec.tip}/sweep_t_stop"], dtype=float) * 1e3
    t_mid_ms = np.asarray(t_ls, dtype=float) * 1e3
    x_cm = np.asarray(xpos, dtype=float)

    onset_row = onset_table.get(spec.run_num, {}) if onset_table else {}
    t_onset_q25_ms = float(onset_row.get("t_onset_q25_ms", np.nan))
    t_onset_med_ms = float(onset_row.get("t_onset_med_ms", np.nan))
    regimes = classify_sweeps(t_start_ms, t_stop_ms, t_onset_q25_ms, t_onset_med_ms)
    ref_idx, ref_regime = select_reference_pre_onset_sweep(t_mid_ms, regimes)

    Vf, Vf_err = extract_vf_profile(Vswp, Iswp, smooth_samples=vf_smooth_samples)
    lam, lam_err = sheath_factor(Vp, Vf, Te, Vp_err, Vf_err, Te_err)

    grads_raw = radial_profile_gradients(
        x_cm, ne, Te, Vp, ne_err=ne_err, te_err=Te_err, vp_err=Vp_err,
        x_center_cm=x_center_cm, smooth_window=1, axis=0,
    )
    grads_sm = radial_profile_gradients(
        x_cm, ne, Te, Vp, ne_err=ne_err, te_err=Te_err, vp_err=Vp_err,
        x_center_cm=x_center_cm, smooth_window=smooth_window, polyorder=smooth_poly, axis=0,
    )

    # Effective azimuthal wavenumber k_theta = m / r on the inner wall (clamp r >= 1 cm).
    r_eff_cm = np.maximum(np.abs(x_cm - x_center_cm), 1.0)[:, None]
    k_perp_cm = float(MODE_M) / r_eff_cm
    scales = local_drift_scales(
        ne, Te, spec.b0_g, mu=ION_MU, Z=1,
        k_perp_cm=k_perp_cm,
        dln_ne_dr_cm=grads_sm.dln_ne_dx,
        dln_te_dr_cm=grads_sm.dln_te_dx,
        er_v_per_cm=grads_sm.er_v_per_cm,
    )

    out = {
        "tip": spec.tip,
        "gas_puff_ms": float(spec.gas_puff_ms),
        "b0_g": float(spec.b0_g),
        "calibrated": bool(calibrated),
        "x_cm": x_cm,
        "t_mid_ms": t_mid_ms,
        "t_start_ms": t_start_ms,
        "t_stop_ms": t_stop_ms,
        "regimes": np.array(regimes, dtype="U12"),
        "t_onset_q25_ms": t_onset_q25_ms,
        "t_onset_med_ms": t_onset_med_ms,
        "ref_sweep_idx": -1 if ref_idx is None else int(ref_idx),
        "ref_sweep_t_ms": float(t_mid_ms[ref_idx]) if ref_idx is not None else np.nan,
        "ref_sweep_regime": ref_regime or "",
        "Vp": Vp, "Vp_err": Vp_err,
        "Te": Te, "Te_err": Te_err,
        "ne": ne, "ne_err": ne_err,
        "Vf": Vf, "Vf_err": Vf_err,
        "lambda": lam, "lambda_err": lam_err,
        "pe_pa": grads_sm.pe_pa, "pe_err_pa": grads_sm.pe_err_pa,
        "dln_ne_dr_raw": grads_raw.dln_ne_dx,
        "dln_te_dr_raw": grads_raw.dln_te_dx,
        "dln_pe_dr_raw": grads_raw.dln_pe_dx,
        "dln_ne_dr": grads_sm.dln_ne_dx, "dln_ne_dr_err": grads_sm.dln_ne_err,
        "dln_te_dr": grads_sm.dln_te_dx, "dln_te_dr_err": grads_sm.dln_te_err,
        "dln_pe_dr": grads_sm.dln_pe_dx, "dln_pe_dr_err": grads_sm.dln_pe_err,
        "eta_e": grads_sm.eta_e,
        "er_r_v_cm": grads_sm.er_v_per_cm, "er_r_err_v_cm": grads_sm.er_err_v_per_cm,
        "rho_s_cm": scales["rho_s_cm"],
        "nu_ei_s": scales["nu_ei_s"],
        "beta_e": scales["beta_e"],
        "v_ExB_m_s": scales["v_ExB_m_s"],
        "omega_star_n_rad_s": scales["omega_star_n_rad_s"],
        "omega_star_T_rad_s": scales["omega_star_T_rad_s"],
        "omega_star_p_rad_s": scales["omega_star_p_rad_s"],
    }

    if ref_idx is not None:
        # Slice smoothed gradients and scales at the reference pre-onset sweep.
        g_ref = radial_profile_gradients(
            x_cm, ne[:, ref_idx], Te[:, ref_idx], Vp[:, ref_idx],
            ne_err=ne_err[:, ref_idx], te_err=Te_err[:, ref_idx], vp_err=Vp_err[:, ref_idx],
            x_center_cm=x_center_cm, smooth_window=smooth_window, polyorder=smooth_poly, axis=0,
        )
        s_ref = {k: v[:, ref_idx] for k, v in scales.items()}
        summary = _inner_wall_summary(
            x_cm, ne[:, ref_idx], Te[:, ref_idx], Vp[:, ref_idx],
            Vf[:, ref_idx], lam[:, ref_idx], g_ref, s_ref, x_center_cm,
        )
        for sk, sv in summary.items():
            out[f"ref_{sk}"] = sv

    return out


def _run_settings_dict(x_center_cm, smooth_window, smooth_poly, vf_smooth_samples):
    return {
        "x_center_cm": float(x_center_cm),
        "smooth_window": int(smooth_window),
        "smooth_poly": int(smooth_poly),
        "vf_smooth_samples": int(vf_smooth_samples),
    }


def _run_settings_mismatch(npz_data, run_num, want):
    diffs = []
    for field, val in want.items():
        k = f"{run_num}__{field}"
        if k not in npz_data.files:
            diffs.append(f"{field}: missing in npz (requested {val:g})")
            continue
        if isinstance(val, int):
            got = int(npz_data[k])
            if got != val:
                diffs.append(f"{field}: stored {got} != requested {val}")
        else:
            got = float(npz_data[k])
            if not np.isclose(got, val, atol=1e-9, rtol=0.0):
                diffs.append(f"{field}: stored {got:g} != requested {val:g}")
    return diffs


def batch_all(data_dir=DATA_DIR, runs=None, onset_npz=None, out_path=None,
              *, x_center_cm=X_CENTER_CM, smooth_window=SMOOTH_WINDOW,
              smooth_poly=SMOOTH_POLY, vf_smooth_samples=VF_SMOOTH_SAMPLES,
              overwrite=False):
    """Compute pre-onset profile, pressure, gradient & Vf summary -> npz path."""
    specs = (PROFILE_RUNS if runs is None
             else tuple(PROFILE_RUN_MAP[r] for r in runs))
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    onset_path = onset_npz or os.path.join(data_dir, jon.OUT_NAME)
    onset_table = jon.load_onset_table(onset_path) if os.path.isfile(onset_path) else {}
    want = _run_settings_dict(x_center_cm, smooth_window, smooth_poly, vf_smooth_samples)

    if overwrite and runs is None and os.path.isfile(out_path):
        os.remove(out_path)

    existing_runs = set()
    drop_keys = []
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            existing_runs = {
                k.split("__", 1)[0] for k in d.files if k.endswith("__ref_sweep_idx")
            }
            if overwrite:
                prefixes = tuple(f"{s.run_num}__" for s in specs)
                drop_keys = [k for k in d.files if k.startswith(prefixes)]
                existing_runs -= {s.run_num for s in specs}
            else:
                for spec in specs:
                    if spec.run_num in existing_runs:
                        diffs = _run_settings_mismatch(d, spec.run_num, want)
                        if diffs:
                            raise ValueError(
                                f"{out_path}: stored run '{spec.run_num}' differs "
                                f"from requested settings ({'; '.join(diffs)}); "
                                "pass overwrite=True to regenerate")

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    ran_any = False
    for spec in tqdm(specs, desc="Profiles batch", unit="run"):
        if spec.run_num in existing_runs:
            continue
        res = analyze_run(
            spec, data_dir=data_dir, onset_table=onset_table,
            x_center_cm=x_center_cm, smooth_window=smooth_window,
            smooth_poly=smooth_poly, vf_smooth_samples=vf_smooth_samples,
        )
        p = spec.run_num
        for k, v in res.items():
            if isinstance(v, str):
                payload[f"{p}__{k}"] = np.str_(v)
            elif isinstance(v, (bool, np.bool_)):
                payload[f"{p}__{k}"] = np.bool_(v)
            elif isinstance(v, (int, np.integer)):
                payload[f"{p}__{k}"] = np.int64(v)
            elif isinstance(v, (float, np.floating)):
                payload[f"{p}__{k}"] = np.float64(v)
            else:
                payload[f"{p}__{k}"] = np.asarray(v)
        for sk, sv in want.items():
            payload[f"{p}__{sk}"] = np.int64(sv) if isinstance(sv, int) else np.float64(sv)
        ran_any = True

    if ran_any or not os.path.isfile(out_path):
        stale_keys = [k for k in drop_keys if k not in payload]
        merge_save_npz(out_path, payload, drop_keys=stale_keys)
        print(f"\nWrote {out_path}")
    return out_path


def print_pre_onset_summary(npz_path=None):
    """Print pre-onset profile, signed-gradient, Vf/Lambda, and scale comparison."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path) as d:
        order = [s.run_num for s in PROFILE_RUNS if f"{s.run_num}__ref_sweep_idx" in d.files]
        print("\n=== Pre-Onset Inner-Wall Gradients & Thermodynamic Thresholds (P29-R, 800 G) ===")
        print(
            f"{'Run':>3} {'Puff':>5} {'Swp[ms]':>8} {'Regime':>10} {'t_on':>6} "
            f"{'Te_c/r':>11} {'n_r/n_c':>7} {'dln_n':>7} {'dln_T':>7} {'dln_p':>7} "
            f"{'eta_e':>6} {'pe[Pa]':>7}"
        )
        for r in order:
            t_on = float(d[f"{r}__t_onset_med_ms"])
            ton_s = f"{t_on:5.2f}" if np.isfinite(t_on) else " stbl"
            print(
                f"{r:>3} {float(d[f'{r}__gas_puff_ms']):4.0f}m "
                f"{float(d[f'{r}__ref_sweep_t_ms']):8.2f} {str(d[f'{r}__ref_sweep_regime']):>10} {ton_s:>6} "
                f"{float(d[f'{r}__ref_te_core_ev']):4.2f}/{float(d[f'{r}__ref_te_ring_ev']):4.2f}eV "
                f"{float(d[f'{r}__ref_hollow_ratio']):6.2f}x "
                f"{float(d[f'{r}__ref_dln_ne_steep_cm']):+7.2f} "
                f"{float(d[f'{r}__ref_dln_te_steep_cm']):+7.2f} "
                f"{float(d[f'{r}__ref_dln_pe_steep_cm']):+7.2f} "
                f"{float(d[f'{r}__ref_eta_e_steep']):+6.2f} "
                f"{float(d[f'{r}__ref_pe_wall_pa']):7.3f}"
            )

        print("\n=== Potential Validation (Vp vs Vf, Sheath Factor Lambda) & Local Scales ===")
        print(
            f"{'Run':>3} {'Puff':>5} {'Vp_c/r [V]':>12} {'Vf_c/r [V]':>12} "
            f"{'Lam_c/w':>9} {'Er[V/cm]':>9} {'vExB[km/s]':>10} "
            f"{'rhos[cm]':>8} {'nuei[MHz]':>9} {'f*p[kHz]':>8}"
        )
        for r in order:
            print(
                f"{r:>3} {float(d[f'{r}__gas_puff_ms']):4.0f}m "
                f"{float(d[f'{r}__ref_vp_core_v']):5.1f}/{float(d[f'{r}__ref_vp_ring_v']):5.1f} "
                f"{float(d[f'{r}__ref_vf_core_v']):5.1f}/{float(d[f'{r}__ref_vf_ring_v']):5.1f} "
                f"{float(d[f'{r}__ref_lambda_core']):4.2f}/{float(d[f'{r}__ref_lambda_wall']):4.2f} "
                f"{float(d[f'{r}__ref_er_wall_v_cm']):+9.2f} "
                f"{float(d[f'{r}__ref_v_ExB_wall_m_s'])*1e-3:+10.2f} "
                f"{float(d[f'{r}__ref_rho_s_wall_cm']):8.2f} "
                f"{float(d[f'{r}__ref_nu_ei_wall_s'])*1e-6:9.1f} "
                f"{float(d[f'{r}__ref_f_star_p_wall_hz'])*1e-3:+8.2f}"
            )


if __name__ == "__main__":
    out = batch_all()
    print_pre_onset_summary(out)
