"""Jun-2026 instability onset time, linear growth rate, and fine frequency.

Extracts per-shot and ensemble instability dynamics from each run's stationary
ion-saturation-current (Isat) reference channel across:

* Day-1 gas-puff scan (Runs 06, 00-05; 23-45 ms puff at 800 G; stationary
  ``machscope/C2`` at P33),
* Day-2 gas-puff scan (Runs 08-14; 23-45 ms puff at 800 G; stationary
  ``lpscope/C3`` at P29-R),
* Magnetic-field & 2D mode scans (Runs 26-31; 600 / 800 / 1000 G; stationary
  ``lpscope/C3`` or ``scope/C5``).

For every run this computes:

1. Sliding Hann-weighted linearly detrended relative RMS envelope
   ``dI/I_rms(t)`` on ``[0.0, 5.0] ms`` (downsampled to 100 kHz, 0.50 ms Hann
   window = 0.25 ms FWHM, 0.05 ms step -- suppresses 2f phase ripple).
2. Per-shot instability onset time ``t_onset`` (first sustained crossing of
   ``ONSET_THRESH = 1.5%`` after ``t >= 0.4 ms``) and unstable shot fraction.
3. Exponential growth rate ``gamma`` [s^-1]:
   - per-shot log-linear fit on each unstable shot's rising ramp, AND
   - onset-aligned ensemble fit on ``median_shot(env(t - t_onset))`` (removes
     shot-to-shot onset-time jitter without smearing the ramp).
4. High-resolution post-onset dominant frequency ``f_peak_hz``, spectral FWHM,
   Rayleigh resolution ``f_rayleigh_hz = 1 / T_win``, and peak-to-median SNR
   via zero-padded full-window Hann FFT (``1/T_win ~ 300-770 Hz`` depending on
   post-onset window length ``1.3-3.3 ms``, parabolically refined around the
   peak; note ``f_fwhm_hz`` has a Hann window floor of ``~1.44 / T_win``).

Writes ``onset_summary_jun2026.npz`` into ``DATA_DIR`` so downstream profile
and dispersion modules (e.g. pre- vs. post-onset IV sweep selection) can load
the exact onset timing without re-reading raw HDF5 files.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_onset.py
"""

import datetime
import glob
import os
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from data_analysis.io import open_lapd, parse_gas_puff
from data_analysis.signal import (
    clip_time_window,
    detect_onset_time,
    dominant_frequency,
    fit_exponential_growth,
    sliding_detrended_rms,
)
from data_analysis.utils import (
    find_run_hdf5 as run_ifn,
    merge_save_npz,
    pack_npz_entries,
    prepare_incremental_npz,
)

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = r"D:\data\LAPD\jun2026-jia"
OUT_NAME = "onset_summary_jun2026.npz"

TMIN_MS, TMAX_MS = 0.0, 5.0          # bias-on window [ms]
TARGET_DT_S = 1e-5                   # 100 kHz block-mean working rate (dt = 10 us)
WIN_S = 0.50e-3                      # 0.50 ms Hann window (FWHM = 0.25 ms, ~2 periods)
STEP_S = 0.05e-3                     # 0.05 ms envelope step
ONSET_THRESH = 0.015                 # 1.5% relative RMS onset threshold
T_MIN_ONSET_MS = 0.4                 # skip t=0 bias turn-on transient [ms]
SUSTAIN_BINS = 2                     # 2 consecutive bins (= 0.10 ms) sustained
FBAND_HZ = (1500.0, 12000.0)         # fundamental mode search band [Hz]
FREQ_PAD_FACTOR = 4                  # zero-padding factor for peak interpolation
TAU_ALIGN_MS = np.arange(-0.8, 1.55, 0.05)  # onset-aligned lag grid [ms]
CHUNK_SHOTS = 60                     # shots read per HDF5 chunk (caps peak RAM)
DEFAULT_MAX_SHOTS = 120              # evenly spaced cap per run (None = all shots)


class OnsetRunSpec(NamedTuple):
    """Stationary reference channel specification for one campaign run."""
    run_num: str
    group: str
    scope: str
    chan: str
    b0_g: float
    probe_label: str
    gas_puff_ms: float | None = None  # None -> parse from HDF5 run description


# Verified against raw HDF5 attributes (Runs 00-06, 08-14, 26-31).
# Note: Run 31 filename says 800G, but its HDF5 run description is uniform 600G.
ONSET_RUNS = (
    # Day-1 gas-puff scan (IV lines at P29; stationary MP@P33 Isat-X- ref)
    OnsetRunSpec("06", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 23.0),
    OnsetRunSpec("00", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 24.0),
    OnsetRunSpec("01", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 25.0),
    OnsetRunSpec("02", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 30.0),
    OnsetRunSpec("03", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 35.0),
    OnsetRunSpec("04", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 40.0),
    OnsetRunSpec("05", "puff_day1", "machscope", "C2", 800.0, "P33 MP-X-", 45.0),
    # Day-2 gas-puff scan (Mach lines at P33; stationary LP@P29-R ref)
    OnsetRunSpec("08", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 23.0),
    OnsetRunSpec("09", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 24.0),
    OnsetRunSpec("10", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 25.0),
    OnsetRunSpec("11", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 30.0),
    OnsetRunSpec("12", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 35.0),
    OnsetRunSpec("13", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 40.0),
    OnsetRunSpec("14", "puff_day2", "lpscope", "C3", 800.0, "P29-R LP", 45.0),
    # Magnetic-field & 2D mode scans (stationary P25-L or P33-X+ ref)
    OnsetRunSpec("28", "b0_scan", "scope", "C5", 600.0, "P33 MP-X+"),
    OnsetRunSpec("31", "b0_scan", "lpscope", "C3", 600.0, "P25-L LP"),
    OnsetRunSpec("26", "b0_scan", "lpscope", "C3", 800.0, "P25-L LP"),
    OnsetRunSpec("27", "b0_scan", "scope", "C5", 800.0, "P33 MP-X+"),
    OnsetRunSpec("29", "b0_scan", "scope", "C5", 800.0, "P33 MP-X+"),
    OnsetRunSpec("30", "b0_scan", "scope", "C5", 1000.0, "P33 MP-X+"),
)
ONSET_RUN_MAP = {spec.run_num: spec for spec in ONSET_RUNS}


def _select_shots(all_shots, max_shots):
    """Evenly spaced deterministic shot subset (or all shots when ``max_shots`` is None)."""
    shots = np.asarray(all_shots, dtype=int)
    if max_shots is None or shots.size <= max_shots:
        return shots
    idx = np.linspace(0, shots.size - 1, int(max_shots), dtype=int)
    return shots[np.unique(idx)]


def _read_downsampled_window(ifn, scope, chan, tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
                             target_dt_s=TARGET_DT_S, max_shots=DEFAULT_MAX_SHOTS):
    """Read clipped, block-mean-downsampled shots -> ``(t_ds_s, stack_ds, shot_nums, puff_ms)``."""
    run = open_lapd(ifn)
    tarr = run.time_array(scope_name=scope)
    i0, i1 = clip_time_window(tarr, tmin_ms, tmax_ms)
    t_clip = tarr[i0:i1]
    dt_raw = float(t_clip[1] - t_clip[0])
    q = max(1, int(round(target_dt_s / dt_raw)))
    nblk = t_clip.size // q
    if nblk < 10:
        raise ValueError(f"{ifn}: clipped window too short after block-mean (q={q}, nblk={nblk})")
    t_ds = t_clip[:nblk * q].reshape(nblk, q).mean(axis=1)

    all_shots = run.shots(scope_name=scope)
    selected = _select_shots(all_shots, max_shots)

    chunks = []
    kept_shots = []
    for start in range(0, selected.size, CHUNK_SHOTS):
        sub = selected[start:start + CHUNK_SHOTS]
        stack, _ = run.channel(chan, scope_name=scope, shots=sub.tolist())
        if stack is None or stack.size == 0:
            continue
        clipped = np.asarray(stack[:, i0:i0 + nblk * q], dtype=float)
        ds = clipped.reshape(clipped.shape[0], nblk, q).mean(axis=2)
        chunks.append(ds)
        kept_shots.append(sub[:ds.shape[0]])

    if not chunks:
        raise ValueError(f"{ifn}: could not read any shots for {scope}/{chan}")

    puff = parse_gas_puff(run.description().raw)
    puff_ms = float(puff[1]) if puff is not None else np.nan
    return t_ds, np.vstack(chunks), np.concatenate(kept_shots), puff_ms


def analyze_envelope_stack(t_s, stack, *, win_s=WIN_S, step_s=STEP_S,
                           onset_thresh=ONSET_THRESH, t_min_onset_ms=T_MIN_ONSET_MS,
                           sustain=SUSTAIN_BINS, fband_hz=FBAND_HZ,
                           freq_pad_factor=FREQ_PAD_FACTOR,
                           tau_align_ms=TAU_ALIGN_MS):
    """Pure-array onset, growth-rate, and dominant-frequency analysis on a shot stack.

    Separated from HDF5 I/O so unit tests can exercise the full campaign
    reduction directly on synthetic waveforms.
    """
    t_s = np.asarray(t_s, dtype=float)
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 2 or stack.shape[1] != t_s.size:
        raise ValueError("stack must be 2-D (nshot, nt) matching t_s")

    dt_s = float(t_s[1] - t_s[0])
    t_env_s, env = sliding_detrended_rms(stack, t_s, win_s=win_s, step_s=step_s, normalize=True)
    t_env_ms = t_env_s * 1e3

    t_onset_s = detect_onset_time(
        t_env_s, env, threshold=onset_thresh, t_min=t_min_onset_ms * 1e-3, sustain=sustain
    )
    t_onset_ms = t_onset_s * 1e3

    # Quiet baseline noise floor per shot: strictly pre-onset so early onsets
    # (e.g. inside 0.4-0.8 ms) cannot contaminate their own noise floor.
    nshot = stack.shape[0]
    post_mask = t_env_ms >= t_min_onset_ms
    noise_floor = np.full(nshot, np.nan, dtype=float)
    early_default = (t_env_ms >= t_min_onset_ms) & (t_env_ms <= t_min_onset_ms + 0.4)
    for k in range(nshot):
        if np.isfinite(t_onset_ms[k]):
            # Prefer a margin (>= 0.15 ms) before onset so the rising foot of the
            # Hann window does not lift the baseline estimate.
            pre_hi = min(t_min_onset_ms + 0.4, t_onset_ms[k] - 0.15)
            pre = (t_env_ms >= t_min_onset_ms) & (t_env_ms <= pre_hi)
            if np.sum(pre) >= 3 and np.any(np.isfinite(env[k, pre])):
                noise_floor[k] = float(np.nanmedian(env[k, pre]))
            else:
                strict_pre = (t_env_ms >= t_min_onset_ms) & (t_env_ms < t_onset_ms[k])
                if np.any(strict_pre) and np.any(np.isfinite(env[k, strict_pre])):
                    noise_floor[k] = float(np.nanmin(env[k, strict_pre]))
                elif np.any(post_mask) and np.any(np.isfinite(env[k, post_mask])):
                    noise_floor[k] = float(np.nanmin(env[k, post_mask]))
        elif np.any(early_default) and np.any(np.isfinite(env[k, early_default])):
            noise_floor[k] = float(np.nanmedian(env[k, early_default]))
        elif np.any(np.isfinite(env[k])):
            noise_floor[k] = float(np.nanpercentile(env[k], 10))

    # Peak relative RMS after t_min_onset_ms (excludes t=0 switching transient).
    peak_rel_rms = np.nanmax(env[:, post_mask], axis=1) if np.any(post_mask) else np.nanmax(env, axis=1)

    gamma_s = np.full(nshot, np.nan, dtype=float)
    gamma_err_s = np.full(nshot, np.nan, dtype=float)
    gamma_r2 = np.full(nshot, np.nan, dtype=float)
    for k in range(nshot):
        if not np.isfinite(t_onset_s[k]):
            continue
        fit = fit_exponential_growth(
            t_env_s, env[k], t_onset=t_onset_s[k], noise_floor=noise_floor[k]
        )
        gamma_s[k] = fit.gamma
        gamma_err_s[k] = fit.gamma_err
        gamma_r2[k] = fit.r2

    # Onset-aligned ensemble envelope (removes shot-to-shot onset jitter).
    tau_ms = np.asarray(tau_align_ms, dtype=float)
    unstable = np.isfinite(t_onset_ms)
    n_unstable = int(np.sum(unstable))
    if n_unstable > 0:
        aligned = np.full((n_unstable, tau_ms.size), np.nan, dtype=float)
        for j, k in enumerate(np.flatnonzero(unstable)):
            q_ms = t_onset_ms[k] + tau_ms
            in_range = (q_ms >= t_env_ms[0]) & (q_ms <= t_env_ms[-1])
            if np.any(in_range):
                aligned[j, in_range] = np.interp(q_ms[in_range], t_env_ms, env[k])
        env_aligned_med = np.full(tau_ms.size, np.nan, dtype=float)
        valid_col = np.any(np.isfinite(aligned), axis=0)
        if np.any(valid_col):
            env_aligned_med[valid_col] = np.nanmedian(aligned[:, valid_col], axis=0)
        pre_tau = tau_ms <= -0.15
        aligned_floor = (float(np.nanmedian(env_aligned_med[pre_tau]))
                         if np.any(pre_tau) and np.any(np.isfinite(env_aligned_med[pre_tau]))
                         else float(np.nanmedian(noise_floor)))
        aligned_fit = fit_exponential_growth(
            tau_ms * 1e-3, env_aligned_med, t_onset=-0.2e-3, noise_floor=aligned_floor
        )
    else:
        env_aligned_med = np.full(tau_ms.size, np.nan, dtype=float)
        aligned_fit = fit_exponential_growth(tau_ms * 1e-3, env_aligned_med)

    # High-resolution post-onset dominant frequency.
    t_onset_med_ms = float(np.nanmedian(t_onset_ms)) if n_unstable > 0 else np.nan
    f_tmin_ms = min(max(t_onset_med_ms, 1.5), 3.5) if np.isfinite(t_onset_med_ms) else 1.5
    f_tmax_ms = min(float(t_s[-1] * 1e3), 4.8)
    f0, f1 = clip_time_window(t_s, f_tmin_ms, f_tmax_ms)
    f_win_dur_s = float((f1 - f0) * dt_s)
    f_rayleigh_hz = 1.0 / f_win_dur_s if f_win_dur_s > 0.0 else np.nan
    freq_res = dominant_frequency(
        stack[:, f0:f1], dt_s, fband=fband_hz, pad_factor=freq_pad_factor
    )

    def _quantiles(arr):
        fin = arr[np.isfinite(arr)]
        if fin.size == 0:
            return np.nan, np.nan, np.nan
        q25, med, q75 = np.percentile(fin, [25.0, 50.0, 75.0])
        return float(med), float(q25), float(q75)

    t_med, t_q25, t_q75 = _quantiles(t_onset_ms)
    g_med, g_q25, g_q75 = _quantiles(gamma_s)
    pk_med, pk_q25, pk_q75 = _quantiles(peak_rel_rms)

    return {
        "n_shots": int(nshot),
        "n_unstable": n_unstable,
        "unstable_frac": float(n_unstable / nshot) if nshot > 0 else np.nan,
        "t_env_ms": t_env_ms,
        "env_med": np.nanmedian(env, axis=0),
        "env_q25": np.nanpercentile(env, 25.0, axis=0),
        "env_q75": np.nanpercentile(env, 75.0, axis=0),
        "t_onset_ms": t_onset_ms,
        "t_onset_med_ms": t_med,
        "t_onset_q25_ms": t_q25,
        "t_onset_q75_ms": t_q75,
        "peak_rel_rms": peak_rel_rms,
        "peak_rel_rms_med": pk_med,
        "peak_rel_rms_q25": pk_q25,
        "peak_rel_rms_q75": pk_q75,
        "gamma_s": gamma_s,
        "gamma_err_s": gamma_err_s,
        "gamma_r2": gamma_r2,
        "gamma_med_s": g_med,
        "gamma_q25_s": g_q25,
        "gamma_q75_s": g_q75,
        "tau_align_ms": tau_ms,
        "env_aligned_med": env_aligned_med,
        "gamma_aligned_s": float(aligned_fit.gamma),
        "gamma_aligned_err_s": float(aligned_fit.gamma_err),
        "gamma_aligned_r2": float(aligned_fit.r2),
        "f_win_tmin_ms": float(f_tmin_ms),
        "f_win_tmax_ms": float(f_tmax_ms),
        "f_rayleigh_hz": float(f_rayleigh_hz),
        "f_peak_hz": float(freq_res.f_peak_hz),
        "f_fwhm_hz": float(freq_res.f_fwhm_hz),
        "f_peak_snr": float(freq_res.peak_snr),
        "psd_freq_hz": freq_res.freq_hz,
        "psd_mean": freq_res.psd_mean,
    }


def analyze_run(spec, data_dir=DATA_DIR, max_shots=DEFAULT_MAX_SHOTS):
    """Read one campaign run and compute its onset, growth-rate, and frequency summary."""
    ifn = run_ifn(data_dir, spec.run_num)
    t_ds, stack_ds, shot_nums, parsed_puff = _read_downsampled_window(
        ifn, spec.scope, spec.chan, max_shots=max_shots
    )
    res = analyze_envelope_stack(t_ds, stack_ds)
    res["shot_nums"] = shot_nums
    res["source_file"] = os.path.basename(ifn)
    res["group"] = spec.group
    res["scope"] = spec.scope
    res["chan"] = spec.chan
    res["probe_label"] = spec.probe_label
    res["b0_g"] = float(spec.b0_g)
    res["gas_puff_ms"] = float(spec.gas_puff_ms) if spec.gas_puff_ms is not None else float(parsed_puff)
    return res


def _run_settings_dict(max_shots):
    """Per-run analysis settings persisted under ``<run>__*`` for staleness checks."""
    return {
        "onset_thresh": float(ONSET_THRESH),
        "win_ms": float(WIN_S * 1e3),
        "step_ms": float(STEP_S * 1e3),
        "t_min_onset_ms": float(T_MIN_ONSET_MS),
        "sustain_bins": int(SUSTAIN_BINS),
        "max_shots": -1 if max_shots is None else int(max_shots),
    }


def batch_all(data_dir=DATA_DIR, runs=None, max_shots=DEFAULT_MAX_SHOTS,
              out_path=None, overwrite=False):
    """Run onset/growth/frequency extraction across :data:`ONSET_RUNS` -> npz path.

    ``runs`` may be a subset of run numbers (e.g. ``["06", "00", "02"]``).
    Settings are stored **per run** (``<run>__onset_thresh``, ``<run>__win_ms``,
    etc.) so merging never mislabels earlier runs. When ``overwrite=False``,
    already-stored requested runs are validated against the current settings
    and any mismatch raises :class:`ValueError` up front. When
    ``overwrite=True``, passing ``runs=None`` rebuilds the whole file while
    passing a subset replaces only those runs' keys (preserving other runs).
    """
    specs = (ONSET_RUNS if runs is None
             else tuple(ONSET_RUN_MAP[r] for r in runs))
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    want = _run_settings_dict(max_shots)
    existing_runs, drop_keys = prepare_incremental_npz(
        out_path, [s.run_num for s in specs], "__t_onset_med_ms",
        overwrite=overwrite, full_overwrite=(runs is None), want=want,
    )
    # Drop legacy file-level setting keys if present; settings live per run.
    drop_keys.extend(["onset_thresh", "win_ms", "step_ms", "t_min_onset_ms", "sustain_bins"])

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    ran_any = False
    for spec in tqdm(specs, desc="Onset batch", unit="run"):
        if spec.run_num in existing_runs:
            continue
        res = analyze_run(spec, data_dir=data_dir, max_shots=max_shots)
        payload.update(pack_npz_entries(spec.run_num, res))
        payload.update(pack_npz_entries(spec.run_num, want))
        ran_any = True

    if ran_any or not os.path.isfile(out_path):
        stale_keys = [k for k in drop_keys if k not in payload]
        merge_save_npz(out_path, payload, drop_keys=stale_keys)
        print(f"\nWrote {out_path}")
    return out_path


def load_onset_table(npz_path=None):
    """Load per-run scalar onset/growth/frequency summary rows -> dict keyed by run_num."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    table = {}
    with np.load(npz_path) as d:
        run_nums = sorted({
            k.split("__", 1)[0] for k in d.files if k.endswith("__t_onset_med_ms")
        })
        for r in run_nums:
            table[r] = {
                "run_num": r,
                "group": str(d[f"{r}__group"]),
                "probe_label": str(d[f"{r}__probe_label"]),
                "gas_puff_ms": float(d[f"{r}__gas_puff_ms"]),
                "b0_g": float(d[f"{r}__b0_g"]),
                "n_shots": int(d[f"{r}__n_shots"]),
                "n_unstable": int(d[f"{r}__n_unstable"]),
                "unstable_frac": float(d[f"{r}__unstable_frac"]),
                "t_onset_med_ms": float(d[f"{r}__t_onset_med_ms"]),
                "t_onset_q25_ms": float(d[f"{r}__t_onset_q25_ms"]),
                "t_onset_q75_ms": float(d[f"{r}__t_onset_q75_ms"]),
                "peak_rel_rms_med": float(d[f"{r}__peak_rel_rms_med"]),
                "gamma_med_s": float(d[f"{r}__gamma_med_s"]),
                "gamma_q25_s": float(d[f"{r}__gamma_q25_s"]),
                "gamma_q75_s": float(d[f"{r}__gamma_q75_s"]),
                "gamma_aligned_s": float(d[f"{r}__gamma_aligned_s"]),
                "gamma_aligned_err_s": float(d[f"{r}__gamma_aligned_err_s"]),
                "gamma_aligned_r2": float(d[f"{r}__gamma_aligned_r2"]),
                "f_win_tmin_ms": float(d[f"{r}__f_win_tmin_ms"]),
                "f_win_tmax_ms": float(d[f"{r}__f_win_tmax_ms"]),
                "f_rayleigh_hz": (float(d[f"{r}__f_rayleigh_hz"])
                                  if f"{r}__f_rayleigh_hz" in d.files else np.nan),
                "f_peak_hz": float(d[f"{r}__f_peak_hz"]),
                "f_fwhm_hz": float(d[f"{r}__f_fwhm_hz"]),
                "f_peak_snr": float(d[f"{r}__f_peak_snr"]),
            }
    return table


def print_summary(npz_path=None):
    """Print a compact summary table grouped by campaign scan."""
    table = load_onset_table(npz_path)
    order = [s.run_num for s in ONSET_RUNS if s.run_num in table]
    extras = [r for r in sorted(table) if r not in order]
    last_group = None
    header = (
        f"{'Run':>3} {'Puff':>5} {'B0':>5} {'Ref':>9} {'Unst%':>6} "
        f"{'t_on [ms] (IQR)':>19} {'Peak%':>6} {'gamma_med':>10} {'gamma_aln (R2)':>17} "
        f"{'f0 [kHz]':>9} {'FWHM':>6} {'SNR':>6}"
    )
    for r in order + extras:
        row = table[r]
        if row["group"] != last_group:
            last_group = row["group"]
            print(f"\n=== {last_group} ===")
            print(header)
        t_str = (
            f"{row['t_onset_med_ms']:4.2f} ({row['t_onset_q25_ms']:4.2f}-{row['t_onset_q75_ms']:4.2f})"
            if np.isfinite(row["t_onset_med_ms"]) else "      ---          "
        )
        g_med = f"{row['gamma_med_s']*1e-3:6.2f} /ms" if np.isfinite(row["gamma_med_s"]) else "       ---"
        g_aln = (
            f"{row['gamma_aligned_s']*1e-3:5.2f} /ms ({row['gamma_aligned_r2']:.2f})"
            if np.isfinite(row["gamma_aligned_s"]) else "              ---"
        )
        f_str = f"{row['f_peak_hz']*1e-3:6.3f}" if np.isfinite(row["f_peak_hz"]) else "   ---"
        w_str = f"{row['f_fwhm_hz']*1e-3:5.2f}" if np.isfinite(row["f_fwhm_hz"]) else "  ---"
        snr_str = f"{row['f_peak_snr']:6.1f}" if np.isfinite(row["f_peak_snr"]) else "   ---"
        puff_str = f"{row['gas_puff_ms']:4.0f}m" if np.isfinite(row["gas_puff_ms"]) else "  ---"
        print(
            f"{r:>3} {puff_str:>5} {row['b0_g']:5.0f} {row['probe_label']:>9} "
            f"{row['unstable_frac']*100:5.1f}% {t_str:>19} "
            f"{row['peak_rel_rms_med']*100:5.2f}% {g_med:>10} {g_aln:>17} "
            f"{f_str:>9} {w_str:>6} {snr_str:>6}"
        )


if __name__ == "__main__":
    out = batch_all()
    print_summary(out)
