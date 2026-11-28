"""Jun-2026 nonlinear wave-wave coupling via auto- and cross-bicoherence.

Tests whether the observed harmonics (``2*f0``, ``3*f0``) and broadband
fluctuations are quadratically phase-coupled (``f1 + f2 -> f3``) or merely
independent/trigger-locked oscillations:

1. **Stationary Isat auto-bicoherence** across all 20 campaign runs
   (Day-1 puff 23-45 ms ``06, 00-05``; Day-2 puff ``08-14``; ``B0`` scan
   ``26-31``) on a fixed late-bias window ``[2.24, 4.80] ms``
   (``nperseg = 256`` at 100 kHz -> ``df = 390.625 Hz``, 1 independent segment
   per shot so the analytic ``bicoherence_threshold`` is exact). Keeping one
   fixed window preserves identical ``df`` and Hann main-lobe resolution across
   all runs; runs whose ``t_onset_med_ms`` is after ``2.24 ms`` (or ``nan`` for
   stable runs) include pre-onset/growth intervals and are explicitly flagged
   via ``window_post_onset`` and ``t_onset_med_ms``.
2. **Trigger-locking controls** on every run:
   - Shot-to-shot phase consistency ``R(f) = |<X(f)/|X(f)|>|`` at ``f0``,
     ``2*f0``, and ``3*f0`` vs. the Rayleigh 95% threshold ``sqrt(-ln(p)/N)``.
   - Shot-shuffled surrogate bicoherence ``b2_shuf`` (deterministic Sattolo
     derangement on ``Z``, breaking adjacent-shot drift). A low ``b2_shuf``
     alongside high ``b2`` rules out trigger-locking as the source of ``b2``;
     a high ``b2_shuf`` (when ``R`` is elevated) means trigger-locking
     confounds the test and cannot distinguish genuine coupling from
     trigger-locked coincidence.
3. **End-electrode current auto- and cross-bicoherence** (Runs ``06, 00-05``):
   - Auto-bicoherence of ``biasscope`` ``I_2"`` (``C1``) and ``I_4"`` (``C2``),
   - Same-scope cross-bicoherence ``(I_2", I_2", I_4")``, and
   - Cross-scope ``(Isat@P33, Isat@P33, I_2")`` and ``(I_2", I_2", Isat@P33)``
     on anti-aliased common shots (``+12.0 ms`` trigger offset).

Note on interpretation: auto-bicoherence alone cannot distinguish resonant
three-wave coupling from static waveform distortion (e.g. a non-sinusoidal
probe response or steepened advected profile). High cross-bicoherence between
P33 ``Isat`` and Port-35 shunt currents ``I_2"``/``I_4"`` (alongside the P28
B-dot ``m_J = -2`` 2nd-harmonic filament in ``Jun2026_modes.py``) rules out a
purely local ``Isat`` probe-response artifact.

Saves ``bicoherence_jun2026.npz`` into ``DATA_DIR`` and prints a concise
synthesis table.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_bicoherence.py
"""

import datetime
import os

import numpy as np
from scipy import signal as sps
from tqdm import tqdm

from data_analysis.io import open_lapd
from data_analysis.signal import (
    bicoherence,
    bicoherence_surrogate,
    clip_time_window,
    cross_bicoherence,
    shot_phase_consistency,
    summed_bicoherence,
)
from data_analysis.utils import merge_save_npz, npz_settings_mismatch

import Jun2026_modes as jmo
import Jun2026_onset as jon

# --- Campaign constants -----------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "bicoherence_jun2026.npz"

# Saturated post-onset window in bias-relative (machscope/lpscope) time [ms].
# T = 2.56 ms at dt = 10 us (100 kHz) -> nperseg = 256, df = 390.625 Hz.
# Adding a 0.02 ms margin guarantees >= 256 samples after block-mean clipping;
# taking 1 segment per shot (n_avg == n_shots) makes the null threshold exact.
TMIN_MS = 2.24
TMAX_MS = 4.82
TARGET_DT_S = 1e-5          # 100 kHz anti-aliased working grid
NPERSEG = 256               # 2.56 ms -> df = 390.625 Hz (Hann lobe = 781 Hz)
WINDOW = "hann"
DETREND = "linear"          # removes slow bias-envelope ramp across the 2.56 ms window
ALPHA_P = 0.05

F1_BAND_HZ = (1000.0, 16000.0)
F2_BAND_HZ = (-10000.0, 10000.0)
FUND_SEARCH_BAND_HZ = (2000.0, 6000.0)

DEFAULT_MAX_SHOTS = 180     # independent shots per run (p=0.05 threshold ~ 0.017)
SURROGATE_RNG_SEED = 0      # deterministic Sattolo derangement (breaks adjacent-shot drift)
BIASSCOPE_RUNS = ("06", "00", "01", "02", "03", "04", "05")
BIASSCOPE_OFFSET_MS = jmo.BIASSCOPE_OFFSET_MS
BIASSCOPE_MAX_SHOTS = 180


def _harmonic_triad_bins(f0_hz, df_hz):
    """Deterministic a-priori FFT bin indices for ``f0 + f0 -> 2*f0`` and ``2*f0 + f0 -> 3*f0``.

    Splitting ``k2_sum = round(2*f0/df)`` into ``(k2_sum // 2, k2_sum - k2_sum // 2)``
    lands squarely on the ``2*f0`` bin even when ``f0`` lies halfway between
    adjacent FFT bins (e.g. ``f0/df = 10.5 -> (10, 11) -> 21``), with zero
    look-elsewhere bias.
    """
    k1 = max(1, int(np.rint(float(f0_hz) / df_hz)))
    k2_sum = max(2, int(np.rint(2.0 * float(f0_hz) / df_hz)))
    k2_a = k2_sum // 2
    k2_b = k2_sum - k2_a
    # Ensure k2_a >= k2_b so the point sits in the f2 <= f1 principal domain
    if k2_b > k2_a:
        k2_a, k2_b = k2_b, k2_a
    k3_sum = max(3, int(np.rint(3.0 * float(f0_hz) / df_hz)))
    k3_a = k2_sum
    k3_b = k3_sum - k2_sum
    return k1, (k2_a, k2_b, k2_sum), (k3_a, k3_b, k3_sum)


def _lookup_triad(res, f1_hz, f2_hz, tol_hz):
    """Read ``(b2, biphase_deg)`` at ``(f1_hz, f2_hz)`` if present on grid within ``tol_hz``."""
    i = int(np.argmin(np.abs(res.f1 - f1_hz)))
    j = int(np.argmin(np.abs(res.f2 - f2_hz)))
    if abs(res.f1[i] - f1_hz) > tol_hz or abs(res.f2[j] - f2_hz) > tol_hz:
        return np.nan, np.nan
    b2_val = float(res.b2[i, j])
    if not np.isfinite(b2_val):
        return np.nan, np.nan
    biphase = float(np.degrees(np.angle(res.bispectrum[i, j])))
    return b2_val, biphase


def _lookup_1d(freq, vals, target_hz, tol_hz):
    i = int(np.argmin(np.abs(freq - target_hz)))
    if abs(freq[i] - target_hz) > tol_hz:
        return np.nan
    return float(vals[i])


def _mean_windowed_psd(stack, dt_s, nperseg, window=WINDOW, detrend=DETREND):
    """Average one-sided amplitude^2 spectrum on the exact ``bicoherence`` segment grid."""
    arr = np.asarray(stack, dtype=float)
    ok = np.all(np.isfinite(arr), axis=1)
    arr = arr[ok, :nperseg]
    if arr.shape[0] == 0:
        raise ValueError("no finite shots for PSD")
    if detrend:
        arr = sps.detrend(arr, axis=-1, type=detrend)
    win = sps.get_window(window, nperseg)
    scale = 2.0 / win.sum()
    F = np.fft.rfft(arr * win, axis=-1) * scale
    psd = np.mean(np.abs(F) ** 2, axis=0)
    freq = np.fft.rfftfreq(nperseg, d=dt_s)
    return freq, psd


def _peak_triad(res, res_shuf, *, symmetric=True, positive_only=False):
    """Strongest finite triad in the non-redundant ``f3 > 0`` domain."""
    f1g, f2g = np.meshgrid(res.f1, res.f2, indexing="ij")
    f3g = f1g + f2g
    mask = np.isfinite(res.b2) & (f3g > 0.0)
    if positive_only:
        mask &= f2g > 0.0
    if symmetric:
        mask &= f2g <= f1g + 1e-9
    if not np.any(mask):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    vals = np.where(mask, res.b2, -np.inf)
    idx = np.unravel_index(int(np.argmax(vals)), vals.shape)
    return (
        float(res.b2[idx]),
        float(res.f1[idx[0]]),
        float(res.f2[idx[1]]),
        float(f3g[idx]),
        float(res_shuf.b2[idx]),
    )


def analyze_bicoherence_stack(t_s, stack, *, f0_hz=None, nperseg=NPERSEG,
                              f1_band_hz=F1_BAND_HZ, f2_band_hz=F2_BAND_HZ,
                              window=WINDOW, detrend=DETREND, p=ALPHA_P,
                              surrogate_rng=SURROGATE_RNG_SEED):
    """Pure-array auto-bicoherence, phase-consistency, and surrogate analysis.

    Separated from HDF5 I/O so unit tests can exercise the complete campaign
    reduction directly on synthetic waveforms.
    """
    t_s = np.asarray(t_s, dtype=float)
    stack = np.asarray(stack, dtype=float)
    if stack.ndim != 2 or stack.shape[1] != t_s.size:
        raise ValueError("stack must be 2-D (nshot, nt) matching t_s")
    if t_s.size < nperseg:
        raise ValueError(f"record length {t_s.size} is shorter than nperseg={nperseg}")

    dt_s = float(t_s[1] - t_s[0])
    # Restrict to exactly 1 segment per shot so realizations are strictly independent.
    seg_stack = stack[:, :nperseg]
    df_hz = 1.0 / (nperseg * dt_s)
    tol_hz = 0.51 * df_hz

    psd_freq, psd = _mean_windowed_psd(seg_stack, dt_s, nperseg, window=window, detrend=detrend)
    if f0_hz is None or not np.isfinite(f0_hz):
        m = (psd_freq >= FUND_SEARCH_BAND_HZ[0]) & (psd_freq <= FUND_SEARCH_BAND_HZ[1])
        if not np.any(m):
            raise ValueError("no FFT bins inside FUND_SEARCH_BAND_HZ")
        sub_idx = np.flatnonzero(m)
        f0_hz = float(psd_freq[sub_idx[int(np.argmax(psd[sub_idx]))]])
    else:
        f0_hz = float(f0_hz)

    k1, (k2_a, k2_b, k2_sum), (k3_a, k3_b, k3_sum) = _harmonic_triad_bins(f0_hz, df_hz)

    res = bicoherence(
        seg_stack, dt_s, nperseg=nperseg, noverlap=0,
        f1_band=f1_band_hz, f2_band=f2_band_hz, window=window, detrend=detrend, p=p,
    )
    res_shuf = bicoherence_surrogate(
        seg_stack, seg_stack, seg_stack, dt_s, nperseg=nperseg, noverlap=0,
        f1_band=f1_band_hz, f2_band=f2_band_hz, window=window, detrend=detrend,
        p=p, rng=surrogate_rng,
    )
    pc = shot_phase_consistency(
        seg_stack, dt_s, nperseg=nperseg, noverlap=0,
        f_band=f1_band_hz, window=window, detrend=detrend, p=p,
    )

    f3_hz, b2_sum_mean, b2_sum_max, b2_sum_count = summed_bicoherence(
        res, symmetric=True, positive_only=True,
    )
    _, b2_shuf_sum_mean, b2_shuf_sum_max, _ = summed_bicoherence(
        res_shuf, symmetric=True, positive_only=True,
    )

    b2_h2, biphase_h2 = _lookup_triad(res, k2_a * df_hz, k2_b * df_hz, tol_hz)
    b2_h2_shuf, _ = _lookup_triad(res_shuf, k2_a * df_hz, k2_b * df_hz, tol_hz)
    b2_h3, biphase_h3 = _lookup_triad(res, k3_a * df_hz, k3_b * df_hz, tol_hz)
    b2_h3_shuf, _ = _lookup_triad(res_shuf, k3_a * df_hz, k3_b * df_hz, tol_hz)

    pk_b2, pk_f1, pk_f2, pk_f3, pk_shuf = _peak_triad(
        res, res_shuf, symmetric=True, positive_only=True,
    )

    # Harmonic amplitude ratios sqrt(P(n*f0)/P(f0)) on the same windowed FFT
    p1 = psd[k1] if k1 < psd.size else np.nan
    p2 = psd[k2_sum] if k2_sum < psd.size else np.nan
    p3 = psd[k3_sum] if k3_sum < psd.size else np.nan
    amp_h2_over_h1 = float(np.sqrt(p2 / p1)) if (np.isfinite(p1) and p1 > 0 and np.isfinite(p2)) else np.nan
    amp_h3_over_h1 = float(np.sqrt(p3 / p1)) if (np.isfinite(p1) and p1 > 0 and np.isfinite(p3)) else np.nan

    return {
        "dt_s": dt_s,
        "df_hz": df_hz,
        "nperseg": int(nperseg),
        "n_avg": int(res.n_avg),
        "threshold": float(res.threshold),
        "f0_hz": f0_hz,
        "f_h1_bin_hz": float(k1 * df_hz),
        "f_h2_triad_hz": np.array([k2_a * df_hz, k2_b * df_hz, k2_sum * df_hz], dtype=float),
        "f_h3_triad_hz": np.array([k3_a * df_hz, k3_b * df_hz, k3_sum * df_hz], dtype=float),
        "f1_hz": res.f1,
        "f2_hz": res.f2,
        "b2": res.b2,
        "biphase_deg": np.degrees(np.angle(res.bispectrum)),
        "b2_shuf": res_shuf.b2,
        "f3_hz": f3_hz,
        "b2_sum_mean": b2_sum_mean,
        "b2_sum_max": b2_sum_max,
        "b2_sum_count": b2_sum_count,
        "b2_shuf_sum_mean": b2_shuf_sum_mean,
        "b2_shuf_sum_max": b2_shuf_sum_max,
        "pc_freq_hz": pc.freq,
        "pc_R": pc.consistency,
        "pc_threshold": float(pc.threshold),
        "pc_R_h1": _lookup_1d(pc.freq, pc.consistency, k1 * df_hz, tol_hz),
        "pc_R_h2": _lookup_1d(pc.freq, pc.consistency, k2_sum * df_hz, tol_hz),
        "pc_R_h3": _lookup_1d(pc.freq, pc.consistency, k3_sum * df_hz, tol_hz),
        "b2_h2": b2_h2,
        "b2_h2_shuf": b2_h2_shuf,
        "biphase_h2_deg": biphase_h2,
        "b2_h3": b2_h3,
        "b2_h3_shuf": b2_h3_shuf,
        "biphase_h3_deg": biphase_h3,
        "b2_peak": pk_b2,
        "b2_peak_f1_hz": pk_f1,
        "b2_peak_f2_hz": pk_f2,
        "b2_peak_f3_hz": pk_f3,
        "b2_peak_shuf": pk_shuf,
        "psd_freq_hz": psd_freq,
        "psd": psd,
        "amp_h2_over_h1": amp_h2_over_h1,
        "amp_h3_over_h1": amp_h3_over_h1,
    }


def analyze_cross_bicoherence_stacks(t_s, stack_x, stack_y, stack_z, *, f0_hz,
                                     nperseg=NPERSEG, f1_band_hz=F1_BAND_HZ,
                                     f2_band_hz=F2_BAND_HZ, window=WINDOW,
                                     detrend=DETREND, p=ALPHA_P,
                                     surrogate_rng=SURROGATE_RNG_SEED):
    """Pure-array cross-bicoherence ``<X(f1) Y(f2) Z*(f1+f2)>`` + shot-shuffled surrogate."""
    t_s = np.asarray(t_s, dtype=float)
    dt_s = float(t_s[1] - t_s[0])
    df_hz = 1.0 / (nperseg * dt_s)
    tol_hz = 0.51 * df_hz
    _, (k2_a, k2_b, k2_sum), (k3_a, k3_b, k3_sum) = _harmonic_triad_bins(f0_hz, df_hz)

    sx = np.asarray(stack_x, dtype=float)[:, :nperseg]
    sy = np.asarray(stack_y, dtype=float)[:, :nperseg]
    sz = np.asarray(stack_z, dtype=float)[:, :nperseg]

    res = cross_bicoherence(
        sx, sy, sz, dt_s, nperseg=nperseg, noverlap=0,
        f1_band=f1_band_hz, f2_band=f2_band_hz, window=window, detrend=detrend, p=p,
    )
    res_shuf = bicoherence_surrogate(
        sx, sy, sz, dt_s, nperseg=nperseg, noverlap=0,
        f1_band=f1_band_hz, f2_band=f2_band_hz, window=window, detrend=detrend,
        p=p, rng=surrogate_rng,
    )
    b2_h2, biphase_h2 = _lookup_triad(res, k2_a * df_hz, k2_b * df_hz, tol_hz)
    b2_h2_shuf, _ = _lookup_triad(res_shuf, k2_a * df_hz, k2_b * df_hz, tol_hz)
    b2_h3, biphase_h3 = _lookup_triad(res, k3_a * df_hz, k3_b * df_hz, tol_hz)
    b2_h3_shuf, _ = _lookup_triad(res_shuf, k3_a * df_hz, k3_b * df_hz, tol_hz)
    pk_b2, pk_f1, pk_f2, pk_f3, pk_shuf = _peak_triad(res, res_shuf, symmetric=False)
    return {
        "n_avg": int(res.n_avg),
        "threshold": float(res.threshold),
        "b2_h2": b2_h2,
        "b2_h2_shuf": b2_h2_shuf,
        "biphase_h2_deg": biphase_h2,
        "b2_h3": b2_h3,
        "b2_h3_shuf": b2_h3_shuf,
        "biphase_h3_deg": biphase_h3,
        "b2_peak": pk_b2,
        "b2_peak_f1_hz": pk_f1,
        "b2_peak_f2_hz": pk_f2,
        "b2_peak_f3_hz": pk_f3,
        "b2_peak_shuf": pk_shuf,
    }


def _load_onset_info(data_dir):
    """Load per-run ``f_peak_hz`` and ``t_onset_med_ms`` from ``onset_summary_jun2026.npz``."""
    p = os.path.join(data_dir, jon.OUT_NAME)
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"missing {p}; run Jun2026_onset.py first so bicoherence uses the "
            "verified per-run fundamental frequency f0 and onset timing"
        )
    f0_map = {}
    tonset_map = {}
    with np.load(p, allow_pickle=False) as d:
        for spec in jon.ONSET_RUNS:
            fk = f"{spec.run_num}__f_peak_hz"
            if fk in d:
                f0_map[spec.run_num] = float(d[fk])
            tk = f"{spec.run_num}__t_onset_med_ms"
            if tk in d:
                tonset_map[spec.run_num] = float(d[tk])
    return p, f0_map, tonset_map


def analyze_run_auto(spec, data_dir=DATA_DIR, *, f0_hz=None, t_onset_med_ms=np.nan,
                     tmin_ms=TMIN_MS, tmax_ms=TMAX_MS, target_dt_s=TARGET_DT_S,
                     nperseg=NPERSEG, max_shots=DEFAULT_MAX_SHOTS, return_stack=False):
    """Run stationary-reference auto-bicoherence for one ``OnsetRunSpec``."""
    ifn = jon.run_ifn(data_dir, spec.run_num)
    t_ds, stack_ds, shot_nums, puff_parsed = jon._read_downsampled_window(
        ifn, spec.scope, spec.chan,
        tmin_ms=tmin_ms, tmax_ms=tmax_ms,
        target_dt_s=target_dt_s, max_shots=max_shots,
    )
    res = analyze_bicoherence_stack(t_ds, stack_ds, f0_hz=f0_hz, nperseg=nperseg)
    t_on = float(t_onset_med_ms) if t_onset_med_ms is not None else np.nan
    res["shot_nums"] = shot_nums
    res["source_file"] = os.path.basename(ifn)
    res["group"] = spec.group
    res["scope"] = spec.scope
    res["chan"] = spec.chan
    res["probe_label"] = spec.probe_label
    res["b0_g"] = float(spec.b0_g)
    res["gas_puff_ms"] = float(spec.gas_puff_ms) if spec.gas_puff_ms is not None else float(puff_parsed)
    res["tmin_ms"] = float(tmin_ms)
    res["tmax_ms"] = float(tmax_ms)
    res["t_onset_med_ms"] = t_on
    res["window_post_onset"] = bool(np.isfinite(t_on) and t_on <= float(tmin_ms))
    if return_stack:
        return res, (t_ds, stack_ds, shot_nums)
    return res


def _read_chunked_aligned(run, scope, chan, shots, i0, i1, t_src, t_grid):
    """Read ``shots`` from ``scope/chan`` in chunks and anti-alias resample onto ``t_grid``."""
    chunks = []
    for start in range(0, shots.size, jon.CHUNK_SHOTS):
        sub = shots[start:start + jon.CHUNK_SHOTS]
        raw, _ = run.channel(chan, scope_name=scope, shots=sub.tolist())
        if raw is None or raw.size == 0:
            raise ValueError(f"empty read for {scope}/{chan} on shots {sub[0]}..{sub[-1]}")
        chunks.append(jmo._blockmean_to_grid(t_src, raw[:, i0:i1], t_grid))
    return np.vstack(chunks)


def analyze_biasscope_bicoherence(data_dir=DATA_DIR, *, onset_f0=None,
                                  runs=BIASSCOPE_RUNS, tmin_ms=TMIN_MS,
                                  tmax_ms=TMAX_MS, target_dt_s=TARGET_DT_S,
                                  nperseg=NPERSEG, max_shots=BIASSCOPE_MAX_SHOTS,
                                  cached_isat=None):
    """Auto- and cross-bicoherence of ``I_2"``, ``I_4"``, and ``P33 Isat`` (Runs 06, 00-05).

    If ``cached_isat`` maps ``rnum -> (t_ds, stack_ds, shot_nums)`` from the
    auto-bicoherence pass, the already-downsampled ``machscope/C2`` traces are
    reused directly (avoiding a second 50 MHz HDF5 decompression).
    """
    t_grid = np.arange(nperseg, dtype=float) * target_dt_s + tmin_ms * 1e-3
    out = {}
    for rnum in tqdm(runs, desc="Biasscope bicoherence", unit="run"):
        ifn = jon.run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        s_b = np.asarray(run.shots(scope_name="biasscope"), dtype=int)
        t_b = run.time_array(scope_name="biasscope")
        pad_ms = 0.05
        ib0, ib1 = clip_time_window(
            t_b,
            tmin_ms + BIASSCOPE_OFFSET_MS - pad_ms,
            tmax_ms + BIASSCOPE_OFFSET_MS + pad_ms,
        )
        t_b_rel = t_b[ib0:ib1] - BIASSCOPE_OFFSET_MS * 1e-3

        if cached_isat is not None and rnum in cached_isat:
            t_ds, stack_ds, s_cached = cached_isat[rnum]
            common, idx_c, _ = np.intersect1d(s_cached, s_b, return_indices=True)
            shots = jon._select_shots(common, max_shots)
            if shots.size < 4:
                raise ValueError(f"run {rnum}: fewer than 4 common machscope/biasscope shots")
            keep = np.isin(s_cached, shots)
            # Ensure shot ordering matches sorted ``shots``
            order = np.argsort(s_cached[keep])
            sub_stack = stack_ds[keep][order]
            isat = np.vstack([np.interp(t_grid, t_ds, row) for row in sub_stack])
        else:
            s_m = np.asarray(run.shots(scope_name="machscope"), dtype=int)
            common = np.intersect1d(s_m, s_b)
            shots = jon._select_shots(common, max_shots)
            if shots.size < 4:
                raise ValueError(f"run {rnum}: fewer than 4 common machscope/biasscope shots")
            t_m = run.time_array(scope_name="machscope")
            im0, im1 = clip_time_window(t_m, tmin_ms - pad_ms, tmax_ms + pad_ms)
            t_m_sub = t_m[im0:im1]
            isat = _read_chunked_aligned(run, "machscope", "C2", shots, im0, im1, t_m_sub, t_grid)

        i2 = _read_chunked_aligned(run, "biasscope", "C1", shots, ib0, ib1, t_b_rel, t_grid)
        i4 = _read_chunked_aligned(run, "biasscope", "C2", shots, ib0, ib1, t_b_rel, t_grid)

        f0 = onset_f0.get(rnum) if onset_f0 else None
        auto_i2 = analyze_bicoherence_stack(t_grid, i2, f0_hz=f0, nperseg=nperseg)
        auto_i4 = analyze_bicoherence_stack(t_grid, i4, f0_hz=auto_i2["f0_hz"], nperseg=nperseg)
        xb_i2_i4 = analyze_cross_bicoherence_stacks(
            t_grid, i2, i2, i4, f0_hz=auto_i2["f0_hz"], nperseg=nperseg,
        )
        xb_isat_i2 = analyze_cross_bicoherence_stacks(
            t_grid, isat, isat, i2, f0_hz=auto_i2["f0_hz"], nperseg=nperseg,
        )
        xb_i2_isat = analyze_cross_bicoherence_stacks(
            t_grid, i2, i2, isat, f0_hz=auto_i2["f0_hz"], nperseg=nperseg,
        )

        out[rnum] = {
            "n_shots": int(shots.size),
            "f0_hz": float(auto_i2["f0_hz"]),
            "threshold": float(auto_i2["threshold"]),
            # I_2" auto
            "i2_b2_h2": float(auto_i2["b2_h2"]),
            "i2_b2_h2_shuf": float(auto_i2["b2_h2_shuf"]),
            "i2_biphase_h2_deg": float(auto_i2["biphase_h2_deg"]),
            "i2_b2_h3": float(auto_i2["b2_h3"]),
            "i2_b2_h3_shuf": float(auto_i2["b2_h3_shuf"]),
            "i2_amp_h2_over_h1": float(auto_i2["amp_h2_over_h1"]),
            "i2_pc_R_h1": float(auto_i2["pc_R_h1"]),
            "i2_pc_R_h2": float(auto_i2["pc_R_h2"]),
            "i2_pc_threshold": float(auto_i2["pc_threshold"]),
            # I_4" auto
            "i4_b2_h2": float(auto_i4["b2_h2"]),
            "i4_b2_h2_shuf": float(auto_i4["b2_h2_shuf"]),
            "i4_biphase_h2_deg": float(auto_i4["biphase_h2_deg"]),
            "i4_amp_h2_over_h1": float(auto_i4["amp_h2_over_h1"]),
            # Cross: (I_2", I_2", I_4")
            "x_i2_i2_i4_b2_h2": float(xb_i2_i4["b2_h2"]),
            "x_i2_i2_i4_b2_h2_shuf": float(xb_i2_i4["b2_h2_shuf"]),
            "x_i2_i2_i4_biphase_h2_deg": float(xb_i2_i4["biphase_h2_deg"]),
            # Cross: (Isat, Isat, I_2")
            "x_isat_isat_i2_b2_h2": float(xb_isat_i2["b2_h2"]),
            "x_isat_isat_i2_b2_h2_shuf": float(xb_isat_i2["b2_h2_shuf"]),
            "x_isat_isat_i2_biphase_h2_deg": float(xb_isat_i2["biphase_h2_deg"]),
            # Cross: (I_2", I_2", Isat)
            "x_i2_i2_isat_b2_h2": float(xb_i2_isat["b2_h2"]),
            "x_i2_i2_isat_b2_h2_shuf": float(xb_i2_isat["b2_h2_shuf"]),
            "x_i2_i2_isat_biphase_h2_deg": float(xb_i2_isat["biphase_h2_deg"]),
        }
    return out


_SETTINGS_META_KEYS = (
    "_meta__tmin_ms",
    "_meta__tmax_ms",
    "_meta__target_dt_s",
    "_meta__nperseg",
    "_meta__window",
    "_meta__detrend",
    "_meta__alpha_p",
    "_meta__max_shots",
    "_meta__surrogate_rng_seed",
)


def _validate_existing_settings(existing, expected, path):
    """Ensure incremental merges do not mix incompatible bicoherence settings."""
    want = {k: expected[k] for k in _SETTINGS_META_KEYS}
    mismatches = npz_settings_mismatch(existing, "", want)
    if mismatches:
        raise ValueError(
            f"existing {path} was built with different settings ({'; '.join(mismatches)}); "
            "pass overwrite=True with run_nums=None to regenerate"
        )


def batch_all(data_dir=DATA_DIR, out_name=OUT_NAME, *, run_nums=None,
              max_shots=DEFAULT_MAX_SHOTS, include_biasscope=True, overwrite=False):
    """Run Jun-2026 bicoherence analyses incrementally and save ``bicoherence_jun2026.npz``.

    - ``overwrite=False`` (default): validates stored settings, skips runs
      already present in ``out_path``, and computes/merges any missing runs.
    - ``overwrite=True``: recomputes the requested ``run_nums`` (or all runs
      when ``run_nums=None``) and replaces only those runs' keys in-place
      (preserving unselected runs on partial ``run_nums`` refreshes).
    """
    out_path = os.path.join(data_dir, out_name)
    onset_path, onset_f0, onset_t_ms = _load_onset_info(data_dir)
    specs = jon.ONSET_RUNS if run_nums is None else tuple(jon.ONSET_RUN_MAP[r] for r in run_nums)
    selected_runs = {spec.run_num for spec in specs}
    all_bias_runs = tuple(r for r in BIASSCOPE_RUNS if r in selected_runs)

    meta = {
        "_meta__created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "_meta__onset_file": os.path.basename(onset_path),
        "_meta__tmin_ms": float(TMIN_MS),
        "_meta__tmax_ms": float(TMAX_MS),
        "_meta__target_dt_s": float(TARGET_DT_S),
        "_meta__nperseg": int(NPERSEG),
        "_meta__window": WINDOW,
        "_meta__detrend": DETREND,
        "_meta__alpha_p": float(ALPHA_P),
        "_meta__max_shots": int(max_shots) if max_shots is not None else -1,
        "_meta__surrogate_rng_seed": int(SURROGATE_RNG_SEED),
    }

    existing_keys = set()
    full_overwrite = bool(overwrite and run_nums is None and include_biasscope)
    if os.path.isfile(out_path):
        with np.load(out_path, allow_pickle=False) as ex:
            existing_keys = set(ex.files)
            if not full_overwrite:
                _validate_existing_settings(ex, meta, out_path)

    if overwrite:
        auto_specs = list(specs)
        bias_runs = all_bias_runs if include_biasscope else ()
    else:
        auto_specs = [s for s in specs if f"auto_{s.run_num}__b2_h2" not in existing_keys]
        bias_runs = (
            tuple(r for r in all_bias_runs if f"bias_{r}__i2_b2_h2" not in existing_keys)
            if include_biasscope else ()
        )

    if not auto_specs and not bias_runs:
        print(f"All requested runs already present in {out_path} (pass overwrite=True to recompute)")
        return out_path

    payload = dict(meta)
    cached_isat = {}
    bias_set = set(bias_runs)
    for spec in tqdm(auto_specs, desc="Auto-bicoherence", unit="run"):
        need_cache = spec.run_num in bias_set
        if need_cache:
            res, stack_info = analyze_run_auto(
                spec, data_dir=data_dir, f0_hz=onset_f0.get(spec.run_num),
                t_onset_med_ms=onset_t_ms.get(spec.run_num, np.nan),
                max_shots=max_shots, return_stack=True,
            )
            cached_isat[spec.run_num] = stack_info
        else:
            res = analyze_run_auto(
                spec, data_dir=data_dir, f0_hz=onset_f0.get(spec.run_num),
                t_onset_med_ms=onset_t_ms.get(spec.run_num, np.nan),
                max_shots=max_shots,
            )
        for k, v in res.items():
            payload[f"auto_{spec.run_num}__{k}"] = v

    if bias_runs:
        bs = analyze_biasscope_bicoherence(
            data_dir=data_dir, onset_f0=onset_f0, runs=bias_runs,
            max_shots=max_shots, cached_isat=cached_isat,
        )
        for rnum, d in bs.items():
            for k, v in d.items():
                payload[f"bias_{rnum}__{k}"] = v

    drop_keys = []
    if full_overwrite:
        drop_keys = [k for k in existing_keys if k not in payload]
    elif overwrite:
        overwritten_prefixes = (
            tuple(f"auto_{s.run_num}__" for s in auto_specs)
            + tuple(f"bias_{r}__" for r in bias_runs)
        )
        drop_keys = [
            k for k in existing_keys
            if k.startswith(overwritten_prefixes) and k not in payload
        ]

    merge_save_npz(out_path, payload, drop_keys=drop_keys)
    print(f"Wrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print concise auto- and cross-bicoherence summary tables from saved NPZ."""
    if npz_path is None:
        npz_path = os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path, allow_pickle=False) as d:
        print("\n=== 1. Stationary Isat Auto-Bicoherence (2.24-4.80 ms, df=390.6 Hz, 1 seg/shot; post=Y if t_on<=2.24 ms) ===")
        print("Run  Grp       Puff   B0 post   N  thr    f0   A2/A1  b2(h2)  shuf  phi2  b2(h3)  shuf  R(f0) R(2f) Rthr | Peak (f1,f2)->f3  b2pk")
        for spec in jon.ONSET_RUNS:
            p = f"auto_{spec.run_num}__"
            if f"{p}b2_h2" not in d:
                continue
            r = spec.run_num
            grp = str(d[f"{p}group"])
            puff = float(d[f"{p}gas_puff_ms"])
            b0 = float(d[f"{p}b0_g"])
            post_flag = "Y" if (f"{p}window_post_onset" in d and bool(d[f"{p}window_post_onset"])) else "N"
            n = int(d[f"{p}n_avg"])
            thr = float(d[f"{p}threshold"])
            f0 = float(d[f"{p}f0_hz"]) / 1e3
            a21 = float(d[f"{p}amp_h2_over_h1"])
            b2 = float(d[f"{p}b2_h2"])
            s2 = float(d[f"{p}b2_h2_shuf"])
            ph2 = float(d[f"{p}biphase_h2_deg"])
            b3 = float(d[f"{p}b2_h3"])
            s3 = float(d[f"{p}b2_h3_shuf"])
            r1 = float(d[f"{p}pc_R_h1"])
            r2 = float(d[f"{p}pc_R_h2"])
            rthr = float(d[f"{p}pc_threshold"])
            pk = float(d[f"{p}b2_peak"])
            pf1 = float(d[f"{p}b2_peak_f1_hz"]) / 1e3
            pf2 = float(d[f"{p}b2_peak_f2_hz"]) / 1e3
            pf3 = float(d[f"{p}b2_peak_f3_hz"]) / 1e3
            ph2_s = f"{ph2:+5.0f}" if np.isfinite(ph2) else "  nan"
            print(
                f" {r:2s}  {grp:9s} {puff:4.0f} {b0:4.0f}   {post_flag}  {n:3d} {thr:.3f} {f0:5.2f}  {a21:5.2f}   "
                f"{b2:5.3f} {s2:5.3f} {ph2_s}   {b3:5.3f} {s3:5.3f}  {r1:5.2f} {r2:5.2f} {rthr:4.2f} | "
                f"({pf1:4.1f},{pf2:+5.1f})->{pf3:4.1f} {pk:5.3f}"
            )

        has_bias = any(f"bias_{r}__i2_b2_h2" in d for r in BIASSCOPE_RUNS)
        if has_bias:
            print("\n=== 2. End-Electrode Current Auto- & Cross-Bicoherence at (f0, f0)->2f0 ===")
            print("Run  Puff   N  thr    f0  | I2\" A2/A1 b2(h2) shuf  phi2  R1   | I4\" b2(h2) shuf | (I2,I2,I4) shuf | (Is,Is,I2) shuf | (I2,I2,Is) shuf")
            for r in BIASSCOPE_RUNS:
                p = f"bias_{r}__"
                if f"{p}i2_b2_h2" not in d:
                    continue
                puff = jon.ONSET_RUN_MAP[r].gas_puff_ms
                n = int(d[f"{p}n_shots"])
                thr = float(d[f"{p}threshold"])
                f0 = float(d[f"{p}f0_hz"]) / 1e3
                print(
                    f" {r:2s}  {puff:4.0f} {n:3d} {thr:.3f} {f0:5.2f} | "
                    f"     {float(d[f'{p}i2_amp_h2_over_h1']):4.2f}  {float(d[f'{p}i2_b2_h2']):5.3f} {float(d[f'{p}i2_b2_h2_shuf']):5.3f} "
                    f"{float(d[f'{p}i2_biphase_h2_deg']):+5.0f} {float(d[f'{p}i2_pc_R_h1']):4.2f} | "
                    f"    {float(d[f'{p}i4_b2_h2']):5.3f} {float(d[f'{p}i4_b2_h2_shuf']):5.3f} | "
                    f"    {float(d[f'{p}x_i2_i2_i4_b2_h2']):5.3f} {float(d[f'{p}x_i2_i2_i4_b2_h2_shuf']):5.3f} | "
                    f"    {float(d[f'{p}x_isat_isat_i2_b2_h2']):5.3f} {float(d[f'{p}x_isat_isat_i2_b2_h2_shuf']):5.3f} | "
                    f"    {float(d[f'{p}x_i2_i2_isat_b2_h2']):5.3f} {float(d[f'{p}x_i2_i2_isat_b2_h2_shuf']):5.3f}"
                )


if __name__ == "__main__":
    p = batch_all(overwrite=True)
    print_summary(p)
