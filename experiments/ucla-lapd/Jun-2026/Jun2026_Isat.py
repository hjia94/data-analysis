"""Jun-2026 LAPD ion-saturation-current (Isat) fluctuation analysis.

Companion to ``Jun2026_IV.py``.  Where that module pulls the *swept* Langmuir
tips (complete I+V pairs) to extract Vp/Te/ne, this one reads a **fixed-bias
ion-saturation channel** and looks at how the Isat signal *fluctuates* in time
(raw trace + FFT), rather than fitting an IV curve.

Which channel is Isat is **not** guessed here -- you read the printed channel
descriptions and the run's probe description
(``data_analysis.io.list_all_channels`` / ``print_run_description``, driven from
``Jun2026_run_overview.ipynb``) and name the scope + channel yourself.

Reading
-------
Same pydaq read path as ``Jun2026_IV``: channels are read by scope-channel name
via ``run.channel(name, scope_name=...)``.  Positions come from
``Jun2026_IV.read_lp_positions``.  Isat is a single fixed-bias current trace per
shot, so there is no sweep detection / reshape -- we read the raw per-shot
signal at one position and FFT it.

Current scaling (the ``RESISTOR`` / ``Aprobe`` knobs from ``Jun2026_IV``) is
currently left OFF -- the signal is kept raw (volts).  For fluctuation work the
*shape* of the spectrum is what matters, not the absolute scaling; the scaling
lines are commented in place (search ``RESISTOR``) to re-enable later.  The IV
pipeline's ``I_SIGN`` is likewise irrelevant to Isat (a sign flip doesn't change
the fluctuation spectrum), so it is not applied here.
"""

import glob
import os

import numpy as np
from tqdm import tqdm

from data_analysis.io import open_lapd, position_shots
from data_analysis.signal import avg_amplitude_spectrum, clip_time_window

# --- Batch FFT configuration (runs 00-06) -----------------------------------
# Runs 00-06 share a fixed (stationary) Isat probe, so the averaged spectrum is
# taken over EVERY shot in the file (no per-position split).  Set the scope +
# channel and the FFT time window here, then call ``batch_fft()``.
DATA_DIR    = r"D:\data\LAPD\jun2026-jia"
RUN_GLOB    = "0[0-6]-*.hdf5"          # runs 00..06
OUT_NPZ     = "isat_fft_00-06.npz"     # written into DATA_DIR

SCOPE_NAME  = "machscope"              # scope group holding the Isat channel
CHAN        = "C2"                     # Isat channel within that scope

FFT_TMIN_MS = 1.5                      # FFT time window start (ms)
FFT_TMAX_MS = 5.0                      # FFT time window stop  (ms)

FFT_CHUNK_SHOTS = 50                   # shots per read in run_avg_fft (caps peak memory)


def get_isat_at_position(run, scope_name, chan, npos, nshot, pos_index):
    """Read the raw Isat signal for ONE probe position.

    Reads only that position's ``nshot`` shots off disk
    (:func:`data_analysis.io.position_shots`).  Returns ``(tarr, Iarr)`` where
    ``Iarr`` is the ``(nshot, nsamples)`` raw signal (volts).  Current scaling
    (``Jun2026_IV.RESISTOR`` / ``Aprobe``) is left out for now -- the
    fluctuation *shape* is what matters, not the absolute scaling.  No sign flip
    either -- it doesn't matter for Isat fluctuations.
    """
    Istack, tarr = run.channel(chan, scope_name=scope_name,
                               shots=position_shots(pos_index, nshot))
    # Istack /= Jun2026_IV.RESISTOR * Jun2026_IV.Aprobe   # current scaling (off)
    return tarr, Istack


def run_avg_fft(fn, scope_name=SCOPE_NAME, chan=CHAN,
                tmin_ms=FFT_TMIN_MS, tmax_ms=FFT_TMAX_MS):
    """Average the Isat FFT over ALL shots in one run file.

    The Isat probe in runs 00-06 is stationary, so every shot in the file is a
    repeat at the same position -- we read them all and incoherently average the
    per-shot amplitude spectra (random shot-to-shot phase cancels in a coherent
    average but not here, so broadband fluctuation power survives).

    Returns ``(freq, amp_mean, n_shots)`` -- ``freq`` in Hz, ``amp_mean`` the
    shot-averaged single-sided amplitude (DC dropped), ``n_shots`` the number of
    shots that contributed (NaN/unreadable shots are skipped).

    Shots are read in chunks of ``FFT_CHUNK_SHOTS`` and the chunk spectra
    averaged weighted by their shot counts -- identical to the all-at-once mean,
    but peak memory is one chunk instead of the whole multi-GB run (each shot is
    ~2.5M samples of which only the FFT window is kept).
    """
    run = open_lapd(fn)
    tarr = run.time_array(scope_name=scope_name)
    dt = tarr[1] - tarr[0]
    i0, i1 = clip_time_window(tarr, tmin_ms, tmax_ms)

    n_all = len(run.shots(scope_name=scope_name))
    freq = amp_sum = None
    n_shots = 0
    for s in range(0, n_all, FFT_CHUNK_SHOTS):
        Istack, _ = run.channel(chan, scope_name=scope_name,
                                shots=slice(s, min(s + FFT_CHUNK_SHOTS, n_all)))
        try:
            freq, amp, n = avg_amplitude_spectrum(Istack[:, i0:i1], dt)
        except ValueError:      # no finite shots in this chunk
            continue
        amp_sum = amp * n if amp_sum is None else amp_sum + amp * n
        n_shots += n
    if amp_sum is None:
        raise ValueError(f"no finite shots for '{scope_name}'/{chan} in {fn!r}")

    # amp_mean = amp_mean / (Jun2026_IV.RESISTOR * Jun2026_IV.Aprobe)
    return freq, amp_sum / n_shots, n_shots


def batch_fft(data_dir=DATA_DIR, run_glob=RUN_GLOB, out_npz=OUT_NPZ,
              scope_name=SCOPE_NAME, chan=CHAN,
              tmin_ms=FFT_TMIN_MS, tmax_ms=FFT_TMAX_MS):
    """Average the Isat FFT over all shots for each run, save to one npz.

    Loops the files matched by ``run_glob`` in ``data_dir`` (runs 00-06),
    computes the all-shot-averaged amplitude spectrum for each via
    :func:`run_avg_fft`, and writes a single npz into ``data_dir``.

    All runs share one window + sampling rate, so the npz holds a single
    ``freq`` array (Hz) plus one ``<run>__amp`` (shot-averaged amplitude) per
    run, keyed by the run's base name (without ``.hdf5``).  A ``runs`` array
    lists the run keys (in order) and ``nshots`` the matching shot counts.
    Reload with ``np.load(path)``: ``d["freq"]`` and ``d[f"{run}__amp"]``.
    """
    files = sorted(glob.glob(os.path.join(data_dir, run_glob)))
    if not files:
        raise FileNotFoundError(f"no files match {run_glob!r} in {data_dir!r}")

    arrays = {}
    runs, nshots = [], []
    # One bar over the run files; %, elapsed, ETA, rate.  Per-run messages go
    # through pbar.write so they don't tear the bar.
    pbar = tqdm(files, desc="FFT", unit="run")
    for fn in pbar:
        key = os.path.splitext(os.path.basename(fn))[0]
        freq, amp, n = run_avg_fft(fn, scope_name, chan, tmin_ms, tmax_ms)
        # Same window + sampling rate across runs -> one shared freq axis.
        if "freq" not in arrays:
            arrays["freq"] = freq
        elif not np.array_equal(freq, arrays["freq"]):
            raise ValueError(f"{key}: freq axis differs from earlier runs")
        arrays[f"{key}__amp"] = amp
        runs.append(key)
        nshots.append(n)
        pbar.write(f"  {key}: averaged {n} shots, {freq.size} freq bins")

    arrays["runs"] = np.array(runs)
    arrays["nshots"] = np.array(nshots)

    out_path = os.path.join(data_dir, out_npz)
    np.savez(out_path, **arrays)
    print(f"\nWrote {out_path} ({len(runs)} runs, "
          f"window {tmin_ms}-{tmax_ms} ms, scope '{scope_name}' {chan})")
    return out_path


if __name__ == "__main__":
    batch_fft()
