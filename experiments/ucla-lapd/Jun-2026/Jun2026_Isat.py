"""Jun-2026 LAPD ion-saturation-current (Isat) fluctuation analysis.

Companion to ``Jun2026_IV.py``.  Where that module pulls the *swept* Langmuir
tips (complete I+V pairs) to extract Vp/Te/ne, this one reads a **fixed-bias
ion-saturation channel** and looks at how the Isat signal *fluctuates* in time
(raw trace + FFT), rather than fitting an IV curve.

Which channel is Isat is **not** guessed here -- you read the printed channel
descriptions (:func:`list_all_channels`) and the run's probe description
(:func:`print_run_description`) and name the scope + channel yourself.

Reading
-------
Same pydaq read path as ``Jun2026_IV``: channels are read by scope-channel name
via ``run.channel(name, scope_name=...)``.  Positions come from
``Jun2026_IV.read_lp_positions``.  Isat is a single fixed-bias current trace per
shot, so there is no sweep detection / reshape -- we read the raw per-shot
signal at one position and FFT it.

Current scaling reuses the ``RESISTOR`` / ``Aprobe`` knobs from ``Jun2026_IV``;
for fluctuation work the *shape* of the spectrum is what matters, not the
absolute scaling.  The IV pipeline's ``I_SIGN`` is irrelevant to Isat (a sign
flip doesn't change the fluctuation spectrum), so it is not applied here.
"""

import glob
import os

import numpy as np
import h5py
from tqdm import tqdm

from data_analysis.io import open_lapd
from data_analysis.io.scope_reader import read_scope_channel_descriptions
from data_analysis.signal import avg_amplitude_spectrum

import Jun2026_IV as jiv

# Scope groups that may hold probe signals.  ``Configuration`` / ``Control`` are
# never scopes; everything else is a scope group.
_NON_SCOPE_GROUPS = {"Configuration", "Control"}

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


def list_all_channels(fn):
    """Print every scope group's channel descriptions and return them.

    Returns ``{scope_name: {chan: description}}`` for every scope group in the
    file (``Configuration`` / ``Control`` excluded).  Use the printout to decide
    which scope + channel is the Isat channel you want -- nothing is classified
    or guessed.
    """
    out = {}
    with h5py.File(fn, "r") as f:
        scope_groups = [g for g in f.keys() if g not in _NON_SCOPE_GROUPS]
        print("Scope groups and channel descriptions:")
        for scope_name in scope_groups:
            desc = read_scope_channel_descriptions(f, scope_name)
            if not desc:
                continue
            out[scope_name] = dict(desc)
            print(f"\n  scope '{scope_name}':")
            for chan in sorted(desc):
                print(f"    {chan}: {desc[chan]!r}")
    if not out:
        print("  (no scope groups with channel descriptions found)")
    return out


def print_run_description(fn):
    """Print the run's hand-written description (plasma / bias / probe settings).

    Reads the pydaq ``description`` attribute via ``open_lapd(fn).description()``
    and prints its raw text so the probe wiring / bias settings the operator
    wrote are visible alongside the channel list.  Returns the parsed
    ``RunDescription`` (or ``None`` if it can't be read).
    """
    try:
        desc = open_lapd(fn).description()
    except (OSError, ValueError, NotImplementedError, KeyError) as e:
        print(f"(could not read run description -- {e})")
        return None
    print("=== Run description ===")
    print(desc.raw)
    return desc


def get_isat_at_position(run, scope_name, chan, npos, nshot, pos_index):
    """Read the scaled Isat signal for ONE probe position.

    Reads only that position's ``nshot`` shots off disk (a positional shot slice,
    same ordering as ``Jun2026_IV._read_reshaped``).  Returns ``(tarr, Iarr)``
    where ``Iarr`` is ``(nshot, nsamples)`` scaled current density (via
    ``jiv.RESISTOR`` / ``jiv.Aprobe``).  No sign flip is applied -- it doesn't
    matter for Isat fluctuations.
    """
    shots = slice(pos_index * nshot, (pos_index + 1) * nshot)
    Istack, tarr = run.channel(chan, scope_name=scope_name, shots=shots)
    Iarr = Istack[:nshot] / (jiv.RESISTOR * jiv.Aprobe)
    return tarr, Iarr


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
    """
    run = open_lapd(fn)
    Istack, tarr = run.channel(chan, scope_name=scope_name)   # shots=None -> all

    # Trim to the FFT window here, then hand the trimmed stack + dt to the
    # shared helper (per-shot FFT + incoherent average is generic DSP).
    dt = tarr[1] - tarr[0]
    i0, i1 = np.searchsorted(tarr, [tmin_ms * 1e-3, tmax_ms * 1e-3])
    if i1 - i0 < 2:
        raise ValueError(
            f"window {tmin_ms}-{tmax_ms} ms selects < 2 samples "
            f"(i0={i0}, i1={i1})")
    freq, amp_mean, n_shots = avg_amplitude_spectrum(Istack[:, i0:i1], dt)

    # amp_mean = amp_mean / (jiv.RESISTOR * jiv.Aprobe)
    return freq, amp_mean, n_shots


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
