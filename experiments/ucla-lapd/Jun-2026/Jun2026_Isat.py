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
"""

import glob
import os

import numpy as np
from tqdm import tqdm

import Jun2026_IV as jiv
import Jun2026_xcorr as jxc
from data_analysis.io import (open_lapd, position_shots, probe_channel_map,
                              moving_group)
from data_analysis.signal import (avg_amplitude_spectrum, clip_time_window,
                                  downsample_blockmean)
from data_analysis.utils import merge_save_npz, run_num_of

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

# --- Per-position FFT configuration ---------------------------------------
# batch_fft_by_position DISCOVERS channels (channel_descriptions) instead of
# taking a hardcoded list, so a run with 1, 2 or 3 tips all work unchanged.
# The default scope is the Mach probe: in runs 00-06 'lpscope' carries the
# *swept* Langmuir probe (FFT of a sweep returns the sweep rate, not a
# fluctuation spectrum) and 'biasscope' carries machine bias/light, so neither
# belongs on an Isat fluctuation page.
ISAT_SCOPE = "machscope"
POS_NPZ_SUFFIX = "-isat-fft-data.npz"

#: Samples kept in the stored raw trace that the page shows the FFT window on.
#: The full record is ~2.5M samples per shot at 50 MHz; the context plot is
#: ~420 px wide, so anything past a few thousand is invisible on the page and
#: pure payload. Block-mean rather than stride, so what survives is the
#: envelope of what was dropped rather than an aliased subset of it.
RAW_TRACE_SAMPLES = 4000


def get_isat_at_position(run, scope_name, chan, pos, pos_index):
    """Read the raw Isat signal for ONE probe position.

    ``pos`` is this channel's drive's :class:`Jun2026_IV.ProbePositions`; only
    that position's shots are read off disk, selected by recorded shot number
    (:func:`data_analysis.io.position_shots`).  Returns ``(tarr, Iarr)`` where
    ``Iarr`` is the ``(nshot, nsamples)`` raw signal (volts).  Current scaling
    (``Jun2026_IV.RESISTOR`` / ``Aprobe``) is left out for now -- the
    fluctuation *shape* is what matters, not the absolute scaling.  No sign flip
    either -- it doesn't matter for Isat fluctuations.
    """
    Istack, tarr = run.channel(
        chan, scope_name=scope_name,
        shots=position_shots(pos.shot_nums, pos_index, pos.nshot))
    # Istack /= Jun2026_IV.RESISTOR * Jun2026_IV.Aprobe   # current scaling (off)
    return tarr, Istack


def isat_npz_path(ifn):
    """Co-located npz path for a run: ``<run dir>/<run_num>-isat-fft-data.npz``.

    Mirrors :func:`Jun2026_xcorr.xcorr_npz_path` -- beside the raw HDF5, named
    from the run number only, so every channel of that run shares one file.
    """
    return os.path.join(os.path.dirname(ifn), f"{run_num_of(ifn)}{POS_NPZ_SUFFIX}")


def _chan_key(scope_name, chan):
    """Key prefix for one channel, e.g. ``'machscope-C2'``.

    Lets one run's npz hold several channels side by side (``<key>__amp``), the
    same encoding :func:`Jun2026_xcorr._pair_key` uses for pairs.
    """
    return f"{scope_name}-{chan}"


def stored_channels(npz_path):
    """The channel keys a run's per-position Isat npz actually holds.

    Reads the ``<scope>-<chan>__amp`` entries :func:`batch_fft_by_position`
    writes, so a caller (the slider emitter) enumerates what was analysed rather
    than guessing channel names.  Returns the sorted key list.

    Note this answers "what was analysed", not "what the file contains" -- the
    latter is :meth:`LapdRun.channel_descriptions`.
    """
    with np.load(npz_path) as d:
        return sorted(k.removesuffix("__amp") for k in d.files
                      if k.endswith("__amp"))


def isat_channels(ifn, scope_name=ISAT_SCOPE):
    """Discover ``[(scope, chan), ...]`` for one scope of a run, sorted by channel.

    Wraps :meth:`LapdRun.channel_descriptions` so the batch plots whatever the
    run happens to carry (1, 2 or 3 Mach tips) instead of a hardcoded list.
    Raises ``ValueError`` if the scope is absent, listing what is there -- a
    silent empty result would render an empty page.
    """
    desc = open_lapd(ifn).channel_descriptions()
    if scope_name not in desc:
        raise ValueError(f"scope {scope_name!r} not in {os.path.basename(ifn)}; "
                         f"available: {sorted(desc)}")
    return [(scope_name, c) for c in sorted(desc[scope_name])]


def batch_fft_by_position(ifn, scope_name=ISAT_SCOPE, chans=None,
                          tmin_ms=FFT_TMIN_MS, tmax_ms=FFT_TMAX_MS,
                          motion_group_name=None):
    """Per-position shot-averaged Isat amplitude spectra -> co-located npz.

    The per-*position* counterpart of :func:`batch_fft` (which averages every
    shot in a run, right for a stationary probe but blind to position).  Reads
    each position's ``nshot`` shots for every channel and incoherently averages
    their amplitude spectra, giving one spectrum per position per channel --
    what the FFT slider page scrubs through.

    ``chans`` defaults to every channel of ``scope_name``
    (:func:`isat_channels`).  All channels of one scope share that scope's time
    axis, so ``dt`` / the clip window / ``freq`` are read **once per scope** --
    but they are *not* run-wide: runs 00-06 sample ``machscope`` at 50 MHz and
    ``biasscope`` at 5 MHz, so a run-level ``dt`` would mislabel one of them by
    10x.  Restricting a call to one scope is what keeps the single stored
    ``freq`` axis valid.

    Writes ``<run>-isat-fft-data.npz`` (:func:`isat_npz_path`) holding ``pos_x``
    / ``pos_y`` (cm) and, per channel, the FFT window
    ``__tmin_ms`` / ``__tmax_ms``,
    ``<scope>-<chan>__amp`` ``(npos, nfreq)``, ``__nshots`` ``(npos,)``,
    ``__freq`` (Hz) and ``__raw`` -- one decimated UNCLIPPED shot
    (:data:`RAW_TRACE_SAMPLES`) that the page draws the window on, so a reader
    can see what the window kept and what it cut. Their shared time axis is
    ``<scope>__raw_t`` (seconds): unlike ``freq``, which a second scope's call
    would overwrite, one call's channels cannot differ in it.  The frequency axis is stored
    **per channel** rather than once: scopes sample at different rates (50 MHz
    for ``machscope``, 5 MHz for ``biasscope``), and a shared key would be
    overwritten by a second scope's call, relabelling the first scope's spectra
    with the second's axis.  Merged into any existing file, so a second scope is
    added by a second call.  Returns the npz path.
    """
    if chans is None:
        chans = isat_channels(ifn, scope_name)
    if not chans:
        raise ValueError(f"no channels to analyse for {os.path.basename(ifn)}")
    if {s for s, _ in chans} != {scope_name}:
        raise ValueError(
            f"chans must all be on scope {scope_name!r} (they share one stored "
            f"freq axis); got {sorted({s for s, _ in chans})}")

    # Positions are per channel, not per run: co-moving probes in a two-drive run
    # sit on different flux surfaces, so one shared pos_x/pos_y would stamp one
    # probe's coordinates onto the other's spectra. One channel map for the file
    # rather than reopening it per channel; positions deduped by drive, since the
    # common case is several channels on one probe.
    chan_map = probe_channel_map(ifn)
    groups = {ch: (motion_group_name or moving_group(chan_map, ch))
              for ch in chans}
    by_group = {g: jiv.read_lp_positions(ifn, g) for g in set(groups.values())}
    positions = {ch: by_group[g] for ch, g in groups.items()}

    npos_set = {p.npos for p in positions.values()}
    if len(npos_set) > 1:
        raise ValueError(
            f"{os.path.basename(ifn)}: channels span drives with different "
            f"position counts {sorted(npos_set)}; one stored axis cannot "
            "describe them. Batch one drive at a time.")
    npos = npos_set.pop()
    # Without this the position loop never runs, freq stays None, and np.savez
    # writes it as an object array -- the file saves and prints success, then
    # fails at load with "Object arrays cannot be loaded".
    if npos == 0:
        raise ValueError(f"{os.path.basename(ifn)} reports 0 positions; "
                         "nothing to analyse")
    run = open_lapd(ifn)
    tarr = run.time_array(scope_name=scope_name)
    dt = tarr[1] - tarr[0]
    i0, i1 = clip_time_window(tarr, tmin_ms, tmax_ms)

    # The window is stored, not just applied: the page shades it on the raw
    # trace, and re-deriving it from the clipped array bounds would be an
    # inference where the number itself is available.
    arrays = {}
    freq = None
    with tqdm(total=len(chans) * npos, desc="Isat FFT", unit="pos") as bar:
        for scope, chan in chans:
            pos = positions[(scope, chan)]
            nshot = pos.nshot
            amps, counts, raw = [], [], None
            for p in range(npos):
                _, Istack = get_isat_at_position(run, scope, chan, pos, p)
                if raw is None:
                    # The FIRST position's first shot, untrimmed: the context
                    # plot exists to show the window against the whole record,
                    # so this is the one array here that is NOT clipped.
                    q = max(1, Istack.shape[1] // RAW_TRACE_SAMPLES)
                    raw_t, raw = downsample_blockmean(tarr, Istack[0], q)
                    arrays[f"{scope_name}__raw_t"] = raw_t
                # get_isat_at_position returns the UNTRIMMED record; the clip to
                # the analysis window happens here, as in run_avg_fft.
                f, amp, n = avg_amplitude_spectrum(Istack[:, i0:i1], dt)
                # Every channel here is on one scope and clipped identically,
                # so this should be impossible; it is checked because a page
                # drawn against the wrong axis fails invisibly.
                if freq is None:
                    freq = f
                elif not np.array_equal(f, freq):
                    raise ValueError(
                        f"{scope}/{chan} position {p}: freq axis differs from "
                        f"earlier positions of the same scope ({scope_name})")
                amps.append(amp)
                counts.append(n)
                bar.update(1)
            key = _chan_key(scope, chan)
            arrays[f"{key}__amp"] = np.asarray(amps)
            arrays[f"{key}__nshots"] = np.asarray(counts)
            arrays[f"{key}__freq"] = freq
            arrays[f"{key}__raw"] = raw
            cx, cy = jxc.position_xy(pos.pos_array, npos, nshot)
            arrays[f"{key}__pos_x"] = cx
            arrays[f"{key}__pos_y"] = cy
            arrays[f"{key}__motion_group"] = np.str_(pos.motion_group or "")
            # File-level keys stay for back-compat and single-drive readers; the
            # first channel's drive defines them.
            arrays.setdefault("pos_x", cx)
            arrays.setdefault("pos_y", cy)
            # Per channel, like __freq: merge_save_npz keeps other channels'
            # arrays, so a file-level window would be overwritten by a second
            # call while the spectra it described stayed as they were --
            # relabelling them with a window they were never computed over.
            arrays[f"{key}__tmin_ms"] = np.float64(tmin_ms)
            arrays[f"{key}__tmax_ms"] = np.float64(tmax_ms)
    out_path = isat_npz_path(ifn)
    merge_save_npz(out_path, arrays)
    print(f"Wrote {out_path} ({len(chans)} channels x {npos} positions, "
          f"window {tmin_ms}-{tmax_ms} ms, scope {scope_name!r})")
    return out_path


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
