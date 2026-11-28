"""Jun-2026 LAPD cross-correlation analysis between two scope channels.

Two channels, three frequency-domain relationships:

1. magnitude-squared coherence ``gamma2(f)`` -- shared frequency bands (0..1)
2. cross-phase ``phase(f)`` -- phase difference per frequency
3. time-lag cross-correlation whose peak is the scalar time delay.

Channel identity is a ``(scope, channel)`` pair, e.g. ``("machscope", "C3")`` --
a run has several scope groups, so the scope must be named.  Channel *inspection*
(which scope/channel is what) is not done here; pick the pair yourself.

Same read path as ``Jun2026_IV`` / ``Jun2026_Isat``: channels via
``run.channel(name, scope_name=...)``; positions via
``Jun2026_IV.read_lp_positions``.  Signal is kept raw (volts) -- the *shape* of
the coherence/phase is what matters, not absolute current scaling.
"""

import os
from typing import NamedTuple
import numpy as np
from tqdm import tqdm

from data_analysis.io import open_lapd, position_shots, probe_channel_map
from data_analysis.io.probe_map import STATIONARY, UNMAPPED
from data_analysis.utils import (
    find_run_hdf5,
    merge_save_npz,
    npz_settings_mismatch,
    run_num_of,
)
from data_analysis.signal import (
    cross_correlation,
    coherence_spectrum,
    cross_phase_spectrum,
    avg_cross_spectrum,
    clip_time_window,
    finite_row_mask,
)

import Jun2026_IV as jiv


# --- Configuration (edit-in-place, like Jun2026_Isat.py) --------------------
DATA_DIR = r"D:\data\LAPD\jun2026-jia"
IFN = os.path.join(DATA_DIR, "26-He-800G-bias40V-Bdot-LP-plane_2026-06-12.hdf5")
CH_A        = ("bdotscope", "C1")
CH_B        = ("lpscope", "C3")
TMIN_MS, TMAX_MS = 1.5, 4.5            # analysis time window (ms)
NPERSEG     = 4096                     # Welch segment length (freq res vs variance)

class PairSpec(NamedTuple):
    """One channel-pair entry in :data:`XCORR_RUNS`.

    ``t_offset_b_ms`` defaults to ``0.0`` (shared trigger); set it when
    ``ch_a`` and ``ch_b`` live on scopes with different trigger delays
    (e.g., ``+12.0`` for ``machscope`` vs. ``biasscope``).
    """
    ch_a: tuple[str, str]
    ch_b: tuple[str, str]
    motion_group_name: str | None = None
    t_offset_b_ms: float = 0.0


# Campaign pair manifest for ``batch_all()``: run_num -> list of PairSpec.
#   * Runs 26 (800G) and 31 (filename says 800G; run description is 600G):
#     ``lpscope`` and ``bdotscope`` share the same trigger (both span
#     ``[-2.0, +8.0] ms`` within 42 ns, verified from HDF5 ``time_array``), so
#     ``t_offset_b_ms=0.0``. B-dot channels are UNMAPPED in the run description,
#     so they pass ``motion_group_name="<Athena>    p28_Bdot"``; LP pairs
#     auto-resolve to ``<Hermes>    p29_LP`` against stationary P25-L.
#   * Runs 27 (800G), 28 (600G), 30 (1kG): co-moving P29 + P25 axial lines plus
#     stationary P33 reference (``scope/C5``).
#   * Run 29 (800G): two-phase 31x31 plane (P29 shots 1-4805, P25 shots
#     4806-9610), so only moving-vs-stationary-P33 pairs are physically valid.
_BDOT_LP_PLANE_PAIRS = [
    PairSpec(("lpscope", "C1"), ("lpscope", "C3")),
    PairSpec(("lpscope", "C2"), ("lpscope", "C3")),
    PairSpec(("lpscope", "C3"), ("bdotscope", "C1"), "<Athena>    p28_Bdot"),
    PairSpec(("lpscope", "C3"), ("bdotscope", "C2"), "<Athena>    p28_Bdot"),
    PairSpec(("lpscope", "C3"), ("bdotscope", "C3"), "<Athena>    p28_Bdot"),
]
_AXIAL_LP_LINE_PAIRS = [
    PairSpec(("scope", "C1"), ("scope", "C3")),  # P29-L vs P25-L (axial phase)
    PairSpec(("scope", "C2"), ("scope", "C4")),  # P29-R vs P25-R (axial phase)
    PairSpec(("scope", "C1"), ("scope", "C5")),  # P29-L vs P33 ref
    PairSpec(("scope", "C2"), ("scope", "C5")),  # P29-R vs P33 ref
    PairSpec(("scope", "C3"), ("scope", "C5")),  # P25-L vs P33 ref
    PairSpec(("scope", "C4"), ("scope", "C5")),  # P25-R vs P33 ref
]
_TWOPHASE_LP_PLANE_PAIRS = [
    PairSpec(("scope", "C1"), ("scope", "C5")),  # P29-L vs P33 ref (phase 1)
    PairSpec(("scope", "C2"), ("scope", "C5")),  # P29-R vs P33 ref (phase 1)
    PairSpec(("scope", "C3"), ("scope", "C5")),  # P25-L vs P33 ref (phase 2)
    PairSpec(("scope", "C4"), ("scope", "C5")),  # P25-R vs P33 ref (phase 2)
]
XCORR_RUNS = {
    "26": _BDOT_LP_PLANE_PAIRS,
    "27": _AXIAL_LP_LINE_PAIRS,
    "28": _AXIAL_LP_LINE_PAIRS,
    "29": _TWOPHASE_LP_PLANE_PAIRS,
    "30": _AXIAL_LP_LINE_PAIRS,
    "31": _BDOT_LP_PLANE_PAIRS,
}

# Tolerance [cm] for treating two pairs' probe coordinates as the same grid
# when deciding whether top-level ``pos_x``/``pos_y`` is unambiguous.
# 0.01 cm (0.1 mm) absorbs ~5 um live-encoder jitter between drives on the same
# scan (runs 27-29) while rejecting real grid offsets (>= 0.5 cm, e.g. run 30)
# and opposite-order scans (runs 26 & 31).
_POS_MATCH_TOL_CM = 1e-2


# =========================================================================== #
#  Reading -- pull a position's shots for both channels onto one time grid
# =========================================================================== #

def _read_pair_at_position(run, ch_a, ch_b, pos_a, pos_b, pos_index,
                           tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
                           t_offset_b_ms=0.0):
    """Read both channels' shots at one probe position onto a common time grid.

    ``ch_a`` / ``ch_b`` are ``(scope, channel)`` pairs; ``pos_a`` / ``pos_b`` are
    their drives' :class:`Jun2026_IV.ProbePositions`.  Only that position's shots
    are read off disk, selected by recorded shot number
    (:func:`data_analysis.io.position_shots`) -- so each channel gets the shots
    ITS probe was at that position for, which for a co-moving pair are the same
    discharges.  Returns ``(stack_a, stack_b, dt)`` with the two
    ``(nshot, nsamples)`` stacks clipped to ``[tmin_ms, tmax_ms]`` (in channel A's
    time base) and on the *same* grid, plus the sample interval ``dt`` (seconds).

    Same-scope pair with ``t_offset_b_ms == 0`` (the common case): both channels
    share the identical scope ``tarr`` recorded in the HDF5, so it is used
    directly -- no resampling.  Cross-scope pair (or non-zero ``t_offset_b_ms``):
    channel B is clipped to ``[tmin_ms + t_offset_b_ms, tmax_ms + t_offset_b_ms]``
    and resampled onto A's clipped grid via ``np.interp``, where
    ``t_offset_b_ms = t_b - t_a`` at the same physical instant (e.g. ``+12.0`` ms
    when A triggers at bias start and B is ``biasscope`` triggered at plasma 1 kA).
    """
    (scope_a, chan_a), (scope_b, chan_b) = ch_a, ch_b

    stack_a, tarr_a = run.channel(
        chan_a, scope_name=scope_a,
        shots=position_shots(pos_a.shot_nums, pos_index, pos_a.nshot))
    stack_b, tarr_b = run.channel(
        chan_b, scope_name=scope_b,
        shots=position_shots(pos_b.shot_nums, pos_index, pos_b.nshot))
    if stack_a is None or stack_b is None:
        raise ValueError(f"could not read {ch_a} or {ch_b} at position {pos_index}")

    ia0, ia1 = clip_time_window(tarr_a, tmin_ms, tmax_ms)
    ta = tarr_a[ia0:ia1]
    sa = stack_a[:, ia0:ia1]
    dt = ta[1] - ta[0]

    if scope_a == scope_b and t_offset_b_ms == 0.0:
        # Same scope -> identical time array; clip B with the same indices.
        sb = stack_b[:, ia0:ia1]
        return sa, sb, dt

    # Cross scope (or trigger-offset): clip B in its own time base, shift into
    # A's time base, then interpolate row-wise onto A's grid for the FFTs.
    ib0, ib1 = clip_time_window(tarr_b, tmin_ms + t_offset_b_ms,
                                tmax_ms + t_offset_b_ms)
    tb = tarr_b[ib0:ib1] - t_offset_b_ms * 1e-3
    sb_clip = stack_b[:, ib0:ib1]
    sb = np.vstack([np.interp(ta, tb, row) for row in sb_clip])
    return sa, sb, dt


# =========================================================================== #
#  Analysis -- per-shot and ensemble-averaged
# =========================================================================== #

def xcorr_per_shot(stack_a, stack_b, dt, nperseg=NPERSEG):
    """The three correlation methods computed **per shot** (for the overlay figure).

    ``stack_a`` / ``stack_b`` are ``(nshot, nsamples)`` on the same grid.  For each
    shot (rows with any non-finite sample skipped) computes the coherence,
    cross-phase, and time-lag cross-correlation.  Returns a dict::

        freq   : (nf,)            frequency axis, Hz  (shared)
        gamma2 : (nshot, nf)      per-shot coherence
        phase  : (nshot, nf)      per-shot cross-phase, radians
        lags   : (nlag,)          lag axis, seconds  (shared)
        xcorr  : (nshot, nlag)    per-shot normalized cross-correlation

    (``nshot`` here is the count of *finite* shots that contributed.)
    """
    stack_a = np.asarray(stack_a, float)
    stack_b = np.asarray(stack_b, float)
    good = finite_row_mask(stack_a, stack_b)

    freq = lags = None
    g2_rows, ph_rows, xc_rows = [], [], []
    for x, y in zip(stack_a[good], stack_b[good]):
        f, g2 = coherence_spectrum(x, y, dt, nperseg=nperseg)
        _, ph = cross_phase_spectrum(x, y, dt, nperseg=nperseg)
        lag, xc = cross_correlation(x, y, dt)
        if freq is None:
            freq, lags = f, lag
        g2_rows.append(g2)
        ph_rows.append(ph)
        xc_rows.append(xc)

    if not g2_rows:
        raise ValueError("no finite shot pairs to correlate")

    return {
        "freq": freq,
        "gamma2": np.vstack(g2_rows),
        "phase": np.vstack(ph_rows),
        "lags": lags,
        "xcorr": np.vstack(xc_rows),
    }


def _ensemble_xcorr(stack_a, stack_b, dt):
    """Time-lag cross-correlation of the two shot-averaged traces at one position.

    Averages the finite shots of each stack, then cross-correlates the two mean
    traces (:func:`cross_correlation`).  Returns ``(lags, xcorr)`` -- the lag axis
    (seconds) and normalized cross-correlation.  Shared by :func:`xcorr_averaged`
    and :func:`batch_xcorr` so the ensemble lag is computed identically.
    """
    stack_a = np.asarray(stack_a, float)
    stack_b = np.asarray(stack_b, float)
    good = finite_row_mask(stack_a, stack_b)
    a_avg = stack_a[good].mean(axis=0)
    b_avg = stack_b[good].mean(axis=0)
    return cross_correlation(a_avg, b_avg, dt)


def xcorr_averaged(stack_a, stack_b, dt, nperseg=NPERSEG):
    """Ensemble-averaged correlation over all shots (for the averaged figure).

    Coherence + cross-phase come from :func:`avg_cross_spectrum` (the spectra are
    averaged across shots before the coherence ratio -- the statistically correct
    ensemble estimate).  The time-lag cross-correlation is computed on the two
    shot-averaged traces.  Returns a dict::

        freq   : (nf,)     frequency axis, Hz
        gamma2 : (nf,)     ensemble coherence
        phase  : (nf,)     ensemble cross-phase, radians
        lags   : (nlag,)   lag axis, seconds
        xcorr  : (nlag,)   cross-correlation of the shot-averaged traces
        n_used : int       number of shots that contributed
    """
    freq, gamma2, phase, n_used = avg_cross_spectrum(
        stack_a, stack_b, dt, nperseg=nperseg)
    lags, xcorr = _ensemble_xcorr(stack_a, stack_b, dt)

    return {
        "freq": freq,
        "gamma2": gamma2,
        "phase": phase,
        "lags": lags,
        "xcorr": xcorr,
        "n_used": int(n_used),
    }


# =========================================================================== #
#  Batch (run after verifying in the notebook) -- one HDF5 file at a time
# =========================================================================== #

# Co-located npz: sits next to the raw HDF5 (like the IV .npz), one file per run.
# A run's npz can hold SEVERAL channel-pairs, each under its own key prefix, so
# more pairs can be added later without a new file. Filename is run-derived only.
OUT_NPZ_SUFFIX = "-xcorr-data.npz"


def xcorr_npz_path(ifn):
    """Co-located npz path for a run: ``<run dir>/<run_num>-xcorr-data.npz``.

    Sits beside the raw HDF5 (same convention as the IV ``.npz``); the name is
    derived from the run number only, so every channel-pair for that run shares
    one file.
    """
    return os.path.join(os.path.dirname(ifn), f"{run_num_of(ifn)}{OUT_NPZ_SUFFIX}")


def _pair_key(ch_a, ch_b):
    """Key prefix for a channel pair, e.g. ``'machscope-C3__machscope-C4'``.

    Lets one run's npz hold several pairs side by side (``<key>__gamma2`` etc.).
    """
    return f"{ch_a[0]}-{ch_a[1]}__{ch_b[0]}-{ch_b[1]}"


def _pair_from_key(key):
    """Inverse of :func:`_pair_key`: ``'lpscope-C2__lpscope-C3'`` -> two tuples.

    Kept beside ``_pair_key`` so the encoding and its decoding cannot drift
    apart.  Splitting on the *last* ``-`` keeps the scope name whole, since a
    scope is the half that realistically carries one (``'bdot-scope'``); the
    channel names it pairs with are short scope-assigned tags (``C1``..``C4``).
    A channel containing ``-`` is the one case this encoding cannot represent.
    """
    return tuple(tuple(part.rsplit("-", 1)) for part in key.split("__"))


def pair_label(ch_a, ch_b):
    """A channel pair as it reads to a human: ``'bdotscope/C1 vs lpscope/C3'``.

    Beside :func:`_pair_key` because a pair has exactly two written forms -- the
    npz key and this one -- and keeping them together is what stops a figure
    title, a slider dropdown, and a stored key from drifting into three
    different spellings of the same pair.
    """
    return f"{ch_a[0]}/{ch_a[1]} vs {ch_b[0]}/{ch_b[1]}"


def stored_settings(npz_path, pair_key=None):
    """The batch settings a run's xcorr npz recorded: ``{window, nperseg}``.

    ``{}`` for a file written before :func:`batch_xcorr` stored them, and for a
    file whose pairs disagree -- one page carries one banner, and a single
    number there would have to pick a pair to be true of. ``pair_key`` reads one
    pair's settings instead of requiring agreement.
    """
    if not npz_path or not os.path.isfile(npz_path):
        return {}
    with np.load(npz_path) as d:
        keys = ([pair_key] if pair_key else
                [k.removesuffix("__nperseg") for k in d.files
                 if k.endswith("__nperseg")])
        found = set()
        for k in keys:
            if f"{k}__nperseg" not in d.files:
                continue
            found.add((float(d[f"{k}__tmin_ms"]), float(d[f"{k}__tmax_ms"]),
                       int(d[f"{k}__nperseg"])))
    if len(found) != 1:
        return {}
    tmin, tmax, nperseg = found.pop()
    return {"window": f"{tmin:g}-{tmax:g} ms", "nperseg": nperseg}


def stored_pairs(npz_path):
    """The channel-pair keys a run's xcorr npz actually holds.

    Reads the ``<pair>__gamma2`` entries :func:`batch_xcorr` writes, so callers
    can enumerate what was batched instead of guessing pair names.  Returns the
    sorted key list; most callers want :func:`stored_pair_tuples` instead.
    """
    with np.load(npz_path) as d:
        return sorted(k.removesuffix("__gamma2") for k in d.files
                      if k.endswith("__gamma2"))


def stored_pair_tuples(npz_path):
    """:func:`stored_pairs` decoded to ``[(ch_a, ch_b), ...]`` tuples.

    What a caller looping over a run's pairs actually wants, so the key
    encoding stays private to this module rather than every caller reaching
    for :func:`_pair_from_key` itself.
    """
    return [_pair_from_key(key) for key in stored_pairs(npz_path)]


def position_xy(pos_array, npos, nshot):
    """(x, y) of each of the ``npos`` positions: the first shot of each block.

    Kept with the spectra so a plane map has real axes (not just a position
    index). Used by :func:`batch_xcorr`.
    """
    return pos_array["x"][::nshot][:npos], pos_array["y"][::nshot][:npos]


def _resolve_pair_positions(ifn, ch_a, ch_b, motion_group_name=None):
    """Resolve ``(pos_a, pos_b, spatial_pos)`` for a channel pair.

    When ``motion_group_name`` is passed, both channels read that drive's shots
    and its coordinates label the pair (the explicit override for ``UNMAPPED``
    channels like B-dot whose descriptions carry no port).

    Otherwise uses :func:`data_analysis.io.probe_channel_map`:

    * moving vs. moving: resolves each drive independently and verifies they
      recorded the same discharges (:func:`assert_pairable`); ``spatial_pos`` is
      channel A's drive.
    * moving vs. stationary (in either argument order): the stationary reference
      was digitized on every discharge, so it inherits the moving probe's shots
      and ``spatial_pos`` is the moving probe's drive -- avoiding a spurious
      multi-drive ambiguity error on runs with 2+ motion groups.
    * either channel ``UNMAPPED``, or both ``STATIONARY``: refuses rather than
      guessing, telling the caller to pass ``motion_group_name``.
    """
    if motion_group_name is not None:
        pos = jiv.read_lp_positions(ifn, motion_group_name)
        return pos, pos, pos

    mapping = probe_channel_map(ifn)
    ka, kb = tuple(ch_a), tuple(ch_b)
    for k in (ka, kb):
        if k not in mapping:
            raise KeyError(
                f"{k!r} has no wiring description in {os.path.basename(ifn)}; "
                f"described channels: {sorted(f'{s}:{c}' for s, c in mapping)}"
            )
    ga, gb = mapping[ka], mapping[kb]

    unmapped = [ch for ch, g in ((ch_a, ga), (ch_b, gb)) if g == UNMAPPED]
    if unmapped:
        raise ValueError(
            f"{os.path.basename(ifn)}: channel(s) {unmapped} name no port in "
            "their wiring description, so their motion group cannot be "
            "inferred. Pass motion_group_name=... explicitly."
        )
    if ga == STATIONARY and gb == STATIONARY:
        raise ValueError(
            f"{os.path.basename(ifn)}: both {ch_a} and {ch_b} are stationary "
            "(no moving probe in this pair). Pass motion_group_name=... if you "
            "want to bin them by a specific drive's scan."
        )

    if gb == STATIONARY:
        pos_a = jiv.read_lp_positions(ifn, ga)
        return pos_a, pos_a, pos_a
    if ga == STATIONARY:
        pos_b = jiv.read_lp_positions(ifn, gb)
        return pos_b, pos_b, pos_b

    pos_a = jiv.read_lp_positions(ifn, ga)
    pos_b = pos_a if gb == ga else jiv.read_lp_positions(ifn, gb)
    if pos_b.motion_group != pos_a.motion_group:
        assert_pairable(ifn, ch_a, pos_a, ch_b, pos_b)
    return pos_a, pos_b, pos_a


def _iter_run_positions(ifn, ch_a, ch_b, tmin_ms, tmax_ms, desc,
                        motion_group_name=None, t_offset_b_ms=0.0):
    """Per-position read loop for :func:`batch_xcorr`.

    Reads the run's positions via :func:`_resolve_pair_positions`, opens the
    file, and returns ``(pos_x, pos_y, npos, gen, motion_group)`` where ``gen``
    yields ``(p, sa, sb, dt)`` for each position -- the two clipped shot stacks
    from :func:`_read_pair_at_position` -- under a per-shot tqdm bar.  A caller
    that ``continue``s past a failed position still ticks the bar.

    The returned ``pos_x``/``pos_y`` describe the moving probe (channel A when
    both move; the moving partner when one is stationary).  Raises if two
    different moving drives did not record the same discharges
    (:func:`assert_pairable`).
    """
    pos_a, pos_b, spatial_pos = _resolve_pair_positions(
        ifn, ch_a, ch_b, motion_group_name)
    npos, nshot = spatial_pos.npos, spatial_pos.nshot
    run = open_lapd(ifn)
    pos_x, pos_y = position_xy(spatial_pos.pos_array, npos, nshot)

    def gen():
        with tqdm(total=npos * nshot, desc=desc, unit="shot") as bar:
            for p in range(npos):
                sa, sb, dt = _read_pair_at_position(
                    run, ch_a, ch_b, pos_a, pos_b, p, tmin_ms, tmax_ms,
                    t_offset_b_ms=t_offset_b_ms)
                yield p, sa, sb, dt
                bar.update(sa.shape[0])

    return pos_x, pos_y, npos, gen(), spatial_pos.motion_group


def assert_pairable(ifn, ch_a, pos_a, ch_b, pos_b):
    """Raise unless two drives' channels can be correlated position by position.

    A cross-spectrum is only physical between signals from the SAME discharge, so
    the two drives must have been at each position for the same shots. Two probes
    can share a file without that being true: in a sequential scan each drive
    owns its own block of shots (probe A shots 1-N, probe B shots N+1-2N), which
    produces identical grids and identical (x, y) at every index -- so a grid or
    shape check passes and the pairing looks valid while every position
    correlates two different plasmas.

    Comparing the recorded shot numbers is the direct test. It also catches the
    partial-overlap case (B starts before A finishes), which a file-wide
    intersection would let through.
    """
    where = (f"{os.path.basename(ifn)}: {ch_a} ({pos_a.motion_group}) and "
             f"{ch_b} ({pos_b.motion_group})")
    if (pos_a.npos, pos_a.nshot) != (pos_b.npos, pos_b.nshot):
        raise ValueError(
            f"{where} scanned different grids ({pos_a.npos}x{pos_a.nshot} vs "
            f"{pos_b.npos}x{pos_b.nshot}); they cannot be paired position by "
            "position.")
    if not np.array_equal(pos_a.shot_nums, pos_b.shot_nums):
        n_same = int(np.sum(np.asarray(pos_a.shot_nums) ==
                            np.asarray(pos_b.shot_nums)))
        raise ValueError(
            f"{where} were not recorded in the same discharges -- only "
            f"{n_same} of {len(pos_a.shot_nums)} shots line up (A covers shots "
            f"{pos_a.shot_nums.min()}-{pos_a.shot_nums.max()}, B covers "
            f"{pos_b.shot_nums.min()}-{pos_b.shot_nums.max()}). The probes "
            "scanned sequentially, not together, so their cross-spectra would "
            "pair different plasmas. Correlate each probe against a channel on "
            "its own drive, or against a stationary reference.")


def resolve_positions(ifn, channel, motion_group_name=None):
    """The :class:`Jun2026_IV.ProbePositions` of the probe that recorded ``channel``.

    The "which drive is this channel on, and where did it scan" step every batch
    path needs; the resolved drive is on the returned record's ``motion_group``.
    ``motion_group_name`` overrides the port join, for channels whose description
    names no port.
    """
    group = motion_group_name or jiv.motion_group_for_channel(ifn, *channel)
    return jiv.read_lp_positions(ifn, group)


def channel_positions(ifn, scope, chan, motion_group_name=None):
    """The ``(pos_x, pos_y)`` of the probe that recorded one channel.

    The per-channel answer the stored file-level ``pos_x``/``pos_y`` cannot give
    in a two-drive run: co-moving probes can sit on different flux surfaces (1 cm
    apart in y, in one run of this campaign), so a page drawing both channels
    must label each with its own drive's coordinates.  Read from the HDF5 rather
    than stored, so it stays right for npz files written before this was tracked.
    """
    pos = resolve_positions(ifn, (scope, chan), motion_group_name)
    return position_xy(pos.pos_array, pos.npos, pos.nshot)


def batch_xcorr(ifn, ch_a=CH_A, ch_b=CH_B, tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
                nperseg=NPERSEG, motion_group_name=None, t_offset_b_ms=0.0):
    """Per-position ensemble coherence + cross-phase for ONE run -> co-located npz.

    The Smith (1974) FFT cross-spectral estimator: reads each probe position's
    ``nshot`` shots for ``ch_a`` / ``ch_b`` and incoherently averages the Welch
    cross/auto spectra over **only that position's shots** (the tqdm bar ticks per
    shot), then forms that position's ensemble coherence
    ``gamma2 = |<Pxy>|**2 / (<Pxx><Pyy>)`` and cross-phase ``phase = angle(<Pxy>)``
    (:func:`avg_cross_spectrum`). The spatial dimension is kept: results are stored
    per position, indexed the same way as ``positions_array`` (position ``p`` is
    ``pos_array[p*nshot]``), so a plane run can be drawn as an xy map.

    The time-lag cross-correlation is intentionally **not** part of the batch: it
    is the Blackman-Tukey correlogram Smith's method supersedes. It stays available
    for interactive use via :func:`xcorr_averaged` / :func:`xcorr_per_shot`.

    Writes into the run's co-located npz (:func:`xcorr_npz_path`), keyed by the
    channel pair (:func:`_pair_key`): ``freq`` (shared axis),
    ``<pair>__pos_x`` / ``<pair>__pos_y`` (the (x, y) of this pair's moving
    probe -- stored **per pair** because co-moving probes in a two-drive run can
    scan in opposite order or sit on offset grids, as in runs 26, 30, 31),
    ``<pair>__gamma2`` ``(npos, nf)``, ``<pair>__phase`` ``(npos, nf)``,
    ``<pair>__nshots`` ``(npos,)``. Top-level ``pos_x`` / ``pos_y`` are written
    only while every stored pair in the file shares the same grid (within
    :data:`_POS_MATCH_TOL_CM`); once a merged pair disagrees, top-level
    ``pos_x`` / ``pos_y`` are dropped so legacy readers cannot silently read
    another drive's coordinates. An existing npz for the run is **merged** (its
    other pairs are kept), so several pairs accumulate in one file. Returns the
    npz path.
    """
    pos_x, pos_y, npos, positions, motion_group = _iter_run_positions(
        ifn, ch_a, ch_b, tmin_ms, tmax_ms, "xcorr", motion_group_name,
        t_offset_b_ms=t_offset_b_ms)

    # One ensemble result PER position: average that position's shots, keep the
    # per-position gamma2/phase so the spatial (x, y) structure is preserved.
    freq = None
    gamma2 = phase = None   # (npos, nf), filled as positions complete
    nshots = np.zeros(npos, dtype=int)
    for p, sa, sb, dt in positions:
        try:
            f, g2, ph, n_used = avg_cross_spectrum(sa, sb, dt, nperseg=nperseg)
        except ValueError:
            # No finite shot pair at this position: leave its row NaN so one
            # dead position doesn't abort the whole plane.
            continue
        if gamma2 is None:
            freq = f
            gamma2 = np.full((npos, f.size), np.nan)
            phase = np.full((npos, f.size), np.nan)
        gamma2[p] = g2
        phase[p] = ph
        nshots[p] = n_used

    if freq is None or nshots.sum() == 0:
        raise ValueError("no finite shot pairs to correlate")

    # Merge into the run's co-located npz (keep any other pairs already stored).
    # Guard the shared freq axis first: pairs in one file must share it.
    out_path = xcorr_npz_path(ifn)
    key = _pair_key(ch_a, ch_b)
    coords_uniform = True
    backfill_pos = {}
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            if "freq" in d.files and not np.array_equal(freq, d["freq"]):
                raise ValueError(f"{out_path}: freq axis differs from stored pairs "
                                 "(different window / sampling rate)")
            other_keys = sorted({
                k[:-len("__gamma2")] for k in d.files
                if k.endswith("__gamma2") and k[:-len("__gamma2")] != key
            })
            legacy_keys = [
                ok for ok in other_keys
                if f"{ok}__pos_x" not in d.files or f"{ok}__pos_y" not in d.files
            ]
            top_px = d["pos_x"] if "pos_x" in d.files else None
            top_py = d["pos_y"] if "pos_y" in d.files else None
            for ok in other_keys:
                ox = d[f"{ok}__pos_x"] if f"{ok}__pos_x" in d.files else top_px
                oy = d[f"{ok}__pos_y"] if f"{ok}__pos_y" in d.files else top_py
                if (ox is None or oy is None
                        or ox.shape != pos_x.shape or oy.shape != pos_y.shape
                        or not np.allclose(ox, pos_x, atol=_POS_MATCH_TOL_CM, equal_nan=True)
                        or not np.allclose(oy, pos_y, atol=_POS_MATCH_TOL_CM, equal_nan=True)):
                    coords_uniform = False
                    break
            if not coords_uniform and legacy_keys:
                # Dropping top-level pos_x/pos_y would orphan pre-fix pairs that
                # lack per-pair coordinates. One legacy pair's coordinates are
                # unambiguously top_px/top_py and can be backfilled; multiple
                # legacy pairs cannot be disambiguated safely.
                if len(legacy_keys) == 1 and top_px is not None and top_py is not None:
                    ok = legacy_keys[0]
                    backfill_pos[f"{ok}__pos_x"] = top_px
                    backfill_pos[f"{ok}__pos_y"] = top_py
                else:
                    raise ValueError(
                        f"{out_path}: pre-fix file has multiple pairs without "
                        f"per-pair coordinates ({legacy_keys}); regenerate with "
                        "overwrite=True before merging a pair on a different grid")
    payload = {
        "freq": freq,
        # Per-pair coordinates: co-moving probes in a two-drive run can scan in
        # opposite order (runs 26 & 31) or sit on offset grids (run 30), so
        # every pair carries its own (pos_x, pos_y).
        f"{key}__pos_x": pos_x, f"{key}__pos_y": pos_y,
        f"{key}__motion_group": np.str_(motion_group or ""),
        # Per pair, not per file: these are arguments, so two pairs in one npz
        # can legitimately differ in them (the freq guard above is blind to the
        # window -- Welch's axis depends only on nperseg and dt). A page that
        # read the module constants instead would state whatever they say at
        # RENDER time, which no stored pair need have been computed with.
        f"{key}__tmin_ms": np.float64(tmin_ms),
        f"{key}__tmax_ms": np.float64(tmax_ms),
        f"{key}__t_offset_b_ms": np.float64(t_offset_b_ms),
        f"{key}__nperseg": np.int64(nperseg),
        f"{key}__gamma2": gamma2, f"{key}__phase": phase,
        f"{key}__nshots": nshots,
    }
    payload.update(backfill_pos)
    if coords_uniform:
        # Back-compat for single-grid / legacy readers: written only while all
        # pairs in the file agree on (pos_x, pos_y).
        payload["pos_x"] = pos_x
        payload["pos_y"] = pos_y
        drop_keys = ()
    else:
        # Multi-grid file: drop top-level pos_x/pos_y so legacy readers fail
        # loudly rather than silently mislabeling another drive's pair.
        drop_keys = ("pos_x", "pos_y")
    merge_save_npz(out_path, payload, drop_keys=drop_keys)
    print(f"\nWrote {out_path}: pair '{key}', {npos} positions "
          f"({int(nshots.sum())} shots total), {freq.size} freq bins, "
          f"window {tmin_ms}-{tmax_ms} ms")
    return out_path


def _as_pair_spec(spec):
    """Normalize a :class:`PairSpec` or 2/3/4-tuple into a :class:`PairSpec`."""
    if isinstance(spec, PairSpec):
        return spec
    return PairSpec(*spec)


def _pair_settings_mismatch(npz_data, key, tmin_ms, tmax_ms, nperseg,
                            t_offset_b_ms, motion_group_name=None):
    """Return human-readable diffs between a stored pair and requested settings."""
    want = {
        "tmin_ms": float(tmin_ms),
        "tmax_ms": float(tmax_ms),
        "t_offset_b_ms": float(t_offset_b_ms),
        "nperseg": int(nperseg),
    }
    if motion_group_name is not None:
        want["motion_group"] = str(motion_group_name)
    return npz_settings_mismatch(npz_data, key, want)


def xcorr_ifn(data_dir, run_num):
    """Resolve the raw HDF5 for a zero-padded run number (e.g. ``"26"``)."""
    return find_run_hdf5(data_dir, run_num)


def batch_all(data_dir=DATA_DIR, runs=None, tmin_ms=TMIN_MS, tmax_ms=TMAX_MS,
              nperseg=NPERSEG, overwrite=False):
    """Batch every run/pair in :data:`XCORR_RUNS` -> written npz paths.

    ``runs`` may be a subset of run numbers (e.g. ``["26", "31"]``) or a
    custom ``{run_num: [PairSpec(...), ...]}`` mapping.
    When ``overwrite=False``, pairs already stored in a run's npz are skipped
    only after verifying their stored ``tmin_ms``, ``tmax_ms``, ``nperseg``,
    ``t_offset_b_ms``, and (when given) ``motion_group_name`` match the
    request; any mismatch raises :class:`ValueError` up front instead of
    silently keeping stale pairs or failing partway on the ``freq`` guard.
    Pass ``overwrite=True`` to remove an existing npz and regenerate cleanly.
    """
    if runs is None:
        manifest = XCORR_RUNS
    elif isinstance(runs, dict):
        manifest = runs
    else:
        manifest = {r: XCORR_RUNS[r] for r in runs}

    written = []
    for run_num, raw_specs in sorted(manifest.items()):
        specs = [_as_pair_spec(s) for s in raw_specs]
        ifn = xcorr_ifn(data_dir, run_num)
        out_path = xcorr_npz_path(ifn)
        if overwrite and os.path.isfile(out_path):
            os.remove(out_path)

        existing = set()
        if os.path.isfile(out_path):
            existing = set(stored_pair_tuples(out_path))
            with np.load(out_path) as d:
                # Validate every already-stored requested pair BEFORE doing any
                # HDF5 reads, and also guard against adding a new pair whose
                # nperseg disagrees with the existing file's stored pairs.
                for spec in specs:
                    key = _pair_key(spec.ch_a, spec.ch_b)
                    if (spec.ch_a, spec.ch_b) in existing:
                        diffs = _pair_settings_mismatch(
                            d, key, tmin_ms, tmax_ms, nperseg,
                            spec.t_offset_b_ms, spec.motion_group_name)
                        if diffs:
                            raise ValueError(
                                f"{out_path}: stored pair '{key}' differs from "
                                f"requested settings ({'; '.join(diffs)}); "
                                "pass overwrite=True to regenerate")
                if any((s.ch_a, s.ch_b) not in existing for s in specs):
                    stored_np = {
                        int(d[k]) for k in d.files if k.endswith("__nperseg")
                    }
                    if stored_np and stored_np != {int(nperseg)}:
                        raise ValueError(
                            f"{out_path}: existing pairs use nperseg={sorted(stored_np)} "
                            f"but requested nperseg={int(nperseg)} (freq axis would mismatch); "
                            "pass overwrite=True to regenerate")

        for spec in specs:
            if (spec.ch_a, spec.ch_b) in existing:
                print(f"  run {run_num}: pair {_pair_key(spec.ch_a, spec.ch_b)} exists, skipping")
                continue
            batch_xcorr(ifn, ch_a=spec.ch_a, ch_b=spec.ch_b,
                        tmin_ms=tmin_ms, tmax_ms=tmax_ms,
                        nperseg=nperseg,
                        motion_group_name=spec.motion_group_name,
                        t_offset_b_ms=spec.t_offset_b_ms)
        written.append(out_path)
    return written


if __name__ == "__main__":

    batch_xcorr(IFN)
