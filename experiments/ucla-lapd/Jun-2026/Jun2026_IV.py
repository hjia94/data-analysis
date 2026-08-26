"""Jun-2026 LAPD Langmuir-probe sweep analysis -- batch processing.

Same analysis workflow as ``Mar-2026/Mar2026_IV.py`` (sweep detection ->
reshape -> smoothing -> per-trace ``analyze_IV`` -> shot-averaged Vp/Te/ne),
but the data is read from the **new LAPD_DAQ (pydaq) HDF5 format** instead of
the bapsflib/C-translator format.

This module is the **processing** half: read raw HDF5 -> reshape/smooth the
sweeps -> batch ``analyze_IV`` -> save the reshaped sweeps and the shot-averaged
Vp/Te/ne to ``.npz``.  It draws no figures.  Plotting lives in
``Jun2026_plot.py`` (the shared Jun-2026 figure module), which reads those
``.npz`` back; keeping the two apart means the interactive ``plt.show`` path
never runs inside the batch loop.

Reading differences vs Mar-2026
-------------------------------
* Mar-2026 used ``open_lapd(ifn).session()`` and read by ``(board, channel)``
  from a SIS crate.  Jun-2026 files are LAPD_DAQ pydaq files: channels live in
  named *scope groups* and are read by scope-channel name (``'C2'`` ...) via
  ``run.channel(name, scope_name=...)``.
* Probe positions: the unified ``run.positions()`` expects a ``motion_list``
  dataset that these files do not have, so it returns ``None``.  We read the
  positions directly from ``Control/Positions/<group>/positions_setup_array``
  (the planned unique positions) plus ``positions_array`` (every shot) instead
  -- see :func:`read_lp_positions`.
* These runs are a 1-D *line* scan (y == 0, x swept), not the xy-plane of
  Mar-2026, so the 2-D ``imshow`` map is not meaningful; the center-line plot
  (``Jun2026_plot.plot_iv_line``) is the primary output.

Which channel is I and which is V
---------------------------------
By default the I and V channels are identified from each channel's own HDF5
``description`` attribute only (e.g. ``'I, LP@P29-R'`` / ``'V, LP@P29-R'``);
the experiment / run description prose is never parsed for this.  We pick the
first probe tip that has a *complete* I+V pair and flag any tip missing a
channel.  **You can override the selection** in the clearly marked block at the
top of the file (``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN``) -- that always wins.

Voltage scaling: on the LAPD_DAQ system the scope auto-detects the probe
attenuation (HV divider) and folds it into the LeCroy ``vertical_gain`` in the
header, so the volts returned by ``run.channel`` are already the true probe
voltage -- there is no separate divider factor to apply here (unlike Mar-2026's
hand-applied ``x100`` for the SIS digitizer, which did not capture it).

Two probe wiring types appear in these runs: a swept Langmuir tip (a complete
I+V pair) and a fixed-bias saturation-current tip (an ``I`` channel only).  The
``I``-only tips are not Langmuir sweeps and are simply not paired here.  Note
also some early runs lost scope channel 1 to a LAPD_DAQ bug, so a tip's current
channel may be absent; such tips are flagged and skipped, but this pattern is
never assumed -- only complete pairs that are actually present are analyzed.

Calibration: only ``RESISTOR`` and ``Aprobe`` matter for the current scaling
(plus ``I_SIGN`` to orient the trace).  Absolute density is calibrated against
the interferometer downstream, so these set only the first-order trend, not a
final ne.
"""

import os
import re
from typing import NamedTuple

import numpy as np
import h5py

from data_analysis.io import (open_lapd, choose_from_list, position_shots,
                              motion_group_for_channel)
from data_analysis.io.scope_reader import read_scope_channel_descriptions
from data_analysis.plasma.langmuir import (
    analyze_IV_safe, prepare_sweep_data, process_iv_and_save, sweep_npz_paths,
    calibrate_plasma_npz,
)
from data_analysis.utils import run_num_of

# ========================================================================== #
#  >>> USER OVERRIDE: set which scope / channels are the LP I and V <<<
#
#  Leave these None to auto-detect from the channel descriptions (recommended).
#  Set them to force a specific mapping -- this ALWAYS takes precedence over
#  auto-detection.  Example:  SCOPE_NAME = "lpscope"; I_CHAN = "C3"; V_CHAN = "C4"
# ========================================================================== #
SCOPE_NAME = None    # e.g. "lpscope" / "scope"; None -> auto-detect the LP scope group
I_CHAN = None        # e.g. "C3"; None -> auto-detect the current channel
V_CHAN = None        # e.g. "C4"; None -> auto-detect the voltage channel

# --------------------------------------------------------------------------- #
# Calibration knobs (current scaling only; voltage is already true probe volts).
# Density is calibrated against the interferometer downstream, so these set the
# first-order trend, not a final ne -- set them to your probe's values.
# --------------------------------------------------------------------------- #
Aprobe = 2e-3        # probe collection area, cm^2
RESISTOR = 25.0      # current shunt resistor, ohm (volts on I-channel -> current)
I_SIGN = -1           # +1 / -1 to orient current (electron current positive at high V)

LP_SCOPE_CANDIDATES = ("lpscope", "scope")  # scope-group names that may hold the LP IV
#===============================================================================================================================================


def find_lp_scope(fn):
    """Return the name of the scope group holding the Langmuir-probe IV channels.

    Tries the known LP scope-group names (``lpscope``, then ``scope``) and picks
    the first one present whose channel descriptions mention a Langmuir probe
    (an ``I,``/``V,`` quantity).  Raises ``ValueError`` if none is found so the
    caller fails loudly rather than analyzing the wrong scope.
    """
    with h5py.File(fn, "r") as f:
        present = list(f.keys())
        for name in LP_SCOPE_CANDIDATES:
            if name in f:
                desc = read_scope_channel_descriptions(f, name)
                if any(_parse_channel_desc(d)[0] in ("I", "V") for d in desc.values()):
                    return name
    raise ValueError(
        f"No Langmuir-probe scope group found in {fn!r}. "
        f"Top-level groups present: {present}. "
        f"Looked for: {LP_SCOPE_CANDIDATES}."
    )


def _parse_channel_desc(desc):
    """Parse a channel description into ``(quantity, tip)``.

    ``quantity`` is ``'I'`` or ``'V'`` (or ``None`` if not an LP IV channel);
    ``tip`` is an uppercased tip label such as ``'L'`` / ``'R'`` (or ``None``).

    Examples
    --------
    ``'I, LP@P29-R'`` -> ``('I', 'R')``
    ``'V, LP@P29-L'`` -> ``('V', 'L')``
    ``'I@P33, R'``    -> ``('I', 'R')``
    ``"Current on 2''"`` -> ``(None, None)`` (not an LP IV channel)
    """
    if not desc:
        return None, None
    text = str(desc).strip()

    # Quantity: a leading 'I' or 'V' token (handles 'I,' / 'I@' / 'I ').
    m = re.match(r"\s*([IV])\b", text)
    quantity = m.group(1) if m else None

    # Tip label: the last short alphanumeric token after a '-' or ',' (e.g. 'R',
    # 'L', 'X+', 'Y-').  We only need it to pair I with V for the same tip.
    tip = None
    tip_match = re.search(r"[-,]\s*([A-Za-z][A-Za-z0-9+\-]{0,3})\s*$", text)
    if tip_match:
        tip = tip_match.group(1).upper()
    return quantity, tip


def discover_lp_channels(fn, scope_name):
    """Identify the I and V scope channels for each probe tip from descriptions.

    Reads the scope group's channel descriptions and groups channels by tip,
    splitting each into its current (``I``) and voltage (``V``) channel.

    Returns
    -------
    pairs : dict[str, dict]
        ``{tip: {'I': chan_name, 'V': chan_name}}`` for every tip that has a
        **complete** I+V pair (these are the tips we can analyze).
    incomplete : dict[str, dict]
        Same shape, for tips missing either I or V (flagged, not analyzed).

    The pairing is data-driven: nothing about which channel is I vs V (or which
    tip) is hardcoded.  A tip missing its current channel -- e.g. the LAPD_DAQ
    "dropped C1" runs -- simply lands in ``incomplete`` and is flagged.
    """
    with h5py.File(fn, "r") as f:
        desc = read_scope_channel_descriptions(f, scope_name)

    tips = {}
    print(f"\nLP scope '{scope_name}' channel descriptions:")
    for chan in sorted(desc):
        d = desc[chan]
        quantity, tip = _parse_channel_desc(d)
        flag = "" if quantity in ("I", "V") else "   <- not an LP IV channel"
        print(f"  {chan}: {d!r}  -> quantity={quantity}, tip={tip}{flag}")
        if quantity in ("I", "V") and tip is not None:
            tips.setdefault(tip, {})[quantity] = chan

    pairs, incomplete = {}, {}
    for tip, chans in tips.items():
        if "I" in chans and "V" in chans:
            pairs[tip] = chans
        else:
            incomplete[tip] = chans

    if pairs:
        print("\nComplete I+V tips (will be analyzed):")
        for tip, chans in pairs.items():
            print(f"  tip {tip}: I={chans['I']}, V={chans['V']}")
    if incomplete:
        print("\n*** FLAG: tips missing a channel (NOT analyzed) ***")
        for tip, chans in incomplete.items():
            have = ", ".join(f"{q}={c}" for q, c in chans.items())
            missing = "I" if "I" not in chans else "V"
            print(f"  tip {tip}: have [{have}], MISSING {missing} "
                  f"(e.g. LAPD_DAQ dropped-C1 bug)")
    if not pairs:
        raise ValueError(
            f"No complete I+V channel pair found in scope '{scope_name}'. "
            "Cannot run Langmuir analysis on this file.")
    return pairs, incomplete


class ProbePositions(NamedTuple):
    """One motion group's scan grid and the shot numbers it was measured at.

    ``shot_nums`` is the ground truth tying positions to discharges: entry
    ``p*nshot + k`` is the shot number of position ``p``'s k-th shot. Pass it to
    :func:`data_analysis.io.position_shots` rather than computing row offsets --
    see that function for why.
    """
    pos_array: np.ndarray   # live rows only (padding filtered)
    xpos: np.ndarray        # unique planned x axis
    ypos: np.ndarray        # unique planned y axis
    npos: int
    nshot: int
    shot_nums: np.ndarray   # this group's shot numbers, position-major
    motion_group: str       # which drive these coordinates describe


def _describe_motion_group(mg):
    """One-line summary of a motion group's grid, for the ambiguity error."""
    setup = mg["positions_setup_array"][:]
    x, y = setup["x"], setup["y"]
    ys = np.unique(np.round(y, 3))
    y_txt = f"{ys[0]:g}" if ys.size == 1 else f"{ys.min():g}..{ys.max():g}"
    return (f"{len(setup)} positions, x {x.min():.2f}..{x.max():.2f}, "
            f"y {y_txt}")


def read_lp_positions(fn, motion_group_name=None, interactive=False):
    """Read line-scan probe positions from a LAPD_DAQ pydaq file.

    The unified ``run.positions()`` needs a ``motion_list`` dataset these files
    lack, so we read ``Control/Positions/<group>`` directly:

    * ``positions_setup_array`` -- the planned unique (x, y) positions.
    * ``positions_array``       -- the (x, y) actually visited for every shot;
      used to count shots-per-position.

    Returns a :class:`ProbePositions`; ``xpos``/``ypos`` are the unique sorted
    axes (for a line scan one of them is a single value) and ``shot_nums`` is the
    shot number of every shot this group owns, position-major.

    ``pos_array`` holds only this group's **live** rows. Where two probes scan
    sequentially into one file, each group's array is padded with ``shot_num==0``
    for the shots the *other* probe owns; those rows sit at the null coordinate,
    so leaving them in makes the leading-run ``nshot`` heuristic below count
    padding as one long first position (measured: 4805 instead of 5 for the
    second probe of a two-phase plane).

    With more than one motion group the caller must say which one it means:
    ``motion_group_name``, or ``interactive=True`` for the notebook picker.
    Defaulting would silently stamp one probe's coordinates onto the other's
    signals -- shapes and x values match, so nothing errors.
    """
    with h5py.File(fn, "r") as f:
        if "Control/Positions" not in f:
            raise ValueError(f"No Control/Positions group in {fn!r}.")
        pos_root = f["Control/Positions"]
        groups = list(pos_root.keys())
        if motion_group_name is None:
            if len(groups) == 1:
                motion_group_name = groups[0]
            elif interactive:
                motion_group_name = choose_from_list(
                    groups, label=lambda g: f'"{g}"', prompt="Motion group index",
                    header="Multiple motion groups; choose one:")
            else:
                listing = "\n".join(
                    f"  {g!r}  ({_describe_motion_group(pos_root[g])})"
                    for g in groups)
                raise ValueError(
                    f"{os.path.basename(fn)} has {len(groups)} motion groups; "
                    f"name one explicitly via motion_group_name=... "
                    f"(or interactive=True to pick):\n{listing}")
        if motion_group_name not in pos_root:
            raise ValueError(
                f"Motion group {motion_group_name!r} not in {os.path.basename(fn)}; "
                f"available: {groups}")
        mg = pos_root[motion_group_name]
        print(f"Using motion group: {motion_group_name!r}")

        raw = mg["positions_array"][:]
        setup = mg["positions_setup_array"][:]  # planned unique positions

    # shot_num == 0 is the pydaq padding convention (verified campaign-wide: the
    # null coordinate never collides with a real measured position). Reads select
    # by shot NUMBER, so the live rows need not be contiguous or start at row 0.
    pos_array = raw[raw["shot_num"] != 0]
    if pos_array.size == 0:
        raise ValueError(
            f"Motion group {motion_group_name!r} in {os.path.basename(fn)} has no "
            "live rows (every shot_num == 0).")
    if pos_array.size != len(raw):
        print(f"  Padding: kept {pos_array.size} live rows of {len(raw)} "
              f"(shots {pos_array['shot_num'].min()}-{pos_array['shot_num'].max()}).")

    npos = len(setup)
    xpos = np.unique(np.round(setup["x"], 3))
    ypos = np.unique(np.round(setup["y"], 3))

    # nshot = number of leading shots at the first position (first row where the
    # position changes; whole run if it never does).
    x = np.round(pos_array["x"], 2)
    y = np.round(pos_array["y"], 2)
    same = (x == x[0]) & (y == y[0])
    nshot = len(same) if same.all() else int(np.argmin(same))

    print(f"Positions: {npos} unique (x: {len(xpos)}, y: {len(ypos)}), "
          f"{nshot} shots/position, {len(pos_array)} live shots.")
    if npos * nshot != len(pos_array):
        print(f"  *** FLAG: npos*nshot ({npos*nshot}) != live shots "
              f"({len(pos_array)}); positions may be irregular. ***")
    return ProbePositions(pos_array, xpos, ypos, npos, nshot,
                          pos_array["shot_num"], motion_group_name)


def _read_reshaped(run, scope_name, I_chan, V_chan, pos, pos_index=None):
    """Read one probe tip's V and I into ``(npos, nshot, nsamples)`` arrays.

    Shared core for :func:`get_IV_arr` and :func:`get_IV_at_position`.  ``pos`` is
    the tip's :class:`ProbePositions`.  Voltage is left as-is (LAPD_DAQ folds the
    probe attenuation into the scope ``vertical_gain``, so this is already true
    probe volts); current is scaled to current density via
    ``I_SIGN``/``RESISTOR``/``Aprobe``.  Returns ``(tarr, V3d, I3d)``, both 3-D.

    Both paths select by recorded shot number
    (:func:`data_analysis.io.position_shots`), so the whole-run read and the
    single-position read return the same rows for the same position.

    * ``pos_index=None`` -> the whole run's ``npos*nshot`` shots, reshaped to
      ``(npos, nshot, nsamples)``.
    * ``pos_index=k``    -> only that position's ``nshot`` shots read off disk,
      reshaped to ``(1, nshot, nsamples)``.
    """
    npos, nshot = pos.npos, pos.nshot
    if pos_index is None:
        shots = [int(s) for s in pos.shot_nums[:npos * nshot]]
        out_npos = npos
    else:
        shots = position_shots(pos.shot_nums, pos_index, nshot)
        out_npos = 1

    Vstack, tarr = run.channel(V_chan, scope_name=scope_name, shots=shots)
    Istack, _ = run.channel(I_chan, scope_name=scope_name, shots=shots)

    V3d = Vstack.reshape((out_npos, nshot, -1))
    I3d = Istack.reshape((out_npos, nshot, -1)) * (I_SIGN / (RESISTOR * Aprobe))
    return tarr, V3d, I3d


def get_IV_at_position(run, scope_name, I_chan, V_chan, pos, pos_index,
                       shot_index=None):
    """Read scaled I and V for ONE probe position (for notebook inspection).

    A single-position view of :func:`_read_reshaped` -- handy for eyeballing a
    trace before committing to the batch pass.  ``pos`` is the tip's
    :class:`ProbePositions`.  Only that position's ``nshot`` shots are read off
    disk (not the whole run), so inspecting one position is cheap.

    * ``shot_index=None`` -> shot-averaged V and all per-shot I for that position:
      ``Vpos`` is ``(nsamples,)`` (mean over shots), ``Ipos`` is ``(nshot, nsamples)``.
    * ``shot_index=k``    -> a single shot: both ``(nsamples,)``.
    """
    tarr, V3d, I3d = _read_reshaped(run, scope_name, I_chan, V_chan, pos,
                                    pos_index=pos_index)
    Vpos, Ipos = V3d[0], I3d[0]                            # (nshot, nsamples)

    if shot_index is None:
        return tarr, np.mean(Vpos, axis=0), Ipos
    return tarr, Vpos[shot_index], Ipos[shot_index]


def get_IV_arr(run, scope_name, I_chan, V_chan, pos):
    """Read the V and I sweeps for one probe tip for the whole run.

    LAPD_DAQ equivalent of Mar-2026's ``get_IV_arr``.  Returns ``(tarr, Vsweep,
    Isweep)`` where ``Vsweep`` is shot-averaged ``(npos, nsamples)`` and
    ``Isweep`` keeps per-shot resolution ``(npos, nshot, nsamples)``.
    """
    tarr, V3d, Isweep = _read_reshaped(run, scope_name, I_chan, V_chan, pos)
    Vsweep = np.mean(V3d, axis=1)                          # (npos, nsamples)
    return tarr, Vsweep, Isweep


def analyze_tip_at_position(run, scope_name, I_chan, V_chan, pos,
                            pos_index, sweep_idx, **prep_kwargs):
    """Analyze one tip at one position/sweep: the batch pipeline
    (:func:`prepare_sweep_data`) on that position's shots, then
    ``analyze_IV_safe`` on the shot-averaged sweep.  ``prep_kwargs``
    (``padding`` / ``trim_percent`` / ``smooth_sigma``) forward to
    ``prepare_sweep_data``, which owns the defaults -- so inspection and batch
    cannot drift apart.  Returns ``(Vp, Te, ne)`` (NaNs on failure).
    """
    tarr, Vpos, Ipos = get_IV_at_position(run, scope_name, I_chan, V_chan, pos,
                                          pos_index)
    V_rs, I_rs, *_ = prepare_sweep_data(tarr, Vpos[None, :], Ipos[None, :, :],
                                        **prep_kwargs)
    I_trace = I_rs[0, :, sweep_idx, :].mean(0)
    return analyze_IV_safe(V_rs[0, sweep_idx], I_trace)


def resolve_iv_channel_map(ifn):
    """Every analyzable tip's channels: ``{tip: (scope_name, I_chan, V_chan)}``.

    Honors the top-of-file ``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN`` override
    first (which collapses the map to ``{None: ...}`` -- the untagged single-tip
    case); otherwise auto-detects every complete I+V pair from the channel
    descriptions.  The single owner of the override-vs-discover decision, shared
    by :func:`resolve_iv_channels` and :func:`process_run`.
    """
    scope_name = SCOPE_NAME if SCOPE_NAME is not None else find_lp_scope(ifn)

    if I_CHAN is not None and V_CHAN is not None:
        print(f"\nUsing USER-OVERRIDE channels: I={I_CHAN}, V={V_CHAN} "
              f"(scope '{scope_name}')")
        return {None: (scope_name, I_CHAN, V_CHAN)}
    if (I_CHAN is None) != (V_CHAN is None):
        raise ValueError("Set BOTH I_CHAN and V_CHAN to override, or leave both None.")

    pairs, _ = discover_lp_channels(ifn, scope_name)
    return {tip: (scope_name, chans["I"], chans["V"]) for tip, chans in pairs.items()}


def resolve_iv_channels(ifn, tip=None):
    """Decide which scope group and I/V channels to use for ONE tip.

    Honors the top-of-file ``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN`` override
    first; else picks ``tip`` (default: the first complete I+V pair) from
    :func:`resolve_iv_channel_map`.  Returns ``(scope_name, I_chan, V_chan)``.
    """
    channel_map = resolve_iv_channel_map(ifn)
    if None in channel_map:                      # override always wins
        return channel_map[None]
    if tip is None:
        tip = next(iter(channel_map))
    if tip not in channel_map:
        raise ValueError(f"Tip {tip!r} has no complete I+V pair; available: {list(channel_map)}")
    scope_name, I_chan, V_chan = channel_map[tip]
    print(f"\nAuto-selected tip {tip}: I={I_chan}, V={V_chan} (scope '{scope_name}')")
    return scope_name, I_chan, V_chan


def save_IV_data(ifn, save_path, tip=None, run=None, channels=None, positions=None,
                 motion_group_name=None):
    """Detect sweeps, reshape + smooth the IV traces, and save to ``.npz``.

    Same workflow as Mar-2026's ``save_IV_data`` but for the pydaq format and
    for a single tip.  Channels come from :func:`resolve_iv_channels` (top-of-file
    override, else auto-detected first complete I+V pair).

    ``run`` / ``channels`` / ``positions`` let a caller that already opened the
    file, resolved ``(scope_name, I_chan, V_chan)``, and read a
    :class:`ProbePositions` -- :func:`process_run` -- pass them in instead of
    re-running discovery for every tip; each ``None`` is resolved here (the
    standalone / notebook path).

    Returns ``(Vswp_arr_rs, Iswp_arr_rs)`` -- the reshaped sweep arrays just
    saved -- so the caller can feed :func:`data_analysis.plasma.langmuir.process_iv_and_save` directly instead of
    re-loading the multi-GB npz it just wrote.
    """
    if run is None:
        run = open_lapd(ifn)
        print(f"backend: {run.backend}")

    scope_name, I_chan, V_chan = (channels if channels is not None
                                  else resolve_iv_channels(ifn, tip=tip))

    pos = positions if positions is not None else read_lp_positions(
        ifn, motion_group_name)

    tarr, Vswp_arr, Iswp_arr = get_IV_arr(run, scope_name, I_chan, V_chan, pos)

    # Sweep detection -> reshape -> smoothing (shared batch pipeline).
    Vswp_arr_rs, Iswp_arr_rs, data_timestamp, sweep_t_start, sweep_t_stop = \
        prepare_sweep_data(tarr, Vswp_arr, Iswp_arr)

    # motion_group: which drive xpos/ypos describe. In a two-drive run the other
    # tip's sweeps sit in a sibling npz with different coordinates, so a reader
    # that only has the file cannot otherwise tell which probe it is holding.
    np.savez(save_path, Vswp_arr_rs=Vswp_arr_rs, Iswp_arr_rs=Iswp_arr_rs,
             data_timestamp=data_timestamp, sweep_t_start=sweep_t_start,
             sweep_t_stop=sweep_t_stop, xpos=pos.xpos, ypos=pos.ypos,
             npos=pos.npos, nshot=pos.nshot, I_chan=I_chan, V_chan=V_chan,
             motion_group=np.str_(pos.motion_group or ""))
    print(f"Saved to: {save_path}")
    return Vswp_arr_rs, Iswp_arr_rs


# Batch analysis + the npz loaders (load_sweep_data / load_sweep_axes /
# load_plasma_data) live in data_analysis.plasma.langmuir now; Jun2026_plot
# imports them from there directly.


def process_run(ifn, motion_group_name=None, calibrated=True):
    """Run the full batch pipeline for **every complete-pair tip** in a run.

    For each tip with a complete I+V pair (from :func:`discover_lp_channels`):
    sweep-detect + reshape + smooth (:func:`save_IV_data`) -> batch
    ``analyze_IV`` over all positions/shots (:func:`data_analysis.plasma.langmuir.process_iv_and_save`).  Outputs
    are saved **per tip** (filenames carry ``-tip<T>``) so the two probes never
    mix; an override (``I_CHAN``/``V_CHAN``) collapses to the single overridden
    tip.  Tips missing a channel are flagged and skipped (their results stay
    absent, never filled from the other probe).

    Positions are resolved **per tip**, not once per run: in a two-drive run each
    tip can sit on a different probe, and one shared position array would stamp
    one drive's coordinates onto the other's sweeps
    (:func:`data_analysis.io.motion_group_for_channel`).  ``motion_group_name``
    forces one group for every tip, for files whose channels name no port.

    No figures are drawn here -- plot from the saved ``.npz`` with
    ``Jun2026_plot.plot_iv_line_run``.  Returns ``{tip: (sweep_path, plasma_path)}``.

    Current orientation comes from the module-level ``I_SIGN`` (``-1`` for this
    experiment); change it at the top of the file if a run needs the other sign.

    ``calibrated`` True (default) saves the [A/cm^2] proxy for
    :func:`calibrate_plasma_npz` to scale; False saves a Te-based density
    (:func:`data_analysis.plasma.langmuir.ne_from_esat`) for runs with no
    interferometer, which must NOT then be calibrated.
    """
    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)

    # Tip-invariant work is done ONCE: the open handle and the scope/channel map
    # (a None tip = the override, flowing through sweep_npz_paths/save_IV_data as
    # the untagged single-tip case).  Positions are NOT tip-invariant -- see the
    # docstring -- so they are read inside the loop.
    run = open_lapd(ifn)
    print(f"backend: {run.backend}")
    channel_map = resolve_iv_channel_map(ifn)

    results = {}
    for tip, channels in channel_map.items():
        print("\n" + "=" * 70)
        print(f"PROCESSING tip {tip if tip is not None else 'override'}")
        print("=" * 70)

        scope_name, I_chan, _ = channels
        group = motion_group_name or motion_group_for_channel(ifn, scope_name, I_chan)

        sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num, tip)

        Vswp_arr_rs, Iswp_arr_rs = save_IV_data(
            ifn, sweep_path, tip=tip, run=run, channels=channels,
            positions=read_lp_positions(ifn, group))
        process_iv_and_save(Vswp_arr_rs, Iswp_arr_rs, plasma_path,
                            calibrated=calibrated)

        results[tip if tip is not None else "override"] = (sweep_path, plasma_path)

    print(f"\nDone. Processed tips: {list(results)}")
    return results
#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':

    ifn = r"D:\data\LAPD\jun2026-jia\06-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"

    # False for a run with no merged interferometer data: ne comes from
    # ne_from_esat(I_esat, Te) instead, and the calibration step is skipped.
    CALIBRATED = True

    # Batch-process every complete-pair tip and save the .npz results.  Draw the
    # figures afterwards from the saved .npz with Jun2026_plot.plot_iv_line_run(ifn).
    results = process_run(ifn, calibrated=CALIBRATED)

    # Calibrate each processed tip's ne against the interferometer chord
    INTERF_CHAN = "phase_p29"
    T_OFFSET = 0.012
    if CALIBRATED:
        for tip in results:
            calibrate_plasma_npz(ifn, INTERF_CHAN,
                                 tip=None if tip == "override" else tip,
                                 t_offset=T_OFFSET)
    else:
        print("\nCALIBRATED=False: ne is a Te-based density [cm^-3]; "
              "skipping interferometer calibration.")

