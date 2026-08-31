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
    analyze_IV_safe, prepare_sweep_data, process_sweep_run, SweepRecord,
    channel_prefix, match_tip, OVERRIDE_PREFIX,
)

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

# None = calibrate iff the run carries interferometer data (the usual case).
# True forces it and RAISES on a run with none, rather than quietly storing an
# uncalibrated density under a calibrated name; False stores the uncalibrated
# density.  Either way ne is [cm^-3] -- the npz records which it is.
CALIBRATE = None

# Cast for the stored sweep arrays only; analysis always runs in the source
# precision.  float32 halves the npz (these are smoothed ~12-14-bit digitizer
# values, so its 24-bit mantissa holds everything physically real); None keeps
# float64 and the files stay ~2.4x larger.
SWEEP_STORE_DTYPE = np.float32

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
    """Parse a channel description into ``(quantity, port, tip)``.

    ``quantity`` is ``'I'`` or ``'V'`` (``None`` if not an LP IV channel);
    ``port`` is the ``@PNN`` port number as a string (``None`` if unstated);
    ``tip`` is an uppercased tip label such as ``'L'`` / ``'R'`` (or ``None``).

    Examples
    --------
    ``'I, LP@P29-R'`` -> ``('I', '29', 'R')``
    ``'V, LP@P29-L'`` -> ``('V', '29', 'L')``
    ``'I@P33, R'``    -> ``('I', '33', 'R')``
    ``"Current on 2''"`` -> ``(None, None, None)`` (not an LP IV channel)

    The port is what makes a tip unique: two probes each carrying an L and an R
    give four channels but only two distinct tip labels, so keying by tip alone
    silently pairs one probe's I with the other's V.
    """
    if not desc:
        return None, None, None
    text = str(desc).strip()

    # Quantity: a leading 'I' or 'V' token (handles 'I,' / 'I@' / 'I ').
    m = re.match(r"\s*([IV])\b", text)
    quantity = m.group(1) if m else None

    # Port: the '@P<digits>' token, wherever it sits ('LP@P29-R', 'I@P33, R').
    port_match = re.search(r"@\s*P(\d+)", text, re.IGNORECASE)
    port = port_match.group(1) if port_match else None

    # Tip label: the last short alphanumeric token after a '-' or ',' (e.g. 'R',
    # 'L', 'X+', 'Y-').  We only need it to pair I with V for the same tip.
    tip = None
    tip_match = re.search(r"[-,]\s*([A-Za-z][A-Za-z0-9+\-]{0,3})\s*$", text)
    if tip_match:
        tip = tip_match.group(1).upper()
    return quantity, port, tip




def discover_lp_channels(fn, scope_name):
    """Identify the I and V scope channels for each probe tip from descriptions.

    Reads the scope group's channel descriptions and groups channels by
    ``(port, tip)``, splitting each into its current (``I``) and voltage (``V``)
    channel.

    Returns
    -------
    pairs : dict[str, dict]
        ``{key: {'I': chan, 'V': chan, 'port': port, 'tip': tip}}`` for every
        tip with a **complete** I+V pair, keyed by :func:`data_analysis.plasma.langmuir.channel_prefix`.
    incomplete : dict[str, dict]
        Same shape, for tips missing either I or V (flagged, not analyzed).

    The pairing is data-driven: nothing about which channel is I vs V (or which
    tip) is hardcoded.  A tip missing its current channel -- e.g. the LAPD_DAQ
    "dropped C1" runs -- simply lands in ``incomplete`` and is flagged.

    Raises ``ValueError`` if two channels claim the same quantity for one
    ``(port, tip)``: that is an ambiguity in the file, and picking either would
    silently pair one probe tip's current with another's voltage.
    """
    with h5py.File(fn, "r") as f:
        desc = read_scope_channel_descriptions(f, scope_name)

    tips = {}
    print(f"\nLP scope '{scope_name}' channel descriptions:")
    for chan in sorted(desc):
        d = desc[chan]
        quantity, port, tip = _parse_channel_desc(d)
        flag = "" if quantity in ("I", "V") else "   <- not an LP IV channel"
        print(f"  {chan}: {d!r}  -> quantity={quantity}, port={port}, "
              f"tip={tip}{flag}")
        if quantity not in ("I", "V") or tip is None:
            continue
        key = channel_prefix(port, tip)
        entry = tips.setdefault(key, {"port": port, "tip": tip})
        if quantity in entry:
            raise ValueError(
                f"Two channels both describe {quantity} for {key} in scope "
                f"'{scope_name}': {entry[quantity]} "
                f"({desc[entry[quantity]]!r}) and {chan} ({d!r}). Cannot tell "
                "which probe tip each belongs to -- fix the channel "
                "descriptions, or set the I_CHAN/V_CHAN override to analyze "
                "one pair explicitly.")
        entry[quantity] = chan

    pairs, incomplete = {}, {}
    for key, chans in tips.items():
        (pairs if "I" in chans and "V" in chans else incomplete)[key] = chans

    if pairs:
        print("\nComplete I+V tips (will be analyzed):")
        for key, chans in pairs.items():
            print(f"  {key}: I={chans['I']}, V={chans['V']}")
    if incomplete:
        print("\n*** FLAG: tips missing a channel (NOT analyzed) ***")
        for key, chans in incomplete.items():
            have = ", ".join(f"{q}={chans[q]}" for q in ("I", "V") if q in chans)
            missing = "I" if "I" not in chans else "V"
            print(f"  {key}: have [{have}], MISSING {missing} "
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

    With more than one motion group the caller must say which one it means:
    ``motion_group_name``, or ``interactive=True`` for the notebook picker.
    ``pos_array`` holds only the selected motion group's **live** rows.
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

    Both paths select by recorded shot number
    (:func:`data_analysis.io.position_shots`), so the whole-run read and the
    single-position read return the same rows for the same position.

    * ``pos_index=None`` -> the whole run's ``npos*nshot`` shots, reshaped to ``(npos, nshot, nsamples)``.
    * ``pos_index=k``    -> only that position's ``nshot`` shots, reshaped to ``(1, nshot, nsamples)``.
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


class IVChannels(NamedTuple):
    """One tip's scope channels plus the port that identifies which probe it is.

    ``port`` is ``None`` when the descriptions name none (single-probe runs, and
    the user override); the interferometer chord is derived from it, so a run
    whose port is unknown must be given a chord explicitly.
    """
    scope_name: str
    I_chan: str
    V_chan: str
    port: str | None


def resolve_iv_channel_map(ifn):
    """Every analyzable tip's channels: ``{key: IVChannels}``.

    Keys come from :func:`data_analysis.plasma.langmuir.channel_prefix`
    (``'P29-R'``, or ``'R'`` when the
    descriptions name no port), so two probes carrying the same tip label stay
    distinct.

    Honors the top-of-file ``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN`` override
    first (which collapses the map to ``{OVERRIDE_PREFIX: ...}`` -- one channel
    naming neither tip nor port); otherwise auto-detects every complete I+V pair
    from the channel descriptions.  The single owner of the override-vs-discover
    decision, shared by :func:`resolve_iv_channels` and :func:`process_run`.
    """
    scope_name = SCOPE_NAME if SCOPE_NAME is not None else find_lp_scope(ifn)

    if I_CHAN is not None and V_CHAN is not None:
        print(f"\nUsing USER-OVERRIDE channels: I={I_CHAN}, V={V_CHAN} "
              f"(scope '{scope_name}')")
        return {OVERRIDE_PREFIX: IVChannels(scope_name, I_CHAN, V_CHAN, None)}
    if (I_CHAN is None) != (V_CHAN is None):
        raise ValueError("Set BOTH I_CHAN and V_CHAN to override, or leave both None.")

    pairs, _ = discover_lp_channels(ifn, scope_name)
    return {key: IVChannels(scope_name, c["I"], c["V"], c["port"])
            for key, c in pairs.items()}


def resolve_iv_channels(ifn, tip=None):
    """Decide which scope group and I/V channels to use for ONE tip.

    Honors the top-of-file ``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN`` override
    first; else picks ``tip`` (default: the first complete I+V pair) from
    :func:`resolve_iv_channel_map`.  Returns an :class:`IVChannels`.

    ``tip`` is a channel label (``'P29-R'``); a bare tip label (``'R'``)
    is accepted when it matches exactly one channel, and raises when a run
    carries that label on more than one probe.
    """
    channel_map = resolve_iv_channel_map(ifn)
    if OVERRIDE_PREFIX in channel_map:              # override always wins
        return channel_map[OVERRIDE_PREFIX]
    if tip is None:
        tip = next(iter(channel_map))
    else:
        tip = match_tip(channel_map, tip, ifn)
    chans = channel_map[tip]
    print(f"\nAuto-selected tip {tip}: I={chans.I_chan}, V={chans.V_chan} "
          f"(scope '{chans.scope_name}')")
    return chans


def read_iv_record(ifn, key, chans, run=None, positions=None,
                   motion_group_name=None):
    """Read one probe tip's raw sweep data into a :class:`SweepRecord`.

    The Jun-2026 half of the adapter: pydaq scope channels -> the units the
    shared driver expects (volts, [A/cm^2] with ``I_SIGN`` applied).  Everything
    after -- detection, analysis, storage, calibration -- is
    :func:`data_analysis.plasma.langmuir.process_sweep_run`.

    ``run`` / ``positions`` let :func:`process_run` reuse the open handle and
    the per-tip :class:`ProbePositions` instead of re-reading them per tip.
    """
    if run is None:
        run = open_lapd(ifn)
        print(f"backend: {run.backend}")

    pos = positions if positions is not None else read_lp_positions(
        ifn, motion_group_name)

    tarr, Vswp_arr, Iswp_arr = get_IV_arr(
        run, chans.scope_name, chans.I_chan, chans.V_chan, pos)

    # motion_group: which drive xpos/ypos describe.  In a two-drive run the
    # other tip's coordinates differ, so a reader holding only the npz cannot
    # otherwise tell which probe a profile belongs to.
    return SweepRecord(
        prefix=key, tarr=tarr, Vswp_arr=Vswp_arr, Iswp_arr=Iswp_arr,
        xpos=pos.xpos, ypos=pos.ypos, npos=pos.npos, nshot=pos.nshot,
        port=chans.port,
        meta={"I_chan": chans.I_chan, "V_chan": chans.V_chan,
              "motion_group": pos.motion_group or ""})




def process_run(ifn, motion_group_name=None, t_offset=0.0, calibrate=CALIBRATE,
                store_dtype=SWEEP_STORE_DTYPE, port=None):
    """Read every complete-pair tip in a run, then hand them to the shared driver.

    This is the Jun-2026 adapter and nothing more: discover the tips
    (:func:`discover_lp_channels`), read each into a ``SweepRecord``
    (:func:`read_iv_record`), and call
    :func:`data_analysis.plasma.langmuir.process_sweep_run`, which owns sweep
    detection, batch ``analyze_IV``, the npz layout and interferometer
    calibration.  Tips missing a channel are flagged and skipped by discovery --
    their results stay absent, never filled from the other probe.

    Positions are resolved **per tip**, not once per run: in a two-drive run each
    tip can sit on a different probe, and one shared position array would stamp
    one drive's coordinates onto the other's sweeps
    (:func:`data_analysis.io.motion_group_for_channel`).  ``motion_group_name``
    forces one group for every tip, for files whose channels name no port.

    ``port`` overrides the port parsed from the channel descriptions, for runs
    whose descriptions name a port the probe was not on.  It changes both the
    channel prefix (``'P33-R'`` -> ``'P29-R'``) and the interferometer chord, so
    it is a claim about which probe took the data -- pass it only with evidence
    outside the channel description (the motion group, the run description),
    and record that evidence at the call site.

    No figures are drawn here -- plot from the saved ``.npz`` with
    ``Jun2026_plot.plot_iv_line_run``.  Returns ``(sweep_path, plasma_path)``.

    Current orientation comes from the module-level ``I_SIGN`` (``-1`` for this
    experiment); change it at the top of the file if a run needs the other sign.
    ``calibrate`` / ``store_dtype`` default to the module-level toggles.
    """
    # Tip-invariant work is done ONCE: the open handle and the scope/channel
    # map.  Positions are NOT tip-invariant -- see the docstring -- so they are
    # read inside the loop.
    run = open_lapd(ifn)
    print(f"backend: {run.backend}")
    channel_map = resolve_iv_channel_map(ifn)

    if port is not None:
        # The tip label is the part of the prefix the port does not own, so it
        # survives the re-key: 'P33-R' -> 'P29-R', bare 'R' -> 'P29-R'.
        channel_map = {channel_prefix(port, key.rpartition("-")[2] or key):
                       chans._replace(port=port)
                       for key, chans in channel_map.items()}
        print(f"\n*** PORT OVERRIDE: channels re-keyed to port {port} "
              f"-> {list(channel_map)} ***")

    records = []
    for key, chans in channel_map.items():
        print(f"\n--- reading {key}")
        group = motion_group_name or motion_group_for_channel(
            ifn, chans.scope_name, chans.I_chan)
        records.append(read_iv_record(
            ifn, key, chans, run=run,
            positions=read_lp_positions(ifn, group)))

    return process_sweep_run(ifn, records, t_offset=t_offset,
                             calibrate=calibrate, store_dtype=store_dtype)
#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':

    ifn = r"D:\data\LAPD\jun2026-jia\06-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"

    # Scope-vs-interferometer trigger offset [s].  The one calibration input with
    # no in-file representation -- the chord itself is derived from each probe's
    # port, and whether to calibrate at all from the run's interferometer data.
    T_OFFSET = 0.012

    # Batch-processes every complete-pair tip, saves one npz pair for the run,
    # and calibrates in the same pass.  Draw figures afterwards from the saved
    # npz with Jun2026_plot.plot_iv_line_run(ifn).
    process_run(ifn, t_offset=T_OFFSET)

