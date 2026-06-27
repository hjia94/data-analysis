"""Jun-2026 LAPD Langmuir-probe sweep analysis.

Same analysis workflow as ``Mar-2026/Mar2026_IV.py`` (sweep detection ->
reshape -> smoothing -> per-trace ``analyze_IV`` -> shot-averaged Vp/Te/ne),
but the data is read from the **new LAPD_DAQ (pydaq) HDF5 format** instead of
the bapsflib/C-translator format.

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
  (:func:`plot_result_line`) is the primary output.

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

import time
import datetime
import os
import re

import numpy as np
import h5py
import matplotlib.pyplot as plt

from data_analysis.io import open_lapd
from data_analysis.io.paths import output_path
from data_analysis.io.scope import read_scope_channel_descriptions
from data_analysis.plasma.langmuir import find_sweep_indices, reshape_IV, analyze_IV_safe

from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

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


def read_lp_positions(fn, motion_group_name=None):
    """Read line-scan probe positions from a LAPD_DAQ pydaq file.

    The unified ``run.positions()`` needs a ``motion_list`` dataset these files
    lack, so we read ``Control/Positions/<group>`` directly:

    * ``positions_setup_array`` -- the planned unique (x, y) positions.
    * ``positions_array``       -- the (x, y) actually visited for every shot;
      used to count shots-per-position.

    Returns ``(pos_array, xpos, ypos, npos, nshot)`` where ``xpos``/``ypos`` are
    the unique sorted axes (for a line scan one of them is a single value).
    """
    with h5py.File(fn, "r") as f:
        if "Control/Positions" not in f:
            raise ValueError(f"No Control/Positions group in {fn!r}.")
        pos_root = f["Control/Positions"]
        groups = list(pos_root.keys())
        if motion_group_name is None:
            if len(groups) != 1:
                raise ValueError(
                    f"Multiple motion groups {groups}; pass motion_group_name=.")
            motion_group_name = groups[0]
        mg = pos_root[motion_group_name]
        print(f"Using motion group: {motion_group_name!r}")

        pos_array = mg["positions_array"][:]
        setup = mg["positions_setup_array"][:]  # planned unique positions

    npos = len(setup)
    xpos = np.unique(np.round(setup["x"], 3))
    ypos = np.unique(np.round(setup["y"], 3))

    # nshot = number of contiguous shots at the first position.
    x = np.round(pos_array["x"], 2)
    y = np.round(pos_array["y"], 2)
    x0, y0 = x[0], y[0]
    nshot = 0
    for i in range(len(pos_array)):
        if x[i] == x0 and y[i] == y0:
            nshot += 1
        else:
            break

    print(f"Positions: {npos} unique (x: {len(xpos)}, y: {len(ypos)}), "
          f"{nshot} shots/position, {len(pos_array)} total shots.")
    if npos * nshot != len(pos_array):
        print(f"  *** FLAG: npos*nshot ({npos*nshot}) != total shots "
              f"({len(pos_array)}); positions may be irregular. ***")
    return pos_array, xpos, ypos, npos, nshot


def _read_reshaped(run, scope_name, I_chan, V_chan, npos, nshot, pos_index=None):
    """Read one probe tip's V and I into ``(npos, nshot, nsamples)`` arrays.

    Shared core for :func:`get_IV_arr` and :func:`get_IV_at_position`.  Voltage is
    left as-is (LAPD_DAQ folds the probe attenuation into the scope
    ``vertical_gain``, so this is already true probe volts); current is scaled to
    current density via ``I_SIGN``/``RESISTOR``/``Aprobe``.  Returns
    ``(tarr, V3d, I3d)``, both 3-D.

    * ``pos_index=None`` -> read the whole run: the first ``npos*nshot`` shots,
      reshaped to ``(npos, nshot, nsamples)``.
    * ``pos_index=k``    -> read **only** that position's ``nshot`` shots (passing
      a positional ``shots=slice`` so ``run.channel`` fetches just those rows off
      disk instead of the whole run), reshaped to ``(1, nshot, nsamples)``.  The
      slice indexes the sorted shot list positionally -- the same ordering the
      ``Vstack[:nuse]`` whole-run slice relies on -- so the rows match the batch
      path exactly.
    """
    if pos_index is None:
        shots = None
        out_npos = npos
        keep = slice(0, npos * nshot)
    else:
        shots = slice(pos_index * nshot, (pos_index + 1) * nshot)
        out_npos = 1
        keep = slice(0, nshot)          # the channel read already returns just these

    Vstack, tarr = run.channel(V_chan, scope_name=scope_name, shots=shots)
    Istack, _ = run.channel(I_chan, scope_name=scope_name, shots=shots)

    V3d = Vstack[keep].reshape((out_npos, nshot, -1))
    I3d = Istack[keep].reshape((out_npos, nshot, -1)) * I_SIGN / (RESISTOR * Aprobe)
    return tarr, V3d, I3d


def get_IV_at_position(run, scope_name, I_chan, V_chan, npos, nshot, pos_index,
                       shot_index=None):
    """Read scaled I and V for ONE probe position (for notebook inspection).

    A single-position view of :func:`_read_reshaped` -- handy for eyeballing a
    trace before committing to the batch pass.  Only that position's ``nshot``
    shots are read off disk (not the whole run), so inspecting one position is
    cheap.

    * ``shot_index=None`` -> shot-averaged V and all per-shot I for that position:
      ``Vpos`` is ``(nsamples,)`` (mean over shots), ``Ipos`` is ``(nshot, nsamples)``.
    * ``shot_index=k``    -> a single shot: both ``(nsamples,)``.
    """
    tarr, V3d, I3d = _read_reshaped(run, scope_name, I_chan, V_chan, npos, nshot,
                                    pos_index=pos_index)
    Vpos, Ipos = V3d[0], I3d[0]                            # (nshot, nsamples)

    if shot_index is None:
        return tarr, np.mean(Vpos, axis=0), Ipos
    return tarr, Vpos[shot_index], Ipos[shot_index]


def get_IV_arr(run, scope_name, I_chan, V_chan, npos, nshot):
    """Read the V and I sweeps for one probe tip for the whole run.

    LAPD_DAQ equivalent of Mar-2026's ``get_IV_arr``.  Returns ``(tarr, Vsweep,
    Isweep)`` where ``Vsweep`` is shot-averaged ``(npos, nsamples)`` and
    ``Isweep`` keeps per-shot resolution ``(npos, nshot, nsamples)``.
    """
    tarr, V3d, Isweep = _read_reshaped(run, scope_name, I_chan, V_chan, npos, nshot)
    Vsweep = np.mean(V3d, axis=1)                          # (npos, nsamples)
    return tarr, Vsweep, Isweep


def analyze_tip_at_position(run, scope_name, I_chan, V_chan, npos, nshot,
                            pos_index, sweep_idx, padding=10, trim_percent=5,
                            smooth_sigma=10):
    """Run the full sweep pipeline for one tip at one position/sweep.

    Reads -> detects sweeps -> ``reshape_IV`` -> smooths -> shot-averages the
    chosen sweep -> ``analyze_IV_safe``.  Returns ``(Vp, Te, ne)`` (NaNs on
    failure).  This is the per-tip body used to inspect both probes at a
    position; the steps mirror :func:`save_IV_data` so what you see matches the
    batch pass.
    """
    _, Vpos, Ipos = get_IV_at_position(run, scope_name, I_chan, V_chan,
                                       npos, nshot, pos_index)
    start_ls, stop_ls = find_sweep_indices(Vpos, padding=padding)
    V_rs, I_rs = reshape_IV(Vpos[None, :], Ipos[None, :, :], start_ls, stop_ls,
                            trim_percent)
    I_rs = gaussian_filter1d(I_rs, smooth_sigma, axis=-1)
    I_trace = I_rs[0, :, sweep_idx, :].mean(0)
    return analyze_IV_safe(V_rs[0, sweep_idx], I_trace)


def resolve_iv_channels(ifn, tip=None):
    """Decide which scope group and I/V channels to use.

    Honors the top-of-file ``SCOPE_NAME`` / ``I_CHAN`` / ``V_CHAN`` override
    first; any field left ``None`` is auto-detected from the channel
    descriptions only.  Returns ``(scope_name, I_chan, V_chan)``.
    """
    scope_name = SCOPE_NAME if SCOPE_NAME is not None else find_lp_scope(ifn)

    if I_CHAN is not None and V_CHAN is not None:
        print(f"\nUsing USER-OVERRIDE channels: I={I_CHAN}, V={V_CHAN} "
              f"(scope '{scope_name}')")
        return scope_name, I_CHAN, V_CHAN
    if (I_CHAN is None) != (V_CHAN is None):
        raise ValueError("Set BOTH I_CHAN and V_CHAN to override, or leave both None.")

    pairs, _ = discover_lp_channels(ifn, scope_name)
    if tip is None:
        tip = next(iter(pairs))
    if tip not in pairs:
        raise ValueError(f"Tip {tip!r} has no complete I+V pair; available: {list(pairs)}")
    I_chan, V_chan = pairs[tip]["I"], pairs[tip]["V"]
    print(f"\nAuto-selected tip {tip}: I={I_chan}, V={V_chan} (scope '{scope_name}')")
    return scope_name, I_chan, V_chan


def save_IV_data(ifn, save_path, tip=None):
    """Detect sweeps, reshape + smooth the IV traces, and save to ``.npz``.

    Same workflow as Mar-2026's ``save_IV_data`` but for the pydaq format and
    for a single tip.  Channels come from :func:`resolve_iv_channels` (top-of-file
    override, else auto-detected first complete I+V pair).
    """
    run = open_lapd(ifn)
    print(f"backend: {run.backend}")

    scope_name, I_chan, V_chan = resolve_iv_channels(ifn, tip=tip)

    pos_array, xpos, ypos, npos, nshot = read_lp_positions(ifn)

    tarr, Vswp_arr, Iswp_arr = get_IV_arr(run, scope_name, I_chan, V_chan, npos, nshot)

    # Reshape arrays to include swept traces only
    start_t_ls, stop_t_ls = find_sweep_indices(Vswp_arr[0], padding=10)

    # Middle index of each sweep -> a representative timestamp per sweep.
    mid_indices = [(start + stop) // 2 for start, stop in zip(start_t_ls, stop_t_ls)]
    data_timestamp = tarr[mid_indices]
    print(f"Number of sweeps: {len(data_timestamp)}")

    Vswp_arr_rs, Iswp_arr_rs = reshape_IV(Vswp_arr, Iswp_arr, start_t_ls, stop_t_ls, 5)

    print("Applying smoothing to current array...")
    Iswp_arr_rs = gaussian_filter1d(Iswp_arr_rs, 10, axis=-1)

    np.savez(save_path, Vswp_arr_rs=Vswp_arr_rs, Iswp_arr_rs=Iswp_arr_rs,
             data_timestamp=data_timestamp, xpos=xpos, ypos=ypos,
             npos=npos, nshot=nshot, I_chan=I_chan, V_chan=V_chan)
    print(f"Saved to: {save_path}")


def process_and_save(voltage_data, current_data, save_path):
    """
    Loops through the multi-dimensional Langmuir probe dataset, extracting
    plasma parameters. Averages the valid shots for each location/sweep combination,
    calculates the standard error, and outputs 2D arrays.
    Saves progress incrementally to prevent data loss.

    (Unchanged from Mar-2026: the reshaped sweep arrays have the same shape
    convention, so the batch loop is reused as-is.)
    """
    n_locs, n_shots, n_sweeps, _ = current_data.shape

    # Pre-allocate 2D output arrays with NaNs (locs, sweeps)
    Vp_arr = np.full((n_locs, n_sweeps), np.nan)
    Te_arr = np.full((n_locs, n_sweeps), np.nan)
    ne_arr = np.full((n_locs, n_sweeps), np.nan)

    # Pre-allocate 2D error arrays for the error bars
    Vp_err = np.full((n_locs, n_sweeps), np.nan)
    Te_err = np.full((n_locs, n_sweeps), np.nan)
    ne_err = np.full((n_locs, n_sweeps), np.nan)

    total_traces = n_locs * n_shots * n_sweeps
    print(f"Starting batch processing of {total_traces} traces across {n_locs} locations...")
    print(f"Averaging {n_shots} shots per sweep...")

    start_time = time.time()
    traces_completed = 0
    fail_count = 0

    # Progress bar over locations; the per-trace fail rate is shown in the postfix
    # and refreshed each location.  tqdm provides the %, elapsed, ETA and rate.
    pbar = tqdm(range(n_locs), desc="Analyzing", unit="loc")
    for loc in pbar:
        for swp in range(n_sweeps):
            # The voltage trace applies to all shots at this location/sweep
            V_trace = voltage_data[loc, swp, :]

            # Temporary storage for the shots in this specific sweep
            temp_Vp = []
            temp_Te = []
            temp_ne = []

            for sht in range(n_shots):
                I_trace = current_data[loc, sht, swp, :]
                trace_id = f"Loc:{loc}|Shot:{sht}|Swp:{swp}"

                # Analyze trace
                Vp, Te, ne = analyze_IV_safe(V_trace, I_trace, file_name=trace_id)

                # Track failures on a per-trace basis
                if np.isnan(Vp):
                    fail_count += 1
                else:
                    # Only keep valid numbers for averaging
                    temp_Vp.append(Vp)
                    temp_Te.append(Te)
                    temp_ne.append(ne)

                traces_completed += 1

            # --- AVERAGING LOGIC ---
            # Calculate the mean and standard error of the mean (SEM) for valid shots

            # Vp
            if len(temp_Vp) > 0:
                Vp_arr[loc, swp] = np.mean(temp_Vp)
                # If more than 1 valid shot, calculate error: std_dev / sqrt(N)
                Vp_err[loc, swp] = np.std(temp_Vp, ddof=1) / np.sqrt(len(temp_Vp)) if len(temp_Vp) > 1 else np.nan

            # Te
            if len(temp_Te) > 0:
                Te_arr[loc, swp] = np.mean(temp_Te)
                Te_err[loc, swp] = np.std(temp_Te, ddof=1) / np.sqrt(len(temp_Te)) if len(temp_Te) > 1 else np.nan

            # ne
            # Because ne can be NaN if Te was forced to 0, we need to filter NaNs out of temp_ne specifically
            valid_ne = [n for n in temp_ne if not np.isnan(n)]
            if len(valid_ne) > 0:
                ne_arr[loc, swp] = np.mean(valid_ne)
                ne_err[loc, swp] = np.std(valid_ne, ddof=1) / np.sqrt(len(valid_ne)) if len(valid_ne) > 1 else np.nan

        # ==========================================
        # INCREMENTAL SAVE
        # ==========================================
        # Save all 6 arrays to the file
        np.savez(save_path,
                 Vp_arr=Vp_arr, Te_arr=Te_arr, ne_arr=ne_arr,
                 Vp_err=Vp_err, Te_err=Te_err, ne_err=ne_err)

        # Live running fail rate alongside the progress bar
        pbar.set_postfix(fails=f"{fail_count} ({fail_count / traces_completed * 100:.1f}%)")

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    total_time = time.time() - start_time
    final_time_str = str(datetime.timedelta(seconds=int(total_time)))
    final_fail_rate = (fail_count / total_traces) * 100

    print("\n" + "=" * 55)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 55)
    print(f"Total Time:    {final_time_str}")
    print(f"Total Traces:  {total_traces}")
    print(f"Total Fails:   {fail_count}")
    print(f"Fail Rate:     {final_fail_rate:.2f}%")
    print(f"Data saved to: {save_path}")
    print("=" * 55)

    return Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err


def _tip_tag(tip):
    """Filename fragment for a tip (``"-tipR"``); empty for the legacy single-tip."""
    return f"-tip{tip}" if tip else ""


def load_sweep_data(data_dir, run_num, tip=None):
    """Load the reshaped sweep arrays + axes saved by :func:`save_IV_data`."""
    save_path = os.path.join(data_dir, f"{run_num}{_tip_tag(tip)}-sweep-data.npz")
    data = np.load(save_path)
    return (data["Vswp_arr_rs"], data["Iswp_arr_rs"], data["data_timestamp"],
            data["xpos"], data["ypos"], int(data["npos"]), int(data["nshot"]))


def load_data(data_dir, run_num, tip=None):
    """Load saved plasma parameters + sweep timestamps for plotting."""
    save_path = os.path.join(data_dir, f"{run_num}{_tip_tag(tip)}-sweep-data.npz")
    data = np.load(save_path)
    t_ls = data["data_timestamp"]

    ps_path = os.path.join(data_dir, f"{run_num}{_tip_tag(tip)}-plasma-data.npz")
    ps_data = np.load(ps_path)
    Vp_arr = ps_data["Vp_arr"]
    Te_arr = ps_data["Te_arr"]
    ne_arr = ps_data["ne_arr"]
    Vp_err = ps_data["Vp_err"]
    Te_err = ps_data["Te_err"]
    ne_err = ps_data["ne_err"]

    return Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls


def plot_result_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, tndx_list, save_fig=None,
                     show=True, title=None):
    """
    Plot the line scan: Vp, Te, and ne vs x at selected sweep (time) indices.

    Jun-2026 LP runs are a 1-D line scan (y == 0), so each row of the parameter
    arrays already corresponds to one x position -- no center-line extraction is
    needed (unlike Mar-2026's xy-plane).  ``Vp_arr`` etc. are (n_locs, n_sweeps)
    with ``n_locs == len(xpos)``.

    ``save_fig`` -- a path to write the figure to (PNG); ``None`` skips saving.
    ``show`` -- call ``plt.show()`` (set False for headless/batch saving only).
    ``title`` -- figure title; the caller passes the run's gas-puff label (run
    number + the puff setting, e.g. ``"01-Puff voltage 75V for 25ms"``).  ``None``
    uses a generic default.
    """
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(tndx_list)))

    for color, t_idx in zip(colors, tndx_list):
        if t_idx >= Vp_arr.shape[1]:
            print(f"Warning: time index {t_idx} exceeds array size {Vp_arr.shape[1]}")
            continue

        t_val = t_ls[t_idx] * 1e3  # Convert to ms
        label = f"t = {t_val:.2f} ms"

        axs[0].plot(xpos, Vp_arr[:, t_idx], "o-", color=color, label=label,
                    linewidth=2, markersize=5)
        axs[1].plot(xpos, Te_arr[:, t_idx], "s-", color=color, label=label,
                    linewidth=2, markersize=5)
        axs[2].plot(xpos, ne_arr[:, t_idx], "^-", color=color, label=label,
                    linewidth=2, markersize=5)

    # Title is the experiment-difference string from the caller (run number +
    # the one changed setting vs the baseline, e.g. "01: puff voltage 75V for
    # 25ms"); fall back to a generic label when no difference was resolved.
    base_title = title or "Plasma Parameters along LP line scan (y = 0)"
    axs[0].set_title(base_title, fontsize=12, fontweight="bold")
    axs[0].set_ylabel("Vp [V]", fontsize=11)
    axs[1].set_ylabel("Te [eV]", fontsize=11)
    axs[2].set_ylabel("ne [cm$^{-3}$]", fontsize=11)
    axs[2].set_xlabel("X Position [cm]", fontsize=11)
    for ax in axs:
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_fig}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _fig_path(run_num, tip):
    """Centralized figure location for a run/tip (under the repo-external output root).

    Routes through :func:`data_analysis.io.paths.output_path`, so figures land in
    ``$DATA_ANALYSIS_OUTPUT/figures/Jun2026_IV/`` (default ``~/data-analysis-output``)
    rather than next to the raw data or in the repo.
    """
    tag = _tip_tag(None if tip in (None, "override") else tip)
    return output_path("figures", "Jun2026_IV", f"{run_num}{tag}-line.png")


# The gas-puff setting the operator writes into the run ``description`` -- e.g.
# "Helium backside pressure 40 Psi, Puff voltage 75V for 25ms West+East".  We
# title each plot with the "Puff voltage ... ms" span of that line; it is the
# knob that varies run to run.
_PUFF_RE = re.compile(r"puff voltage.*?\d+\s*ms", re.IGNORECASE)


def _run_title(ifn, run_num):
    """Plot title for a run: ``"<run_num>-<puff voltage ... ms>"``.

    Reads this run's own hand-written ``description`` and pulls the gas-puff span
    (``"Puff voltage 75V for 25ms"``) -- the setting that changes run to run -- so
    e.g. run 01 titles as ``"01-Puff voltage 75V for 25ms"``.  Returns just the
    run number when the description has no recognizable puff line, and ``None``
    when the file can't be read / isn't pydaq, so the plot falls back to its
    generic title rather than being blocked by a description hiccup.
    """
    try:
        desc = open_lapd(ifn).description()
    except (OSError, ValueError, NotImplementedError, KeyError) as e:
        print(f"  (title: could not read run description -- {e})")
        return None
    for sec in desc.sections.values():
        for it in sec.items:
            m = _PUFF_RE.search(it.raw)
            if m:
                return f"{run_num}-{m.group(0)}"
    return run_num
1

def _plot_tip(data_dir, run_num, tip, ifn=None, tndx_list=None, save_fig=True,
              show=True):
    """Load one tip's saved arrays and draw the line-scan plot (no reprocessing).

    Shared by :func:`process_run` and :func:`replot_run`.  ``tip`` is the
    discovered tip label, or ``"override"``/``None`` for the un-tagged single-tip
    files.  ``ifn`` (the run's HDF5 path) drives the descriptive title (just the
    changed setting vs the baseline) via :func:`_run_title`; pass ``None`` for the
    plot's generic title.  ``save_fig`` True routes the PNG through
    :func:`_fig_path`; pass a path to override, or False to skip saving.
    """
    load_tip = None if tip in (None, "override") else tip
    Vp_arr, Te_arr, ne_arr, *_errs, t_ls = load_data(data_dir, run_num, tip=load_tip)
    _, _, _, xpos, _, _, _ = load_sweep_data(data_dir, run_num, tip=load_tip)

    ndx = tndx_list if tndx_list is not None else list(
        range(0, Vp_arr.shape[1], max(1, Vp_arr.shape[1] // 4)))

    # Title = "<run_num>-<puff voltage ... ms>" from this run's description (e.g.
    # "01-Puff voltage 75V for 25ms"); no tip.  None -> the plot's generic
    # default.  (Output filenames still carry run_num + tip, via _fig_path.)
    title = _run_title(ifn, run_num) if ifn is not None else None

    fig_path = _fig_path(run_num, tip) if save_fig is True else (save_fig or None)
    plot_result_line(Vp_arr, Te_arr, ne_arr, xpos, t_ls, ndx,
                     save_fig=fig_path, show=show, title=title)


def replot_run(ifn, tndx_list=None, save_fig=True, show=True):
    """Re-draw the line-scan plot(s) for a run from already-saved ``.npz`` -- no
    HDF5 read, no reprocessing.

    Discovers the same complete-pair tips as :func:`process_run` and plots each
    from its saved ``-tip<T>-*.npz`` files.  Use this to revisit / re-save the
    figure after the batch has run.  ``save_fig`` True writes the PNG via
    :func:`_fig_path`; pass a path to override or False to skip.
    """
    data_dir = os.path.dirname(ifn)
    run_num = os.path.basename(ifn).split("-")[0]

    if I_CHAN is not None and V_CHAN is not None:
        tips = ["override"]
    else:
        scope_name = SCOPE_NAME if SCOPE_NAME is not None else find_lp_scope(ifn)
        pairs, _ = discover_lp_channels(ifn, scope_name)
        tips = list(pairs)

    for tip in tips:
        print(f"Re-plotting tip {tip} from saved data...")
        _plot_tip(data_dir, run_num, tip, ifn=ifn, tndx_list=tndx_list,
                  save_fig=save_fig, show=show)


def process_run(ifn, plot=True, tndx_list=None):
    """Run the full batch pipeline for **every complete-pair tip** in a run.

    For each tip with a complete I+V pair (from :func:`discover_lp_channels`):
    sweep-detect + reshape + smooth (:func:`save_IV_data`) -> batch
    ``analyze_IV`` over all positions/shots (:func:`process_and_save`) -> optional
    line-scan plot.  Outputs are saved **per tip** (filenames carry ``-tip<T>``)
    so the two probes never mix; an override (``I_CHAN``/``V_CHAN``) collapses to
    the single overridden tip.  Tips missing a channel are flagged and skipped
    (their results stay absent, never filled from the other probe).

    Current orientation comes from the module-level ``I_SIGN`` (``-1`` for this
    experiment); change it at the top of the file if a run needs the other sign.
    """
    data_dir = os.path.dirname(ifn)
    run_num = os.path.basename(ifn).split("-")[0]

    scope_name = SCOPE_NAME if SCOPE_NAME is not None else find_lp_scope(ifn)
    if I_CHAN is not None and V_CHAN is not None:
        tips = ["override"]
        resolve_tip = None            # resolve_iv_channels uses the override block
    else:
        pairs, _ = discover_lp_channels(ifn, scope_name)
        tips = list(pairs)
        resolve_tip = True

    results = {}
    for tip in tips:
        tag = _tip_tag(None if tip == "override" else tip)
        print("\n" + "=" * 70)
        print(f"PROCESSING tip {tip}")
        print("=" * 70)

        sweep_path = os.path.join(data_dir, f"{run_num}{tag}-sweep-data.npz")
        plasma_path = os.path.join(data_dir, f"{run_num}{tag}-plasma-data.npz")

        save_IV_data(ifn, sweep_path, tip=None if resolve_tip is None else tip)

        Vswp_arr_rs, Iswp_arr_rs, *_ = \
            load_sweep_data(data_dir, run_num, tip=None if tip == "override" else tip)
        process_and_save(Vswp_arr_rs, Iswp_arr_rs, plasma_path)

        if plot:
            _plot_tip(data_dir, run_num, tip, ifn=ifn, tndx_list=tndx_list)

        results[tip] = (sweep_path, plasma_path)

    print(f"\nDone. Processed tips: {list(results)}")
    return results
#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':

    ifn = r"D:\data\LAPD\jun2026-jia\02-He-800G-bias40V-LP-p29-line_2026-06-10.hdf5"
    
    if not ifn:
        raise SystemExit(
            "Set `ifn` to a run file first (or use Jun2026_IV_explore.ipynb to "
            "inspect one position before running the batch pipeline).")



    # To revisit the plot later WITHOUT reprocessing (reads only the saved .npz):
    # replot_run(ifn)                       # show + re-save the figure(s)
    replot_run(ifn, save_fig=False, tndx_list=[-9,-7,-5,-3,-1])       # just show, don't overwrite

