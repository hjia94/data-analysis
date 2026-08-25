"""Calibrate the P29 4-tip Mach probe from the Mar-2026 rotation runs 055/056.

Runs 055 ("normal") and 056 ("rotated 180 degree along x-axis") record the same
stationary probe at (0,0) with the bias off, 100 shots each. Rotating 180 deg
swaps which physical tip of a pair looks upstream while the DAQ labels stay put,
so the **product** of the two orientations' face ratios isolates the collecting
area ratio ``kappa = A_+/A_-`` and the **ratio** isolates the flow. The estimator
maths lives in :mod:`data_analysis.plasma.mach`; this script is the campaign
wiring -- paths, runs, the window, and which pairs actually swapped.

**Both pairs swap, and that is measured, not assumed.** The description says the
rotation is about the x-axis, which would leave the X pair on the rotation axis
and unswapped -- in which case no pairing would exist for X and only a no-flow
assumption could calibrate it (Jun-2026's situation, and the origin of its
x/÷1.46 bar on kappa_X). The data reject that: ``ln R_X`` moves from -0.550 to
+0.146 between the runs, a 459-sigma change against a 0.0015 shot error. A pair
that did not swap would have reported the same ratio twice. So the X tips do
exchange under this rotation and :func:`main` pairs both axes; the check is
re-run on every execution and printed.

This matters because it **replaces the weakest number** in the Jun-2026
calibration that Mar-2026 was borrowing. kappa_X there is 1.9077 x/÷1.458 from
the no-x-flow assumption; here it is measured against a rotation. The difference
is most of the uniform pre-bias offset that made run 054's flow maps look wrong:
M_X shifts by +0.191 and M_Y by +0.034 when this file replaces that one.

Writes ``<STEM>.npz``/``.png`` **into the raw data directory**, following
``Jun2026_mach_cal``: the figure is the evidence the kappa can be trusted, so it
travels with the data.

    .venv/Scripts/python.exe experiments/ucla-lapd/Mar-2026/Mar2026_mach_cal.py
"""

import datetime
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.io import open_lapd
from data_analysis.io.paths import output_path
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.mach import (K_HUTCHINSON, area_ratio, combine_log,
                                       face_ratio, mach_number,
                                       valid_current_mask)
from data_analysis.viz.plot_utils import finalize_figure

DATA_DIR = r"D:\data\LAPD\Mar26-data"
FILES = {
    55: "055-mach-4tip-p29-calibration-normal-orientation-Nobias "
        "2026-03-06 14.40.08.hdf5",
    56: "056-mach-4tip-p29-calibration-Yflipped-orientation-Nobias "
        "2026-03-06 14.41.36.hdf5",
}
ORIENTATIONS = {55: "normal", 56: "180-about-x"}
#: ``(normal, rotated)``. Order is load-bearing: the flow sign in
#: :func:`~data_analysis.plasma.mach.mach_number` is quoted in the first run's
#: frame.
PAIRING = (55, 56)

STEM = "055-056-mach-calibration"
AXES = ("X", "Y")

#: Calibration window [ms, machine time]. The DAQ triggers at 0 ms here (unlike
#: run 054's 4.5 ms), so this is machine time directly.
#:
#: Chosen on this run's own currents: the summed tip current is ~8 mA at 0-2 ms
#: while the discharge is still coming up, plateaus at 17-19 mA over 4-10 ms, and
#: collapses to 2 mA by 12 ms. kappa is flat over the plateau (kappa_X 0.80-0.85,
#: kappa_Y 1.03-1.05) and pulls low over the ramp (kappa_X 0.76 at 2-3 ms), which
#: is why the window starts at 4 rather than 2 ms.
CAL_WINDOW_MS = (4.0, 10.0)

#: Halves the split-half systematic is computed over -- the only systematic a
#: SINGLE pairing supports. Jun-2026 had two pairings and used their spread;
#: with one pairing the available check is whether kappa is stable across the
#: window, which is what this measures. It is NOT a shot-noise bar.
SPLIT_HALVES = ((4.0, 7.0), (7.0, 10.0))

#: Shunt resistance [ohm]. Every tip: "All four tips used ground referemce,
#: -98 V bias, and 45 Ohm resistor". Unlike the Jun-2026 runs (75/300/75/75/43/43
#: ohm, written per tip and parsed by ``parse_shunts``), this description states
#: one value in prose for all four, so ``parse_shunts`` finds nothing and this
#: constant carries it. Checked against the description text on every run.
SHUNT_OHM = 45.0
_SHUNT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*Ohm\s*resistor", re.IGNORECASE)

#: Which digitizer channel each tip is on comes from the run's own wiring
#: descriptions ('Vx+_p29'), never a hardcoded table -- a swapped +/- assignment
#: would silently report 1/kappa, which looks entirely plausible.
_TIP_RE = re.compile(r"\bV([XY])([+-])", re.IGNORECASE)

RAW_PANEL_SHOT = 0


def tip_channels(path):
    """``{'X+': (board, chan), ...}`` from the run's wiring descriptions."""
    out = {}
    for (_adc, chan), desc in channel_wiring(path).items():
        m = _TIP_RE.search(desc)
        if m:
            out[f"{m.group(1).upper()}{m.group(2)}"] = chan
    missing = [f"{a}{s}" for a in AXES for s in "+-" if f"{a}{s}" not in out]
    if missing:
        raise ValueError(f"{path}: no digitizer channel found for {missing}")
    return out


def shunt_ohm(raw, path):
    """The tips' shunt [ohm], read from the description and checked.

    Raises rather than defaulting: a guessed resistance yields a wrong current
    with nothing downstream complaining. kappa is a *ratio* of two tips sharing
    one value, so it survives a wrong shunt -- the printed currents and the
    amplitude panel do not.
    """
    found = {float(m) for m in _SHUNT_RE.findall(raw)}
    if not found:
        raise ValueError(f"{path}: no 'NN Ohm resistor' in the description; "
                         f"cannot confirm the {SHUNT_OHM:g} ohm shunt")
    if found != {SHUNT_OHM}:
        raise ValueError(f"{path}: description says {sorted(found)} ohm, "
                         f"this script assumes {SHUNT_OHM:g}")
    return SHUNT_OHM


def load_run(run):
    """One run's windowed currents -> ``(currents, t_win_ms, tarr, raw_trace, chans)``.

    ``currents`` is ``{tip: (nshot, nt_win) amps}``. These runs are stationary
    (the description's "no motion, mach probe at P29 sits at (0,0)"), and carry
    no 6K Compumotor group at all -- so every shot is the same position and the
    whole stack is read directly, with no position handling.
    """
    path = os.path.join(DATA_DIR, FILES[run])
    r = open_lapd(path)
    ohm = shunt_ohm(r.description().raw, path)
    chans = tip_channels(path)

    currents, raw_trace = {}, {}
    with r.session() as sess:
        adc, _digi = sess.digitizer_config()
        for tip in (f"{a}{s}" for a in AXES for s in "+-"):
            board, chan = chans[tip]
            data, tarr = sess.read_data(board, chan, adc=adc)
            stack = np.asarray(data["signal"], dtype=float) / ohm
            t_ms = tarr * 1e3
            keep = (t_ms >= CAL_WINDOW_MS[0]) & (t_ms < CAL_WINDOW_MS[1])
            # .copy(): a boolean-column slice of a (100, 108544) stack would
            # otherwise pin the whole 87 MB parent for the life of the run.
            currents[tip] = stack[:, keep].copy()
            raw_trace[tip] = stack[RAW_PANEL_SHOT].copy()
    return currents, t_ms[keep], t_ms, raw_trace, chans


def main():
    data, raw_traces, chans, t_win, t_full = {}, {}, {}, None, None
    for run in PAIRING:
        data[run], t_win, t_full, raw_traces[run], chans[run] = load_run(run)
        print(f"run {run} ({ORIENTATIONS[run]:>12}): "
              f"{data[run]['X+'].shape[0]} shots, "
              f"{data[run]['X+'].shape[1]} samples in window")

    if chans[PAIRING[0]] != chans[PAIRING[1]]:
        raise ValueError(f"the two runs wire tips to different channels: "
                         f"{chans}; pairing them would compare different tips")

    print(f"\nwindow {CAL_WINDOW_MS[0]:g}-{CAL_WINDOW_MS[1]:g} ms machine "
          f"({t_win.size} samples), shunt {SHUNT_OHM:g} ohm\n")

    # --- per-run, per-axis face ratios --------------------------------------
    ra, rb = PAIRING
    lnR = np.full((len(AXES), 2), np.nan)
    sem = np.full_like(lnR, np.nan)
    for i, axis in enumerate(AXES):
        for j, run in enumerate(PAIRING):
            R = face_ratio(data[run][f"{axis}+"], data[run][f"{axis}-"])
            lg, sd, n = combine_log(R)
            lnR[i, j], sem[i, j] = lg, sd / np.sqrt(n)

    # --- did each pair actually swap? ---------------------------------------
    # The description says the rotation is about x, which would leave the X pair
    # unswapped and its ratio UNCHANGED between runs. Tested rather than
    # believed: an unswapped pair has no pairing, and sqrt(Ra*Rb) for it would
    # be a meaningless number wearing a plausible magnitude.
    print("=== did the pair swap? (an unswapped pair has no pairing) ===")
    swapped = {}
    for i, axis in enumerate(AXES):
        d = lnR[i, 0] - lnR[i, 1]
        e = float(np.hypot(sem[i, 0], sem[i, 1]))
        swapped[axis] = abs(d) > 3 * e
        print(f"  {axis}: ln R {lnR[i, 0]:+.4f} -> {lnR[i, 1]:+.4f}, "
              f"change {d:+.4f} +- {e:.4f} ({abs(d) / e:.0f} sigma) -> "
              f"{'SWAPPED' if swapped[axis] else 'UNCHANGED: no pairing'}")
    unswapped = [a for a in AXES if not swapped[a]]
    if unswapped:
        raise ValueError(
            f"axes {unswapped} show the same face ratio in both orientations, "
            f"so their tips did not exchange and the product sqrt(Ra*Rb) is not "
            f"an area ratio. Calibrating them needs a no-flow assumption, which "
            f"this script deliberately does not make.")

    # --- kappa and flow from the pairing ------------------------------------
    kappa = np.array([area_ratio(np.exp(lnR[i, 0]), np.exp(lnR[i, 1]))
                      for i in range(len(AXES))])
    mach = np.array([mach_number(np.exp(lnR[i, 0]), np.exp(lnR[i, 1]))
                     for i in range(len(AXES))])

    # --- error contributions, all multiplicative (kappa x/÷ factor) ---------
    # (1) shot-to-shot, worst of the two runs.
    err_stat = np.exp(sem.max(axis=1))
    # (2) split-half: kappa recomputed on each half of the window. With one
    #     pairing this is the only systematic available, and it is the honest
    #     one -- a kappa that moves between halves is contaminated by something
    #     that changes during the shot.
    err_sys = np.full(len(AXES), np.nan)
    kappa_half = np.full((len(AXES), 2), np.nan)
    for i, axis in enumerate(AXES):
        for h, (lo, hi) in enumerate(SPLIT_HALVES):
            m = (t_win >= lo) & (t_win < hi)
            Ra = np.exp(combine_log(face_ratio(data[ra][f"{axis}+"][:, m],
                                               data[ra][f"{axis}-"][:, m]))[0])
            Rb = np.exp(combine_log(face_ratio(data[rb][f"{axis}+"][:, m],
                                               data[rb][f"{axis}-"][:, m]))[0])
            kappa_half[i, h] = area_ratio(Ra, Rb)
        err_sys[i] = np.exp(abs(np.log(kappa_half[i, 0] / kappa_half[i, 1])) / 2)

    # (3) kappa(t): the curve the split-half number summarizes. One mask over
    #     all four tips so both runs' ratios are taken at the same samples.
    kappa_t = np.full((len(AXES), t_win.size), np.nan)
    n_valid = np.zeros((len(AXES), t_win.size), dtype=int)
    err_time = np.full(len(AXES), np.nan)
    for i, axis in enumerate(AXES):
        pa, ma = data[ra][f"{axis}+"], data[ra][f"{axis}-"]
        pb, mb = data[rb][f"{axis}+"], data[rb][f"{axis}-"]
        ok = valid_current_mask(pa, ma, pb, mb)
        n_valid[i] = ok.sum(axis=0)
        kappa_t[i] = area_ratio(face_ratio(pa, ma, axis=0, mask=ok),
                                face_ratio(pb, mb, axis=0, mask=ok))
        f = kappa_t[i][np.isfinite(kappa_t[i]) & (kappa_t[i] > 0)]
        err_time[i] = np.exp(np.log(f).std(ddof=1)) if f.size > 1 else np.nan

    # --- printed checks -----------------------------------------------------
    print("\n=== kappa ===")
    for i, axis in enumerate(AXES):
        print(f"  kappa_{axis} = {kappa[i]:.4f}   sys x/÷{err_sys[i]:.4f}   "
              f"[rotation-pair]")
        print(f"      stat x/÷{err_stat[i]:.4f}, time x/÷{err_time[i]:.4f}, "
              f"halves {kappa_half[i, 0]:.4f}/{kappa_half[i, 1]:.4f}")

    print("\n=== flow at (0,0), bias off "
          f"(sign in run {ra}'s frame) ===")
    for i, axis in enumerate(AXES):
        print(f"  M_{axis} = {mach[i]:+.4f}")
    big = [a for a, m in zip(AXES, mach) if abs(m) > 0.3]
    if big:
        print(f"  <-- WARN: |M| > 0.3 (supersonic?) for {big}")

    out_of_bounds = [a for a, k in zip(AXES, kappa) if not 0.5 <= k <= 2.0]
    if out_of_bounds:
        print(f"\n  <-- WARN: kappa outside 0.5-2.0 for {out_of_bounds}")
    print(f"\n  valid shots, min per axis: " +
          ", ".join(f"{a}={n_valid[i].min()}" for i, a in enumerate(AXES)))

    # --- outputs ------------------------------------------------------------
    npz_path = output_path(f"{STEM}.npz", explicit=DATA_DIR)
    png_path = output_path(f"{STEM}.png", explicit=DATA_DIR)

    # Key names match Jun2026_mach_cal's npz so Mar2026_mach_flow.load_calibration
    # reads either file unchanged. kappa_err_fit is all-NaN here (there is no
    # fit: two orientations give kappa in closed form), NOT 0.0 -- a zero would
    # read as "fitted and negligible", the opposite of the truth.
    results = dict(
        tips=np.array(AXES),
        kappa=kappa,
        kappa_calibrated=np.ones(len(AXES), dtype=bool),
        kappa_method=np.array(["rotation-pair"] * len(AXES)),
        kappa_err_fit=np.full(len(AXES), np.nan),
        kappa_err_stat=err_stat,
        kappa_err_time=err_time,
        kappa_err_sys=err_sys,
        kappa_halves=kappa_half,
        split_halves_ms=np.array(SPLIT_HALVES),
        mach=mach,
        lnR=lnR,
        lnR_sem=sem,
        kappa_t=kappa_t,
        t_win_ms=t_win,
        n_valid=n_valid.astype(np.int16),
        shunt_ohm=np.float64(SHUNT_OHM),
        tip_channels=np.array([f"{t}=board{chans[ra][t][0]}ch{chans[ra][t][1]}"
                               for t in (f"{a}{s}" for a in AXES for s in "+-")]),
        window_ms=np.array(CAL_WINDOW_MS),
        K_hutchinson=np.float64(K_HUTCHINSON),
        runs=np.array(PAIRING),
        orientations=np.array([ORIENTATIONS[r] for r in PAIRING]),
        source_files=np.array([FILES[r] for r in PAIRING]),
        created=np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    )
    np.savez(npz_path, **results)
    print(f"\nWrote {npz_path}")

    _figure(png_path, results, raw_traces, t_full)


def _figure(path, results, raw_traces, t_full):
    """Four panels: the evidence that the kappa in the npz is trustworthy.

    Reads ``results`` -- the exact dict written to the npz -- by name, so the
    figure cannot drift from the saved numbers.
    """
    kappa, kappa_t = results["kappa"], results["kappa_t"]
    t_win, n_valid = results["t_win_ms"], results["n_valid"]
    halves, err_sys = results["kappa_halves"], results["kappa_err_sys"]
    runs = results["runs"]

    fig, axs = plt.subplots(4, 1, figsize=(9, 12),
                            gridspec_kw={"height_ratios": [2, 2, 1.6, 1]})
    ax_raw, ax_kt, ax_k, ax_n = axs

    for run, style in zip(runs, ("-", "--")):
        for tip, trace in raw_traces[int(run)].items():
            ax_raw.plot(t_full, trace * 1e3, style, lw=0.4,
                        label=f"{tip} ({run})")
    ax_raw.set_ylabel("Isat [mA]")
    ax_raw.set_title(f"Runs {runs[0]}/{runs[1]}, shot {RAW_PANEL_SHOT} - "
                     f"single-shot traces (shunt-corrected)")
    ax_raw.legend(fontsize=7, ncol=4)
    ax_raw.set_xlim(0, 14)

    for i, axis in enumerate(AXES):
        ax_kt.plot(t_win, kappa_t[i], lw=0.6, label=f"$\\kappa_{axis}(t)$")
        ax_kt.axhline(kappa[i], color=f"C{i}", lw=0.8, ls=":")
    ax_kt.set_ylabel(r"$\kappa(t)$")
    ax_kt.legend(fontsize=8)
    ax_kt.set_xlim(0, 14)

    x = np.arange(len(AXES))
    for i, axis in enumerate(AXES):
        ax_k.errorbar(i, kappa[i], yerr=kappa[i] * (err_sys[i] - 1), fmt="o",
                      ms=8, capsize=4, color="C0",
                      label="rotation pair (sys bar)" if i == 0 else None)
        ax_k.plot([i] * 2, halves[i], "x", color="C1", ms=8,
                  label="window halves" if i == 0 else None)
    ax_k.set_xticks(x, [f"$\\kappa_{a}$" for a in AXES])
    ax_k.set_ylabel(r"$\kappa = A_+/A_-$")
    ax_k.axhline(1.0, color="k", lw=0.5, ls=":")
    ax_k.legend(fontsize=8)

    for i, axis in enumerate(AXES):
        ax_n.plot(t_win, n_valid[i], lw=0.6, label=axis)
    ax_n.set_ylabel("valid shots")
    ax_n.set_xlabel("Time [ms]  (machine time; DAQ trigger at 0 ms)")
    ax_n.legend(fontsize=8)
    ax_n.set_xlim(0, 14)

    for ax in (ax_raw, ax_kt, ax_n):
        ax.axvspan(*CAL_WINDOW_MS, color="0.85", zorder=0)
        ax.grid(alpha=0.3)

    finalize_figure(fig, save_fig=path, dpi=140)


if __name__ == "__main__":
    main()
