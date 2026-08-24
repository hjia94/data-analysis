"""Calibrate the P33 6-tip Mach probe from the Jun-2026 rotation runs 17-20.

Each run records the same probe in a different orientation about its shaft
(normal, 90cw, 180, 90ccw). Rotating 180 deg swaps which physical tip of the Y
and Z pairs looks upstream while the DAQ labels stay put, so combining opposing
orientations separates the collecting-area ratio ``kappa = A_+/A_-`` from the
plasma flow. The estimator maths lives in ``data_analysis.plasma.mach``; this
script is the campaign wiring -- paths, runs, the time window, and the rotation
sign table.

**X is different.** It never swaps faces, so no pairing exists: it is calibrated
from the assumption of no flow along x, making ``X+/X-`` a direct read of the
area ratio. The four runs then disagree by a factor of 2.5 against a ~0.3%
shot-to-shot error, so that disagreement -- not the shot noise -- is kappa_X's
error bar. Treat kappa_X as order-of-magnitude.

Writes ``<STEM>.npz``/``.png`` **into the raw data directory**, an exception to
the centralized-output convention: the figure is the evidence that the kappa can
be trusted, so it travels with the data.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_cal.py
"""

import datetime
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.io import open_lapd, parse_shunts
from data_analysis.io.paths import output_path
from data_analysis.io.probe_map import channel_wiring
from data_analysis.plasma.mach import (K_HUTCHINSON, area_ratio, combine_log,
                                       face_ratio, fit_calibration,
                                       mach_number, valid_current_mask)
from data_analysis.signal import clip_time_window
from data_analysis.viz.plot_utils import finalize_figure

DATA_DIR = r"D:\data\LAPD\jun2026-jia"
FILE_FMT = "{run}-Mach-p33-calibration_2026-06-12.hdf5"
RUNS = {17: "normal", 18: "90cw", 19: "180", 20: "90ccw"}

# Jun-2026 only. t=0 is bias start; 0-5 ms is the settled plateau (the +/-
# ratios drift before t=0 during breakdown, and the signal decays after ~10 ms).
# A property of this campaign's bias timing, not of Mach calibration -- which is
# why plasma.mach takes no window and this constant lives here.
CAL_WINDOW_MS = (0.0, 5.0)

STEM = "17-20-mach-calibration"
PAIRINGS = [(17, 19), (18, 20)]          # orientations 180 deg apart
AXES = ("X", "Y", "Z")
FIT_AXES = ("Y", "Z")                    # X contributes no row -- see module docstring
# How each axis' kappa was obtained; travels into the npz so a consumer never has
# to infer that X's provenance differs.
METHOD = {"X": "no-x-flow-assumption", "Y": "joint-fit", "Z": "joint-fit"}
RAW_PANEL_TIPS = ("X+", "Y+", "Z+")
RAW_PANEL_SHOT = 0

# Which lab flow component each face pair faces, and with what sign, per
# orientation: (component index, s) where s=+1 means the '+' tip looks upstream.
# Rotating the probe moves the Y pair into the lab axis Z occupied and back.
#
# Only two possibilities exist per entry, so this is fixed by running rather than
# by deriving: if the fit disagrees with the per-pair cross-check (which cannot
# be fooled by a sign error, since the product cancels the exponent either way),
# flip the signs of one axis. The table used is stored in the npz.
ORIENTATION = {
    #        Y pair          Z pair
    17: {"Y": (0, +1), "Z": (1, +1)},
    18: {"Y": (1, -1), "Z": (0, +1)},
    19: {"Y": (0, -1), "Z": (1, -1)},
    20: {"Y": (1, +1), "Z": (0, -1)},
}
N_FLOW = 2                               # lab components resolved: M_1, M_2

# Tip name at the end of a wiring description: 'MP@P33, Isat-X-' -> 'X-'.
_TIP_RE = re.compile(r"([XYZ][+-])\s*$", re.IGNORECASE)


def load_run(run, keep_full=()):
    """Read one run, windowed -> ``(currents, tarr_win, shunts, tip_channels, full)``.

    ``currents`` is ``{tip: (nshot, nt_win) amps}``, converted from volts across
    each tip's own shunt and cut to :data:`CAL_WINDOW_MS`. Channel->tip comes from
    the recorded wiring descriptions, never a hardcoded C1=X+ table: a swapped
    +/- assignment would silently report 1/kappa, which looks entirely plausible.

    Windows per tip inside the read loop, not after all six are resident: a full
    stack is (20, 500001) float64 = 80 MB, so holding six costs ~480 MB to keep
    ~8 MB each.

    ``full`` returns ``{tip: (tarr, one shot)}`` for ``keep_full``'s tips (the
    raw-trace panel) -- **copied**, because a row slice is a view that would
    otherwise pin its whole 80 MB parent for the run.
    """
    path = os.path.join(DATA_DIR, FILE_FMT.format(run=run))
    r = open_lapd(path)
    shunts = parse_shunts(r.description().raw)

    tip_channels = {}
    for (_scope, chan), desc in channel_wiring(path).items():
        # Match the tip at the end of 'MP@P33, Isat-X-'. Splitting on '-' does
        # not work: the trailing sign is itself a '-'.
        m = _TIP_RE.search(desc)
        if m:
            tip_channels[m.group(1).upper()] = chan

    missing = [f"{a}{s}" for a in AXES for s in "+-" if f"{a}{s}" not in tip_channels]
    if missing:
        raise ValueError(f"run {run}: no channel found for {missing}")
    no_shunt = [t for t in tip_channels if t not in shunts]
    if no_shunt:
        # Never default: the tips differ (300 ohm on X-, 75/43 elsewhere), so a
        # guessed value yields a wrong current with nothing downstream complaining.
        raise ValueError(f"run {run}: no shunt in description for {no_shunt}")

    currents, full, tarr_win = {}, {}, None
    for tip, chan in tip_channels.items():
        stack, tarr = r.channel(chan)
        i0, i1 = clip_time_window(tarr, *CAL_WINDOW_MS)   # takes ms; raises if < 2 samples
        # One window expression for every run and tip: kappa compares currents
        # across runs, so a window that differed between them would compare
        # different plasma conditions and deposit the difference into kappa.
        currents[tip] = stack[:, i0:i1] / shunts[tip]     # volts across shunt -> amps
        if tip in keep_full:
            full[tip] = (tarr, stack[RAW_PANEL_SHOT].copy() / shunts[tip])
        tarr_win = tarr[i0:i1]
    return currents, tarr_win, shunts, tip_channels, full


def main():
    runs = sorted(RUNS)
    data, shunts, tip_channels, raw_traces = {}, None, None, None
    tarr_win = None

    for run in runs:
        keep = RAW_PANEL_TIPS if run == runs[0] else ()
        data[run], tarr_win, shunts, tip_channels, full = load_run(run, keep_full=keep)
        if full:
            raw_traces = full
        print(f"run {run:>3} ({RUNS[run]:>6}): shunts " +
              ", ".join(f"{t}={shunts[t]:g}" for t in sorted(shunts)))

    print(f"\nwindow {CAL_WINDOW_MS[0]:g}-{CAL_WINDOW_MS[1]:g} ms "
          f"({tarr_win.size} samples), {len(data[runs[0]]['X+'])} shots/run\n")

    # --- per-run, per-axis face ratios -------------------------------------
    # (naxis, nrun): one ln R and its standard error per measurement. Arrays
    # rather than (axis, run)-keyed dicts because every consumer below wants a
    # row or a column: R_x is lnR[0], the fit reads them in order.
    lnR = np.full((len(AXES), len(runs)), np.nan)
    sig = np.full_like(lnR, np.nan)
    for i, axis in enumerate(AXES):
        for j, run in enumerate(runs):
            R = face_ratio(data[run][f"{axis}+"], data[run][f"{axis}-"])
            lnR[i, j], log_std, n = combine_log(R)
            sig[i, j] = log_std / np.sqrt(n)

    # --- primary: joint weighted fit over Y and Z --------------------------
    # 8 measurements, 4 unknowns [ln kappa_Y, ln kappa_Z, M_1, M_2]. The same lab
    # flow is seen by the Y pair in one pairing and the Z pair in the other, so
    # the measurements are not independent; the fit enforces that, an average of
    # pairings would ignore it.
    param_names = [f"ln_kappa_{a}" for a in FIT_AXES] + [f"M_{i+1}" for i in range(N_FLOW)]
    rows, y, s, row_labels = [], [], [], []
    for axis in FIT_AXES:
        for j, run in enumerate(runs):
            comp, sign = ORIENTATION[run][axis]
            row = np.zeros(len(param_names))
            row[FIT_AXES.index(axis)] = 1.0
            row[len(FIT_AXES) + comp] = (2.0 / K_HUTCHINSON) * sign
            rows.append(row)
            y.append(lnR[AXES.index(axis), j])
            s.append(sig[AXES.index(axis), j])
            row_labels.append(f"{axis}/{run}")
    design = np.array(rows)
    params, cov, residuals, chi2_dof = fit_calibration(np.array(y), np.array(s), design)

    fit_kappa = {a: np.exp(params[i]) for i, a in enumerate(FIT_AXES)}
    fit_err = {a: np.exp(np.sqrt(cov[i, i])) for i, a in enumerate(FIT_AXES)}
    fit_mach_lab = params[len(FIT_AXES):]

    # --- cross-check: per-pair product -------------------------------------
    # Immune to a design-matrix sign error, so it is the arbiter of one. X has no
    # pairing (it never flips), so its slots stay NaN rather than being filled
    # with a number that would look comparable.
    kappa_pair = {a: [np.nan] * len(PAIRINGS) for a in AXES}
    mach_pair = {a: [np.nan] * len(PAIRINGS) for a in AXES}
    for axis in FIT_AXES:
        i = AXES.index(axis)
        for p, (ra, rb) in enumerate(PAIRINGS):
            Ra, Rb = (np.exp(lnR[i, runs.index(ra)]), np.exp(lnR[i, runs.index(rb)]))
            kappa_pair[axis][p] = area_ratio(Ra, Rb)
            mach_pair[axis][p] = mach_number(Ra, Rb)

    # --- X: geometric mean over the four runs ------------------------------
    R_x = np.exp(lnR[AXES.index("X")])
    lnx_mean, lnx_std, _ = combine_log(R_x)
    kappa_x = np.exp(lnx_mean)

    kappa = np.array([kappa_x, fit_kappa["Y"], fit_kappa["Z"]])

    # --- error contributions, all multiplicative (kappa x/÷ factor) --------
    err_stat, err_time, err_sys, err_fit = (np.full(len(AXES), np.nan) for _ in range(4))
    kappa_t = np.full((len(AXES), len(PAIRINGS), tarr_win.size), np.nan)
    n_valid = np.zeros((len(AXES), len(PAIRINGS), tarr_win.size), dtype=int)

    for i, axis in enumerate(AXES):
        # (1) shot-to-shot, worst over the runs feeding this axis.
        err_stat[i] = np.exp(sig[i].max())
        # (2) in-window drift of kappa(t): a rising trend means the window is wrong.
        # For Y,Z that is the pairing product. For X it must NOT be: X never
        # flips, so the flow exponent does not cancel in a product of two runs
        # and sqrt(R_a*R_b) would be a meaningless number. Under the no-x-flow
        # assumption the per-run ratio *is* kappa, so track that instead -- one
        # curve per paired run, so the array shape stays uniform.
        for j, (ra, rb) in enumerate(PAIRINGS):
            pa, ma = data[ra][f"{axis}+"], data[ra][f"{axis}-"]
            pb, mb = data[rb][f"{axis}+"], data[rb][f"{axis}-"]
            # One mask over all four tips, so the two ratios stay sample-aligned
            # and their product is taken at the same samples.
            ok = valid_current_mask(pa, ma, pb, mb)
            n_valid[i, j] = ok.sum(axis=0)
            Ra_t = face_ratio(pa, ma, axis=0, mask=ok)   # -> one ratio per sample
            kappa_t[i, j] = Ra_t if axis == "X" else area_ratio(
                Ra_t, face_ratio(pb, mb, axis=0, mask=ok))
        # Per curve, then worst: pooling the curves would fold the
        # between-pairing (X: between-run) offset into what must be a purely
        # temporal number. That offset is contribution (3)'s job.
        drift = []
        for j in range(len(PAIRINGS)):
            f = kappa_t[i, j][np.isfinite(kappa_t[i, j]) & (kappa_t[i, j] > 0)]
            if f.size > 1:
                drift.append(np.log(f).std(ddof=1))
        err_time[i] = np.exp(max(drift)) if drift else np.nan

    # (3) systematic. Two different formulas into one slot, deliberately:
    #     Y,Z = half the spread between pairings; X = spread across the four
    #     orientations, of order x/÷1.5 rather than x/÷1.03.
    # Indexed by name, not by position: a reordered AXES must not silently
    # attach Y's error bar to X.
    for axis in FIT_AXES:
        ka, kb = kappa_pair[axis]
        err_sys[AXES.index(axis)] = np.exp(abs(np.log(ka / kb)) / 2.0)
        err_fit[AXES.index(axis)] = fit_err[axis]
    err_sys[AXES.index("X")] = np.exp(lnx_std)
    # err_fit[X] stays NaN: X has no fit row. A 0.0 would read as "measured and
    # negligible", the opposite of the truth. X's (1) and (2) are real but
    # describe one run's stability, not kappa_X's uncertainty -- only err_sys is.

    # --- printed checks (none abort) ---------------------------------------
    print("=== kappa ===")
    for i, axis in enumerate(AXES):
        method = METHOD[axis]
        bar = (f"sys x/÷{err_sys[i]:.3f}" if axis == "X"
               else f"fit x/÷{err_fit[i]:.4f}, sys x/÷{err_sys[i]:.3f}")
        print(f"  kappa_{axis} = {kappa[i]:.4f}   {bar}   [{method}]")
        print(f"      stat x/÷{err_stat[i]:.4f}, time x/÷{err_time[i]:.4f}")

    print("\n=== fit vs per-pair cross-check ===")
    for i, axis in enumerate(FIT_AXES, start=1):
        ka, kb = kappa_pair[axis]
        agree = abs(np.log(kappa[i] / np.sqrt(ka * kb)))
        flag = "  <-- DISAGREES: suspect the design matrix, not the data" \
            if agree > abs(np.log(err_sys[i])) else ""
        print(f"  {axis}: fit {kappa[i]:.4f} vs pairs {ka:.4f} / {kb:.4f}{flag}")
    # chi2 >> 1 is expected here and is not a fit failure: the weights are
    # shot-to-shot errors (~0.003 in ln R) while the residuals carry the
    # orientation-repeatability systematic (~0.06), which the 4-parameter model
    # has no term for. What would indict the fit is disagreement with the
    # per-pair check above, or a coherent sign pattern in the residuals.
    print(f"  reduced chi2 = {chi2_dof:.1f} ({len(y) - len(param_names)} dof)")
    print(f"      rms residual {np.sqrt((residuals ** 2).mean()):.4f} vs stat sigma "
          f"{max(s):.4f} in ln R -> systematic-dominated, as expected;")
    print(f"      read kappa's bar from 'sys', not 'fit'. Suspect the design "
          f"matrix only if the cross-check above disagrees.")

    print("\n=== flow ===")
    print(f"  fit lab components: " +
          ", ".join(f"M_{i+1} = {m:+.4f}" for i, m in enumerate(fit_mach_lab)))
    for axis in FIT_AXES:
        for (ra, rb), m in zip(PAIRINGS, mach_pair[axis]):
            print(f"  {axis} tips ({ra}/{rb}): M = {m:+.4f}")
    big = [m for a in FIT_AXES for m in mach_pair[a] if abs(m) > 0.3]
    if big:
        print(f"  <-- WARN: |M| > 0.3 (supersonic?) for {len(big)} pairing(s)")

    print("\n=== X spread (this IS the error bar) ===")
    for run, R in zip(runs, R_x):
        print(f"  run {run} ({RUNS[run]:>6}): R_X = {R:.4f}   ln = {np.log(R):+.4f}")
    spread = np.log(R_x).max() - np.log(R_x).min()
    print(f"  spread in ln R_X = {spread:.3f} (factor {np.exp(spread):.2f}), "
          f"kappa_X = {kappa_x:.4f} x/÷{err_sys[0]:.3f}")
    if spread > 1.5 * 0.918:
        print("  <-- WARN: spread exceeds 1.5x what runs 17-20 showed")
    print(f"  kappa_X {kappa_x:.2f} vs kappa_Y {kappa[1]:.2f}, kappa_Z {kappa[2]:.2f} "
          f"-- X differs by METHOD (assumed, not fitted), which is what its "
          f"x/÷{err_sys[0]:.2f} bar measures.")

    out_of_bounds = [a for a, k in zip(AXES, kappa) if not 0.5 <= k <= 2.0]
    if out_of_bounds:
        print(f"\n  <-- WARN: kappa outside 0.5-2.0 for {out_of_bounds}")
    print(f"\n  valid shots, min per axis: " +
          ", ".join(f"{a}={n_valid[i].min()}" for i, a in enumerate(AXES)))

    # --- outputs ------------------------------------------------------------
    npz_path = output_path(f"{STEM}.npz", explicit=DATA_DIR)
    png_path = output_path(f"{STEM}.png", explicit=DATA_DIR)

    # One payload for both the npz and the figure, so the figure provably plots
    # the numbers that were saved -- and so adding a key cannot desynchronize them.
    results = dict(
        tips=np.array(AXES),
        kappa=kappa,
        kappa_calibrated=np.ones(len(AXES), dtype=bool),
        kappa_method=np.array([METHOD[a] for a in AXES]),
        kappa_err_fit=err_fit,
        kappa_err_stat=err_stat,
        kappa_err_time=err_time,
        kappa_err_sys=err_sys,
        kappa_pairwise=np.array([kappa_pair[a] for a in AXES]),
        pairing_runs=np.array(PAIRINGS),
        mach=np.array([mach_pair[a] for a in AXES]),
        fit_mach_lab=fit_mach_lab,
        fit_cov=cov,
        fit_param_names=np.array(param_names),
        fit_residuals=residuals,
        fit_row_labels=np.array(row_labels),
        fit_design=design,
        fit_chi2_dof=chi2_dof,
        R_x_raw=R_x,
        kappa_t=kappa_t,
        tarr_win=tarr_win,
        # int16: n_valid is a shot count bounded by ~20, and (3, 2, nt_win) at
        # int64 would be 2.4 MB of the npz for numbers under 100.
        n_valid=n_valid.astype(np.int16),
        shunts=np.array([shunts[f"{a}{s}"] for a in AXES for s in "+-"]),
        tip_channels=np.array([f"{a}{s}={tip_channels[f'{a}{s}']}"
                               for a in AXES for s in "+-"]),
        window_ms=np.array(CAL_WINDOW_MS),
        K_hutchinson=K_HUTCHINSON,
        runs=np.array(runs),
        orientations=np.array([RUNS[r] for r in runs]),
        source_files=np.array([FILE_FMT.format(run=r) for r in runs]),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    np.savez(npz_path, **results)
    print(f"\nWrote {npz_path}")

    _figure(png_path, results, raw_traces)


def _figure(path, results, raw_traces):
    """Five panels: the evidence that the kappa in the npz is trustworthy.

    Reads ``results`` -- the exact dict written to the npz -- by name, so the
    figure cannot drift from the saved numbers.
    """
    kappa, kappa_t = results["kappa"], results["kappa_t"]
    err_fit, n_valid = results["kappa_err_fit"], results["n_valid"]
    kappa_pairwise, tarr_win = results["kappa_pairwise"], results["tarr_win"]
    residuals, row_labels = results["fit_residuals"], results["fit_row_labels"]
    fig, axs = plt.subplots(5, 1, figsize=(9, 14),
                            gridspec_kw={"height_ratios": [2, 2, 2, 1.5, 1]})
    ax_raw, ax_kt, ax_k, ax_res, ax_n = axs
    for ax in (ax_raw, ax_kt, ax_n):     # sharex by hand: ax_k/ax_res are not time
        ax.sharex(ax_raw)

    for tip, (tarr_full, trace) in raw_traces.items():
        ax_raw.plot(tarr_full * 1e3, trace * 1e3, lw=0.4, label=tip)
    ax_raw.set_ylabel("Isat [mA]")
    ax_raw.set_title(f"Run {sorted(RUNS)[0]}, shot {RAW_PANEL_SHOT} - "
                     f"single-shot traces (shunt-corrected)")
    ax_raw.legend(fontsize=8, ncol=len(raw_traces))

    for i, axis in enumerate(AXES):
        for j, (ra, rb) in enumerate(PAIRINGS):
            if np.all(np.isnan(kappa_t[i, j])):
                continue
            # X carries no pairing (see kappa_t above) -- label it by the single
            # run it actually came from, not as a pair.
            ax_kt.plot(tarr_win * 1e3, kappa_t[i, j], lw=0.6,
                       label=f"{axis} {ra}" if axis == "X" else f"{axis} {ra}/{rb}")
    ax_kt.set_ylabel(r"$\kappa(t)$")
    ax_kt.legend(fontsize=8, ncol=2)

    x = np.arange(len(AXES))
    for i, axis in enumerate(AXES):
        if axis == "X":
            # Open marker, no error bar: different provenance (assumed, not
            # fitted), and its bar is x/÷1.5 -- drawing it beside x/÷1.03 bars
            # would imply a comparability that does not exist.
            ax_k.plot(i, kappa[i], "o", mfc="none", ms=10, color="C3",
                      label="X: assumed no x-flow")
        else:
            ax_k.errorbar(i, kappa[i], yerr=kappa[i] * (err_fit[i] - 1),
                          fmt="o", ms=8, capsize=4, color="C0",
                          label="joint fit" if axis == "Y" else None)
            ax_k.plot([i] * len(PAIRINGS), kappa_pairwise[i], "x", color="C1", ms=8,
                      label="per-pair check" if axis == "Y" else None)
    ax_k.set_xticks(x, [f"$\\kappa_{a}$" for a in AXES])
    ax_k.set_ylabel(r"$\kappa = A_+/A_-$")
    ax_k.axhline(1.0, color="k", lw=0.5, ls=":")
    ax_k.legend(fontsize=8)

    ax_res.bar(np.arange(len(residuals)), residuals, color="C0")
    ax_res.set_xticks(np.arange(len(residuals)), row_labels, fontsize=8)
    ax_res.axhline(0, color="k", lw=0.5)
    ax_res.set_ylabel(r"fit residual [$\ln R$]")

    for i, axis in enumerate(AXES):
        for j, (ra, rb) in enumerate(PAIRINGS):
            ax_n.plot(tarr_win * 1e3, n_valid[i, j], lw=0.6,
                      label=f"{axis} {ra}/{rb}")
    ax_n.set_ylabel("valid shots")
    ax_n.set_xlabel("Time [ms]  (t=0 = bias start)")

    for ax in (ax_raw, ax_kt, ax_n):
        ax.axvspan(*CAL_WINDOW_MS, color="0.85", zorder=0)
        ax.grid(alpha=0.3)
    ax_raw.set_xlim(-5, 15)

    finalize_figure(fig, save_fig=path, dpi=140)


if __name__ == "__main__":
    main()
