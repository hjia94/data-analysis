"""Aug-2026 swept Langmuir probe: time-resolved ne, Te, Vp over a probe plane.

For one LP sweep run (run 34: P26 RF-compensated single tip, 37x37 plane at
1 cm, 9 shots per position, ``lp_scope`` C1 = I across 50 ohm, C4 = V) this:

1. **Finds the sweeps** on the voltage trace. The sweeper is a sawtooth: a rest
   at about -93 V, then a ramp to +80 V, every 0.8 ms. A sweep is one ramp, from
   leaving the rest level to the fall back. The timing is set by the trigger,
   so the windows are found once (first position, shot-averaged V). Every
   chunk checks that its own V still matches them.
2. **Removes the antenna pickup.** During the burst (8-13.3 ms) both V and I
   carry the 188 kHz drive. A boxcar of exactly one drive period nulls it and
   all its harmonics. It also smooths V by ~1.4 V (the ramp moves 0.26 V/us),
   well under Te. The trace is then decimated 10x (to 200 ns).
3. **Labels each shot's antenna mode** (A = south, B = north, AB = both) from
   the antenna currents on ``bdot_scope`` C7/C8.
4. **Fits** with the shared :func:`data_analysis.plasma.langmuir.analyze_IV`.
   - Each mode is fitted on its shot-averaged current (V is common to all
     shots).
   - "all" is fitted on the average over every shot, regardless of mode.
   - Every single shot is also fitted, which gives the shot-to-shot scatter.
5. **Calibrates ne** against the interferometer. The probe area is not known,
   so ``ne`` is first computed for 1 cm^2 (``ne_uncal``). For each sweep,
   ``ne_uncal`` is integrated along x on the row through y = 0, using the "all"
   fit. That integral is scaled to the interferometer's line-integrated
   density in the sweep window (nearest chord, phase_p29).
   - One factor per sweep is applied to every mode and position, so modes and
     positions stay comparable.
   - The scan spans only +-18 cm. Where the profile has not reached zero at the
     plane's edge, the integral undercounts and ne is biased high. The script
     prints that edge fraction.

The electron current reads **negative** on this digitizer (verified on the
center position: the large current is at high V), so ``I_POLARITY = -1``.

    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_IV.py 34 --workers 24

Output: ``<out-dir>/<run>-iv-plane.npz``. Plot and slider page:
``Aug2026_IV_slider.py``.
"""

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from data_analysis.io import open_lapd, read_interferometer
from data_analysis.plasma.langmuir import analyze_IV_safe, interferometer_chords, chord_for_port
from data_analysis.signal import line_integral

from Aug2026_bdot import CH_I, OUT_DIR, run_file

LP_SCOPE = "lp_scope"
CH_LP_I, CH_LP_V = "C1", "C4"
PORT = "26"
RESISTOR = 50.0                     # ohm, current shunt
I_POLARITY = -1                     # electron current reads negative on this digitizer
AREA_CM2 = 1.0                      # placeholder; the interferometer fixes the scale
F_DRIVE = 188e3                     # antenna drive, Hz -- the boxcar nulls it
DECIMATE = 10
TRIM = 0.03                         # fraction of each ramp dropped at both ends
REST_MARGIN_V = 5.0                 # V above the rest level that counts as "sweeping"
MODES = ("all", "A", "B", "AB")
QTY = ("Vp", "Te", "ne")
T_OFFSET = 0.0                      # scope t = 0 vs interferometer t = 0, s (both: plasma start)


def shot_groups(path):
    """Live shots grouped by position: ``(shots_per_position, xy)``, scan order."""
    with h5py.File(path, "r") as f:
        (mg,) = f["Control/Positions"].values()
        a = mg["positions_array"][:]
        a = a[a["shot_num"] != 0]
        # run 34 ends with 10 shots that have a position but no scope data
        g = f[LP_SCOPE]
        a = a[[f"{CH_LP_V}_data" in g.get(f"shot_{s}", {}) for s in a["shot_num"]]]
    xy = np.round(np.c_[a["x"], a["y"]], 1)
    new = np.r_[True, np.any(xy[1:] != xy[:-1], axis=1)]
    starts = np.flatnonzero(new)
    groups = np.split(a["shot_num"].astype(int), starts[1:])
    return groups, xy[starts]


def sweep_windows(v):
    """``(start, stop)`` sample indices of every full ramp in a smoothed V trace."""
    rest = np.percentile(v, 1)
    up = (v > rest + REST_MARGIN_V).astype(np.int8)
    d = np.diff(up)
    starts, stops = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    win = [(a, stops[stops > a][0]) for a in starts if (stops > a).any()]
    n = np.array([b - a for a, b in win])
    win = [w for w, k in zip(win, n) if k > 0.9 * n.max()]      # drop a ramp cut by the record
    L = min(b - a for a, b in win)                               # one common length, so sweeps stack
    cut = int(TRIM * L)
    return [(a + cut, a + L - cut) for a, _ in win]


def smooth(x, box):
    return uniform_filter1d(np.asarray(x, np.float64), box, axis=-1)


def antenna_mode(Ia, Ib):
    """Per-shot mode from the two antenna currents (nshot, nt)."""
    amp = np.stack([Ia.std(1), Ib.std(1)])
    on = amp > 0.25 * max(amp.max(), 1e-12)
    return np.where(on[0] & on[1], "AB", np.where(on[0], "A", np.where(on[1], "B", "off")))


def position_job(args):
    path, shots, win, box, v_ref = args
    warnings.simplefilter("ignore")                 # curve_fit overflow on failed fits -> NaN anyway
    run = open_lapd(path)
    V, _ = run.channel(CH_LP_V, shots=shots, scope_name=LP_SCOPE)
    I, _ = run.channel(CH_LP_I, shots=shots, scope_name=LP_SCOPE)
    Ia, _ = run.channel(CH_I[0], shots=shots, scope_name="bdot_scope")
    Ib, _ = run.channel(CH_I[1], shots=shots, scope_name="bdot_scope")
    mode = antenna_mode(Ia, Ib)
    V = smooth(V, box)
    I = smooth(I, box) * (I_POLARITY / (RESISTOR * AREA_CM2))      # A/cm^2, electron-positive
    v = V.mean(0)
    sw = np.stack([v[a:b:DECIMATE] for a, b in win])               # (nsweep, n)
    ref = np.stack([v_ref[a:b:DECIMATE] for a, b in win])
    drift = float(np.max(np.abs(sw - ref)))
    Isw = np.stack([I[:, a:b:DECIMATE] for a, b in win], axis=1)   # (nshot, nsweep, n)

    nsw = len(win)
    by_mode = np.full((len(MODES), nsw, 3), np.nan)
    for m, name in enumerate(MODES):
        sel = mode != "off" if name == "all" else mode == name
        if sel.any():
            Im = Isw[sel].mean(0)
            for k in range(nsw):
                by_mode[m, k] = analyze_IV_safe(sw[k], Im[k])
    per_shot = np.full((len(shots), nsw, 3), np.nan)
    for s in range(len(shots)):
        for k in range(nsw):
            per_shot[s, k] = analyze_IV_safe(sw[k], Isw[s, k])
    return by_mode, per_shot, mode, drift


def calibrate(path, ne_all, xy, t_start, t_stop):
    """Per-sweep factor: interferometer int(ne dl) / probe int(ne dx) on the y = 0 row."""
    chord, away = chord_for_port(interferometer_chords(path), PORT, path)
    ch = read_interferometer(path, channels=[chord])[chord]
    row = np.flatnonzero(np.abs(xy[:, 1]) < 0.05)
    row = row[np.argsort(xy[row, 0])]
    x = xy[row, 0]
    if row.size < 2:
        print("ne calibration skipped: no y = 0 row among the processed positions")
        return np.full(len(t_start), np.nan), chord, np.full(len(t_start), np.nan)
    ti, ni = ch.t_ms * 1e-3, ch.ne_line_integral_avg_cm2
    fac, edge = np.full(len(t_start), np.nan), np.full(len(t_start), np.nan)
    for k, (a, b) in enumerate(zip(t_start + T_OFFSET, t_stop + T_OFFSET)):
        prof = ne_all[row, k]
        ok = np.isfinite(prof)
        integral = line_integral(prof[ok], x[ok])
        win = (ti >= a) & (ti <= b)
        if win.any() and integral > 0:
            fac[k] = ni[win].mean() / integral
            edge[k] = max(prof[ok][0], prof[ok][-1]) / prof[ok].max()
    print(f"ne calibration: chord {chord} ({away} ports from P{PORT}), row y = 0, "
          f"x {x.min():.0f}..{x.max():.0f} cm; profile edge / peak up to {np.nanmax(edge):.2f}")
    return fac, chord, edge


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="process only the first N positions (testing)")
    args = ap.parse_args()
    t0 = time.time()

    path = run_file(args.run)
    groups, xy = shot_groups(path)
    if args.limit:
        groups, xy = groups[:args.limit], xy[:args.limit]
    run = open_lapd(path)
    V0, t = run.channel(CH_LP_V, shots=groups[0], scope_name=LP_SCOPE)
    dt = float(t[1] - t[0])
    box = int(round(1 / (F_DRIVE * dt)))
    v_ref = smooth(V0, box).mean(0)
    win = sweep_windows(v_ref)
    t_start = np.array([t[a] for a, _ in win])
    t_stop = np.array([t[b - 1] for _, b in win])
    t_mid = (t_start + t_stop) / 2
    Ia, ta = run.channel(CH_I[0], shots=groups[0][:3], scope_name="bdot_scope")
    Ib, _ = run.channel(CH_I[1], shots=groups[0][:3], scope_name="bdot_scope")
    env = np.abs(np.r_[Ia, Ib] - np.r_[Ia, Ib][:, :100].mean(1, keepdims=True)).max(0)
    on = np.flatnonzero(uniform_filter1d(env, int(round(1 / (F_DRIVE * (ta[1] - ta[0]))))) > 0.5 * env.max())
    burst = (float(ta[on[0]]), float(ta[on[-1]]))
    print(f"{path.name}: {len(groups)} positions, {sum(map(len, groups))} shots; "
          f"{len(win)} sweeps {t_start[0]*1e3:.2f}-{t_stop[-1]*1e3:.2f} ms, "
          f"{(win[0][1]-win[0][0])//DECIMATE} points each; boxcar {box} samples; "
          f"burst {burst[0]*1e3:.2f}-{burst[1]*1e3:.2f} ms", flush=True)

    jobs = [(path, g, win, box, v_ref) for g in groups]
    with ProcessPoolExecutor(args.workers) as ex:
        out = []
        for i, r in enumerate(ex.map(position_job, jobs, chunksize=2)):
            out.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)} positions, {time.time()-t0:.0f} s", flush=True)

    by_mode = np.stack([o[0] for o in out])                        # (npos, nmode, nsweep, 3)
    nshot = max(len(g) for g in groups)
    per_shot = np.full((len(groups), nshot, len(win), 3), np.nan)
    mode = np.full((len(groups), nshot), "", dtype="<U3")
    for p, o in enumerate(out):
        per_shot[p, :len(o[1])] = o[1]
        mode[p, :len(o[2])] = o[2]
    drift = np.array([o[3] for o in out])
    print(f"sweep-voltage drift from the reference windows: max {drift.max():.2f} V")
    fail = np.isnan(by_mode[..., 2]).mean((0, 2))
    print("fit failure rate by mode:", dict(zip(MODES, np.round(fail, 3))))

    fac, chord, edge = calibrate(path, by_mode[:, 0, :, 2], xy, t_start, t_stop)
    res = dict(
        run=args.run, xy=xy, shots=np.array([g[0] for g in groups]), mode=mode, modes=np.array(MODES),
        t_mid=t_mid, t_start=t_start, t_stop=t_stop, burst=np.array(burst),
        Vp=by_mode[..., 0], Te=by_mode[..., 1], ne_uncal=by_mode[..., 2],
        ne=by_mode[..., 2] * fac, cal_factor=fac, cal_chord=chord, cal_edge=edge,
        shot_Vp=per_shot[..., 0], shot_Te=per_shot[..., 1], shot_ne=per_shot[..., 2] * fac,
        drift=drift, box=box, decimate=DECIMATE, trim=TRIM, area_cm2=AREA_CM2,
    )
    out_path = args.out_dir / f"{args.run:02d}-iv-plane.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **res)
    print(f"saved {out_path} ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
