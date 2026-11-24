"""Aug-2026 B-dot bicoherence: three-wave coupling between the ~188 kHz wave and low frequencies.

For one B-dot run this computes, on the steady part of the antenna burst:

1. **Fixed probe P30** (C4-C6), pooled over every shot of each antenna mode
   (A / B / AB / all): the auto-bicoherence ``b2(f1, f2)`` with ``f1`` a low
   frequency and ``f2`` near the pump, so the third wave is a sideband
   ``f2 + f1`` -- or the pump itself when ``f2 = f_pump - f1``. P30 does not
   move, so all its shots share one biphase and pool.
2. **Trigger-locking null.** The pump is phase-locked to the trigger, and so can
   be other content; a triad of locked components gives ``b2 -> 1`` with no
   interaction. The surrogate ``cross_bicoherence(x[perm], x, x)`` takes ``f1``
   from a different shot of the same mode: it keeps everything that is locked
   to the trigger and destroys coupling to the shot's own low-frequency
   fluctuation. A triad is **coupled** when ``b2`` exceeds the chance
   threshold and the surrogate stays at or below the threshold. Each listed triad
   also reports the shot-to-shot phase consistency ``|<X/|X|>|`` of its three
   components.
3. **Pump-line summary.** Run 30 showed coupling at *every* ``f1`` in
   0.5-20 kHz (broadband modulation: sidebands ``f_pump +- f1`` whose phase
   follows the probe's own low-frequency fluctuation), so a single triad is
   not representative. The summary is the mean ``b2`` on the pump line
   ``f2 = f_pump`` (upper sidebands) over the ``f1`` band, against its
   no-coupling value ``1/N``.
4. **Moving probe P28** (C1-C3): that pump-line mean per position, pooling
   the shots and modes there (24 realizations per position on run 30). With
   ``--triad f1 f2`` the scan uses that single triad instead.

``b2`` is unchanged by per-frequency gain, so raw dB/dt volts are used (no
integration, no coil calibration needed).

    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_bicoherence.py 30
    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_bicoherence.py 35 --workers 24

Output: ``<out-dir>/<run>-bicoherence.npz`` and ``figures/<run>-bicoherence.png``.
"""

import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.io import open_lapd
from data_analysis.signal import bicoherence, cross_bicoherence

from Aug2026_bdot import CH_I, OUT_DIR, SCOPE, burst_windows, run_file

CH_P30 = ["C4", "C5", "C6"]
CH_P28 = ["C1", "C2", "C3"]
COMP = ["Bx", "By", "Bz"]
MODES = ("A", "B", "AB", "all")


def read_chunk(args):
    path, shots, sl = args
    run = open_lapd(path)
    out = {}
    for ch in CH_P30 + CH_P28 + CH_I:
        v, _ = run.channel(ch, shots=shots, scope_name=SCOPE)
        if v is not None:                                     # e.g. run 35 recorded only P30 By
            out[ch] = v[:, sl].astype(np.float32)
    return out


def antenna_freq(I, dt):
    """Peak frequency (Hz) of a current trace, zero-padded FFT in 150-250 kHz."""
    n = 8 * I.size
    s = np.abs(np.fft.rfft((I - I.mean()) * np.hanning(I.size), n))
    f = np.fft.rfftfreq(n, dt)
    band = (f > 150e3) & (f < 250e3)
    return float(f[band][np.argmax(s[band])])


def phase_consistency(x, dt, nperseg, freqs):
    """``|<X/|X|>|`` over shots at each frequency, mean over segment index."""
    nseg = x.shape[1] // nperseg
    seg = x[:, :nseg * nperseg].reshape(x.shape[0], nseg, nperseg)
    seg = seg - seg.mean(-1, keepdims=True)
    t = np.arange(nperseg) * dt
    w = np.hanning(nperseg)
    out = []
    for f in freqs:
        z = (seg * w) @ np.exp(-2j * np.pi * f * t)          # (nshot, nseg)
        out.append(np.abs(np.mean(z / np.abs(z), axis=0)).mean())
    return np.array(out)


_D = {}            # channel -> (nshot, nt) float32 traces, inherited by forked workers
_MODE = None


def p30_job(args):
    """Bicoherence, surrogate and top triads of one P30 component for one mode."""
    m, c, ch, dt, kw, f_pump, seed = args
    sel = mode_select(_MODE, m)
    x = _D[ch][sel]
    r = bicoherence(x, dt, **kw)
    s = cross_bicoherence(x[shuffle_within(_MODE[sel], np.random.default_rng(seed))], x, x, dt, **kw)
    coupled = (r.b2 > r.threshold) & (s.b2 <= r.threshold)
    b = np.where(coupled, r.b2, -1).ravel()
    top = []
    for k in [k for k in np.argsort(b)[::-1][:3] if b[k] > 0]:
        i, j = np.unravel_index(k, r.b2.shape)
        pc = phase_consistency(x, dt, kw["nperseg"], (r.f1[i], r.f2[j], r.f1[i] + r.f2[j]))
        top.append((r.b2[i, j], s.b2[i, j], r.f1[i], r.f2[j], *pc))
    j = np.argmin(abs(r.f2 - f_pump))
    line = (np.nanmean(r.b2[:, j]), np.nanmean(s.b2[:, j]))
    return m, c, r, s.b2, coupled, top, line


def mode_select(mode, m):
    """Shots of antenna mode ``m``. For ``"all"``: every mode that is on and has
    at least 2 shots, so that :func:`shuffle_within` can pair each shot with
    another shot of its own mode."""
    if m != "all":
        return mode == m
    names, counts = np.unique(mode, return_counts=True)
    return np.isin(mode, names[(counts >= 2) & (names != "off")])


def shuffle_within(groups, rng):
    """Permutation with no fixed point that only swaps shots within the same group."""
    q = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if idx.size < 2:
            raise ValueError(f"group {str(g)!r} has 1 shot; it has no other shot to pair with")
        p = rng.permutation(idx.size)
        q[idx[p]] = idx[np.roll(p, 1)]                        # cyclic shift: nobody stays put
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=36)
    ap.add_argument("--df", type=float, default=500.0, help="frequency resolution, Hz")
    ap.add_argument("--f1", type=float, nargs=2, default=(500.0, 20e3), help="low-frequency band, Hz")
    ap.add_argument("--f2", type=float, nargs=2, default=(168e3, 208e3), help="pump-side band, Hz")
    ap.add_argument("--triad", type=float, nargs=2, help="(f1, f2) in Hz for the P28 position scan")
    args = ap.parse_args()
    t0 = time.time()

    path = run_file(args.run)
    run = open_lapd(path)
    _setup, pos, _npos, _nshot = run.positions()
    pos = pos[pos["shot_num"] != 0]
    shots = [int(s) for s in pos["shot_num"]]
    t = run.time_array(CH_I[0], scope_name=SCOPE)
    dt = float(t[1] - t[0])
    I = np.stack([run.channel(c, shots=shots[:3], scope_name=SCOPE)[0] for c in CH_I]).reshape(-1, t.size)
    w = burst_windows(I, t)
    st = w["steady"]
    L = st.stop - st.start
    nperseg = min(int(round(1 / (args.df * dt))), L)
    df = 1 / (nperseg * dt)

    chunks = [shots[i:i + args.chunk] for i in range(0, len(shots), args.chunk)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(read_chunk, [(path, c, st) for c in chunks]))
    D = {ch: np.concatenate([p[ch] for p in parts]) for ch in parts[0]}
    amp = np.stack([D[c].std(1) for c in CH_I])
    on = amp > 0.25 * amp.max(1, keepdims=True)
    mode = np.where(on[0] & on[1], "AB", np.where(on[0], "A", np.where(on[1], "B", "off")))
    fA = antenna_freq(D[CH_I[0]][np.argmax(amp[0])], dt)
    fB = antenna_freq(D[CH_I[1]][np.argmax(amp[1])], dt)
    print(f"{path.name}: {len(shots)} shots, modes {dict(zip(*np.unique(mode, return_counts=True)))}\n"
          f"  steady {t[st.start]*1e3:.2f}-{t[st.stop-1]*1e3:.2f} ms, {L // nperseg} segment(s)/shot, "
          f"df {df:.0f} Hz; antenna A {fA/1e3:.2f} kHz, B {fB/1e3:.2f} kHz; read {time.time()-t0:.0f}s",
          flush=True)

    kw = dict(nperseg=nperseg, f1_band=tuple(args.f1), f2_band=tuple(args.f2))
    f_ant = fA if amp[0].max() >= amp[1].max() else fB
    f_pump = round(f_ant / df) * df                          # nearest FFT bin
    global _MODE
    _D.update(D)
    _MODE = mode
    f_mode = {"A": round(fA / df) * df, "B": round(fB / df) * df}   # each mode's own pump line
    jobs = [(m, c, ch, dt, kw, f_mode.get(m, f_pump), 10 * k + c) for k, m in enumerate(MODES)
            if mode_select(mode, m).sum() >= 2
            for c, ch in enumerate(CH_P30) if ch in D]
    if not jobs:
        raise SystemExit(f"run {args.run}: no P30 channel recorded, or no antenna mode with "
                         f"2+ shots; nothing to analyse")
    with ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=multiprocessing.get_context("fork")) as ex:
        out = list(ex.map(p30_job, jobs))
    res = {(m, c): (r, sb2, coupled) for m, c, r, sb2, coupled, _, _ in out}

    f1g, f2g = next(iter(res.values()))[0].f1, next(iter(res.values()))[0].f2
    missing = [f"P30 {COMP[c]}" for c, ch in enumerate(CH_P30) if ch not in D]
    missing += [f"P28 {COMP[c]}" for c, ch in enumerate(CH_P28) if ch not in D]
    if missing:
        print("  not recorded in this run:", ", ".join(missing))
    print(f"\nP30 auto-bicoherence, f1 {args.f1[0]/1e3:g}-{args.f1[1]/1e3:g} kHz x f2 "
          f"{args.f2[0]/1e3:g}-{args.f2[1]/1e3:g} kHz ({f1g.size}x{f2g.size} points); pump line f2 = "
          f"{f_mode['A']/1e3:.1f} kHz (A), {f_mode['B']/1e3:.1f} kHz (B), {f_pump/1e3:.1f} kHz (AB, all)")
    print("  mode comp     N  thresh  1/N    | pump-line mean b2 [surrogate] | coupled / locked points"
          " | top coupled triads: b2 [surr] f1+f2 kHz (phase consistency f1/f2/f3)")
    lines = {}
    for m, c, r, sb2, coupled, top, line in out:
        lines[m, c] = line
        locked = (r.b2 > r.threshold) & ~coupled
        txt = "; ".join(f"{b:.2f} [{sb:.3f}] {f1/1e3:.1f}+{f2/1e3:.1f} ({p1:.2f}/{p2:.2f}/{p3:.2f})"
                        for b, sb, f1, f2, p1, p2, p3 in top) or "none"
        print(f"  {m:4s} {COMP[c]}  {r.n_avg:5d} {r.threshold:.4f} {1/r.n_avg:.4f} | "
              f"{line[0]:.3f} [{line[1]:.4f}]               | {coupled.sum():4d} / {locked.sum():4d}"
              f"             | {txt}")

    if args.triad:
        f1b, f2b = (args.triad[0],) * 2, (args.triad[1],) * 2
        what = f"b2 at {args.triad[0]/1e3:.1f} + {args.triad[1]/1e3:.1f} kHz"
    else:
        f1b, f2b = tuple(args.f1), (f_pump, f_pump)
        what = f"pump-line mean b2 (f1 {args.f1[0]/1e3:g}-{args.f1[1]/1e3:g} kHz, f2 {f_pump/1e3:.1f} kHz)"
    # antennas at different frequencies (e.g. run 17): pool only shots that share the pump
    pool = (mode != "off") if abs(fA - fB) < df else np.isin(mode, ["A", "AB"] if f_ant == fA else ["B", "AB"])
    px, py = np.round(pos["x"], 1), np.round(pos["y"], 1)
    xs, ys = np.unique(px), np.unique(py)
    scan = np.full((3, len(ys), len(xs)), np.nan)
    nrz = np.zeros((len(ys), len(xs)), int)
    pkw = dict(nperseg=nperseg, f1_band=f1b, f2_band=f2b)
    for jy, y in enumerate(ys):
        for ix, xv in enumerate(xs):
            sel = (px == xv) & (py == y) & pool
            if sel.sum() * (L // nperseg) < 3:
                continue
            for c, ch in enumerate(CH_P28):
                if ch not in D:
                    continue
                r = bicoherence(D[ch][sel], dt, **pkw)
                scan[c, jy, ix] = np.nanmean(r.b2)
                nrz[jy, ix] = r.n_avg
    if (nrz > 0).any():
        n_typ = int(np.median(nrz[nrz > 0]))
        print(f"\nP28 per position: {what}; {n_typ} realizations/position, "
              f"no-coupling level 1/N = {1/n_typ:.3f}")
    else:
        n_typ = 0
        print("\nP28 per position: skipped (P28 not recorded, or < 3 realizations at every position)")
    for c in range(3):
        if not np.isfinite(scan[c]).any():
            continue
        jy, ix = np.unravel_index(np.nanargmax(scan[c]), scan[c].shape)
        print(f"  P28 {COMP[c]}: median {np.nanmedian(scan[c]):.3f}, max {scan[c, jy, ix]:.3f} "
              f"at (x, y) = ({xs[ix]:g}, {ys[jy]:g}) cm, positions > 2/N: "
              f"{np.sum(scan[c] > 2 / n_typ)}/{np.isfinite(scan[c]).sum()}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.run:02d}-bicoherence.npz"
    save = dict(source=str(path), dt=dt, nperseg=nperseg, f1=f1g, f2=f2g, fA=fA, fB=fB,
                f_pump=f_pump, t_steady=(t[st.start], t[st.stop - 1]), xs=xs, ys=ys,
                p28_b2=scan, p28_n=nrz, p28_what=what)
    for (m, c), (r, sb2, coupled) in res.items():
        save[f"b2_{m}_{COMP[c]}"] = r.b2
        save[f"surr_{m}_{COMP[c]}"] = sb2
        save[f"n_{m}_{COMP[c]}"] = r.n_avg
        save[f"line_{m}_{COMP[c]}"] = lines[m, c]
    np.savez(out, **save)

    plot(args.run, res, what, n_typ, xs, ys, scan, args.out_dir)
    print(f"\nsaved {out}  ({time.time() - t0:.0f}s)")


def plot(run, res, what, n_typ, xs, ys, scan, out_dir):
    modes = [m for m in MODES if any((m, c) in res for c in range(3))]
    line = len(xs) < 3 or len(ys) < 3
    fig, ax = plt.subplots(len(modes) + 1, 3, figsize=(15, 3.3 * (len(modes) + 1)),
                           constrained_layout=True, squeeze=False)
    vmax = max(np.nanmax(res[k][0].b2) for k in res)
    for row, m in enumerate(modes):
        for c in range(3):
            a = ax[row, c]
            if (m, c) not in res:
                a.set_title(f"P30 {COMP[c]}: not recorded")
                a.axis("off")
                continue
            r, sb2, coupled = res[m, c]
            ext = [r.f2[0] / 1e3, r.f2[-1] / 1e3, r.f1[0] / 1e3, r.f1[-1] / 1e3]
            im = a.imshow(r.b2, origin="lower", aspect="auto", extent=ext, cmap="Blues", vmin=0, vmax=vmax)
            lk = (r.b2 > r.threshold) & ~coupled
            F2, F1 = np.meshgrid(r.f2 / 1e3, r.f1 / 1e3)
            a.scatter(F2[lk], F1[lk], s=6, marker="x", color="0.35", linewidths=0.6,
                      label="explained by trigger locking")
            a.set(title=f"P30 {COMP[c]}, mode {m}  (N={r.n_avg})", xlabel="f2 (kHz)", ylabel="f1 (kHz)")
            if row == 0 and c == 0 and lk.any():
                a.legend(loc="upper left", fontsize=7, framealpha=0.8)
        fig.colorbar(im, ax=ax[row, :], label="b²", shrink=0.9)
    for c in range(3):
        a = ax[-1, c]
        if not np.isfinite(scan[c]).any():
            a.axis("off")
            continue
        title = f"P28 {COMP[c]} per position (1/N = {1 / n_typ:.3f})"
        if line:
            coord, vals = (xs, scan[c, 0]) if len(xs) >= len(ys) else (ys, scan[c, :, 0])
            a.plot(coord, vals, "o-", color="#2a6fbb", lw=2, ms=4)
            a.axhline(1 / n_typ, color="0.5", lw=1, ls="--")
            a.set(xlabel="position (cm)", ylabel="b²", ylim=(0, None), title=title)
        else:
            d = (xs[1] - xs[0]) / 2 if len(xs) > 1 else 0.5
            im = a.imshow(scan[c], origin="lower", extent=[xs[0] - d, xs[-1] + d, ys[0] - d, ys[-1] + d],
                          cmap="Blues", vmin=0, vmax=np.nanmax(scan))
            a.plot(0, 0, "+", color="0.3", ms=8)
            a.set(xlabel="x (cm)", ylabel="y (cm)", title=title)
            if c == 2:
                fig.colorbar(im, ax=ax[-1, :], label=what.split(" (")[0], shrink=0.9)
    fig.suptitle(f"Aug-2026 run {run:02d}: B-dot bicoherence, low frequency f1 × pump-side f2 → f1 + f2")
    p = out_dir / "figures" / f"{run:02d}-bicoherence.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=80)
    print("figure:", p)


if __name__ == "__main__":
    main()
