"""Aug-2026 LP plane: slider page and summary figure from ``<run>-iv-plane.npz`` (Aug2026_IV.py).

The page (``slider_html``, the same renderer as the Jun-2026 sliders):
- **Slider:** scrubs the sweep time.
- **Dropdown:** picks the antenna mode (all / A / B / AB).
- **Panels:** ne, Te and Vp as heatmaps over the probe plane, plus their change from the
  last sweep before the burst (dne/ne in %, dTe, dVp).
- **Context plot:** ne, Te and Vp at the plane center versus time for every mode, with the
  antenna burst shaded.

The figure: the "all" maps at three times (before, during, after the burst), with the
time traces per mode at the points in ``POINTS`` (the nearest recorded position to
each; marked on the maps).

    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_IV_slider.py 34

Output: ``<out-dir>/figures/<run>-iv-plane.html`` (+ ``-bundle.npz``) and ``<run>-iv-plane.png``.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.viz.plot_utils import grid_frames
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

from Aug2026_bdot import OUT_DIR

FIELDS = (("ne", "1e12 cm^-3", "viridis"), ("Te", "eV", "inferno"), ("Vp", "V", "cividis"))
#: change from the last sweep before the burst: (name, source, unit, relative?)
CHANGES = (("dne/ne", "ne", "%", True), ("dTe", "Te", "eV", False), ("dVp", "Vp", "V", False))
CENTER_R_CM = 1.5            # positions within this radius of (0, 0) make the page's "center" trace
POINTS = ((0, 0), (3.5, 3.5), (5, 5))    # figure time traces, cm; the nearest recorded position is used
MARKERS = ("o", "s", "^")


def load(path):
    r = dict(np.load(path))
    r["ne"] = r["ne"] / 1e12
    t = r["t_mid"]
    r["ref"] = int(np.flatnonzero(r["t_stop"] < r["burst"][0])[-1])   # last sweep before the burst
    for name, q, _, rel in CHANGES:
        d = r[q] - r[q][..., r["ref"]][..., None]
        r[name] = 100 * d / r[q][..., r["ref"]][..., None] if rel else d
    return r


def scale(a, sym=False):
    """Fixed colour range shared by every mode and time: the 2-98 % spread, so the
    noisy fits at the low-density edge do not set it."""
    if sym:
        h = float(np.nanpercentile(np.abs(a), 96))
        return -h, h
    lo, hi = np.nanpercentile(a, [2, 98])
    return float(lo), float(hi)


def center_trace(r, q, m):
    near = np.hypot(*r["xy"].T) <= CENTER_R_CM
    return np.nanmedian(r[q][near, m], axis=0)


def nearest_position(xy, p):
    """Index of the recorded position nearest ``p``, and every position tied with it.

    On a 1 cm grid a half-integer request such as (3.5, 3.5) is equidistant from
    four positions. The tie goes to the one nearest the line through the origin
    and ``p`` (so a point on the diagonal stays on it), then to the smaller
    radius: deterministic and stated, rather than whichever came first in scan order.
    """
    p = np.asarray(p, float)
    d = np.hypot(*(xy - p).T)
    tied = np.flatnonzero(np.isclose(d, d.min()))
    u = p / (np.hypot(*p) or 1.0)
    off_line = np.abs(xy[tied, 0] * u[1] - xy[tied, 1] * u[0])
    best = tied[np.lexsort((np.hypot(*xy[tied].T), np.round(off_line, 6)))[0]]
    return int(best), tied


def bundle(r, run):
    t = r["t_mid"] * 1e3
    modes = [str(m) for m in r["modes"]]
    counts = {m: int((r["mode"] == m).sum()) for m in ("A", "B", "AB")}
    groups = []
    for m, name in enumerate(modes):
        fields = []
        specs = [(q, unit, cmap, scale(r[q])) for q, unit, cmap in FIELDS]
        specs += [(n, unit, "RdBu_r", scale(r[n], sym=True)) for n, _, unit, _ in CHANGES]
        for q, unit, cmap, (lo, hi) in specs:
            frames, xs, ys = grid_frames(r["xy"][:, 0], r["xy"][:, 1], r[q][:, m])
            fields.append({"name": q, "unit": unit, "cmap": cmap, "vmin": lo, "vmax": hi,
                           "frames": frames})
        label = "all shots" if name == "all" else f"{name} ({counts[name]} shots)"
        groups.append({"name": label, "fields": fields})
    burst = r["burst"] * 1e3
    ctx = {"title": f"ne at the plane center (r <= {CENTER_R_CM:g} cm) vs time",
           "x": {"label": "time", "unit": "ms", "values": t},
           "y": {"label": "ne", "unit": "1e12 cm^-3"},
           "traces": [{"name": name, "values": center_trace(r, "ne", m)} for m, name in enumerate(modes)],
           "span": [float(burst[0]), float(burst[1])], "span_label": "antenna burst"}
    labels = [f"{tt:.2f} ms" + (" (burst)" if burst[0] <= tt <= burst[1] else "") for tt in t]
    edge = np.nanmax(r["cal_edge"])
    return {
        "schema": SCHEMA_VERSION,
        "title": f"Aug-2026 run {run:02d}: P26 swept LP, ne / Te / Vp over the plane vs time",
        "geometry": "plane",
        "axis": {"name": "time", "unit": "ms", "values": t, "labels": labels},
        "x": {"label": "x", "unit": "cm", "values": xs},
        "y": {"label": "y", "unit": "cm", "values": ys},
        "groups": groups,
        "group_label": "Antenna mode",
        "context": ctx,
        "provenance": {
            "source": f"run {run:02d}, {len(r['xy'])} positions x {r['mode'].shape[1]} shots",
            "params": {"sweep": f"{(r['t_stop'][0]-r['t_start'][0])*1e3:.2f} ms ramp, -93..+80 V",
                       "pickup filter": f"{int(r['box'])}-sample boxcar (one 188 kHz period)",
                       "ne": f"interferometer-scaled ({r['cal_chord']}, y = 0 row)",
                       "fit": "langmuir.analyze_IV on the mode-averaged current"},
            "details": [
                "A = south antenna, B = north, AB = both; 'all' averages every shot.",
                f"ne scale: per-sweep factor from the y = 0 row integrated over x = -18..18 cm. The profile at "
                f"the row ends is up to {edge*100:.0f}% of its peak, so the integral misses plasma outside the "
                "scan and ne is biased high by roughly that much.",
                f"dne/ne, dTe, dVp: change from the last sweep before the burst "
                f"({r['t_mid'][r['ref']]*1e3:.2f} ms), same shots and mode.",
                "Colour scales are fixed across modes and times (2-98 % of all values), so switching "
                "the mode is a direct comparison.",
                "Grey cells: positions not reached (the scan stopped at y = -16 cm) or fits that failed.",
            ],
        },
        "warning": None,
    }


def figure(r, run, out, points=POINTS):
    t = r["t_mid"] * 1e3
    burst = r["burst"] * 1e3
    pick = [int(np.argmin(abs(t - x))) for x in (burst[0] - 0.7, (burst[0] + burst[1]) / 2, burst[1] + 1.5)]
    loc = []
    for p in points:
        i, tied = nearest_position(r["xy"], p)
        loc.append(i)
        note = f" (tied with {', '.join(f'({x:g}, {y:g})' for x, y in r['xy'][tied] if (x, y) != tuple(r['xy'][i]))})" \
            if tied.size > 1 else ""
        print(f"time trace for ({p[0]:g}, {p[1]:g}) cm -> recorded ({r['xy'][i, 0] + 0.0:g}, {r['xy'][i, 1] + 0.0:g}) cm{note}")
    npt = len(loc)
    trace_modes = [m for m, name in enumerate(r["modes"]) if name != "all"]   # per-antenna traces only
    # sweeps that the burst turns on or off partway through (drawn hollow)
    edge = np.array([(a < b0 < c) or (a < b1 < c) for a, c in zip(r["t_start"], r["t_stop"])
                     for b0, b1 in [r["burst"]]])
    fig, ax = plt.subplots(4, 3 + npt, figsize=(5 * (3 + npt), 16), constrained_layout=True)
    for row, (q, unit, cmap) in enumerate(FIELDS[:3]):
        frames, xs, ys = grid_frames(r["xy"][:, 0], r["xy"][:, 1], r[q][:, 0])
        sel = frames[pick]
        lo, hi = np.nanpercentile(sel, [2, 98])
        ext = [xs[0] - .5, xs[-1] + .5, ys[0] - .5, ys[-1] + .5]
        for c, k in enumerate(pick):
            im = ax[row, c].imshow(frames[k], origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
            ax[row, c].set(title=f"{q} at {t[k]:.2f} ms", xlabel="x (cm)", ylabel="y (cm)")
            for j, i in enumerate(loc):
                ax[row, c].plot(*r["xy"][i], MARKERS[j % len(MARKERS)], ms=7, mfc="none", mec="w", mew=1.5)
        fig.colorbar(im, ax=ax[row, :3], label=f"{q} ({unit})", shrink=0.8)
        for j, i in enumerate(loc):
            a = ax[row, 3 + j]
            x0, y0 = r["xy"][i] + 0.0                              # +0.0: no "-0" in labels
            for m in trace_modes:
                ln, = a.plot(t, r[q][i, m], "o-", ms=3, color=f"C{m}", label=str(r["modes"][m]))
                a.plot(t[edge], r[q][i, m, edge], "o", ms=6, mfc="w", mec=ln.get_color())
            a.axvspan(*burst, color="0.85", zorder=0)
            a.set(xlabel="t (ms)", ylabel=f"{q} ({unit})",
                  title=f"{q} at ({x0:g}, {y0:g}) cm  [{MARKERS[j % len(MARKERS)]}]")
            a.legend(fontsize=8)
        # One y-range per row so the points compare directly, taken from the sweeps
        # wholly before, inside or after the burst: a sweep the burst switches on or
        # off in the middle of is no single plasma state, and its fit can be far off.
        v = np.concatenate([r[q][i][trace_modes][:, ~edge].ravel() for i in loc])
        lo, hi = np.nanmin(v), np.nanmax(v)
        lo, hi = lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)
        for j, i in enumerate(loc):
            a = ax[row, 3 + j]
            a.set_ylim(lo, hi)
            seen = r[q][i][trace_modes][:, edge]
            off = np.isfinite(seen) & ((seen < lo) | (seen > hi))
            if off.any():
                a.annotate(f"off scale at {t[edge][np.any(off, 0)][0]:.1f} ms: "
                           + ", ".join(f"{r['modes'][trace_modes[k]]} {seen[k][off[k]][0]:.3g}"
                                       for k in np.flatnonzero(off.any(1))),
                           (0.02, 0.03), xycoords="axes fraction", fontsize=8, color="0.3")
    # radial profiles of the "all" fit, azimuthally binned, at every 2nd sweep
    rr = np.hypot(*r["xy"].T)
    bins = np.arange(0, 19, 1.0)
    idx = np.digitize(rr, bins)
    cm = plt.get_cmap("viridis")
    for c, (q, unit, _) in enumerate(FIELDS[:3]):
        a = ax[3, c]
        for k in range(0, len(t), 2):
            prof = [np.nanmedian(r[q][idx == i, 0, k]) if (idx == i).any() else np.nan for i in range(1, len(bins))]
            a.plot(bins[:-1] + .5, prof, color=cm(k / len(t)), lw=1.5 if burst[0] <= t[k] <= burst[1] else .8,
                   ls="-" if burst[0] <= t[k] <= burst[1] else "--")
        a.set(xlabel="r (cm)", ylabel=f"{q} ({unit})", title=f"{q}(r), all shots; solid = burst")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(t[0], t[-1]))
    fig.colorbar(sm, ax=ax[3, :3], label="t (ms)", shrink=0.8)
    a = ax[3, 3]
    a.plot(t, r["cal_factor"] * r["area_cm2"], "o-")
    a.set(xlabel="t (ms)", ylabel="factor", title="ne calibration factor (interferometer / probe)")
    for a in ax[3, 4:]:
        a.axis("off")
    fig.suptitle(f"Aug-2026 run {run:02d}: P26 swept LP. Maps and r-profiles: all shots. Burst {burst[0]:.1f}-{burst[1]:.1f} ms "
                 "shaded; hollow markers: sweeps the burst switches on/off during")
    fig.savefig(out, dpi=70)
    print("figure:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    r = load(args.out_dir / f"{args.run:02d}-iv-plane.npz")
    figs = args.out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    figure(r, args.run, figs / f"{args.run:02d}-iv-plane.png")
    write_slider_html(bundle(r, args.run), figs / f"{args.run:02d}-iv-plane.html")


if __name__ == "__main__":
    main()
