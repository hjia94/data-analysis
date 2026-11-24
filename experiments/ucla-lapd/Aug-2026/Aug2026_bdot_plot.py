"""Aug-2026 B-dot: checks and plane maps from ``<run>-bdot-phasors.npz`` (made by Aug2026_bdot.py).

Prints the real-data checks of the integration and field operators, then grids
the phasors by antenna mode (A / B / AB) and plots, per mode: |B_perp| with
Re(B_perp) arrows, J_z = curl_z(B)/mu_0, |div B_perp|, and the wave envelope
over the burst. Checks:

1. B(f) = Bdot(f) / (i w) per shot -- the time integration on real traces.
2. The f-phasor does not depend on the baseline (pretrigger / smooth / periodic).
3. Offset drift at record end, raw integral vs pretrigger baseline.
4. Wave-band B after the burst returns to the pre-burst floor.
5. Linearity: B_AB vs B_A + B_B.
6. div(B_perp) << curl_z(B) for the measured map; the same ratio with Bx's sign
   flipped is the control -- it catches a probe-axis or grid-axis sign error.

Line scans (one unique x or y) get checks 1-5 only.

    .venv/bin/python experiments/ucla-lapd/Aug-2026/Aug2026_bdot_plot.py 30
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_analysis.plasma import fields

OUT_DIR = Path.home() / "data-analysis-output" / "Aug2026"
MODES = ("A", "B", "AB")


def load(path):
    r = dict(np.load(path))
    pmap = {int(s): (x, y) for s, x, y in zip(r["pos_shot"], r["pos_x"], r["pos_y"])}
    xy = np.array([pmap[int(s)] for s in r["shots"]])
    r["px"], r["py"] = np.round(xy[:, 0], 1), np.round(xy[:, 1], 1)
    on = r["Iamp"] > 0.25 * r["Iamp"].max()                     # (2, nshot)
    r["mode"] = np.where(on[0] & on[1], "AB", np.where(on[0], "A", np.where(on[1], "B", "off")))
    return r


def checks(r):
    w = 2 * np.pi * float(r["freq"])
    m = r["mode"]
    print("modes:", {k: int((m == k).sum()) for k in np.unique(m)})
    e = np.abs(r["Bph"] / (r["Bdph"] / (1j * w)) - 1)
    print(f"[1] |B(f) - Bdot(f)/(iw)| / |B(f)|: median {np.median(e):.1e}, max {e.max():.1e}")
    for k, lab in (("Bph_sm", "smooth"), ("Bph_per", "periodic")):
        d = np.abs(r[k] / r["Bph"] - 1)
        print(f"[2] {lab:8s} vs pretrigger phasor: median {np.median(d):.1e}, max {d.max():.1e}")
    print(f"[3] drift at record end / |B(f)|: raw {np.median(r['end_raw']):.0f}, "
          f"pretrigger {np.median(r['end_pre']):.1f}")
    print(f"[4] wave-band B / |B(f)| (Bx, By): before burst {np.nanmedian(r['pre_sm'][:2]):.3f}, "
          f"after {np.nanmedian(r['post_sm'][:2]):.3f}")
    for k, a in (("A", 0), ("B", 1)):
        ph = np.unwrap(np.angle(r["Iph"][a, m == k]))
        print(f"    antenna {k}: |I| {np.abs(r['Iph'][a, m == k]).mean():.0f} A, "
              f"phase std {np.degrees(ph.std()):.2f} deg")


def grid_modes(r):
    xs, ys = np.unique(r["px"]), np.unique(r["py"])
    out = {}
    for k in MODES:
        G = np.full((3, len(ys), len(xs)), np.nan + 0j)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                sel = (r["mode"] == k) & (r["px"] == x) & (r["py"] == y)
                if sel.any():
                    G[:, j, i] = r["Bph"][:, sel].mean(1)
        out[k] = G
    return xs, ys, out


def curl_div(G, h, sx=1.0):
    Bx, By = sx * G[0], G[1]
    _, _, cz = fields.curl(Bx, By, 0, h, h, None, axes=(-1, -2, None))
    return cz, np.gradient(Bx, h, axis=-1) + np.gradient(By, h, axis=-2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=int)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    r = load(args.out_dir / f"{args.run:02d}-bdot-phasors.npz")
    checks(r)

    xs, ys, grid = grid_modes(r)
    print("[5] linearity |B_AB - (B_A + B_B)| / |B_AB|:",
          ", ".join(f"B{c} {np.nansum(np.abs(grid['AB'][i] - grid['A'][i] - grid['B'][i])) / np.nansum(np.abs(grid['AB'][i])):.3f}"
                    for i, c in enumerate("xyz")))
    if len(xs) < 3 or len(ys) < 3:
        print("line scan: no plane maps")
        return
    h = float(np.median(np.diff(xs))) * 1e-2                      # m
    if not (np.allclose(np.diff(xs), h * 100) and np.allclose(np.diff(ys), h * 100)):
        raise ValueError("plane maps assume a uniform square grid")
    for k in MODES:
        ratio = []
        for sx in (1, -1):
            cz, dv = curl_div(grid[k], h, sx)
            ratio.append(np.sqrt(np.nanmean(np.abs(dv) ** 2) / np.nanmean(np.abs(cz) ** 2)))
        print(f"[6] mode {k:2s}: rms|div B_perp| / rms|curl_z B| = {ratio[0]:.2f} (Bx flipped: {ratio[1]:.2f})")

    fig, ax = plt.subplots(3, 4, figsize=(17, 12), constrained_layout=True)
    d = (xs[1] - xs[0]) / 2
    ext = [xs[0] - d, xs[-1] + d, ys[0] - d, ys[-1] + d]
    X, Y = np.meshgrid(xs, ys)
    step = max(1, len(xs) // 16)
    for row, k in enumerate(MODES):
        G = grid[k]
        Jz = fields.current_density(G[0], G[1], 0, h, h, None, axes=(-1, -2, None))[2]
        cz, dv = curl_div(G, h)
        bp = np.sqrt(np.abs(G[0]) ** 2 + np.abs(G[1]) ** 2)
        rot = np.exp(-1j * np.angle(Jz.flat[np.nanargmax(np.abs(Jz))]))   # phase of peak J_z
        a = ax[row, 0]
        im = a.imshow(bp / np.nanmax(bp), origin="lower", extent=ext, cmap="viridis")
        s = (slice(None, None, step),) * 2
        a.quiver(X[s], Y[s], (G[0] * rot).real[s], (G[1] * rot).real[s], color="w")
        a.set_title(f"mode {k}: |B_perp| (norm.), Re(B_perp) at J_z-peak phase")
        fig.colorbar(im, ax=a)
        lim = np.nanmax(np.abs((Jz * rot).real))
        a = ax[row, 1]
        im = a.imshow((Jz * rot).real / lim, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_title("Re(J_z) = curl_z B / mu_0 (norm.)")
        fig.colorbar(im, ax=a)
        a = ax[row, 2]
        im = a.imshow(np.abs(dv) / np.nanmax(np.abs(cz)), origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
        a.set_title("|div B_perp| / max|curl_z B|")
        fig.colorbar(im, ax=a)
        a = ax[row, 3]
        env = np.sqrt(r["cyc"][:2][:, r["mode"] == k].sum(0).mean(0))
        tc = (float(r["t_burst0"]) + np.arange(env.size) / float(r["freq"])) * 1e3
        a.plot(tc, env / env[env.size // 2:].mean())
        a.set(xlabel="t (ms)", ylim=(0, 1.6), title="wave |B_perp| envelope (cycle_means)")
    for a in ax[:, :3].ravel():
        a.set(xlabel="x (cm)", ylabel="y (cm)")
    # area may be a scalar or a 3x3 calibration matrix; 1.0 is the uncalibrated placeholder
    uncalibrated = np.ndim(r["area"]) == 0 and float(r["area"]) == 1.0
    fig.suptitle(f"Aug-2026 run {args.run:02d}: {float(r['freq'])/1e3:.1f} kHz Alfvén wave, P28 B-dot"
                 + (" (unit coil area: relative scale)" if uncalibrated else ""))
    out = args.out_dir / "figures" / f"{args.run:02d}-bdot-plane.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    print("figure:", out)


if __name__ == "__main__":
    main()
