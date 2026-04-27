#!/usr/bin/env python3
"""Pixel-to-cm calibration from the ball's free-fall trajectory.

Two entry points (selected by CLI flag):

evaluate_runs (default): for each cine file in BASE_DIR, fit a parabola to the
    tracked vertical position y_px(tau) = a + b*tau + c*tau^2 and use g as
    ground truth: cm/px = -0.5 * g*100 / c_px. Emits one diagnostic PNG per
    shot and a terminal summary. The chamber-derived cm/px (18 cm /
    chamber_radius_px) is reported alongside, but the two values are NOT
    expected to match: the chamber wall is the back wall of the vacuum
    chamber, while the ball plane is determined by the port the cine was
    shot from. Use plot_port_ratio for the cross-port comparison.

plot_port_ratio (--port-ratio): scatter chamber_cm_px / gravity_cm_px vs
    port number, loaded from saved E:/calibration_factor_P*.npy summaries.
    A monotonic decrease toward 1 as port number increases confirms the
    ball plane moves toward the camera and justifies deriving cm/px per
    shot from the free-fall fit instead of reusing the chamber value.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g

from read_cine import read_cine
from track_object import (
    track_object_per_frame, ensure_avi,
    get_chamber, chamber_cm_per_px,
)


# ------------------------------ config ---------------------------------------
BASE_DIR = r"E:\good_data\kapton\He3kA_B380G800G_pl0t20_uw15t35"
CINE_GLOB = "*.cine"
MAX_VIDEOS = 2

CALIB_FACTOR_GLOB = r"E:/calibration_factor_P*.npy"

G_CM_PER_S2 = g * 100.0                       # 981.0 cm/s^2


# ------------------------------ per-shot evaluation --------------------------
def evaluate_shot(cine_path):
    """Track a cine file and deduce cm/px from the parabolic y(t) trajectory.

    Returns a dict with the gravity-derived cm/px plus the chamber-derived
    cm/px (different plane — kept for sanity checking, not for agreement).
    """
    cine_name = os.path.basename(cine_path)

    try:
        tarr, _frarr, _dt = read_cine(cine_path)
    except Exception as e:
        print(f"[SKIP] {cine_name}: read_cine failed: {e}")
        return None

    try:
        avi_path = ensure_avi(cine_path)
    except Exception as e:
        print(f"[SKIP] {cine_name}: avi conversion failed: {e}")
        return None

    _cx, _cy, ch_radius = get_chamber()
    cm_per_px_chamber = chamber_cm_per_px(ch_radius)

    result = track_object_per_frame(avi_path)
    positions = np.asarray(result.positions)
    frame_numbers = np.asarray(result.frame_numbers)
    min_ydiff_frame = result.min_ydiff_frame

    if positions.size == 0 or min_ydiff_frame is None:
        print(f"[SKIP] {cine_name}: tracker returned no usable frames")
        return None

    x_px = positions[:, 0].astype(float)
    y_px = positions[:, 1].astype(float)
    t = tarr[frame_numbers]
    t0 = float(tarr[min_ydiff_frame])
    tau = t - t0

    if tau.size < 5:
        print(f"[SKIP] {cine_name}: too few tracked frames ({tau.size})")
        return None

    c_px, b_px, a_px = np.polyfit(tau, y_px, 2)
    y_pred_px = a_px + b_px * tau + c_px * tau * tau
    fit_rms_px = float(np.sqrt(np.mean((y_px - y_pred_px) ** 2)))

    # Ball falls => c_px < 0 (y upward-positive); cm/px = -0.5 * g*100 / c_px.
    if c_px >= 0:
        print(f"[WARN] {cine_name}: non-falling quadratic (c_px={c_px:.2f}); cm/px unreliable")
        cm_per_px = float("nan")
    else:
        cm_per_px = float(-0.5 * G_CM_PER_S2 / c_px)

    # Implied drop height from v0 = -b_px * cm_per_px / 100 (b_px<0 at t0).
    if not np.isnan(cm_per_px) and b_px < 0:
        v0_m_per_s = -b_px * cm_per_px / 100.0
        h_implied_m = v0_m_per_s ** 2 / (2.0 * g)
    else:
        h_implied_m = float("nan")

    if not np.isnan(cm_per_px):
        x_cm = x_px * cm_per_px
        x_slope, _ = np.polyfit(tau, x_cm, 1)
        x_p2p = float(np.ptp(x_cm))
    else:
        x_slope, x_p2p = float("nan"), float("nan")

    return {
        "cine_name": cine_name,
        "t0": t0,
        "n_frames": int(positions.shape[0]),
        "c_px": float(c_px),
        "b_px": float(b_px),
        "cm_per_px": cm_per_px,
        "cm_per_px_chamber": cm_per_px_chamber,
        "h_implied_m": h_implied_m,
        "fit_rms_px": fit_rms_px,
        "x_slope_cm_per_s": float(x_slope),
        "x_p2p_cm": x_p2p,
        "tau": tau,
        "y_px": y_px,
        "x_px": x_px,
        "y_pred_px": y_pred_px,
    }


def plot_shot(res):
    """Two-panel diagnostic: y_px vs tau with parabola fit, and x_cm drift."""
    name = res["cine_name"]
    tau_ms = res["tau"] * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{name}   |   cm/px = {res['cm_per_px']:.5f}   "
        f"|   h_implied = {res['h_implied_m']:.3f} m"
    )

    ax = axes[0]
    ax.plot(tau_ms, res["y_px"], 'o', ms=3, label="tracked y")
    ax.plot(tau_ms, res["y_pred_px"], '-', color='tab:red',
            label=f"parabola fit (c={res['c_px']:.2f} px/s²)")
    ax.axvline(0, color='k', ls=':', lw=0.8, label="t0")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel("t - t0 (ms)")
    ax.set_ylabel("y (px)")
    ax.set_title(f"Gravity fit  |  RMS residual = {res['fit_rms_px']:.2f} px")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    x_cm = res["x_px"] * res["cm_per_px"] if not np.isnan(res["cm_per_px"]) else res["x_px"]
    ax.plot(tau_ms, x_cm, 'o', ms=3)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', ls=':', lw=0.8)
    ax.set_xlabel("t - t0 (ms)")
    ax.set_ylabel("x (cm, via deduced cm/px)")
    ax.set_title(
        f"Horizontal drift  |  slope={res['x_slope_cm_per_s']:.2f} cm/s  "
        f"p-p={res['x_p2p_cm']:.2f} cm"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout(rect=(0, 0, 1, 0.94))


def print_summary(results):
    if not results:
        print("No shots evaluated.")
        return

    header = (
        f"{'shot':<36} {'n':>4} {'c_px':>10} {'cm/px_grav':>11} "
        f"{'cm/px_cham':>11} {'ratio':>7} {'h(m)':>6} {'RMS(px)':>8} {'xdrift':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        cham = r.get("cm_per_px_chamber", float("nan"))
        cham_str = f"{cham:.5f}" if not np.isnan(cham) else "    n/a"
        if not np.isnan(cham) and not np.isnan(r["cm_per_px"]):
            ratio_str = f"{r['cm_per_px']/cham:.3f}"
        else:
            ratio_str = "  n/a"
        print(
            f"{r['cine_name']:<36} {r['n_frames']:>4d} {r['c_px']:>10.2f} "
            f"{r['cm_per_px']:>11.5f} {cham_str:>11} {ratio_str:>7} "
            f"{r['h_implied_m']:>6.3f} {r['fit_rms_px']:>8.2f} "
            f"{r['x_slope_cm_per_s']:>8.2f}"
        )

    vals = [r["cm_per_px"] for r in results if not np.isnan(r["cm_per_px"])]
    if len(vals) >= 2:
        print("-" * len(header))
        print(
            f"cm/px (gravity) across {len(vals)} shots: "
            f"mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  "
            f"spread={max(vals)-min(vals):.5f}  "
            f"(max/min={max(vals)/min(vals):.4f})"
        )


def evaluate_runs():
    if not BASE_DIR:
        print("BASE_DIR is empty. Set it to the directory containing the .cine files.")
        return

    cine_files = sorted(glob.glob(os.path.join(BASE_DIR, CINE_GLOB)))
    if MAX_VIDEOS is not None:
        cine_files = cine_files[:MAX_VIDEOS]

    if not cine_files:
        print(f"No cine files in {BASE_DIR} matching {CINE_GLOB}")
        return

    print(f"Evaluating {len(cine_files)} cine file(s) from {BASE_DIR}")

    results = []
    for i, path in enumerate(cine_files, 1):
        print(f"\n[{i}/{len(cine_files)}] {os.path.basename(path)}")
        res = evaluate_shot(path)
        if res is None:
            continue
        plot_shot(res)
        results.append(res)

    plt.show()
    print_summary(results)


# ------------------------------ cross-port ratio plot ------------------------
def plot_port_ratio():
    """Scatter chamber_cm_px / gravity_cm_px vs port number.

    Loads E:/calibration_factor_P*.npy summaries (each holds the gravity-fit
    mean/std for one port) and divides the chamber-derived cm/px by the
    gravity mean. A ratio > 1 means the ball plane is closer to the camera
    than the chamber wall; a monotonic trend in port number reflects the
    perspective shift between port planes.
    """
    _cx, _cy, ch_radius = get_chamber()
    cm_per_px_chamber = chamber_cm_per_px(ch_radius)

    rows = []
    for path in sorted(glob.glob(CALIB_FACTOR_GLOB)):
        m = re.search(r"_P(\d+)\.npy$", os.path.basename(path))
        if not m:
            continue
        port = int(m.group(1))
        d = np.load(path, allow_pickle=True).item()
        mean = d["cm_per_px_mean"]
        std = d["cm_per_px_std"]
        n = d["n_used"]
        ratio = cm_per_px_chamber / mean
        ratio_err = ratio * (std / mean) / np.sqrt(n)
        rows.append((port, mean, std, n, ratio, ratio_err))
        print(
            f"P{port:>2d}  cm/px_grav={mean:.5f}±{std:.5f} (n={n})  "
            f"chamber/grav={ratio:.3f}±{ratio_err:.3f}"
        )

    if not rows:
        raise SystemExit(f"No files matched {CALIB_FACTOR_GLOB}")

    rows.sort(key=lambda r: r[0])
    ports = np.array([r[0] for r in rows])
    ratios = np.array([r[4] for r in rows])
    rerr = np.array([r[5] for r in rows])

    _, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        ports, ratios, yerr=rerr,
        fmt="o", ms=8, capsize=4, color="tab:blue",
        label="chamber cm/px ÷ gravity cm/px",
    )
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="equal-plane (ratio = 1)")

    if len(ports) >= 2:
        coef = np.polyfit(ports, ratios, 1)
        xs = np.linspace(ports.min() - 0.5, ports.max() + 0.5, 50)
        ax.plot(xs, np.polyval(coef, xs), "-", color="tab:red", alpha=0.6,
                label=f"linear fit: slope={coef[0]:+.3f}/port")

    for port, _, _, _, ratio, _ in rows:
        ax.annotate(f"P{port}", (port, ratio), xytext=(6, 6),
                    textcoords="offset points", fontsize=9)

    ax.set_xlabel("Port number")
    ax.set_ylabel("cm/px (chamber) / cm/px (gravity)")
    ax.set_title(
        f"Chamber cm/px = {cm_per_px_chamber:.5f}  "
        f"(radius {ch_radius} px, 36 cm chamber)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            "chamber_vs_gravity_ratio.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved {out_path}")
    plt.show()


# ------------------------------ entry ----------------------------------------
def main():
    if "--port-ratio" in sys.argv[1:]:
        plot_port_ratio()
    else:
        evaluate_runs()


if __name__ == "__main__":
    main()
