#!/usr/bin/env python3
"""
Deduce the pixel-to-cm calibration from the ball's free-fall trajectory.

For each cine file, this script:
    1. Runs the tracker from object_tracking/.
    2. Fits a parabola y_px(tau) = a + b*tau + c*tau^2 to the tracked
       vertical positions (tau = t - t0, t0 = center-crossing time).
    3. Uses g as ground truth: 2*c*(cm/px)/100 = -g  =>  cm/px = -g*100/(2*c).
    4. Emits one PNG per shot and a compact terminal report comparing the
       deduced cm/px against extract_calibration() when available.

The deduced cm/px is decoupled from the drop height: only g is assumed.
"""

import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import g

repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..')
)
sys.path = [repo_root, f"{repo_root}/read", f"{repo_root}/object_tracking"] + sys.path

from object_tracking.read_cine import read_cine, convert_cine_to_avi
from object_tracking.track_object import track_object, extract_calibration


# ------------------------------ config ---------------------------------------
BASE_DIR = r""                               # set when drive is mounted
CINE_GLOB = "*.cine"
MAX_VIDEOS = 2
OUT_DIR = os.path.join(repo_root, "reports", "freefall_eval")

G_CM_PER_S2 = g * 100.0                       # 981.0 cm/s^2


# ------------------------------ helpers --------------------------------------
def ensure_avi(cine_path):
    """Convert cine to AVI next to it if the AVI does not yet exist."""
    avi_path = os.path.splitext(cine_path)[0] + ".avi"
    if os.path.exists(avi_path):
        return avi_path
    _tarr, frarr, _dt = read_cine(cine_path)
    convert_cine_to_avi(frarr, avi_path)
    return avi_path


def evaluate_shot(cine_path):
    """Track a cine file and deduce cm/px from the parabolic y(t) trajectory.

    Returns a dict with the derived calibration and arrays for plotting,
    or None on failure.
    """
    cine_name = os.path.basename(cine_path)

    try:
        cm_per_px_ref = float(extract_calibration(cine_name))
    except (ValueError, Exception):
        cm_per_px_ref = None

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

    result = track_object(avi_path)
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

    # Fit parabola in pixel space: y_px = a + b*tau + c*tau^2
    c_px, b_px, a_px = np.polyfit(tau, y_px, 2)
    y_pred_px = a_px + b_px * tau + c_px * tau * tau
    fit_rms_px = float(np.sqrt(np.mean((y_px - y_pred_px) ** 2)))

    # Deduce cm/px from gravity. Ball falls => c_px < 0 (y upward-positive).
    #   y_m(tau)  = -0.5 * g * tau^2 + v0 * tau         (with y(t0)=0)
    #   y_px(tau) = y_m / (cm/px * 0.01)
    #   => coefficient of tau^2 in px: c_px = -0.5 * g * 100 / (cm/px)
    #   => cm/px = -0.5 * (g*100) / c_px
    if c_px >= 0:
        print(f"[WARN] {cine_name}: non-falling quadratic (c_px={c_px:.2f}); cm/px unreliable")
        cm_per_px = float("nan")
    else:
        cm_per_px = float(-0.5 * G_CM_PER_S2 / c_px)

    # Implied drop height (distance fallen before center): v0 = b_px * cm_per_px / 100
    # v0 = sqrt(2*g*h) => h = v0^2 / (2g). With y upward-positive and ball falling,
    # at t0 the velocity is negative, so b_px < 0 and v0 = -b_px * cm_per_px / 100.
    if not np.isnan(cm_per_px) and b_px < 0:
        v0_m_per_s = -b_px * cm_per_px / 100.0
        h_implied_m = v0_m_per_s ** 2 / (2.0 * g)
    else:
        h_implied_m = float("nan")

    # Horizontal drift using the derived calibration
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
        "cm_per_px_ref": cm_per_px_ref,
        "h_implied_m": h_implied_m,
        "fit_rms_px": fit_rms_px,
        "x_slope_cm_per_s": float(x_slope),
        "x_p2p_cm": x_p2p,
        # arrays for plotting
        "tau": tau,
        "y_px": y_px,
        "x_px": x_px,
        "y_pred_px": y_pred_px,
    }


def plot_shot(res, out_dir):
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
    png_path = os.path.join(out_dir, os.path.splitext(name)[0] + "_gravity_cal.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    return png_path


def print_summary(results):
    if not results:
        print("No shots evaluated.")
        return

    header = (
        f"{'shot':<36} {'n':>4} {'c_px':>10} {'cm/px_fit':>11} "
        f"{'cm/px_ref':>11} {'ratio':>7} {'h(m)':>6} {'RMS(px)':>8} {'xdrift':>8}"
    )
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        ref = r["cm_per_px_ref"]
        ref_str = f"{ref:.5f}" if ref is not None else "    n/a"
        if ref and not np.isnan(r["cm_per_px"]):
            ratio_str = f"{r['cm_per_px']/ref:.3f}"
        else:
            ratio_str = "  n/a"
        print(
            f"{r['cine_name']:<36} {r['n_frames']:>4d} {r['c_px']:>10.2f} "
            f"{r['cm_per_px']:>11.5f} {ref_str:>11} {ratio_str:>7} "
            f"{r['h_implied_m']:>6.3f} {r['fit_rms_px']:>8.2f} "
            f"{r['x_slope_cm_per_s']:>8.2f}"
        )

    vals = [r["cm_per_px"] for r in results if not np.isnan(r["cm_per_px"])]
    if len(vals) >= 2:
        print("-" * len(header))
        print(
            f"cm/px across {len(vals)} shots: "
            f"mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  "
            f"spread={max(vals)-min(vals):.5f}  "
            f"(max/min={max(vals)/min(vals):.4f})"
        )


# ------------------------------ main -----------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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
    print(f"Output dir: {OUT_DIR}")

    results = []
    for i, path in enumerate(cine_files, 1):
        print(f"\n[{i}/{len(cine_files)}] {os.path.basename(path)}")
        res = evaluate_shot(path)
        if res is None:
            continue
        plot_path = plot_shot(res, OUT_DIR)
        print(f"    cm/px deduced = {res['cm_per_px']:.5f}   plot: {plot_path}")
        results.append(res)

    print_summary(results)


if __name__ == "__main__":
    main()
