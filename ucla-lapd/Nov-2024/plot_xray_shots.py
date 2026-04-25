"""X-ray traces (a)(b) and camera overlay (c) in a three-panel figure."""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\object_tracking")

from read_scope_data import read_trc_data
from read_cine import read_cine, convert_cine_to_avi, overlay_motion_frames
from track_object import track_object, get_chamber


PATH_A = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\background\C3--E-ring-p30-z13-x200-xray--00003.trc"
PATH_B = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\C3--E-ring-p30-z13-x200-xray--00022.trc"
CINE_PATH = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\Y20241102_P30_z13_x200_y0@-40_022.cine"

OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
             '#56B4E9', '#F0E442', '#000000']

FN = None   # set to a frame number to mark the ball position with a red square


def configure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'lines.linewidth': 0.9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })


def _file_number(path):
    m = re.search(r'--(\d{5})\.trc$', os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot extract shot number from: {path}")
    return m.group(1)


def plot_panels(save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)

    paths = [PATH_A, PATH_B]
    cached = {path: read_trc_data(path) for path in paths if path}
    y_max = max(np.max(-data) for data, _ in cached.values()) * 1.2

    for i, (ax, path, label, color) in enumerate(zip(axes, paths, ['(a)', '(b)'], OKABE_ITO)):
        if path:
            data, tarr = cached[path]
            t_ms = tarr * 1e3
            if i == 0:
                n = int(0.4 * len(t_ms))
                ax.plot(t_ms[:n], -data[:n], color=color, lw=0.8)
            else:
                ax.plot(t_ms, -data, color=color, lw=0.8)

        ax.set_ylim(0, y_max)
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("X-ray signal (a.u.)")
        ax.set_xlabel("Time (ms)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(0.01, 0.95, label, transform=ax.transAxes, ha='left', va='top', fontweight='bold')

    ax_c = axes[2]
    if CINE_PATH:
        _, frarr, _ = read_cine(CINE_PATH)
        avi_path = CINE_PATH.replace('.cine', '.avi')
        if not os.path.exists(avi_path):
            convert_cine_to_avi(frarr, avi_path)
        # result = track_object(avi_path)
        cf = 500 # result.min_ydiff_frame
        if cf is not None:
            cx, cy, chamber_radius = get_chamber()
            overlay_motion_frames(frarr, center_frame=cf, n_frames=400, step=40,
                                  ax=ax_c, show_window=False)
            ax_c.set_xticks([])
            ax_c.set_yticks([])
            ax_c.text(0.01, 0.95, '(c)', transform=ax_c.transAxes, ha='left', va='top', fontweight='bold', color='white')
            calib = np.load(r"E:\calibration_factor_P30.npy", allow_pickle=True).item()
            cm_per_px = calib["cm_per_px_mean"]
            bar_px = 1.0 / cm_per_px
            H, W = frarr.shape[1], frarr.shape[2]
            print(f"[scale bar] cm_per_px={cm_per_px:.6f} cm/px -> 1 cm = {bar_px:.2f} px "
                  f"(frame {W}x{H}, bar = {bar_px / W * 100:.2f}% of width)")
            x0 = W * 0.05
            y0 = H * 0.05
            ax_c.plot([x0, x0 + bar_px], [y0, y0], color='white', lw=2)
            ax_c.text(x0 + bar_px / 2, y0 + H * 0.02, '1 cm',
                      color='white', ha='center', va='bottom', fontsize=8)


    if save_path:
        for ext in ('pdf', 'png'):
            fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}.pdf / {save_path}.png")

    plt.show()
    return fig


def main(save=False):
    configure_style()
    out_path = None
    if save:
        out_dir = r"C:\Users\hjia9\Documents\lapd\e-ring\diagnostic_fig"
        out_path = os.path.join(out_dir, "xray_wt_camera_overlay")
    plot_panels(save_path=out_path)


if __name__ == "__main__":
    main(save=True)
