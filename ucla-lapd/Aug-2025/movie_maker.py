#!/usr/bin/env python3
"""
Interactive animation of X-ray counts vs position using bar charts.

- X-axis: position r_arr (cm)
- Y-axis: X-ray counts (bar height)
- Frames: integer-ms time windows centered at t (ms), selecting all raw points with bin_center in [t-0.5, t+0.5)
"""

# Standard libs
import os
import sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..'))
sys.path = [repo_root, f"{repo_root}/read", f"{repo_root}/object_tracking"] + sys.path

from data_analysis_utils import counts_per_bin  # noqa: E402
from object_tracking.track_object import get_pos_freefall  # noqa: E402


def draw_frame(ax, fig, all_t_ms, all_r_cm, all_c, t_ms, pos_min, pos_max, xpad):
    ax.clear()
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('X-ray Counts')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pos_min - xpad, pos_max + xpad)

    # Select points in [t_ms-0.5, t_ms+0.5)
    mask = (all_t_ms >= (t_ms - 0.5)) & (all_t_ms < (t_ms + 0.5))
    r_vals = all_r_cm[mask]
    c_vals = all_c[mask]

    if r_vals.size > 0:
        # Determine a sensible bar width from local spacing
        if r_vals.size > 1:
            rs = np.sort(r_vals)
            dr = np.median(np.diff(rs))
            if not np.isfinite(dr) or dr <= 0:
                dr = 0.02 * max(1.0, pos_max - pos_min)
        else:
            dr = 0.02 * max(1.0, pos_max - pos_min)
        bar_w = max(0.8 * dr, 0.2)

        ax.bar(r_vals, c_vals, width=bar_w, align='center', alpha=0.85, edgecolor='k', color='tab:blue')

        # Dynamic Y range per frame
        ymax = float(np.max(c_vals))
        ax.set_ylim(0, max(1.0, ymax * 1.2))
        ax.set_title(f'X-ray Counts vs Position  —  t = {t_ms:.1f} ms   (n={r_vals.size})')
    else:
        ax.set_ylim(0, 1.0)
        ax.set_title(f'X-ray Counts vs Position  —  t = {t_ms:.1f} ms   (n=0)')

    fig.canvas.draw_idle()


class Player:
    def __init__(self, slider, frame_ticks, fig, ax, all_t_ms, all_r_cm, all_c, pos_min, pos_max, xpad):
        self.play = False
        self.idx = 0
        self.slider = slider
        self.frame_ticks = frame_ticks
        self.fig = fig
        self.ax = ax
        self.all_t_ms = all_t_ms
        self.all_r_cm = all_r_cm
        self.all_c = all_c
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.xpad = xpad

    def toggle(self, event=None):
        self.play = not self.play
        if self.play:
            self.idx = int(np.argmin(np.abs(self.frame_ticks - self.slider.val)))
            self.loop()

    def loop(self):
        if not self.play:
            return
        t_ms = self.frame_ticks[self.idx]
        self.slider.set_val(t_ms)
        draw_frame(self.ax, self.fig, self.all_t_ms, self.all_r_cm, self.all_c, t_ms, self.pos_min, self.pos_max, self.xpad)
        self.idx = (self.idx + 1) % len(self.frame_ticks)
        self.fig.canvas.start_event_loop(0.1)
        self.loop()


def plot_result(base_dir, uw_start=30):
    """
    Interactive animation of counts vs position over time.
    - X-axis: position r_arr (cm)
    - Y-axis: X-ray counts
    - Frames: time slices defined by bin_centers (ms)
    """
    # Load cached results
    analysis_file = os.path.join(base_dir, 'analysis_results.npy')
    tracking_file = os.path.join(base_dir, 'tracking_result.npy')

    analysis_dict = np.load(analysis_file, allow_pickle=True).item()
    tracking_dict = np.load(tracking_file, allow_pickle=True).item()

    # Flatten all points: arrays of time (ms), radius (cm), counts (raw)
    all_t_ms = []
    all_r_cm = []
    all_c = []

    for key, item in tracking_dict.items():
        if item[1] is None:
            # missing ct for this video
            continue
        prefixes = os.path.basename(key)[:2]
        shot_numbers = int(os.path.basename(key).split('_shot')[1][:3])
        t0 = item[1]  # seconds

        analysis_key = f"{prefixes}_{shot_numbers:03d}"
        pulse_tarr, pulse_amp = analysis_dict.get(analysis_key, ([], []))
        if len(pulse_tarr) == 0:
            continue

        # bin_centers in ms; counts are per-bin photon counts
        bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=1)
        # Convert to seconds for kinematics, then to cm
        time_seconds = (bin_centers + uw_start) * 1e-3
        r_arr_cm = get_pos_freefall(time_seconds, t0) * 100.0

        all_t_ms.extend(bin_centers.tolist())
        all_r_cm.extend(r_arr_cm.tolist())
        all_c.extend(counts.tolist())

    if len(all_t_ms) == 0:
        print("No data available to plot.")
        return

    all_t_ms = np.asarray(all_t_ms, dtype=float)
    all_r_cm = np.asarray(all_r_cm, dtype=float)
    all_c = np.asarray(all_c, dtype=float)

    # Fixed X-axis across frames; dynamic Y-axis per frame
    pos_min, pos_max = float(np.min(all_r_cm)), float(np.max(all_r_cm))
    xpad = max(1.0, 0.05 * (pos_max - pos_min))

    tmin = float(np.min(all_t_ms))
    tmax = float(np.max(all_t_ms))
    frame_ticks = np.arange(np.floor(tmin), np.ceil(tmax) + 1.0, 1.0)  # integer ms

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Time (ms)',
                    tmin, tmax,
                    valinit=frame_ticks[0], valstep=frame_ticks)

    def on_slide(val):
        draw_frame(ax, fig, all_t_ms, all_r_cm, all_c, slider.val, pos_min, pos_max, xpad)

    slider.on_changed(on_slide)

    # Initial frame
    draw_frame(ax, fig, all_t_ms, all_r_cm, all_c, frame_ticks[0], pos_min, pos_max, xpad)

    player = Player(slider, frame_ticks, fig, ax, all_t_ms, all_r_cm, all_c, pos_min, pos_max, xpad)
    ax_btn = plt.axes([0.85, 0.1, 0.1, 0.05])
    btn = Button(ax_btn, 'Play/Pause')
    btn.on_clicked(player.toggle)

    plt.show(block=True)


if __name__ == '__main__':
    base_dir = r"F:\AUG2025\P24"
    plot_result(base_dir, uw_start=30)