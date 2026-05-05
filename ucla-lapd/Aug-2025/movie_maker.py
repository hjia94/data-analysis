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
from matplotlib.animation import FuncAnimation

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__ if '__file__' in globals() else os.getcwd()), '../..'))
sys.path = [repo_root, f"{repo_root}/read", f"{repo_root}/object_tracking"] + sys.path

from object_tracking.generate_tracking import count_y_passes  # noqa: E402
from data_analysis_utils import counts_per_bin  # noqa: E402
from tracking_utils import (  # noqa: E402
    analysis_key_for_basename,
    evaluate_y_cm,
    iter_valid_tracking,
)

# Set default font size for all labels
plt.rcParams.update({'font.size': 24})


def draw_frame(ax, fig, all_t_ms, all_r_cm, all_c, t_ms, half_bin_width=0.5,
               ymax=None, all_m=None):
    ax.clear()
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('Normalized X-ray Counts')
    ax.grid(True, alpha=0.3)
    if ymax is None:
        ymax = float(np.max(all_c))
    ax.set_xlim(-50, 50)
    ax.set_ylim(0, ymax * 1.01)

    mask = (all_t_ms >= (t_ms - half_bin_width)) & (all_t_ms < (t_ms + half_bin_width))
    r_vals = all_r_cm[mask]
    c_vals = all_c[mask]
    title = f't = {t_ms:.1f} ms'
    if all_m is not None:
        m_vals = np.asarray(all_m)[mask]
        if m_vals.size > 0:
            m_min = int(np.min(m_vals))
            m_max = int(np.max(m_vals))
            m_text = f'M = {m_min}' if m_min == m_max else f'M = {m_min}-{m_max}'
            print(f"t = {t_ms:.1f} ms: {m_text}")
            title = f'{title}, {m_text}'
        else:
            print(f"t = {t_ms:.1f} ms: M = none")
            title = f'{title}, M = none'

    if r_vals.size > 0:
        ax.bar(r_vals, c_vals, width=0.5, align='center', alpha=0.85, edgecolor='k', color='tab:blue')
    ax.set_title(title)

    fig.canvas.draw_idle()


def draw_coverage_frame(ax, fig, y_centers_cm, coverage_counts, t_ms, ymax=None):
    ax.clear()
    ax.set_xlabel('y (cm)')
    ax.set_ylabel('Valid Trajectory Crossings')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, 50)
    if ymax is None:
        ymax = float(np.max(coverage_counts))
    ax.set_ylim(0, max(ymax * 1.1, 1))
    ax.bar(
        y_centers_cm,
        coverage_counts,
        width=np.diff(y_centers_cm).mean() if y_centers_cm.size > 1 else 0.5,
        align='center',
        alpha=0.85,
        edgecolor='k',
        color='tab:orange',
    )
    ax.set_title(f't = {t_ms:.1f} ms')
    fig.canvas.draw_idle()


class Player:
    def __init__(self, slider, frame_ticks, fig, ax, all_t_ms, all_r_cm, all_c,
                 ymax=None, all_m=None):
        self.play = False
        self.idx = 0
        self.slider = slider
        self.frame_ticks = frame_ticks
        self.fig = fig
        self.ax = ax
        self.all_t_ms = all_t_ms
        self.all_r_cm = all_r_cm
        self.all_c = all_c
        self.ymax = ymax
        self.all_m = all_m

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
        frame_step_ms = self.frame_ticks[1] - self.frame_ticks[0] if len(self.frame_ticks) > 1 else 1.0
        draw_frame(
            self.ax, self.fig, self.all_t_ms, self.all_r_cm, self.all_c,
            t_ms, frame_step_ms / 2, ymax=self.ymax, all_m=self.all_m,
        )
        self.idx = (self.idx + 1) % len(self.frame_ticks)
        self.fig.canvas.start_event_loop(0.1)
        self.loop()


def _assemble_xray_points(base_dir, uw_start, frame_step_ms):
    """Build the x-ray-vs-position scatter for every valid tracking shot.

    Pure data: no matplotlib. Returns ``None`` when no shots contribute.
    """
    analysis_dict = np.load(os.path.join(base_dir, 'analysis_results.npy'),
                            allow_pickle=True).item()
    tracking_dict = np.load(os.path.join(base_dir, 'tracking_result.npy'),
                            allow_pickle=True).item()

    half_width = frame_step_ms / 2
    all_t_ms, all_r_cm, all_c, all_m = [], [], [], []
    valid_tracking_shots = 0
    contributing_shots = 0

    for cine_path, entry in iter_valid_tracking(tracking_dict, strict_schema=True):
        valid_tracking_shots += 1

        analysis_key_str = analysis_key_for_basename(os.path.basename(cine_path))
        pulse_tarr, pulse_amp = analysis_dict.get(analysis_key_str, ([], []))
        if len(pulse_tarr) == 0:
            continue
        contributing_shots += 1

        bin_centers, counts = counts_per_bin(pulse_tarr, pulse_amp, bin_width=frame_step_ms)

        y_cm = evaluate_y_cm(entry, bin_centers, uw_start)
        r_arr_cm = -y_cm  # explicit: r is distance below chamber centre

        shot_counts = count_y_passes(
            base_dir, r_arr_cm, bin_centers - half_width, bin_centers + half_width,
            tracking_dict=tracking_dict,
        )
        counts = counts.astype(float) / np.where(shot_counts > 0, shot_counts, 1)

        all_t_ms.extend(bin_centers.tolist())
        all_r_cm.extend(r_arr_cm.tolist())
        all_c.extend(counts.tolist())
        all_m.extend(shot_counts.tolist())

    print(
        f"Valid tracking shots: {valid_tracking_shots} / "
        f"{len(tracking_dict)} entries in tracking_result.npy"
    )
    print(f"Shots contributing x-ray counts to movie: {contributing_shots}")

    if len(all_t_ms) == 0:
        return None

    all_t_ms = np.asarray(all_t_ms, dtype=float)
    all_r_cm = np.asarray(all_r_cm, dtype=float)
    all_c = np.asarray(all_c, dtype=float)
    all_m = np.asarray(all_m, dtype=int)

    tmin = float(np.min(all_t_ms))
    tmax = float(np.max(all_t_ms))
    frame_ticks = np.arange(np.floor(tmin), np.ceil(tmax) + frame_step_ms, frame_step_ms)
    ymax = float(np.max(all_c))

    return {
        "all_t_ms": all_t_ms,
        "all_r_cm": all_r_cm,
        "all_c": all_c,
        "all_m": all_m,
        "frame_ticks": frame_ticks,
        "tmin": tmin,
        "tmax": tmax,
        "ymax": ymax,
        "half_width": half_width,
    }


def _save_xray_animation(points, output_path, fps):
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame_idx):
        draw_frame(
            ax, fig,
            points["all_t_ms"], points["all_r_cm"], points["all_c"],
            points["frame_ticks"][frame_idx], points["half_width"],
            ymax=points["ymax"], all_m=points["all_m"],
        )

    anim = FuncAnimation(fig, update, frames=len(points["frame_ticks"]), blit=False)

    print(f"Saving animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=fps)
    print(f"Animation saved successfully to {output_path}")
    plt.close(fig)
    return output_path


def _show_xray_animation(points):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Time (ms)', points["tmin"], points["tmax"],
        valinit=points["frame_ticks"][0], valstep=points["frame_ticks"],
    )

    def on_slide(val):
        draw_frame(
            ax, fig,
            points["all_t_ms"], points["all_r_cm"], points["all_c"],
            slider.val, points["half_width"],
            ymax=points["ymax"], all_m=points["all_m"],
        )

    slider.on_changed(on_slide)

    draw_frame(
        ax, fig,
        points["all_t_ms"], points["all_r_cm"], points["all_c"],
        points["frame_ticks"][0], points["half_width"],
        ymax=points["ymax"], all_m=points["all_m"],
    )

    player = Player(
        slider, points["frame_ticks"], fig, ax,
        points["all_t_ms"], points["all_r_cm"], points["all_c"],
        ymax=points["ymax"], all_m=points["all_m"],
    )
    ax_btn = plt.axes([0.85, 0.1, 0.1, 0.05])
    btn = Button(ax_btn, 'Play/Pause')
    btn.on_clicked(player.toggle)

    plt.show(block=True)


def plot_result(base_dir, uw_start=30, frame_step_ms=1.0, save_mp4=False,
                output_filename="animation.mp4", fps=10):
    """
    Interactive animation of counts vs position over time.
    - X-axis: position r_arr (cm)  — distance below chamber centre, evaluated
      from the per-shot line fit saved by
      object_tracking/generate_tracking.py:track_shots (sparse tracker).
    - Y-axis: X-ray counts
    - Frames: time slices defined by bin_centers (ms)
    - frame_step_ms: time step between frames (default: 1.0 ms)
    - save_mp4: Whether to save the animation as an MP4 file (requires ffmpeg)
    - output_filename: The filename for the saved animation (if save_mp4 is True)
    - fps: Frames per second for the saved animation (if save_mp4 is True)

    Inputs (npy files): analysis_results.npy and tracking_result.npy must
    already exist in base_dir — produced by process_xray.py and
    object_tracking/generate_tracking.py respectively. The tracking entries
    carry their own cm_per_px so no calibration file is loaded here.
    """
    points = _assemble_xray_points(base_dir, uw_start, frame_step_ms)
    if points is None:
        print("No data available to plot.")
        return

    if save_mp4:
        return _save_xray_animation(points, os.path.join(base_dir, output_filename), fps)
    _show_xray_animation(points)


def _assemble_coverage_grid(base_dir, frame_step_ms, y_min, y_max,
                            y_bin_width, t_min, t_max):
    """Build the (frame, y) coverage grid from tracking_result.npy."""
    tracking_dict = np.load(os.path.join(base_dir, 'tracking_result.npy'),
                            allow_pickle=True).item()

    valid_tracking_shots = sum(
        1 for _ in iter_valid_tracking(tracking_dict, strict_schema=True)
    )

    half_width = frame_step_ms / 2
    y_centers_cm = np.arange(y_min, y_max + 0.5 * y_bin_width, y_bin_width)
    frame_ticks = np.arange(t_min, t_max + frame_step_ms, frame_step_ms)
    coverage_by_frame = []

    for t_ms in frame_ticks:
        coverage_counts = count_y_passes(
            base_dir,
            y_centers_cm,
            np.full_like(y_centers_cm, t_ms - half_width, dtype=float),
            np.full_like(y_centers_cm, t_ms + half_width, dtype=float),
            tracking_dict=tracking_dict,
        )
        coverage_by_frame.append(coverage_counts)

    coverage_by_frame = np.asarray(coverage_by_frame, dtype=int)
    ymax = int(np.max(coverage_by_frame)) if coverage_by_frame.size else 1

    print(
        f"Valid tracking shots: {valid_tracking_shots} / "
        f"{len(tracking_dict)} entries in tracking_result.npy"
    )
    print(
        f"Coverage movie bins: {y_bin_width:g} cm in y, "
        f"{frame_step_ms:g} ms in time"
    )
    print(f"Maximum trajectory crossings in a plotted bin: {ymax}")

    return {
        "y_centers_cm": y_centers_cm,
        "coverage_by_frame": coverage_by_frame,
        "frame_ticks": frame_ticks,
        "t_min": t_min,
        "t_max": t_max,
        "ymax": ymax,
    }


def _save_coverage_animation(grid, output_path, fps):
    fig, ax = plt.subplots(figsize=(12, 8))

    def update(frame_idx):
        draw_coverage_frame(
            ax, fig, grid["y_centers_cm"], grid["coverage_by_frame"][frame_idx],
            grid["frame_ticks"][frame_idx], ymax=grid["ymax"],
        )

    anim = FuncAnimation(fig, update, frames=len(grid["frame_ticks"]), blit=False)
    print(f"Saving trajectory coverage animation to {output_path}...")
    anim.save(output_path, writer='ffmpeg', fps=fps)
    print(f"Trajectory coverage animation saved successfully to {output_path}")
    plt.close(fig)
    return output_path


def _show_coverage_animation(grid):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Time (ms)', grid["t_min"], grid["t_max"],
        valinit=grid["frame_ticks"][0], valstep=grid["frame_ticks"],
    )

    def on_slide(val):
        frame_idx = int(np.argmin(np.abs(grid["frame_ticks"] - val)))
        draw_coverage_frame(
            ax, fig, grid["y_centers_cm"], grid["coverage_by_frame"][frame_idx],
            grid["frame_ticks"][frame_idx], ymax=grid["ymax"],
        )

    slider.on_changed(on_slide)
    draw_coverage_frame(
        ax, fig, grid["y_centers_cm"], grid["coverage_by_frame"][0],
        grid["frame_ticks"][0], ymax=grid["ymax"],
    )
    plt.show(block=True)


def plot_trajectory_coverage(base_dir, frame_step_ms=1.0, y_min=-50, y_max=50,
                             y_bin_width=0.5, t_min=0, t_max=45,
                             save_mp4=False,
                             output_filename="trajectory_coverage_animation.mp4",
                             fps=10):
    """
    Plot or save a movie of valid fitted trajectory crossings per y-time bin.

    This is a coverage diagnostic for the ensemble reconstruction. It uses only
    ``tracking_result.npy`` and applies the same per-bin crossing count used by
    ``plot_result`` for x-ray-count normalization.
    """
    grid = _assemble_coverage_grid(base_dir, frame_step_ms, y_min, y_max,
                                   y_bin_width, t_min, t_max)

    if save_mp4:
        return _save_coverage_animation(grid, os.path.join(base_dir, output_filename), fps)
    _show_coverage_animation(grid)


if __name__ == '__main__':
    base_dir = r"E:\AUG2025\P24"

    # Example 1: Interactive plot
    # plot_result(base_dir, uw_start=30, frame_step_ms=1)

    # Example 2: Save as MP4 (uncomment to use)
    plot_result(base_dir, uw_start=30, frame_step_ms=1, save_mp4=True, output_filename="xray_counts_animation.mp4", fps=15)

    # Example 3: Save trajectory-coverage movie (uncomment to use)
    # plot_trajectory_coverage(base_dir, frame_step_ms=1, save_mp4=False)
