'''
Plotting Utilities Module
------------------------

This module provides various plotting functions for data analysis. Functions are organized
into the following categories:

1. Frequency Analysis
    - plot_fft: Plot Fast Fourier Transform of signals
    - plot_stft: Plot one STFT spectrogram panel (ms/MHz axes, optional LogNorm)
    - plot_stft_wt_photon_counts: Plot STFT spectrogram with photon-count overlay
    - floor_for_lognorm: Make a matrix safe for matplotlib LogNorm

2. Multi-Shot Display
    - select_monitor / position_window: place figures across monitors
    - plot_shots_grid: Display multiple shots in a grid layout across monitors
    - cleanup_figures: Clean up matplotlib figures and free memory

3. Photon Counting
    - plot_counts_per_bin: Plot histogram of photon counts in time bins
    - plot_photon_detection / plot_original_and_baseline / plot_subtracted_signal

4. Figure Finalisation and Output
    - finalize_figure: tight_layout -> save -> show/close, in one place
    - fig_path / resolve_save: centralized figure paths under the output root
    - pack_shared_x: pack shared-x panels so they abut vertically

5. Style
    - configure_publication_style: publication-quality rcParams
    - OKABE_ITO: colorblind-safe categorical palette

6. Plane Maps and Animation
    - grid_by_position: per-position scalars -> regular (y, x) grid + imshow extent
    - Player: play/pause controller stepping a Slider through frame ticks

Each function includes detailed documentation of its parameters and returns.
'''
import matplotlib.pyplot as plt
from typing import Callable, Optional
from data_analysis.io.scope_reader import read_trc_data
import gc  # For garbage collection
import numpy as np

# NOTE: `tkinter` and `screeninfo` are GUI/monitor dependencies used only by the
# multi-shot display helpers (select_monitor, plot_shots_grid). They are imported
# lazily inside those functions so the pure-plotting helpers (plot_fft,
# plot_counts_per_bin, ...) import cleanly on headless machines without them.
# `screeninfo` is an optional dependency (the `gui` extra in pyproject.toml).


#==============================================================================
# Multi-Shot Display Functions
#==============================================================================

def cleanup_figures():
    """
    Helper function to properly clean up matplotlib figures and references.
    Useful for freeing memory in Jupyter notebooks.
    """
    # Close all figures
    plt.close('all')
    # Clear the current figure
    plt.clf()
    # Clear the current axes
    plt.cla()
    # Force garbage collection
    gc.collect()

def position_window(window_manager, x_pos: int, y_pos: int, 
                   window_width: Optional[int] = None, 
                   window_height: Optional[int] = None) -> None:
    """
    Position a matplotlib window across different backends.
    
    Args:
        window_manager: plt.get_current_fig_manager() instance
        x_pos (int): Window x position
        y_pos (int): Window y position
        window_width (Optional[int]): Window width. If None, keeps current width
        window_height (Optional[int]): Window height. If None, keeps current height
    """
    try:
        # For Qt backend
        if window_width and window_height:
            window_manager.window.setGeometry(x_pos, y_pos, window_width, window_height)
        else:
            window_manager.window.move(x_pos, y_pos)
    except:
        try:
            # For TkAgg backend
            window_manager.window.wm_geometry(f"+{x_pos}+{y_pos}")
        except:
            try:
                # For WX backend
                window_manager.window.SetPosition((x_pos, y_pos))
            except:
                print("Could not position window - unsupported backend")

def select_monitor(monitor_idx: Optional[int] = None, 
                  window_scale: tuple = (1.0, 1.0),
                  position_fig: bool = True) -> tuple:
    """
    Select a monitor and optionally position the current matplotlib figure.
    
    Args:
        monitor_idx (Optional[int]): Monitor index to use. If None, will prompt user.
        window_scale (tuple): Scale factors (width, height) for window size.
        position_fig (bool): If True, positions the current matplotlib figure.
    
    Returns:
        tuple: (monitor_object, x_pos, y_pos, window_width, window_height)
    """
    from screeninfo import get_monitors
    from data_analysis.io import choose_from_list
    monitors = get_monitors()

    # If monitor_idx not provided, show available monitors and prompt user
    if monitor_idx is None:
        monitor = choose_from_list(
            monitors,
            label=lambda m: f"{m.width}x{m.height} at position ({m.x}, {m.y})",
            prompt="Monitor number", header="\nAvailable monitors:")
    else:
        if not 0 <= monitor_idx < len(monitors):
            raise ValueError(f"Invalid monitor index {monitor_idx}. Must be between 0 and {len(monitors)-1}")
        monitor = monitors[monitor_idx]
    
    # Calculate window dimensions
    width_scale, height_scale = window_scale
    window_width = int(monitor.width * width_scale)
    window_height = int(monitor.height * height_scale)
    
    # Calculate centered position on the monitor
    x_pos = monitor.x + (monitor.width - window_width) // 2
    y_pos = monitor.y + (monitor.height - window_height) // 2
    
    # Position the current figure if requested
    if position_fig:
        mngr = plt.get_current_fig_manager()
        position_window(mngr, x_pos, y_pos, window_width, window_height)
    
    return monitor, x_pos, y_pos, window_width, window_height

#==============================================================================
# Display multiple shots on the screen
#==============================================================================
def plot_shots_grid(data_path_template: str, shot_range: range, n_cols: int = 2,
                   data_processor: Optional[Callable] = None, monitor_idx: Optional[int] = None) -> None:
    """Plot multiple shots in windows with 2x2 subplots."""
    import tkinter as tk

    # Clean up any existing figures first
    cleanup_figures()

    # Get monitor information using full window size
    monitor, base_x, base_y, window_width, window_height = select_monitor(
        monitor_idx=monitor_idx,
        window_scale=(0.9, 0.9)  # Use 90% of monitor size for each window
    )

    # Calculate number of windows needed (each window shows 4 plots)
    n_shots = len(shot_range)
    n_windows = (n_shots + 3) // 4  # Ceiling division by 4

    # Find common path prefix for naming windows
    first_path = data_path_template.format(shot=shot_range[0])
    path_parts = first_path.split('/')
    common_prefix = '/'.join(path_parts[:-1]) + '/'
    
    figures = []  # Keep track of figures for cleanup
    
    # Create windows with 2x2 subplots
    for window_idx in range(n_windows):
        # Create new figure with 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        figures.append(fig)  # Add to list for cleanup
        axs = axs.ravel()  # Flatten axes array for easier indexing
        
        # Calculate window position
        row = window_idx // n_cols
        col = window_idx % n_cols
        x_pos = base_x + (col * window_width)
        y_pos = base_y + (row * window_height)
        
        # Plot 4 shots in this window
        for subplot_idx in range(4):
            shot_idx = window_idx * 4 + subplot_idx
            if shot_idx < n_shots:
                shot = shot_range[shot_idx]
                
                # Read data
                data_path = data_path_template.format(shot=shot)
                data, tarr = read_trc_data(data_path, False)
                
                # Process data if processor provided
                if data_processor:
                    data = data_processor(data)
                
                # Plot data
                axs[subplot_idx].plot(tarr, -data)
                axs[subplot_idx].set_title(f'Shot {shot}')
                axs[subplot_idx].set_xlabel('Time')
                axs[subplot_idx].set_ylabel('Signal')
                
                # Clear data from memory
                del data
                del tarr
            else:
                # Hide empty subplots
                axs[subplot_idx].set_visible(False)

        # Position window
        mngr = plt.get_current_fig_manager()
        # Set window title using common path
        try:
            mngr.set_window_title(common_prefix)
        except Exception:
            print(f"Could not set window title for {common_prefix}")

        position_window(mngr, x_pos, y_pos, window_width, window_height)

        plt.show(block=False)
        
        # Force garbage collection of data (but keep the figure)
        gc.collect()
    
    # Create a small tkinter window to keep the plots alive without blocking
    root = tk.Tk()
    root.withdraw()  # Hide the tkinter window
    root.quit()  # Allow the script to terminate while keeping plots open
    
    # Return the list of figures so they can be properly cleaned up later
    return figures

#==============================================================================
# Photon Counting Functions
#==============================================================================

def plot_counts_per_bin(bin_centers, counts, bin_width_ms, ax=None):
    """
    Plot number of pulses in each time bin.
    
    Args:
        bin_centers (np.ndarray): Array of bin center times in milliseconds
        counts (np.ndarray): Array of pulse counts per bin
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used.
    """
    if ax is None:
        ax = plt.gca()
        
    # Plot counts per bin
    ax.plot(bin_centers, counts, label=f'Counts per {bin_width_ms} ms')
    
    # Add labels and title
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Counts per Bin')
    ax.legend(loc='upper right')
    ax.grid(True)
    

#==============================================================================
# Photon Detection Functions
#==============================================================================

def plot_photon_detection(tarr, data, pulse_times, detector, ax=None):
    """
    Plot photon pulses from x-ray detector data with dynamic baseline.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signal_data (np.ndarray): Signal data to analyze
        pulse_times (np.ndarray): Array of detected pulse times in milliseconds
        detector (Photons): Photon detector object containing baseline and threshold
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, current axes will be used
    """
    if ax is None:
        plt.figure()
        ax = plt.gca()

    # Plot original signal
    ax.plot(tarr, data, 'b-', label='Original', alpha=0.7)
    
    # Plot dynamic baseline
    ax.plot(tarr, detector.baseline, 'g-', label='Baseline', alpha=0.7)
    
    # Plot baseline-subtracted signal
    baseline_mean = detector.baseline.mean()
    subtracted_signal = data - detector.baseline
    ax.plot(tarr, subtracted_signal + baseline_mean, 'k-',
            label='Subtracted', alpha=0.5)

    # Plot detected pulses on the subtracted signal
    pulse_heights = subtracted_signal[np.searchsorted(tarr, pulse_times)] + baseline_mean
    ax.plot(pulse_times, pulse_heights, 'r.', label='Pulses')

    # Plot threshold level
    ax.axhline(y=baseline_mean + detector.threshold, color='r',
               linestyle='--', alpha=0.5, label='Threshold')
    
    ax.set_xlabel('Time (ms)')
    ax.set_title(f'Detected {len(pulse_times)} pulses')
    ax.legend(loc='upper right')
    ax.grid(True)

#==============================================================================
# Frequency Analysis Functions
#==============================================================================

def plot_fft(time_array, signals_dict, window=None):
    """
    Compute and plot FFT of signals with proper frequency units.
    
    Args:
        time_array (np.ndarray): Time array in seconds
        signals_dict (dict): Dictionary of signals to plot with their labels
        window (str, optional): Window function to use (e.g., 'hanning', 'blackman')
    
    Returns:
        tuple: (frequencies in MHz, magnitude spectrum)
    """
    # Calculate sampling parameters
    dt = time_array[1] - time_array[0]  # Time step
    n = len(time_array)
    
    # Create frequency array in MHz
    freq = np.fft.rfftfreq(n, dt) / 1e6  # Convert to MHz
    
    plt.figure(figsize=(10, 6))
    
    for label, signal in signals_dict.items():
        # Apply window if specified
        if window is not None:
            if window.lower() == 'hanning':
                win = np.hanning(len(signal))
            elif window.lower() == 'blackman':
                win = np.blackman(len(signal))
            signal = signal * win
            
        # Compute FFT
        fft_result = np.fft.rfft(signal)
        # Compute magnitude spectrum (normalized)
        magnitude = 2.0/n * np.abs(fft_result)
        
        # Plot
        plt.plot(freq, magnitude, label=label)
    
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.legend(loc='upper right')  # Fixed legend location
    
    return freq, magnitude

def plot_original_and_baseline(tarr, data, detector, ax=None):
    """Plot original signal and its calculated baseline.
    
    Args:
        tarr: Time array in milliseconds
        data: Original signal data
        detector: Photons detector object containing baseline
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot original signal and baseline
    ax.plot(tarr, data, 'b-', alpha=0.7, label='Original')
    ax.plot(tarr, detector.baseline, 'r-', alpha=0.7, label='Baseline')
    
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_subtracted_signal(tarr, data, pulse_times, detector, ax=None):
    """Plot baseline-subtracted signal with detected pulses.
    
    Args:
        tarr: Time array in milliseconds
        data: Original signal data (not used, kept for API consistency)
        pulse_times: Not used, kept for API consistency
        detector: Photons detector object containing baseline-subtracted signal
        ax: Optional matplotlib axis to plot on
    """
    if ax is None:
        ax = plt.gca()
    
    # Plot baseline-subtracted signal
    ax.plot(tarr, detector.baseline_subtracted, 'b-', alpha=0.3, label='Raw - Baseline')
    
    # Plot filtered signal
    ax.plot(tarr, detector.filtered_signal, 'g-', alpha=0.7, label='Filtered Signal')
    
    # Plot detected peaks
    if hasattr(detector, 'pulses'):
        # Plot peak points
        ax.plot(detector.pulse_times, detector.pulse_amplitudes, 'r.', 
                markersize=8, label='Detected Peaks')
        
        # Plot pulse areas as stems
        for pulse in detector.pulses:
            ax.plot([pulse.time, pulse.time], [0, pulse.area/pulse.width], 
                   'r-', alpha=0.3, linewidth=1)
    
    # Plot threshold level
    ax.axhline(y=detector.threshold, color='r', linestyle='--', 
               alpha=0.5, label='Threshold')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)

    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    ax.grid(True)

def plot_stft_wt_photon_counts(tarr, fft_arr, freq_arr, bin_centers, counts, fig=None, ax=None):
    """Plot STFT spectrogram with photon counts overlay.
    
    Args:
        tarr: Time array for STFT in seconds
        fft_arr: 2D array of STFT values
        freq_arr: Frequency array in Hz
        bin_centers: Time array for photon counts in milliseconds
        counts: Array of photon counts
        fig: Figure object for colorbar
        ax: Axis to plot on
    """
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
    # Plot STFT
    im = ax.imshow(fft_arr.T, 
                   aspect='auto',
                   origin='lower',
                   extent=[tarr[0]*1e3, tarr[-1]*1e3, freq_arr[0]/1e6, freq_arr[-1]/1e6],
                   interpolation='None',
                   cmap='jet')
    
    # Add colorbar
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    cax = fig.add_axes([pos.x0 + pos.width * 0.92, pos.y0, 0.02, pos.height])
    fig.colorbar(im, cax=cax)
    
    # Add counts overlay
    ax_twin = ax.twinx()
    ax_twin.plot(bin_centers, counts, 'w-', linewidth=1.5, alpha=0.7)
    ax_twin.set_yticks([])
    ax_twin.set_ylim(0, np.max(counts))
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('STFT with Photon Counts Overlay')

    ax.set_xlim(tarr[0]*1e3, tarr[-1]*1e3)

#==============================================================================
# Spectrogram Helpers
#==============================================================================

def floor_for_lognorm(matrix):
    """Replace non-positive entries with the smallest positive value so the
    matrix is safe for matplotlib LogNorm.

    Args:
        matrix (np.ndarray): STFT/spectrogram magnitude matrix.

    Returns:
        tuple: (safe_matrix, vmin) where safe_matrix is a floored copy and vmin
        is the floor value (usable as the LogNorm vmin).
    """
    safe = matrix.copy()
    vmin = safe[safe > 0].min() if np.any(safe > 0) else 1e-10
    safe[safe <= 0] = vmin
    return safe, vmin


def plot_stft(tarr, freq_arr, stft_matrix, ax=None, fig=None, log_norm=True,
              norm=None, cmap='jet', colorbar=True, cbar_label='Magnitude',
              title=None):
    """Plot one STFT spectrogram panel with the ms/MHz axis convention.

    Args:
        tarr (np.ndarray): STFT time array in seconds (displayed in ms)
        freq_arr (np.ndarray): Frequency array in Hz (displayed in MHz)
        stft_matrix (np.ndarray): (n_time, n_freq) magnitude matrix as returned
            by ``data_analysis.signal.calculate_stft``; plotted transposed
        ax: Axes to plot on. If None, current axes will be used.
        fig: Figure for the colorbar. If None, taken from ax.
        log_norm (bool): Floor non-positive values (floor_for_lognorm) and use a
            LogNorm built from this matrix.
        norm: Externally shared matplotlib norm (e.g. for side-by-side panels
            that must share one color scale); overrides the per-matrix LogNorm.
        colorbar (bool): Draw a colorbar labelled cbar_label next to ax.
        title (str, optional): Axes title.

    Sets the y label ("Frequency (MHz)"); the x label is left to the caller so
    stacked shared-x panels can label only the bottom one.

    Returns:
        matplotlib.image.AxesImage: The imshow image (for external colorbars).
    """
    import matplotlib.colors as mcolors

    if ax is None:
        ax = plt.gca()
    if fig is None:
        fig = ax.get_figure()

    matrix = stft_matrix
    if log_norm or norm is not None:
        matrix, vmin = floor_for_lognorm(stft_matrix)
        if norm is None:
            norm = mcolors.LogNorm(vmin=vmin, vmax=matrix.max())

    im = ax.imshow(matrix.T, aspect='auto', origin='lower',
                   extent=[tarr[0]*1e3, tarr[-1]*1e3,
                           freq_arr[0]/1e6, freq_arr[-1]/1e6],
                   interpolation='None', cmap=cmap, norm=norm)
    ax.set_ylabel('Frequency (MHz)')
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, label=cbar_label)
    return im

#==============================================================================
# Figure Finalisation and Output
#==============================================================================

def finalize_figure(fig, save_fig=None, show=False, compact_axes=None, dpi=150):
    """Save and/or show a finished figure, then release it.

    Args:
        fig: The finished figure.
        save_fig: Path to write the figure to; None skips saving.
        show (bool): Call ``plt.show()`` (the interactive window owns the figure
            until closed); otherwise the figure is closed immediately
            (headless/batch saving).
        compact_axes: Optional group of shared-x axes to pack tight *after*
            ``tight_layout`` (which would otherwise re-space them); see
            :func:`pack_shared_x`.
        dpi (int): Resolution for the saved figure.
    """
    fig.tight_layout()
    if compact_axes is not None:
        pack_shared_x(compact_axes)
    if save_fig is not None:
        fig.savefig(save_fig, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_fig}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def fig_path(name, subdir):
    """Centralized figure location under the repo-external output root.

    Routes through :func:`data_analysis.io.paths.output_path`, so figures land in
    ``$DATA_ANALYSIS_OUTPUT/figures/<subdir>/`` (default ``~/data-analysis-output``)
    rather than next to the raw data or in the repo.  ``name`` is the filename
    stem already including any run/tip/figure tags (e.g. ``"02-tipR-line"``); a
    ``.png`` extension is added if absent.

    Drivers follow the convention: ``save_fig is True`` -> ``fig_path(name, subdir)``;
    a string -> that explicit path; anything falsey -> don't save.
    """
    import os
    from data_analysis.io.paths import output_path

    if not os.path.splitext(name)[1]:
        name += ".png"
    return output_path("figures", subdir, name)


def resolve_save(save_fig, name, subdir):
    """Resolve a plot driver's ``save_fig`` convention (see :func:`fig_path`).

    ``True`` -> the centralized ``fig_path(name, subdir)``; a string -> that
    explicit path; anything falsey -> ``None`` (don't save).
    """
    return fig_path(name, subdir) if save_fig is True else (save_fig or None)


def pack_shared_x(axes, gap=0.012):
    """Stack ``axes`` so they abut vertically with only ``gap`` between them.

    For a group of panels that share one x-axis the default ``subplots`` spacing
    leaves wasted blank rows between them.  This repositions the panels to fill
    their *current* combined top->bottom extent, split into equal heights with a
    thin ``gap`` (figure fraction) between neighbours.  Only the passed ``axes``
    move, so any other panels on the same figure keep their original positions
    and stay visually separated.
    """
    boxes = [ax.get_position() for ax in axes]
    top = max(b.y1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    left = boxes[0].x0
    width = boxes[0].width
    n = len(axes)
    h = (top - bottom - gap * (n - 1)) / n
    for i, ax in enumerate(axes):
        y0 = top - (i + 1) * h - i * gap
        ax.set_position([left, y0, width, h])

#==============================================================================
# Style
#==============================================================================

# Okabe-Ito colorblind-safe categorical palette.
OKABE_ITO = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00',
             '#56B4E9', '#F0E442', '#000000']


def configure_publication_style():
    """Set rcParams for compact publication-quality figures (Arial, inward
    ticks on all four sides, thin lines)."""
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

#==============================================================================
# Plane Maps and Animation
#==============================================================================

def grid_by_position(pos_x, pos_y, values):
    """Scatter per-position ``values`` onto a regular (y, x) grid for ``imshow``.

    ``pos_x`` / ``pos_y`` / ``values`` are 1-D, one entry per probe position.
    Returns ``(grid, extent)`` where ``grid`` is ``(ny, nx)`` (NaN at any unvisited
    cell) laid out for ``imshow(origin="lower")`` and ``extent`` is
    ``(xmin, xmax, ymin, ymax)`` in cm.  Positions are snapped to the sorted unique
    x / y axes, so an irregular visiting order still lands on the right cell.
    """
    xs = np.unique(np.round(pos_x, 3))
    ys = np.unique(np.round(pos_y, 3))
    grid = np.full((ys.size, xs.size), np.nan)
    ix = np.searchsorted(xs, np.round(pos_x, 3))
    iy = np.searchsorted(ys, np.round(pos_y, 3))
    grid[iy, ix] = values

    # extent spans cell centers +/- half a step so pixels are centered on positions.
    def _halfspan(a):
        step = np.diff(a).mean() if a.size > 1 else 1.0
        return a[0] - step / 2, a[-1] + step / 2
    xmin, xmax = _halfspan(xs)
    ymin, ymax = _halfspan(ys)
    return grid, (xmin, xmax, ymin, ymax)


class Player:
    """Play/pause controller that steps a matplotlib Slider through frame ticks.

    Wire the actual drawing to the slider's ``on_changed`` callback; each play
    tick just sets the slider value, so play mode and manual scrubbing share
    one draw path.  Connect :meth:`toggle` to a ``Button`` for play/pause.

    Args:
        slider: matplotlib.widgets.Slider whose on_changed callback redraws.
        frame_ticks (np.ndarray): The slider values to step through.
        fig: The figure (its canvas event loop paces the playback).
        interval_s (float): Delay between frames while playing.
    """

    def __init__(self, slider, frame_ticks, fig, interval_s=0.1):
        self.play = False
        self.idx = 0
        self.slider = slider
        self.frame_ticks = np.asarray(frame_ticks)
        self.fig = fig
        self.interval_s = interval_s

    def toggle(self, event=None):
        self.play = not self.play
        if self.play:
            self.idx = int(np.argmin(np.abs(self.frame_ticks - self.slider.val)))
            self.loop()

    def loop(self):
        # Iterative, not recursive: playback can run indefinitely (the ticks
        # wrap around), so recursing per frame would exhaust the stack.
        while self.play:
            self.slider.set_val(self.frame_ticks[self.idx])
            self.idx = (self.idx + 1) % len(self.frame_ticks)
            self.fig.canvas.start_event_loop(self.interval_s)

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == "__main__":
    data_path = r"E:\x-ray\20241029\C3--E-ring-wt-Tungsten2mm-xray--{shot:05d}.trc"
    figs = plot_shots_grid(data_path_template=data_path, shot_range=range(0, 5))

