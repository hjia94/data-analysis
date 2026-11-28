"""Campaign-agnostic single-axis panel painters for static publication figures.

Every painter in this module operates purely on NumPy arrays and matplotlib
``Axes`` objects — zero hardcoded run numbers, file paths, or scope names.
Campaign figure scripts (e.g. ``Jun2026_figures.py``) load data from ``.npz``
or HDF5, compose 1–4 of these painters onto a figure grid, and save through
:func:`data_analysis.viz.plot_utils.finalize_figure`.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
from scipy.ndimage import map_coordinates


def windowed_fft_amp(sig, dt_s, mask=None, pad_factor=8):
    """Single-sided Hann-windowed FFT amplitude [same units as ``sig``] vs kHz.

    Args:
        sig: 1-D real signal array.
        dt_s: Sampling interval in seconds.
        mask: Optional boolean/index mask selecting the window to transform.
        pad_factor: Zero-padding multiplier (>= 1) for smooth spectral display.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(freq_khz, amp)``.
    """
    y = np.asarray(sig if mask is None else np.asarray(sig)[mask], dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError(f"windowed_fft_amp needs a 1-D window of >= 2 samples, got shape {y.shape}")
    y = y - float(np.mean(y))
    n = y.size
    win = np.hanning(n)
    w_sum = float(np.sum(win))
    if w_sum <= 0.0:
        win = np.ones(n, dtype=float)
        w_sum = float(n)
    nfft = n * max(int(pad_factor), 1)
    spec = 2.0 * np.abs(np.fft.rfft(y * win, n=nfft)) / w_sum
    freq_khz = np.fft.rfftfreq(nfft, d=float(dt_s)) * 1e-3
    return freq_khz, spec


def upsample_grid_2d(xs, ys, arr, *, n=129, order=3, mode="nearest"):
    """Spline-upsample a uniform 2-D grid ``arr(ny, nx)`` (real or complex).

    Args:
        xs, ys: 1-D uniform coordinate arrays of lengths ``nx``, ``ny``.
        arr: 2-D real or complex array of shape ``(len(ys), len(xs))``.
        n: Target samples per axis (default ``129``).
        order: Spline interpolation order forwarded to ``scipy.ndimage.map_coordinates``.
        mode: Boundary mode forwarded to ``map_coordinates``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(XF, YF, vals)`` on the ``(n, n)`` grid.
    """
    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    data = np.asarray(arr)
    xf = np.linspace(float(xs_a[0]), float(xs_a[-1]), int(n))
    yf = np.linspace(float(ys_a[0]), float(ys_a[-1]), int(n))
    XF, YF = np.meshgrid(xf, yf)
    ix = (XF - float(xs_a[0])) / float(xs_a[1] - xs_a[0])
    iy = (YF - float(ys_a[0])) / float(ys_a[1] - ys_a[0])
    if np.iscomplexobj(data):
        re = map_coordinates(np.real(data), [iy, ix], order=order, mode=mode)
        im = map_coordinates(np.imag(data), [iy, ix], order=order, mode=mode)
        return XF, YF, re + 1j * im
    vals = map_coordinates(np.asarray(data, dtype=float), [iy, ix], order=order, mode=mode)
    return XF, YF, vals


def radial_binned_quantiles(xs, ys, values, *, edges, center=(0.0, 0.0),
                            min_count=3, q=(25.0, 75.0)):
    """Compute radial-annulus median and quantile envelope of a 2-D gridded field.

    Args:
        xs, ys: 1-D coordinate arrays.
        values: 2-D scalar array of shape ``(len(ys), len(xs))``.
        edges: Monotonic 1-D sequence of radial annulus edges.
        center: ``(cx, cy)`` origin for radial distance.
        min_count: Minimum finite samples required to include an annulus.
        q: ``(q_lo, q_hi)`` percentile pair (default IQR ``(25.0, 75.0)``).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ``(r_med, y_med, y_lo, y_hi)``.
    """
    XX, YY = np.meshgrid(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    cx, cy = float(center[0]), float(center[1])
    RR = np.hypot(XX - cx, YY - cy)
    vals = np.asarray(values, dtype=float)
    r_med, y_med, y_lo, y_hi = [], [], [], []
    q_lo, q_hi = float(q[0]), float(q[1])
    for a, b in zip(edges[:-1], edges[1:]):
        m = (RR >= float(a)) & (RR < float(b)) & np.isfinite(vals)
        if int(np.sum(m)) >= int(min_count):
            r_med.append(float(np.median(RR[m])))
            y_med.append(float(np.median(vals[m])))
            y_lo.append(float(np.percentile(vals[m], q_lo)))
            y_hi.append(float(np.percentile(vals[m], q_hi)))
    return (
        np.asarray(r_med, dtype=float),
        np.asarray(y_med, dtype=float),
        np.asarray(y_lo, dtype=float),
        np.asarray(y_hi, dtype=float),
    )


def prepare_folded_cycle(phase_rad, wave):
    """Map a ``[-pi, pi)`` phase-folded waveform onto a normalized closed ``[0, 1]`` cycle.

    Sorts by phase shifted to ``[0, 1)``, subtracts the mean, scales by the
    half peak-to-peak amplitude, and closes the cycle by appending ``(1.0, w[0])``.
    """
    ph = (np.asarray(phase_rad, dtype=float) + np.pi) / (2.0 * np.pi)
    order = np.argsort(ph)
    ph = ph[order]
    w = np.asarray(wave, dtype=float)[order]
    w = w - float(np.mean(w))
    amp = 0.5 * float(np.max(w) - np.min(w))
    w = w / max(amp, 1e-12)
    return np.append(ph, 1.0), np.append(w, w[0])


def prepare_bdot_2d_fields(
    Bx,
    By,
    Jz,
    *,
    calibrated=True,
    phase_align=True,
    interior_phase_align=False,
    j_unit_mode="auto",
):
    """Normalize/scale 2-D complex or real ``(Bx, By, Jz)`` fields for plotting.

    Shared by :func:`draw_bdot_2d_map` and
    :func:`data_analysis.viz.animations.animate_lab_vs_plasma_frame` so unit
    selection (``kA/m^2`` vs ``A/m^2`` vs normalized), phase alignment, and
    quiver scaling stay consistent between static maps and propagation GIFs.

    Args:
        Bx, By, Jz: 2-D arrays of shape ``(ny, nx)`` (complex phasors or real).
            ``Bx`` and ``By`` may be ``None`` for ``Jz``-only maps.
        calibrated: True if ``Bx, By`` are in Tesla and ``Jz`` in A/m^2;
            False for uncalibrated relative units.
        phase_align: If True and ``Jz`` is complex, rotate ``(Bx, By, Jz)`` by
            ``exp(-i * angle(Jz[argmax |Jz|]))`` so peak ``Re[J_par]`` is positive.
        interior_phase_align: If True, find the phase-alignment peak on
            ``Jz[1:-1, 1:-1]`` (ignoring finite-difference edges).
        j_unit_mode: ``"auto"`` (kA/m^2 when ``j_max >= 1e3``, else A/m^2),
            ``"kA"``, or ``"A"`` when ``calibrated=True``.

    Returns:
        dict: ``Bx``, ``By``, ``Jz`` (phase-aligned raw arrays), ``Jz_plot``
        (scaled to display units), ``j_vlim``, ``j_scale``, ``j_unit``,
        ``cb_label``, ``Bx_plot``, ``By_plot``, ``b_scale`` (95th percentile
        magnitude in display units), ``calibrated``.
    """
    Bx_a = None if Bx is None else np.asarray(Bx)
    By_a = None if By is None else np.asarray(By)
    Jz_a = np.asarray(Jz)

    if phase_align and np.iscomplexobj(Jz_a):
        ref_arr = (
            Jz_a[1:-1, 1:-1]
            if (interior_phase_align and Jz_a.shape[0] > 2 and Jz_a.shape[1] > 2)
            else Jz_a
        )
        abs_j = np.abs(ref_arr)
        if np.any(np.isfinite(abs_j)):
            ph_max = float(np.angle(ref_arr[np.unravel_index(int(np.nanargmax(abs_j)), ref_arr.shape)]))
            rot = np.exp(-1j * ph_max)
            Jz_a = Jz_a * rot
            if Bx_a is not None:
                Bx_a = Bx_a * rot
            if By_a is not None:
                By_a = By_a * rot

    j_max = float(np.nanmax(np.abs(Jz_a))) if np.any(np.isfinite(Jz_a)) else 0.0
    if not np.isfinite(j_max) or j_max <= 0.0:
        j_max = 1.0

    if calibrated:
        if j_unit_mode == "kA":
            j_scale, j_unit = 1e-3, r"kA/m$^2$"
        elif j_unit_mode == "A":
            j_scale, j_unit = 1.0, r"A/m$^2$"
        elif j_unit_mode == "auto":
            j_scale, j_unit = (1e-3, r"kA/m$^2$") if j_max >= 1e3 else (1.0, r"A/m$^2$")
        else:
            raise ValueError(f"j_unit_mode must be 'auto', 'kA', or 'A', got {j_unit_mode!r}")
        Jz_plot = Jz_a * j_scale
        j_vlim = j_max * j_scale
        cb_label = rf"$\mathrm{{Re}}[J_\parallel]$ [{j_unit}]"
        # Convert Tesla -> Gauss for quiver display
        Bx_plot = None if Bx_a is None else (Bx_a * 1e4)
        By_plot = None if By_a is None else (By_a * 1e4)
    else:
        j_scale, j_unit = 1.0 / j_max, ""
        Jz_plot = Jz_a / j_max
        j_vlim = 1.0
        cb_label = r"$\mathrm{Re}[J_\parallel] / J_{\max}$"
        Bx_plot = Bx_a
        By_plot = By_a

    if Bx_plot is not None and By_plot is not None:
        b_mag = np.hypot(np.abs(Bx_plot), np.abs(By_plot))
        b_scale = float(np.nanpercentile(b_mag, 95)) if np.any(np.isfinite(b_mag)) else 0.0
        if not np.isfinite(b_scale):
            b_scale = 0.0
    else:
        b_scale = 0.0

    return {
        "Bx": Bx_a,
        "By": By_a,
        "Jz": Jz_a,
        "Jz_plot": Jz_plot,
        "j_vlim": j_vlim,
        "j_scale": j_scale,
        "j_unit": j_unit,
        "cb_label": cb_label,
        "Bx_plot": Bx_plot,
        "By_plot": By_plot,
        "b_scale": b_scale,
        "calibrated": bool(calibrated),
    }


def _unpack_trace(item):
    """Normalize a trace entry (dict or 3/4-tuple) into a dict."""
    if isinstance(item, dict):
        return item
    if len(item) == 3:
        t, y, label = item
        return {"t": t, "y": y, "label": label}
    if len(item) == 4:
        t, y, label, color = item
        return {"t": t, "y": y, "label": label, "color": color}
    raise ValueError(f"Trace entry must be a dict or (t, y, label[, color]) tuple; got len={len(item)}")


def draw_stacked_traces(
    ax,
    traces,
    *,
    xlim=None,
    ylim=None,
    yticks=None,
    offset=None,
    xlabel="Time [ms]",
    ylabel=None,
    ylabel_index=None,
    title=None,
    lw=0.95,
    grid_alpha=0.22,
    label_pos=(0.02, 0.12),
    label_fontsize=10.2,
    ylim_pad=(0.20, 0.12),
):
    """Draw multi-run raw time traces stacked vertically.

    Supports two layouts through the same API:

    1. **Single ``Axes``:** traces are vertically offset top-to-bottom on ``ax``
       with inline colored labels and clean offset spacing.
    2. **Sequence of ``Axes``** (one per trace, e.g. from ``subgridspec``):
       each trace is drawn on its own shared-x sub-panel with common ``ylim``
       so true signal amplitudes stay directly readable on the y-axis ticks.

    Args:
        ax: A single :class:`matplotlib.axes.Axes` or a sequence of ``Axes``
            with ``len(ax) == len(traces)``.
        traces: Sequence of dicts ``{"t", "y", "label", "color", ...}`` or
            ``(t, y, label[, color])`` tuples.
        xlim: Optional ``(xmin, xmax)``.
        ylim: Optional ``(ymin, ymax)`` (for multi-axes mode, or explicit single-axes).
        yticks: Optional y-tick locations (multi-axes mode).
        offset: Vertical step between traces in single-axes mode (auto if None).
        xlabel: Bottom x-axis label.
        ylabel: Y-axis label (placed on the middle panel in multi-axes mode).
        ylabel_index: Optional panel index for ``ylabel`` in multi-axes mode.
        title: Title (placed on top panel in multi-axes mode).
        lw: Default line width.
        grid_alpha: Grid line opacity.
        label_pos: ``(x_frac, y_frac)`` in axes coordinates for multi-axes labels.
        label_fontsize: Font size for per-trace labels.
        ylim_pad: ``(bottom_frac, top_frac)`` auto-padding in multi-axes mode.

    Returns:
        list[matplotlib.lines.Line2D]: Plotted line artists.
    """
    entries = [_unpack_trace(tr) for tr in traces]
    if not entries:
        raise ValueError("draw_stacked_traces requires at least one trace")

    lines = []
    if isinstance(ax, Axes):
        # Single-axes vertically offset stack (top-to-bottom order)
        spans = []
        for e in entries:
            y = np.asarray(e["y"], dtype=float)
            finite = y[np.isfinite(y)]
            if finite.size:
                spans.append(float(np.max(finite) - np.min(finite)))
        base_span = max(spans) if spans else 1.0
        step = float(offset) if offset is not None else (1.25 * base_span if base_span > 0.0 else 1.0)
        n = len(entries)

        t_lo = min(float(np.nanmin(e["t"])) for e in entries)
        t_hi = max(float(np.nanmax(e["t"])) for e in entries)
        x0, x1 = xlim if xlim is not None else (t_lo, t_hi)

        y_centers = []
        for i, e in enumerate(entries):
            t = np.asarray(e["t"], dtype=float)
            y = np.asarray(e["y"], dtype=float)
            y_ref = float(np.nanmedian(y)) if np.any(np.isfinite(y)) else 0.0
            shift = (n - 1 - i) * step - y_ref
            y_plot = y + shift
            col = e.get("color")
            (line,) = ax.plot(
                t, y_plot,
                color=col,
                lw=e.get("lw", lw),
                ls=e.get("ls", "-"),
            )
            lines.append(line)
            yc = (n - 1 - i) * step
            y_centers.append(yc)
            lbl = e.get("label")
            if lbl:
                ax.text(
                    x0 + 0.015 * (x1 - x0),
                    yc + 0.12 * step,
                    lbl,
                    color=col or line.get_color(),
                    fontweight="bold",
                    fontsize=label_fontsize,
                    va="bottom",
                )

        ax.set_xlim(x0, x1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(-0.65 * step, (n - 1 + 0.75) * step)
        if yticks is not None:
            ax.set_yticks(yticks)
        else:
            ax.set_yticks([])
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontweight="bold")
        ax.grid(alpha=grid_alpha)
        return lines

    # Multi-axes stack (one Axes per trace)
    axes_list = list(ax) if isinstance(ax, Sequence) or hasattr(ax, "__iter__") else [ax]
    if len(axes_list) != len(entries):
        raise ValueError(
            f"draw_stacked_traces received {len(axes_list)} axes for {len(entries)} traces"
        )

    if ylim is None:
        vmin_all = min(float(np.nanmin(e["y"])) for e in entries)
        vmax_all = max(float(np.nanmax(e["y"])) for e in entries)
        span = vmax_all - vmin_all
        if span <= 0.0 or not np.isfinite(span):
            span = 1.0
        y_bounds = (vmin_all - ylim_pad[0] * span, vmax_all + ylim_pad[1] * span)
    else:
        y_bounds = tuple(ylim)

    n = len(entries)
    y_idx = (n // 2) if ylabel_index is None else int(ylabel_index)
    for i, (sub_ax, e) in enumerate(zip(axes_list, entries)):
        t = np.asarray(e["t"], dtype=float)
        y = np.asarray(e["y"], dtype=float)
        col = e.get("color")
        (line,) = sub_ax.plot(
            t, y,
            color=col,
            lw=e.get("lw", lw),
            ls=e.get("ls", "-"),
        )
        lines.append(line)
        if xlim is not None:
            sub_ax.set_xlim(*xlim)
        sub_ax.set_ylim(*y_bounds)
        if yticks is not None:
            sub_ax.set_yticks(yticks)
        sub_ax.grid(alpha=grid_alpha)
        lbl = e.get("label")
        if lbl:
            sub_ax.text(
                label_pos[0], label_pos[1], lbl,
                transform=sub_ax.transAxes,
                color=col or line.get_color(),
                fontweight="bold",
                fontsize=label_fontsize,
                va="bottom",
            )
        if i < n - 1:
            plt.setp(sub_ax.get_xticklabels(), visible=False)
        elif xlabel:
            sub_ax.set_xlabel(xlabel)
        if ylabel and i == y_idx:
            sub_ax.set_ylabel(ylabel)

    if title:
        axes_list[0].set_title(title, fontweight="bold")
    return lines


def _unpack_spectrum(item):
    """Normalize a spectrum entry (dict or 3/4-tuple) into a dict."""
    if isinstance(item, dict):
        return item
    if len(item) == 3:
        f, p, label = item
        return {"freq": f, "power": p, "label": label}
    if len(item) == 4:
        f, p, label, color = item
        return {"freq": f, "power": p, "label": label, "color": color}
    raise ValueError(f"Spectrum entry must be a dict or (freq, power, label[, color]) tuple; got len={len(item)}")


def draw_psd_overlay(
    ax,
    spectra,
    *,
    xlim=None,
    ylim=None,
    logy=True,
    xlabel="Frequency [kHz]",
    ylabel=None,
    title=None,
    grid_alpha=0.25,
    legend_kw=None,
    harmonics=None,
    peak_half_width=0.4,
    peak_ms=5.5,
):
    """Draw overlaid FFT/Welch PSD spectra with optional peak & harmonic markers.

    Args:
        ax: Target matplotlib ``Axes``.
        spectra: Sequence of dicts ``{"freq", "power", "label", "color", "ls",
            "lw", "peak_freq"}`` or ``(freq, power, label[, color])`` tuples.
        xlim: Optional ``(fmin, fmax)`` in same units as ``freq``.
        ylim: Optional ``(ymin, ymax)``.
        logy: If True, plot with ``ax.semilogy``; if False, linear ``ax.plot``.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional axes title.
        grid_alpha: Grid opacity.
        legend_kw: Optional dict forwarded to ``ax.legend(...)`` (pass ``False``
            to suppress legend).
        harmonics: Optional float ``f0`` or dict ``{"f0": float, "orders":
            (1, 2, 3, 4), "half_width": 0.5, "show_f0_value": True, "unit": "kHz",
            "text_dx": 0.45, "text_dy": 0.16, "y_cap": 1.13}`` for annotating
            harmonic peaks ``f0, 2f0, ...`` with arrows.
        peak_half_width: Half-window around each spectrum's optional
            ``peak_freq`` used to locate the marker height.
        peak_ms: Marker size for ``peak_freq`` dots.

    Returns:
        list[matplotlib.lines.Line2D]: Plotted spectrum line artists.
    """
    entries = [_unpack_spectrum(sp) for sp in spectra]
    if not entries:
        raise ValueError("draw_psd_overlay requires at least one spectrum")

    plotter = ax.semilogy if logy else ax.plot
    lines = []
    plotted = []
    for e in entries:
        f = np.asarray(e["freq"], dtype=float)
        p = np.asarray(e["power"], dtype=float)
        if xlim is not None:
            m = (f >= float(xlim[0])) & (f <= float(xlim[1]))
            f_use, p_use = f[m], p[m]
        else:
            f_use, p_use = f, p
        col = e.get("color")
        (line,) = plotter(
            f_use, p_use,
            color=col,
            ls=e.get("ls", "-"),
            lw=e.get("lw", 1.8),
            label=e.get("label"),
        )
        lines.append(line)
        plotted.append((f_use, p_use))

        f_pk = e.get("peak_freq")
        if f_pk is not None and np.isfinite(f_pk):
            m_pk = (f >= float(f_pk) - peak_half_width) & (f <= float(f_pk) + peak_half_width)
            if np.any(m_pk):
                y_pk = float(np.nanmax(p[m_pk]))
                ax.plot(float(f_pk), y_pk, "o", color=col or line.get_color(), ms=peak_ms, zorder=5)

    if harmonics is not None:
        hcfg = {"f0": float(harmonics)} if isinstance(harmonics, (int, float, np.floating)) else dict(harmonics)
        f0 = float(hcfg["f0"])
        orders = tuple(hcfg.get("orders", (1, 2, 3, 4)))
        hw = float(hcfg.get("half_width", 0.5))
        show_f0_val = bool(hcfg.get("show_f0_value", True))
        unit = hcfg.get("unit", "kHz")
        fmax_lim = float(xlim[1]) if xlim is not None else max(float(np.nanmax(f)) for f, _ in plotted if f.size)
        dx_txt = float(hcfg.get("text_dx", 0.45))
        dy_txt = float(hcfg.get("text_dy", 0.16))
        y_cap = hcfg.get("y_cap", 1.13 if not logy else None)

        for k in orders:
            fk = k * f0
            if fk > fmax_lim - hw:
                continue
            best_y = -np.inf
            best_f = fk
            for f_arr, p_arr in plotted:
                mk = (f_arr >= fk - hw) & (f_arr <= fk + hw)
                if not np.any(mk):
                    continue
                idx = int(np.nanargmax(p_arr[mk]))
                yk = float(p_arr[mk][idx])
                if yk >= best_y:
                    best_y = yk
                    best_f = float(f_arr[mk][idx])
            if not np.isfinite(best_y) or best_y <= -np.inf:
                continue
            lbl = r"$f_0$" if k == 1 else rf"${k}f_0$"
            txt = f"{lbl} ({best_f:.1f} {unit})" if (k == 1 and show_f0_val) else lbl
            if logy:
                y_txt = best_y * float(hcfg.get("log_text_mult", 2.0))
            else:
                y_txt = best_y + dy_txt
                if y_cap is not None:
                    y_txt = min(y_txt, float(y_cap))
            ax.annotate(
                txt,
                xy=(best_f, best_y),
                xytext=(best_f + dx_txt, y_txt),
                fontsize=9.3,
                fontweight="bold",
                color="0.15",
                arrowprops=dict(arrowstyle="->", color="0.35", lw=0.9),
            )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=grid_alpha)

    if legend_kw is not False:
        has_labels = any(e.get("label") for e in entries)
        if has_labels:
            lkw = {"loc": "upper right", "frameon": True, "framealpha": 0.92}
            if isinstance(legend_kw, dict):
                lkw.update(legend_kw)
            ax.legend(**lkw)
    if title:
        ax.set_title(title, fontweight="bold")
    return lines


def draw_ne_te_profile(
    ax,
    x,
    ne,
    te,
    *,
    xlim=None,
    ne_ylim=None,
    te_ylim=None,
    ne_ylim_min=None,
    te_ylim_min=None,
    xlabel=r"$x - x_c$ [cm]",
    ne_ylabel=r"$n_e$ [$10^{18}\;\mathrm{m}^{-3}$]",
    te_ylabel=r"$T_e$ [eV]",
    ne_color="#1d4ed8",
    te_color="#dc2626",
    ne_label=r"$n_e$",
    te_label=r"$T_e$",
    eta_e=None,
    annotation=None,
    title=None,
    grid_alpha=0.22,
    legend_loc="upper right",
):
    """Draw a dual-axis radial ``n_e(x)`` and ``T_e(x)`` profile panel.

    Args:
        ax: Primary matplotlib ``Axes`` (left y-axis for ``n_e``).
        x, ne, te: 1-D radial coordinate and profile arrays.
        xlim: Optional ``(xmin, xmax)``; also masks plotted data to this window.
        ne_ylim, te_ylim: Explicit y-axis limits; auto-scaled with headroom if None.
        ne_ylim_min, te_ylim_min: Optional minimum upper y-limit when auto-scaling.
        xlabel, ne_ylabel, te_ylabel: Axis labels.
        ne_color, te_color: Curve and y-axis colors.
        ne_label, te_label: Legend labels.
        eta_e: Optional scalar ``eta_e`` to annotate in bottom-left.
        annotation: Optional custom annotation text (overrides ``eta_e`` text).
        title: Optional axes title.
        grid_alpha: Grid opacity.
        legend_loc: Legend location on ``ax``.

    Returns:
        tuple: ``(ax_te, line_ne, line_te)``.
    """
    x_a = np.asarray(x, dtype=float)
    ne_a = np.asarray(ne, dtype=float)
    te_a = np.asarray(te, dtype=float)

    if xlim is not None:
        m = (x_a >= float(xlim[0])) & (x_a <= float(xlim[1]))
        x_use, ne_use, te_use = x_a[m], ne_a[m], te_a[m]
    else:
        x_use, ne_use, te_use = x_a, ne_a, te_a

    (l1,) = ax.plot(x_use, ne_use, "o-", color=ne_color, lw=2.2, ms=5.2, label=ne_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ne_ylabel, color=ne_color)
    ax.tick_params(axis="y", labelcolor=ne_color)
    if xlim is not None:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ne_ylim is not None:
        ax.set_ylim(*ne_ylim)
    elif ne_use.size and np.any(np.isfinite(ne_use)):
        ymax_ne = float(np.nanmax(ne_use)) * 1.18
        if ne_ylim_min is not None:
            ymax_ne = max(ymax_ne, float(ne_ylim_min))
        if ymax_ne > 0.0:
            ax.set_ylim(0.0, ymax_ne)
    ax.grid(alpha=grid_alpha)

    ax2 = ax.twinx()
    (l2,) = ax2.plot(x_use, te_use, "s-", color=te_color, lw=2.2, ms=5.0, label=te_label)
    ax2.set_ylabel(te_ylabel, color=te_color)
    ax2.tick_params(axis="y", labelcolor=te_color)
    if te_ylim is not None:
        ax2.set_ylim(*te_ylim)
    elif te_use.size and np.any(np.isfinite(te_use)):
        ymax_te = float(np.nanmax(te_use)) * 1.24
        if te_ylim_min is not None:
            ymax_te = max(ymax_te, float(te_ylim_min))
        if ymax_te > 0.0:
            ax2.set_ylim(0.0, ymax_te)

    note = annotation
    if note is None and eta_e is not None and np.isfinite(eta_e):
        note = rf"$\eta_e = {float(eta_e):.2f}$"
    if note:
        ax.text(0.04, 0.06, note, transform=ax.transAxes, fontsize=10.5, va="bottom")

    ax.legend(handles=[l1, l2], loc=legend_loc, frameon=True, framealpha=0.92)
    if title:
        ax.set_title(title, fontweight="bold")
    return ax2, l1, l2


def draw_bdot_2d_map(
    ax,
    xs,
    ys,
    Bx,
    By,
    Jz,
    *,
    calibrated=True,
    phase_align=True,
    interior_phase_align=False,
    j_unit_mode="auto",
    j_vlim=None,
    upsample_n=None,
    contour_levels=None,
    cmap="RdBu_r",
    add_colorbar=True,
    cb_label=None,
    cbar_kw=None,
    show_quiver=True,
    quiver_scale_mult=14.0,
    quiver_width=0.0042,
    quiver_alpha=0.82,
    show_quiverkey=True,
    quiverkey_val=None,
    quiverkey_box=(0.72, 0.025, 0.25, 0.065),
    quiverkey_xy=(0.80, 0.058),
    circles=(),
    markers=(),
    note_lines=None,
    xlim=None,
    ylim=None,
    xlabel=r"$x$ [cm]",
    ylabel=r"$y$ [cm]",
    title=None,
    legend_loc="upper left",
    legend_kw=None,
):
    """Draw a 2-D ``Re[J_par]`` pcolormesh + optional contours & ``B_perp`` quiver map.

    Args:
        ax: Target matplotlib ``Axes``.
        xs, ys: 1-D coordinate arrays in cm.
        Bx, By, Jz: 2-D arrays of shape ``(len(ys), len(xs))`` (complex phasors
            or real snapshots). ``Bx``/``By`` may be ``None`` for ``Jz``-only maps.
        calibrated: True for calibrated SI inputs (``Bx, By`` in Tesla -> quiver
            in Gauss with quiver-key box; ``Jz`` in A/m^2 -> A/m^2 or kA/m^2);
            False for normalized relative units.
        phase_align: Rotate complex phasors so peak ``Re[J_par]`` is positive.
        interior_phase_align: Use ``Jz[1:-1, 1:-1]`` to choose alignment phase.
        j_unit_mode: ``"auto"``, ``"kA"``, or ``"A"`` when ``calibrated=True``.
        j_vlim: Optional explicit symmetric color limit ``(-j_vlim, +j_vlim)``.
        upsample_n: Optional int (e.g. ``129``) to bicubic-upsample ``Jz`` for
            smooth ``pcolormesh`` and contours while keeping ``quiver`` on ``(xs, ys)``.
        contour_levels: Optional sequence of scalar levels for solid (``>0``)
            and dashed (``<0``) black contour overlays on ``Re[J_par]``.
        cmap: Diverging colormap name.
        add_colorbar: If True, add a colorbar attached to ``ax``.
        cb_label: Optional override for colorbar label text.
        cbar_kw: Optional dict forwarded to ``fig.colorbar(...)``.
        show_quiver: Draw ``(Bx, By)`` quiver overlay when ``Bx`` and ``By`` are provided.
        quiver_scale_mult: Multiplier on 95th-percentile ``|B_perp|`` for quiver scale.
        quiver_width: Quiver shaft width.
        quiver_alpha: Quiver arrow opacity.
        show_quiverkey: Draw rounded white quiver-key box when ``calibrated=True``.
        quiverkey_val: Explicit quiver-key reference value in Gauss (auto if None).
        quiverkey_box: ``(x, y, w, h)`` in axes coords for the quiver-key backdrop.
        quiverkey_xy: ``(x, y)`` in axes coords for ``ax.quiverkey``.
        circles: Sequence of dicts ``{"radius": r, "center": (cx, cy), "ec": ...,
            "ls": ..., "lw": ..., "label": ...}`` for electrode/orbit overlays.
        markers: Sequence of dicts ``{"x": x, "y": y, "marker": ..., "ms": ...,
            "color": ..., "mec": ..., "mew": ..., "label": ...}`` for probe dots.
        note_lines: Optional sequence of strings placed in bottom-left white box.
        xlim, ylim: Optional ``(lo, hi)`` axis limits (default grid extent).
        xlabel, ylabel, title: Labels and title.
        legend_loc: Legend location if any circles/markers carry labels (None suppresses).
        legend_kw: Optional dict merged into ``ax.legend(...)``.

    Returns:
        dict: ``{"im", "quiver", "colorbar", "j_vlim", "cb_label"}``.
    """
    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    prep = prepare_bdot_2d_fields(
        Bx, By, Jz,
        calibrated=calibrated,
        phase_align=phase_align,
        interior_phase_align=interior_phase_align,
        j_unit_mode=j_unit_mode,
    )

    if upsample_n is not None:
        XJ, YJ, Jz_up = upsample_grid_2d(xs_a, ys_a, prep["Jz_plot"], n=int(upsample_n))
        Jz_r = np.real(Jz_up)
    else:
        XJ, YJ = np.meshgrid(xs_a, ys_a)
        Jz_r = np.real(prep["Jz_plot"])

    vlim = float(j_vlim) if j_vlim is not None else float(prep["j_vlim"])
    if not np.isfinite(vlim) or vlim <= 0.0:
        vlim = 1.0

    im = ax.pcolormesh(XJ, YJ, Jz_r, cmap=cmap, vmin=-vlim, vmax=vlim, shading="auto")
    if contour_levels is not None:
        pos_lvls = [lv for lv in contour_levels if lv > 0]
        neg_lvls = [lv for lv in contour_levels if lv < 0]
        if pos_lvls:
            ax.contour(XJ, YJ, Jz_r, levels=pos_lvls, colors="k", linestyles="-", linewidths=0.85, alpha=0.72)
        if neg_lvls:
            ax.contour(XJ, YJ, Jz_r, levels=neg_lvls, colors="k", linestyles="--", linewidths=0.85, alpha=0.72)

    lbl_cb = prep["cb_label"] if cb_label is None else cb_label
    cb = None
    if add_colorbar:
        ckw = {"fraction": 0.046, "pad": 0.03, "shrink": 0.84}
        if isinstance(cbar_kw, dict):
            ckw.update(cbar_kw)
        cb = ax.get_figure().colorbar(im, ax=ax, **ckw)
        cb.set_label(lbl_cb)

    qv = None
    if show_quiver and prep["Bx_plot"] is not None and prep["By_plot"] is not None:
        XX, YY = np.meshgrid(xs_a, ys_a)
        Bx_r = np.real(prep["Bx_plot"])
        By_r = np.real(prep["By_plot"])
        if calibrated:
            bscale_g = float(np.nanpercentile(np.hypot(Bx_r, By_r), 95)) if np.any(np.isfinite(Bx_r)) else 0.0
            if np.isfinite(bscale_g) and bscale_g > 0.0:
                qv = ax.quiver(
                    XX, YY, Bx_r, By_r,
                    color="k", scale=bscale_g * quiver_scale_mult,
                    width=quiver_width, alpha=quiver_alpha,
                )
                if show_quiverkey:
                    key_g = float(quiverkey_val) if quiverkey_val is not None else (0.2 if bscale_g >= 0.15 else 0.1)
                    bx0, by0, bw, bh = quiverkey_box
                    ax.add_patch(FancyBboxPatch(
                        (bx0, by0), bw, bh,
                        boxstyle="round,pad=0.012", transform=ax.transAxes,
                        fc="white", ec="none", alpha=0.88, zorder=7,
                    ))
                    qk = ax.quiverkey(
                        qv, quiverkey_xy[0], quiverkey_xy[1], key_g, f"{key_g:g} G",
                        labelpos="E", coordinates="axes",
                        fontproperties={"size": 9.4, "weight": "bold"},
                    )
                    qk.set_zorder(8)
        else:
            bscale = float(np.nanpercentile(np.hypot(Bx_r, By_r), 95)) if np.any(np.isfinite(Bx_r)) else 0.0
            if np.isfinite(bscale) and bscale > 0.0:
                qv = ax.quiver(
                    XX, YY, Bx_r / bscale, By_r / bscale,
                    color="k", scale=quiver_scale_mult,
                    width=quiver_width, alpha=quiver_alpha,
                )

    has_legend = False
    for c in circles:
        center = c.get("center", (0.0, 0.0))
        lbl = c.get("label")
        if lbl:
            has_legend = True
        ax.add_patch(Circle(
            center,
            float(c["radius"]),
            fill=c.get("fill", False),
            ec=c.get("ec", "#16a34a"),
            ls=c.get("ls", "-"),
            lw=c.get("lw", 1.6),
            alpha=c.get("alpha", 1.0),
            zorder=c.get("zorder", 5),
            label=lbl,
        ))

    for m in markers:
        lbl = m.get("label")
        if lbl:
            has_legend = True
        ax.plot(
            float(m["x"]),
            float(m["y"]),
            marker=m.get("marker", "o"),
            ms=m.get("ms", 8.0),
            color=m.get("color", "#facc15"),
            mec=m.get("mec", "k"),
            mew=m.get("mew", 1.0),
            ls="none",
            zorder=m.get("zorder", 6),
            label=lbl,
        )

    ax.set_aspect("equal")
    ax.set_xlim(*(xlim if xlim is not None else (float(xs_a[0]), float(xs_a[-1]))))
    ax.set_ylim(*(ylim if ylim is not None else (float(ys_a[0]), float(ys_a[-1]))))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if note_lines:
        ax.text(
            0.04, 0.04, "\n".join(note_lines),
            transform=ax.transAxes, fontsize=9.4, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.88),
        )
    if has_legend and legend_loc is not None:
        lkw = {"loc": legend_loc, "frameon": True, "framealpha": 0.90}
        if isinstance(legend_kw, dict):
            lkw.update(legend_kw)
        ax.legend(**lkw)
    if title:
        ax.set_title(title, fontweight="bold")

    return {
        "im": im,
        "quiver": qv,
        "colorbar": cb,
        "j_vlim": vlim,
        "cb_label": lbl_cb,
    }


def draw_polar_phasor_scatter(
    ax,
    series,
    *,
    rlim=1.05,
    rticks=(0.25, 0.50, 0.75, 1.00),
    rlabel_position=112,
    title=None,
    title_kw=None,
    legend_kw=None,
    grid_alpha=0.28,
):
    """Draw complex amplitude/phase groups on a signed-angle polar ``Axes``.

    Configures ``0 deg`` at East, CCW positive angles labelled in ``(-180, +180]``,
    horizontal ``0 deg / 180 deg`` reference rays, and plots each scatter group
    dict containing ``"r"`` and ``"deg"`` (plus standard ``ax.scatter`` kwargs).

    Returns:
        list[matplotlib.collections.PathCollection]: Scatter artists.
    """
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_rlim(0.0, float(rlim))
    if rticks is not None:
        ax.set_rticks(list(rticks))
    ax.set_rlabel_position(float(rlabel_position))
    ax.tick_params(axis="y", labelsize=8.6, colors="0.35")
    ax.set_thetagrids(
        [0, 45, 90, 135, 180, 225, 270, 315],
        labels=[
            r"$0^\circ$", r"$+45^\circ$", r"$+90^\circ$", r"$+135^\circ$",
            r"$\pm 180^\circ$", r"$-135^\circ$", r"$-90^\circ$", r"$-45^\circ$",
        ],
        fontsize=9.2,
    )
    ax.grid(alpha=grid_alpha)
    ax.plot([0.0, 0.0], [0.0, float(rlim)], color="0.5", ls=":", lw=1.0, zorder=1)
    ax.plot([np.pi, np.pi], [0.0, float(rlim)], color="0.5", ls=":", lw=1.0, zorder=1)

    artists = []
    has_label = False
    for item in series:
        cfg = dict(item)
        r_vals = np.atleast_1d(np.asarray(cfg.pop("r"), dtype=float))
        th_vals = np.radians(np.atleast_1d(np.asarray(cfg.pop("deg"), dtype=float)))
        if cfg.get("label"):
            has_label = True
        artists.append(ax.scatter(th_vals, r_vals, **cfg))

    if has_label and legend_kw is not False:
        lkw = {"loc": "upper center", "bbox_to_anchor": (0.50, -0.10), "ncol": 2,
               "frameon": True, "framealpha": 0.92, "fontsize": 8.5, "columnspacing": 1.1}
        if isinstance(legend_kw, dict):
            lkw.update(legend_kw)
        ax.legend(**lkw)
    if title:
        tkw = {"fontweight": "bold", "pad": 14}
        if isinstance(title_kw, dict):
            tkw.update(title_kw)
        ax.set_title(title, **tkw)
    return artists


def draw_pulse_and_zoom(
    ax_full,
    ax_zoom,
    t_ms,
    traces,
    zoom_window_ms,
    *,
    t_window_ms=None,
    full_xlabel="Time relative to bias turn-on [ms]",
    full_ylabel="Current [A]",
    zoom_xlabel="Time [ms]",
    zoom_ylabel=r"Oscillating current $\delta I$ [A]",
    full_title=None,
    zoom_title=None,
    full_ylim=None,
    zoom_ylim=None,
    zoom_note=None,
    full_legend_loc="lower center",
    zoom_legend_loc="upper right",
    shade_color="0.90",
    shade_alpha=0.65,
    grid_alpha=0.22,
):
    """Draw a full bias-pulse trace pair with shaded zoom + zoomed zero-mean AC panel.

    Args:
        ax_full: Top ``Axes`` showing the full pulse traces and shaded zoom window.
        ax_zoom: Bottom ``Axes`` showing mean-subtracted AC traces inside
            ``zoom_window_ms``.
        t_ms: 1-D shared time array [ms].
        traces: Sequence of dicts ``{"y", "color", "full_label", "zoom_label",
            "full_ls", "zoom_ls", "full_lw", "zoom_lw"}`` (``"label"`` is used
            as a fallback for both panels when separate labels are omitted).
        zoom_window_ms: ``(z_lo, z_hi)`` zoom interval [ms].
        t_window_ms: Optional ``(t_lo, t_hi)`` x-limits for ``ax_full``.
        full_xlabel, full_ylabel, zoom_xlabel, zoom_ylabel: Axis labels.
        full_title, zoom_title: Panel titles.
        full_ylim, zoom_ylim: Explicit y-limits; auto-scaled when None.
        zoom_note: Optional annotation text placed in bottom-left of ``ax_zoom``.
        full_legend_loc, zoom_legend_loc: Legend positions.
        shade_color, shade_alpha: Shaded zoom-window span styling on ``ax_full``.
        grid_alpha: Grid opacity.

    Returns:
        dict: ``{"full_lines": list, "zoom_lines": list}``.
    """
    t_a = np.asarray(t_ms, dtype=float)
    z_lo, z_hi = float(zoom_window_ms[0]), float(zoom_window_ms[1])
    t_lo, t_hi = (
        (float(t_window_ms[0]), float(t_window_ms[1]))
        if t_window_ms is not None
        else (float(t_a[0]), float(t_a[-1]))
    )
    m_z = (t_a >= z_lo) & (t_a <= z_hi)
    if not np.any(m_z):
        raise ValueError(f"zoom_window_ms={zoom_window_ms} selects zero samples from t_ms")

    # (a) Full pulse panel
    ax_full.axvspan(z_lo, z_hi, color=shade_color, alpha=shade_alpha, zorder=0)
    full_lines = []
    full_max = -np.inf
    has_full_lbl = False
    for tr in traces:
        y = np.asarray(tr["y"], dtype=float)
        lbl = tr.get("full_label", tr.get("label"))
        if lbl:
            has_full_lbl = True
        (line,) = ax_full.plot(
            t_a, y,
            ls=tr.get("full_ls", "-"),
            color=tr.get("color"),
            lw=tr.get("full_lw", 1.6),
            label=lbl,
        )
        full_lines.append(line)
        if np.any(np.isfinite(y)):
            full_max = max(full_max, float(np.nanmax(y)))

    ax_full.set_xlim(t_lo, t_hi)
    if full_ylim is not None:
        ax_full.set_ylim(*full_ylim)
    elif np.isfinite(full_max) and full_max > 0.0:
        ax_full.set_ylim(-2.0, full_max * 1.15)
    if full_xlabel:
        ax_full.set_xlabel(full_xlabel)
    if full_ylabel:
        ax_full.set_ylabel(full_ylabel)
    ax_full.grid(alpha=grid_alpha)
    if has_full_lbl and full_legend_loc is not None:
        ax_full.legend(loc=full_legend_loc, frameon=True, framealpha=0.92)
    if full_title:
        ax_full.set_title(full_title, fontweight="bold")

    # (b) Zoomed zero-mean AC panel
    ax_zoom.axhline(0.0, color="0.45", ls=":", lw=1.0)
    zoom_lines = []
    ac_max = 0.0
    has_zoom_lbl = False
    t_z = t_a[m_z]
    for tr in traces:
        y_z = np.asarray(tr["y"], dtype=float)[m_z]
        dy = y_z - float(np.nanmean(y_z))
        lbl = tr.get("zoom_label", tr.get("label"))
        if lbl:
            has_zoom_lbl = True
        (line,) = ax_zoom.plot(
            t_z, dy,
            ls=tr.get("zoom_ls", "-"),
            color=tr.get("color"),
            lw=tr.get("zoom_lw", 2.0),
            label=lbl,
        )
        zoom_lines.append(line)
        if np.any(np.isfinite(dy)):
            ac_max = max(ac_max, float(np.nanmax(np.abs(dy))))

    ax_zoom.set_xlim(z_lo, z_hi)
    if zoom_ylim is not None:
        ax_zoom.set_ylim(*zoom_ylim)
    elif ac_max > 0.0:
        ymax = ac_max * 1.28
        ax_zoom.set_ylim(-ymax, ymax)
    if zoom_xlabel:
        ax_zoom.set_xlabel(zoom_xlabel)
    if zoom_ylabel:
        ax_zoom.set_ylabel(zoom_ylabel)
    ax_zoom.grid(alpha=grid_alpha)
    if zoom_note:
        ax_zoom.text(0.03, 0.06, zoom_note, transform=ax_zoom.transAxes, fontsize=10.0, va="bottom")
    if has_zoom_lbl and zoom_legend_loc is not None:
        ax_zoom.legend(loc=zoom_legend_loc, frameon=True, framealpha=0.92)
    if zoom_title:
        ax_zoom.set_title(zoom_title, fontweight="bold")

    return {"full_lines": full_lines, "zoom_lines": zoom_lines}
