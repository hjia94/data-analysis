"""Campaign-agnostic 2-D complex-phasor propagation animation renderers.

All animation routines take plain NumPy arrays and physics scalars — zero
hardcoded run numbers, file paths, or scope names — and reuse field
normalization/unit scaling from :mod:`data_analysis.viz.panels`.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from data_analysis.viz.panels import prepare_bdot_2d_fields


def _add_curved_arrow(ax, cx, cy, r, ang1_deg, ang2_deg, color, label, rad,
                      text_ang_deg, r_text):
    """Draw an annotated curved rotation-direction arrow around ``(cx, cy)``."""
    a1, a2 = np.deg2rad(ang1_deg), np.deg2rad(ang2_deg)
    arr = FancyArrowPatch(
        (cx + r * np.cos(a1), cy + r * np.sin(a1)),
        (cx + r * np.cos(a2), cy + r * np.sin(a2)),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6.5,head_width=4.0",
        color=color,
        lw=2.3,
        zorder=7,
    )
    ax.add_patch(arr)
    at = np.deg2rad(text_ang_deg)
    ax.text(
        cx + r_text * np.cos(at),
        cy + r_text * np.sin(at),
        label,
        color=color,
        fontsize=9.2,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
        zorder=8,
    )


def animate_lab_vs_plasma_frame(
    xs,
    ys,
    Bx,
    By,
    Jz,
    *,
    f_lab_khz,
    f_exb_khz,
    center=(0.0, 0.0),
    calibrated=True,
    seamless_loop=True,
    n_frames=60,
    fps=15,
    dpi=100,
    r_orbit_cm=3.2,
    fine_n=129,
    title_prefix="",
    lab_title="Lab Frame\n(Mode lags behind CW $E\\times B$ flow)",
    plasma_title_template=(
        "Plasma Rest Frame (co-rotating with $E\\times B$)\n"
        "(Mode moves CCW at {f_pl_abs:.1f} kHz — with $\\nabla T_e$)"
    ),
    lab_arrows=None,
    plasma_arrows=None,
    out_gif=None,
    preview_png=None,
):
    """Render a side-by-side lab-frame vs. co-rotating plasma-rest-frame GIF.

    Animates a measured 2-D complex harmonic phasor ``(Bx, By, Jz)`` evolving
    as ``Re[F(x, y) * exp(-i * 2*pi*f_lab*t)]`` in the lab frame (left) and
    transformed into a frame co-rotating CW at ``f_exb_khz`` about ``center``
    (right).

    Args:
        xs, ys: 1-D uniform coordinate arrays [cm].
        Bx, By, Jz: 2-D complex phasor arrays of shape ``(len(ys), len(xs))``.
        f_lab_khz: Lab-frame oscillation frequency [kHz].
        f_exb_khz: Co-rotating CW ``E x B`` rotation frequency [kHz].
        center: ``(cx, cy)`` rotation center [cm].
        calibrated: True if ``Jz`` is in A/m^2 and ``Bx, By`` in Tesla; False
            for relative units.
        seamless_loop: If True and ``f_exb_khz > f_lab_khz``, snap ``f_exb``
            to the nearest half-integer harmonic of ``f_lab`` for a seamless loop.
        n_frames: Number of frames in the GIF.
        fps: Playback frame rate [frames/s].
        dpi: Figure DPI for rendered frames.
        r_orbit_cm: Radius of the dashed fluid-marker orbit circle [cm].
        fine_n: Upsampled scalar grid resolution per axis for smooth ``imshow``.
        title_prefix: Leading suptitle text before ``"  —  t = X.XX ms"``.
        lab_title: Title for the left (lab-frame) panel.
        plasma_title_template: Title template for the right panel (formatted
            with ``f_pl_khz``, ``f_pl_abs``, ``f_lab_khz``, ``f_exb_khz``).
        lab_arrows: Optional sequence of curved-arrow spec tuples
            ``(r, ang1_deg, ang2_deg, color, label, rad, text_ang_deg, r_text)``.
            Pass ``None`` for default mode arrow; pass ``()`` to omit.
        plasma_arrows: Same as ``lab_arrows`` for the right panel (``None``
            draws default ``grad(Te)`` CCW and ``grad(ne)`` CW drift arrows).
        out_gif: Output ``.gif`` path (if None/empty, skips writing GIF).
        preview_png: Optional path to write a 3-frame vertical preview strip PNG.

    Returns:
        str | None: ``out_gif`` if written, else ``None``.
    """
    xs_a = np.asarray(xs, dtype=float)
    ys_a = np.asarray(ys, dtype=float)
    cx, cy = float(center[0]), float(center[1])
    f_lab = float(f_lab_khz)
    f_exb_meas = float(f_exb_khz)

    if seamless_loop and f_exb_meas > f_lab > 0.0:
        half_mult = max(int(round(2.0 * f_exb_meas / f_lab)), 3)
        f_exb_eff = 0.5 * half_mult * f_lab
        t_loop_ms = 2.0 / f_lab
    else:
        f_exb_eff = f_exb_meas
        t_loop_ms = 1.0 / max(abs(f_exb_eff - f_lab), 0.1)
    f_pl_khz = f_exb_eff - f_lab

    prep = prepare_bdot_2d_fields(Bx, By, Jz, calibrated=calibrated, phase_align=True)
    Jz_n = prep["Jz_plot"]
    j_vlim = float(prep["j_vlim"])
    cb_label = prep["cb_label"]
    b_scale = float(prep["b_scale"])
    if b_scale > 0.0:
        Bx_n = prep["Bx_plot"] / b_scale
        By_n = prep["By_plot"] / b_scale
    else:
        Bx_n = np.zeros_like(prep["Bx_plot"])
        By_n = np.zeros_like(prep["By_plot"])

    x0, dx = float(xs_a[0]), float(xs_a[1] - xs_a[0])
    y0, dy = float(ys_a[0]), float(ys_a[1] - ys_a[0])
    xf = np.linspace(float(xs_a[0]), float(xs_a[-1]), int(fine_n))
    yf = np.linspace(float(ys_a[0]), float(ys_a[-1]), int(fine_n))
    XF, YF = np.meshgrid(xf, yf)
    XQ, YQ = np.meshgrid(xs_a, ys_a)

    def _sample_complex(arr, x_pts, y_pts):
        iy = (y_pts - y0) / dy
        ix = (x_pts - x0) / dx
        re = map_coordinates(np.real(arr), [iy, ix], order=1, mode="constant", cval=0.0)
        im = map_coordinates(np.imag(arr), [iy, ix], order=1, mode="constant", cval=0.0)
        return re + 1j * im

    Jz_lab_f = _sample_complex(Jz_n, XF, YF)

    def _eval_frame(t_ms):
        wt = 2.0 * np.pi * f_lab * t_ms
        alpha = 2.0 * np.pi * f_exb_eff * t_ms
        phase = np.exp(-1j * wt)

        jz_l = np.real(Jz_lab_f * phase)
        bx_l = np.real(Bx_n * phase)
        by_l = np.real(By_n * phase)

        ca, sa = np.cos(alpha), np.sin(alpha)
        dx_f, dy_f = XF - cx, YF - cy
        xl_f = cx + dx_f * ca + dy_f * sa
        yl_f = cy - dx_f * sa + dy_f * ca
        jz_p = np.real(_sample_complex(Jz_n, xl_f, yl_f) * phase)

        dx_q, dy_q = XQ - cx, YQ - cy
        xl_q = cx + dx_q * ca + dy_q * sa
        yl_q = cy - dx_q * sa + dy_q * ca
        bx_lq = np.real(_sample_complex(Bx_n, xl_q, yl_q) * phase)
        by_lq = np.real(_sample_complex(By_n, xl_q, yl_q) * phase)
        bx_p = bx_lq * ca - by_lq * sa
        by_p = bx_lq * sa + by_lq * ca
        return jz_l, bx_l, by_l, jz_p, bx_p, by_p, np.pi / 2.0 - alpha

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 5.5), dpi=int(dpi))
    fig.subplots_adjust(left=0.06, right=0.89, top=0.85, bottom=0.11, wspace=0.16)

    jz_l0, bx_l0, by_l0, jz_p0, bx_p0, by_p0, th0 = _eval_frame(0.0)
    ext = [float(xs_a[0]), float(xs_a[-1]), float(ys_a[0]), float(ys_a[-1])]
    im_l = ax_l.imshow(jz_l0, extent=ext, origin="lower", cmap="RdBu_r", vmin=-j_vlim, vmax=j_vlim)
    qv_l = ax_l.quiver(XQ, YQ, bx_l0, by_l0, color="k", scale=14.0, width=0.0045, alpha=0.78)
    im_r = ax_r.imshow(jz_p0, extent=ext, origin="lower", cmap="RdBu_r", vmin=-j_vlim, vmax=j_vlim)
    qv_r = ax_r.quiver(XQ, YQ, bx_p0, by_p0, color="k", scale=14.0, width=0.0045, alpha=0.78)

    for ax in (ax_l, ax_r):
        ax.add_patch(Circle((cx, cy), float(r_orbit_cm), fill=False, ec="#15803d",
                            ls="--", lw=1.1, alpha=0.65, zorder=4))
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel(r"$x$ [cm]")
    ax_l.set_ylabel(r"$y$ [cm]")
    ax_r.set_yticklabels([])

    (dot_l,) = ax_l.plot(
        cx + r_orbit_cm * np.cos(th0),
        cy + r_orbit_cm * np.sin(th0),
        "o", ms=9.5, color="#22c55e", mec="k", mew=1.2, zorder=9,
        label=rf"$E\times B$ plasma ({f_exb_meas:.1f} kHz CW)",
    )
    ax_r.plot(
        cx, cy + r_orbit_cm,
        "o", ms=9.5, color="#22c55e", mec="k", mew=1.2, zorder=9,
        label=r"$E\times B$ plasma (fixed)",
    )

    if lab_arrows is None:
        lab_arrows = ((5.3, 165, 110, "0.15", f"Mode ({f_lab:.1f} kHz CW)", -0.28, 140, 6.9),)
    if plasma_arrows is None:
        plasma_arrows = (
            (5.3, 110, 165, "#dc2626", r"$\nabla T_e$ drift (CCW)", 0.28, 142, 6.9),
            (5.3, 250, 195, "#1d4ed8", r"$\nabla n_e$ drift (CW)", -0.28, 222, 6.9),
        )
    for spec in lab_arrows:
        _add_curved_arrow(ax_l, cx, cy, *spec)
    for spec in plasma_arrows:
        _add_curved_arrow(ax_r, cx, cy, *spec)

    ax_l.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.6)
    ax_r.legend(loc="lower left", frameon=True, framealpha=0.88, fontsize=8.6)
    ax_l.set_title(lab_title, fontweight="bold")
    ax_r.set_title(
        plasma_title_template.format(
            f_pl_khz=f_pl_khz, f_pl_abs=abs(f_pl_khz), f_lab_khz=f_lab, f_exb_khz=f_exb_meas,
        ),
        fontweight="bold",
    )

    cax = fig.add_axes([0.905, 0.14, 0.018, 0.68])
    cb = fig.colorbar(im_r, cax=cax)
    cb.set_label(cb_label)

    def _suptitle_text(t_ms):
        return f"{title_prefix}  —  t = {t_ms:.2f} ms" if title_prefix else f"t = {t_ms:.2f} ms"

    time_txt = fig.suptitle(_suptitle_text(0.0), fontsize=12.0, fontweight="bold", y=0.975)

    n_fr = max(int(n_frames), 1)
    frames = []
    preview_idx = {0, n_fr // 4, n_fr // 2}
    preview_imgs = []
    for k in range(n_fr):
        t_ms = t_loop_ms * (k / n_fr)
        jz_l, bx_l, by_l, jz_p, bx_p, by_p, th_fl = _eval_frame(t_ms)
        im_l.set_data(jz_l)
        qv_l.set_UVC(bx_l, by_l)
        im_r.set_data(jz_p)
        qv_r.set_UVC(bx_p, by_p)
        dot_l.set_data([cx + r_orbit_cm * np.cos(th_fl)], [cy + r_orbit_cm * np.sin(th_fl)])
        time_txt.set_text(_suptitle_text(t_ms))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        img = Image.fromarray(rgba[:, :, :3])
        frames.append(img.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        if preview_png and k in preview_idx:
            preview_imgs.append(img.copy())
    plt.close(fig)

    if out_gif:
        os.makedirs(os.path.dirname(os.path.abspath(out_gif)), exist_ok=True)
        frames[0].save(
            out_gif,
            save_all=True,
            append_images=frames[1:],
            duration=max(int(round(1000.0 / float(fps))), 20),
            loop=0,
            optimize=True,
        )
        print(f"Animation saved to: {out_gif} ({n_fr} frames)")

    if preview_png and preview_imgs:
        os.makedirs(os.path.dirname(os.path.abspath(preview_png)), exist_ok=True)
        w, h = preview_imgs[0].size
        strip = Image.new("RGB", (w, h * len(preview_imgs)))
        for i, im in enumerate(preview_imgs):
            strip.paste(im, (0, i * h))
        strip.save(preview_png)

    return out_gif or None
