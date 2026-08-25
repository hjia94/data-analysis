"""Renders for a calibrated in-plane flow map: slider page and static frame.

Both renders consume the ``(npos, nbin)`` velocity cubes a Mach-flow batch
writes, resolve them about a fitted rotation centre
(:mod:`data_analysis.plasma.flow`), and draw the same panels in the same order
-- so a figure and the slider page it was chosen from cannot show different
things. That ordering guarantee is the reason these live together rather than
being written per campaign.

What varies between campaigns is passed in, never inferred here:

``axis_label``/``t_axis``
    The time frame a campaign quotes. Jun-2026 run 32 triggers at bias start and
    plots ``t`` directly; Mar-2026 run 054 triggers 4.5 ms into the discharge and
    plots machine time. Neither is converted here -- the caller hands over the
    axis it wants labelled and the label that names it.
``extra_fields``/``extra_panels``
    Campaign-specific panels appended after the three every in-plane map has
    (speed+quiver, v_theta, v_r). Run 32 adds axial ``v_Z``; run 054 adds the
    summed tip current, because its plasma decays inside the record.
``warning``/``params``/``details``
    Provenance and caveat text. Assembled by the caller from its own npz, since
    what is worth warning about is a property of the run, not of the renderer.

Colour-scale convention, shared by both renders: ``vmax=None`` autoscales per
frame -- right while hunting for structure, wrong when comparing frames. Speed is
unsigned and runs sequentially from 0; the signed components diverge about 0,
because for them the sign is the physics -- one sign over the ring is a coherent
rotation, and v_r is the control (an E x B rotation has little radial flow).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from data_analysis.viz.plot_utils import grid_by_position, grid_frames
from data_analysis.viz.slider_html import SCHEMA_VERSION, write_slider_html

#: Panel titles and colourbar labels, in the order both renders draw them. One
#: table, so the slider page and the static figure cannot drift apart.
POLAR_PANELS = (
    ("azimuthal flow (v_theta, +ve CCW)",
     r"azimuthal  ($v_\theta$, +ve CCW)", r"$v_\theta$ [km/s]"),
    ("radial flow (v_r, +ve outward)",
     r"radial  ($v_r$, +ve outward)", r"$v_r$ [km/s]"),
)


def flow_slider_bundle(pos_x, pos_y, vx, vy, v_r, v_th, t_axis, axis_label,
                       title, source, params, details=None, warning=None,
                       quiver_step=1, vmax=None, extra_fields=()):
    """Assemble the slider bundle for an in-plane flow map -> ``dict``.

    ``vx``/``vy``/``v_r``/``v_th`` are ``(npos, nbin)`` [km/s]; ``t_axis`` is the
    slider axis in whatever frame ``axis_label`` names.

    **Panels, not dropdown groups**: they share one time slider, so every
    component is read at the same instant rather than by switching.

    ``extra_fields`` are appended after the three standard panels; each is a
    complete field dict (the caller has already gridded its frames).
    """
    fx, xs, ys = grid_frames(pos_x, pos_y, vx)
    grid = lambda a: grid_frames(pos_x, pos_y, a)[0]
    fy, fs = grid(vy), grid(np.hypot(vx, vy))
    f_th, f_r = grid(v_th), grid(v_r)

    # vmin/vmax are set together (schema rule) -- see the module docstring for
    # why speed is sequential and the signed components diverge.
    fixed = vmax is not None
    fields = [
        {"name": "in-plane flow (v_X, v_Y)", "unit": "km/s", "frames": fs,
         "cmap": "viridis", "vmin": 0.0 if fixed else None, "vmax": vmax,
         "vectors": {"u": fx, "v": fy, "step": quiver_step}},
    ]
    for (name, _title, _label), frames in zip(POLAR_PANELS, (f_th, f_r)):
        fields.append({"name": name, "unit": "km/s", "frames": frames,
                       "cmap": "RdBu_r",
                       "vmin": -vmax if fixed else None, "vmax": vmax})
    fields.extend(extra_fields)

    bundle = {
        "schema": SCHEMA_VERSION,
        "title": title,
        "geometry": "plane",
        "axis": {"name": axis_label, "unit": "ms", "values": t_axis},
        "x": {"label": "X position", "unit": "cm", "values": xs},
        "y": {"label": "Y position", "unit": "cm", "values": ys},
        "fields": fields,
        "provenance": {"source": source, "params": params,
                       "details": details or {}},
    }
    if warning:
        bundle["warning"] = warning
    return bundle


def write_flow_slider(out, **kwargs):
    """:func:`flow_slider_bundle` written to ``out`` -> the written path."""
    return write_slider_html(flow_slider_bundle(**kwargs), out)


def plot_flow_frame(pos_x, pos_y, vx, vy, v_r, v_th, centre, suptitle,
                    quiver_step=1, vmax=None, extra_panels=(),
                    suptitle_color=None):
    """Static figure of one frame: in-plane quiver, v_theta, v_r, then extras.

    Every argument is one value per position (a single time bin, already
    selected). ``extra_panels`` are ``(values, title, colourbar_label)`` drawn
    after the two polar panels on the same diverging scale -- run 32's axial
    ``v_Z``. ``centre`` is marked on the panels resolved about it.

    Returns the figure; the caller saves it (campaign figure paths and naming
    are the caller's business).
    """
    # grid_by_position takes one value per position and returns the imshow
    # extent as cell EDGES; building it from the axis vectors instead would put
    # the limits at cell centres, shrinking the map by half a cell each side.
    grid = lambda v: grid_by_position(pos_x, pos_y, v)
    g_vx, extent = grid(vx)
    g_vy, _ = grid(vy)
    g_th, _ = grid(v_th)
    g_r, _ = grid(v_r)
    extras = [(grid(v)[0], title, label) for v, title, label in extra_panels]
    speed = np.hypot(g_vx, g_vy)
    peak = np.nanmax(speed)

    # 5 in per panel plus 2 in of shared margin: 3 panels -> 17, 4 -> 22, the
    # sizes the per-campaign figures this replaces were tuned to.
    npanel = 3 + len(extras)
    fig, axs = plt.subplots(1, npanel, figsize=(2 + 5 * npanel, 5.4),
                            sharey=True)
    ax_p = axs[0]

    im = ax_p.imshow(speed, origin="lower", extent=extent, cmap="viridis",
                     vmin=None if vmax is None else 0.0, vmax=vmax,
                     interpolation="nearest")
    s = quiver_step
    xs = np.linspace(extent[0], extent[1], g_vx.shape[1], endpoint=False)
    ys = np.linspace(extent[2], extent[3], g_vx.shape[0], endpoint=False)
    # Cell centres: extent gives edges, and an arrow belongs on the position it
    # was measured at, not on the corner of its cell.
    xs += (xs[1] - xs[0]) / 2 if xs.size > 1 else 0
    ys += (ys[1] - ys[0]) / 2 if ys.size > 1 else 0
    X, Y = np.meshgrid(xs[::s], ys[::s])
    q = ax_p.quiver(X, Y, g_vx[::s, ::s], g_vy[::s, ::s], color="w",
                    pivot="mid", scale_units="xy")
    ax_p.quiverkey(q, 0.88, 1.03, peak or 1.0, f"{peak:.1f} km/s",
                   labelpos="E", color="k")
    ax_p.set_title("in-plane flow  (v_X, v_Y)")
    ax_p.set_ylabel("Y [cm]")
    fig.colorbar(im, ax=ax_p, label="|v| in-plane [km/s]")

    polar = [(g, t, lab) for (_n, t, lab), g in zip(POLAR_PANELS, (g_th, g_r))]
    for ax, (g, title, label) in zip(axs[1:], polar + extras):
        lim = vmax or np.nanmax(np.abs(g))
        img = ax.imshow(g, origin="lower", extent=extent, cmap="RdBu_r",
                        vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(title)
        fig.colorbar(img, ax=ax, label=label)

    # The fitted centre, marked on the two panels resolved about it. Machine
    # coordinates throughout -- the marker moves, the axes do not.
    for ax in axs[1:3]:
        ax.plot(centre[0], centre[1], "k+", ms=11, mew=1.6)
    for ax in axs:
        ax.set_xlabel("X [cm]")

    fig.suptitle(suptitle, fontsize=9,
                 **({"color": suptitle_color} if suptitle_color else {}))
    return fig
