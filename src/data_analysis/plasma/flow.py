"""In-plane flow-field geometry: polar resolution and rotation-centre fitting.

Operates on any in-plane vector field sampled at scan positions -- one ``(vx,
vy)`` per position, in whatever units the caller brought. Nothing here knows
about currents, faces, area ratios or Mach numbers; a drift field from a
swept-Langmuir potential map resolves the same way a Mach flow map does.

Both functions take ``centre`` (or fit it) in the *same* coordinates as
``pos_x``/``pos_y``. Nothing is translated: a feature at x = 5 cm stays at
x = 5 cm on every plot, and the centre is used only by the projection.
"""

from __future__ import annotations

import numpy as np


def polar_components(vx, vy, pos_x, pos_y, centre):
    """In-plane flow resolved about ``centre`` -> ``(v_r, v_theta)``, same shape.

    ``v_r`` is positive outward; ``v_theta`` is positive counter-clockwise (the
    +z sense, with z out of the x-y plane along B). An azimuthal E x B flow
    reverses sign across the column in Cartesian components and averages to
    nearly zero; ``v_theta`` states it as one number per cell.

    Accepts ``(npos,)`` or ``(npos, nbin)``.
    """
    dx = np.asarray(pos_x, float) - centre[0]
    dy = np.asarray(pos_y, float) - centre[1]
    r = np.hypot(dx, dy)
    # r == 0 has no defined direction; one cell at most, left NaN rather than
    # given an arbitrary unit vector.
    with np.errstate(invalid="ignore", divide="ignore"):
        ur, ut = np.where(r > 0, dx / r, np.nan), np.where(r > 0, dy / r, np.nan)
    # Broadcast the per-position unit vectors against (npos, nbin) cubes.
    if np.ndim(vx) == 2:
        ur, ut = ur[:, None], ut[:, None]
    return vx * ur + vy * ut, -vx * ut + vy * ur


def find_flow_centre(vx, vy, pos_x, pos_y, search_cm=3.0, step_cm=0.25,
                     r_min=3.0, r_max=15.0, min_cells=50):
    """The rotation centre, fitted from the flow field -> ``(cx, cy, ratio)`` cm.

    ``vx``/``vy`` are one in-plane velocity per position (already reduced over
    whatever time window the caller cares about). A rigid rotation has one
    stagnation point, and about the true centre the flow is purely azimuthal, so
    this scans candidate centres and takes the one minimising
    ``mean|v_r| / mean|v_theta|`` over the annulus ``r_min..r_max``. The returned
    ``ratio`` is that minimum -- small means a well-centred rotation, ~1 means
    there was no rotation to centre, so it is the fit's own quality flag.

    Fitted rather than assumed from a nominal plate position: v_r/v_theta are
    sensitive to the centre in a way the Cartesian components are not.

    Both annulus bounds move the answer (measured, Jun-2026 run 32): below
    ``r_min`` the near-stagnant core, where direction is ill-defined, dominates
    the ratio; at the plane edge low-signal cells win by having no flow to be
    radial. Non-finite cells -- a position the digitizer never wrote -- are
    excluded, and a candidate with fewer than ``min_cells`` valid cells is
    skipped rather than being allowed to win on a handful of points.

    Evaluated as one ``(ncand, npos)`` broadcast rather than a candidate loop.
    Ties go to the first candidate in row-major ``(cx, cy)`` order.
    """
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)
    px = np.asarray(pos_x, float)
    py = np.asarray(pos_y, float)
    finite = np.isfinite(vx) & np.isfinite(vy)
    grid = np.arange(-search_cm, search_cm + step_cm / 2, step_cm)
    cx, cy = (a.ravel() for a in np.meshgrid(grid, grid, indexing="ij"))

    # (ncand, npos): every candidate against every position at once.
    dx = px[None, :] - cx[:, None]
    dy = py[None, :] - cy[:, None]
    r = np.hypot(dx, dy)
    keep = (r > r_min) & (r < r_max) & finite[None, :]
    n_keep = keep.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        ur, ut = dx / r, dy / r
        v_r = np.abs(vx[None, :] * ur + vy[None, :] * ut)
        v_t = np.abs(-vx[None, :] * ut + vy[None, :] * ur)
        # Zero the excluded cells and divide by the kept count: the mean over
        # the annulus, without materializing a ragged per-candidate selection.
        mean_r = np.where(keep, v_r, 0.0).sum(axis=1) / n_keep
        mean_t = np.where(keep, v_t, 0.0).sum(axis=1) / n_keep
        ratio = mean_r / (mean_t + 1e-12)

    # A candidate below min_cells is not merely bad, it is not a candidate:
    # +inf keeps it out of the argmin without special-casing the winner.
    ratio = np.where(n_keep >= min_cells, ratio, np.inf)
    best = int(np.argmin(ratio))
    if not np.isfinite(ratio[best]):
        raise ValueError(
            f"no candidate centre had {min_cells} valid cells in the annulus "
            f"{r_min}-{r_max} cm; {int(finite.sum())} of {finite.size} positions "
            f"are finite.")
    return float(cx[best]), float(cy[best]), float(ratio[best])
