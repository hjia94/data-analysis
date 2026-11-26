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

from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


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


class FlowShear(NamedTuple):
    """1-D cylindrical flow-shear diagnostics [rad/s or s^-1].

    ``dv_dr_s`` is the signed outward radial velocity gradient ``dv_theta/dr``
    [s^-1], ``omega_rad_s`` is ``v_theta / r`` [rad/s], and ``gamma_E_s`` is
    the cylindrical rotational shearing rate ``r * d(v_theta / r) / dr =
    dv_theta/dr - v_theta/r`` [s^-1] (identically zero for rigid-body rotation).
    """
    dv_dr_s: np.ndarray
    omega_rad_s: np.ndarray
    gamma_E_s: np.ndarray


def antisymmetric_rotation_profile(x_cm, v_perp, x_center_cm=0.0, axis=0):
    """Split a 1-D transverse velocity scan into antisymmetric & symmetric parts.

    For a line scan along ``x`` through a rotating column centered at
    ``x_center_cm``, any uniform translation or uniform calibration offset
    shifts ``v_perp(x_c + r)`` and ``v_perp(x_c - r)`` equally while rotation
    flips sign. Interpolating both sides onto positive radii ``r > 0`` gives::

        v_rot(r) = 0.5 * (v_perp(x_c + r) - v_perp(x_c - r))
        v_sym(r) = 0.5 * (v_perp(x_c + r) + v_perp(x_c - r))

    Returns ``(r_cm, v_rot, v_sym)`` where ``r_cm`` is 1-D positive radii [cm]
    covered on both sides of ``x_center_cm``, and ``v_rot`` / ``v_sym`` have
    the same non-``axis`` shape as ``v_perp``.
    """
    x = np.asarray(x_cm, dtype=float)
    v = np.asarray(v_perp, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x_cm must be a 1-D array with at least 3 points")
    if v.shape[axis] != x.size:
        raise ValueError(f"x_cm length ({x.size}) does not match v_perp.shape[{axis}]")

    order = np.argsort(x)
    xs = x[order]
    vm = np.moveaxis(v, axis, 0)[order]  # (nx, ...)

    xc = float(x_center_cm)
    r_max = min(xc - float(xs[0]), float(xs[-1]) - xc)
    if r_max <= 0.0:
        raise ValueError(f"x_center_cm={xc} is outside the scan span [{xs[0]}, {xs[-1]}]")

    # Use positive radii from the right-hand scan points within [0, r_max] so
    # exact grid points are sampled without unnecessary interpolation drift.
    dr_right = xs[xs > xc] - xc
    r_cm = dr_right[dr_right <= r_max + 1e-9]
    if r_cm.size == 0:
        raise ValueError("no symmetric radii available around x_center_cm")

    flat = vm.reshape(xs.size, -1)
    v_rot = np.full((r_cm.size, flat.shape[1]), np.nan, dtype=float)
    v_sym = np.full((r_cm.size, flat.shape[1]), np.nan, dtype=float)
    xp = xc + r_cm
    xm = xc - r_cm

    for j in range(flat.shape[1]):
        col = flat[:, j]
        fin = np.isfinite(col)
        if np.sum(fin) < 2:
            continue
        # Interpolate only inside the finite span on each side so NaN gaps
        # outside the measured range stay NaN.
        xf, cf = xs[fin], col[fin]
        vp = np.interp(xp, xf, cf, left=np.nan, right=np.nan)
        vn = np.interp(xm, xf, cf, left=np.nan, right=np.nan)
        v_rot[:, j] = 0.5 * (vp - vn)
        v_sym[:, j] = 0.5 * (vp + vn)

    out_shape = (r_cm.size,) + vm.shape[1:]
    return (
        r_cm,
        np.moveaxis(v_rot.reshape(out_shape), 0, axis),
        np.moveaxis(v_sym.reshape(out_shape), 0, axis),
    )


def radial_flow_shear(r_cm, v_theta_km_s, *, smooth_window=1, polyorder=2,
                      axis=0, r_min_cm=0.5):
    """Compute velocity gradient ``dv_theta/dr``, ``Omega``, and shear ``gamma_E``.

    Expects a non-negative radial coordinate ``r_cm >= 0`` and azimuthal
    velocity ``v_theta_km_s`` [km/s] (e.g. from :func:`antisymmetric_rotation_profile`
    for a 1-D chord or azimuthally averaged :func:`polar_components` for a 2-D
    plane). Passing negative ``r_cm`` raises :class:`ValueError` so raw signed
    full-chord ``(x_cm, v_y)`` scans cannot be passed by mistake.

    Unit conversion: ``(km/s) / cm = (1e3 m/s) / (1e-2 m) = 1e5 s^-1``.

    Returns :class:`FlowShear` ``(dv_dr_s, omega_rad_s, gamma_E_s)``; points
    with ``r_cm < r_min_cm`` are masked to ``NaN`` in ``omega_rad_s`` and
    ``gamma_E_s`` to avoid ``1/r`` amplification near the rotation axis.
    """
    coord = np.asarray(r_cm, dtype=float)
    v = np.asarray(v_theta_km_s, dtype=float)
    if coord.ndim != 1 or coord.size < 3:
        raise ValueError("r_cm must be a 1-D array with at least 3 points")
    if np.any(coord < 0.0):
        raise ValueError(
            "r_cm must be non-negative radial coordinates; convert a full chord "
            "(x_cm, v_y) via antisymmetric_rotation_profile() first."
        )
    if v.shape[axis] != coord.size:
        raise ValueError(f"r_cm length ({coord.size}) does not match v_theta.shape[{axis}]")

    vm = np.moveaxis(v, axis, 0)
    nx = vm.shape[0]
    if smooth_window <= 1:
        dv_dc = np.gradient(vm, coord, axis=0)
    else:
        if smooth_window % 2 == 0 or smooth_window < 3:
            raise ValueError(f"smooth_window must be <= 1 or odd >= 3, got {smooth_window}")
        if smooth_window > nx:
            raise ValueError(f"smooth_window ({smooth_window}) exceeds grid length ({nx})")
        dx_all = np.diff(coord)
        if not np.allclose(dx_all, dx_all[0], rtol=1e-3, atol=1e-9):
            raise ValueError("Savitzky-Golay differentiation requires a uniform grid")
        dx = float(dx_all[0])
        dv_dc = np.full_like(vm, np.nan, dtype=float)
        flat_v = vm.reshape(nx, -1)
        flat_d = dv_dc.reshape(nx, -1)
        for j in range(flat_v.shape[1]):
            col = flat_v[:, j]
            if np.all(np.isfinite(col)):
                flat_d[:, j] = savgol_filter(col, smooth_window, polyorder, deriv=1, delta=dx)
            else:
                fin = np.flatnonzero(np.isfinite(col))
                if fin.size >= smooth_window:
                    i0, i1 = int(fin[0]), int(fin[-1])
                    if i1 - i0 + 1 >= smooth_window:
                        sub = np.interp(np.arange(i0, i1 + 1), fin, col[fin])
                        sub_d = savgol_filter(sub, smooth_window, polyorder, deriv=1, delta=dx)
                        out = np.full(nx, np.nan)
                        out[i0:i1 + 1] = sub_d
                        out[~np.isfinite(col)] = np.nan
                        flat_d[:, j] = out

    # Unit conversion: (km/s) / cm = 1e5 s^-1.
    KM_S_PER_CM_TO_INV_S = 1e5
    bshape = (nx,) + (1,) * (vm.ndim - 1)
    dv_dr_s = dv_dc * KM_S_PER_CM_TO_INV_S
    r_b = coord.reshape(bshape)

    r_ok = r_b >= float(r_min_cm)
    r_safe = np.where(r_ok, r_b, 1.0)
    omega = np.where(r_ok & np.isfinite(vm), (vm / r_safe) * KM_S_PER_CM_TO_INV_S, np.nan)
    gamma_e = np.where(r_ok & np.isfinite(dv_dr_s) & np.isfinite(omega), dv_dr_s - omega, np.nan)

    return FlowShear(
        dv_dr_s=np.moveaxis(dv_dr_s, 0, axis),
        omega_rad_s=np.moveaxis(omega, 0, axis),
        gamma_E_s=np.moveaxis(gamma_e, 0, axis),
    )


def fit_exponential_relaxation(t, y, *, t_start: float | None = None,
                               t_end: float | None = None,
                               tau_bounds: tuple[float, float] = (1e-3, 1e2),
                               offset_init: float | None = None,
                               min_points: int = 4) -> dict:
    """Fit ``y(t) = amp * exp(-(t - t0) / tau) + offset`` over ``[t_start, t_end]``.

    Diagnostic-agnostic exponential relaxation fitter for flow spin-down or
    profile-decay traces. Time ``t`` and ``tau_bounds`` share the caller's
    time units (e.g., ms -> ``rate = 1/tau`` in ``ms^-1``). Returns a dict with
    ``amp``, ``tau``, ``rate``, ``offset``, ``r2``, ``n_points``, ``t0``, and
    ``t_end`` (all NaN if fewer than ``min_points`` finite samples exist or if
    the nonlinear fit fails).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError("t and y must be 1-D arrays of equal length")
    lo = float(t[0]) if t_start is None else float(t_start)
    hi = float(t[-1]) if t_end is None else float(t_end)
    mask = (t >= lo) & (t <= hi) & np.isfinite(t) & np.isfinite(y)
    n = int(np.sum(mask))
    if n < min_points:
        return {
            "amp": np.nan, "tau": np.nan, "rate": np.nan, "offset": np.nan,
            "r2": np.nan, "n_points": n, "t0": lo, "t_end": hi,
        }
    tw = t[mask]
    yw = y[mask]
    t0 = float(tw[0])
    dt_span = max(float(tw[-1] - tw[0]), 1e-12)
    tau_min, tau_max = float(tau_bounds[0]), float(tau_bounds[1])
    c0 = float(yw[-1]) if offset_init is None else float(offset_init)
    a0 = float(yw[0] - c0)
    tau0 = float(np.clip(0.3 * dt_span, tau_min, tau_max))
    y_span = max(float(np.ptp(yw)), abs(a0), 1e-6)
    try:
        popt, _ = curve_fit(
            lambda tt, a, tau, c: a * np.exp(-(tt - t0) / tau) + c,
            tw, yw, p0=[a0, tau0, c0],
            bounds=(
                [-10.0 * y_span, tau_min, float(np.min(yw)) - 5.0 * y_span],
                [+10.0 * y_span, tau_max, float(np.max(yw)) + 5.0 * y_span],
            ),
            maxfev=8000,
        )
        amp, tau, offset = float(popt[0]), float(popt[1]), float(popt[2])
        pred = amp * np.exp(-(tw - t0) / tau) + offset
        ss_res = float(np.sum((yw - pred) ** 2))
        ss_tot = float(np.sum((yw - float(np.mean(yw))) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan
        return {
            "amp": amp, "tau": tau, "rate": 1.0 / tau, "offset": offset,
            "r2": float(r2), "n_points": n, "t0": t0, "t_end": float(tw[-1]),
        }
    except Exception:
        return {
            "amp": np.nan, "tau": np.nan, "rate": np.nan, "offset": np.nan,
            "r2": np.nan, "n_points": n, "t0": t0, "t_end": float(tw[-1]),
        }
