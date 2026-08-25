"""Mach-probe estimator maths -- campaign-agnostic.

A Mach probe reads ion saturation current on two opposed faces of one axis. In
the magnetised fluid model the faces see

    j_pm = j_sat * kappa_pm * exp(pm M / K)

so their ratio ``R = j_+ / j_-`` carries the flow *and* the unknown ratio of
collecting areas ``kappa = A_+/A_-``. Separating the two needs a second
measurement with the probe rotated 180 deg about its shaft.

**Every current here is hardware-labelled.** ``j_plus`` is a fixed physical tip
on a fixed DAQ channel -- *not* "the face pointing upstream". Under that
convention a 180 deg rotation inverts the flow exponent while leaving kappa
alone::

    R_a = kappa * exp(+2M/K)        R_b = kappa * exp(-2M/K)

so the **product** isolates the area ratio and the **ratio** isolates the flow.
Flow-referenced treatments (where ``J_+`` means the upstream face by definition)
have these the other way round. Feeding flow-labelled data in returns the flow
where kappa was asked for, silently and with a plausible magnitude.

Selecting which samples enter a calibration is campaign policy, so this module
takes no time axis, window, run numbers or orientation table. The reduction
helpers below (:func:`time_bin_edges`, :func:`binned_face_ratio`) are the
exception only in taking a time axis -- they are handed one, and still decide
nothing about which window is interesting.

Two functions here are **not** Mach-probe maths at all:
:func:`polar_components` and :func:`find_flow_centre` operate on any in-plane
vector field sampled at scan positions, and mention no current, face, ratio or
kappa. They live here because the Mach flow maps are the only present caller and
a module for two functions was not worth the churn. Anything else that resolves a
drift field about a centre -- an ExB velocity from a swept-Langmuir potential
map, say -- should call them rather than re-deriving the projection, and is the
signal to move them to their own module.
"""

from __future__ import annotations

import numpy as np

from data_analysis.plasma.formulas import ion_sound_speed

# Magnetised fluid model, Hutchinson, Phys. Fluids 30, 3777 (1987). The
# calibration constant relating ln(j_+/j_-) to the Mach number; unmagnetised and
# kinetic models give other values, so it is a parameter everywhere below.
K_HUTCHINSON = 0.45


def valid_current_mask(*currents):
    """``True`` where every current is finite and strictly positive.

    Isat is positive by construction, so a non-positive sample is missing data,
    not a small measurement. Callers **count** what this excludes: a silently
    shrinking sample looks identical to a clean one.
    """
    mask = np.ones(np.broadcast(*currents).shape, dtype=bool)
    for c in currents:
        mask &= np.isfinite(c) & (c > 0)
    return mask


def face_ratio(j_plus, j_minus, axis=1, mask=None):
    """Face ratio of two current stacks, reduced along ``axis``.

    ``j_plus``/``j_minus`` are ``(nshot, nt_win)`` currents [A], already windowed.

    **Averages first, then divides.** Dividing sample-by-sample divides by the
    instantaneous ``j_minus``, which is small and noisy, growing a heavy tail that
    biases any later average. This function owns the reduction so that cannot be
    written by accident.

    ``axis=1`` -> one ratio per shot (calibration input); ``axis=0`` -> one per
    sample (drift diagnostic). ``mask`` overrides which samples count (default:
    :func:`valid_current_mask`); pass a wider one -- e.g. all four tips of a
    pairing -- to keep two ratios sample-aligned. Nothing valid yields ``nan``.
    """
    j_plus = np.asarray(j_plus, dtype=float)
    j_minus = np.asarray(j_minus, dtype=float)
    ok = valid_current_mask(j_plus, j_minus) if mask is None else mask
    n = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        # n cancels in the ratio, so sum/sum is the mean-ratio without dividing.
        ip = np.where(ok, j_plus, 0.0).sum(axis=axis)
        im = np.where(ok, j_minus, 0.0).sum(axis=axis)
        return np.where(n > 0, ip / im, np.nan)


def area_ratio(R_a, R_b):
    """Area ratio from two opposing orientations -> ``kappa = sqrt(R_a * R_b)``.

    The **product** (hardware labels -- module docstring): the flow exponent
    cancels. Immune to a sign-convention error for the same reason, which makes it
    the cross-check on a fitted kappa.
    """
    return np.sqrt(np.asarray(R_a, dtype=float) * np.asarray(R_b, dtype=float))


def mach_number(R_a, R_b, K=K_HUTCHINSON):
    """Mach number from two opposing orientations -> ``M = (K/4) ln(R_a/R_b)``.

    The **ratio**: kappa cancels, leaving the flow along the lab axis this face
    pair faced in orientation ``a``. Swapping the arguments flips the sign.
    """
    return (K / 4.0) * np.log(np.asarray(R_a, dtype=float)
                              / np.asarray(R_b, dtype=float))


def mach_single(R, kappa, K=K_HUTCHINSON):
    """Mach number from one orientation given a known kappa -> ``(K/2) ln(R/kappa)``.

    Accuracy is limited by kappa's systematic error, not by shot noise.
    """
    return (K / 2.0) * np.log(np.asarray(R, dtype=float)
                              / np.asarray(kappa, dtype=float))


def flow_velocity(M, T_e_eV, mu, gamma=5 / 3, Z=1):
    """Mach number -> flow speed in **km/s**. ``mu`` is the ion mass in m_p (He = 4).

    A Mach probe does not measure ``T_e``, so every velocity carries that
    assumption (``v`` scales as ``sqrt(T_e)``). Callers state the T_e they used
    wherever the velocity is shown.
    """
    # ion_sound_speed returns cm/s; 1e-5 converts to km/s.
    return np.asarray(M, dtype=float) * ion_sound_speed(T_e_eV, mu, gamma, Z) * 1e-5


#: Fewest samples a time bin may hold. A bin is the shot-and-sample pool one
#: face ratio is formed from, so a handful of samples gives a ratio dominated by
#: whichever sample landed in it. Real runs sit far above this -- the floor
#: exists to fail loudly if a bin width is ever set near the sample interval.
MIN_BIN_SAMPLES = 10


def time_bin_edges(tarr, window_ms, bin_ms, min_samples=MIN_BIN_SAMPLES):
    """Sample-index edges of uniform time bins -> ``(edges, centres_ms)``.

    ``tarr`` is the record's time axis [s]; ``window_ms`` is ``(t0, t1)`` and
    ``bin_ms`` the bin width, both in ms. ``edges`` has ``nbin+1`` entries
    indexing ``tarr``, with bin ``k`` spanning ``[edges[k], edges[k+1])``.

    Bins are defined on the *time* axis and then located in samples, so a bin is
    ``bin_ms`` wide by construction even where the sample count per bin is not
    exactly constant. Raises ``ValueError`` naming the offending bin and the
    record's actual span if any bin holds fewer than ``min_samples`` -- the usual
    causes being a window reaching outside the record or a bin shorter than the
    sample interval, both of which otherwise yield plausible noise.
    """
    tarr = np.asarray(tarr, dtype=float)
    t0, t1 = window_ms
    nbin = int(round((t1 - t0) / bin_ms))
    if nbin < 1:
        raise ValueError(f"window {window_ms} ms is narrower than one {bin_ms} ms bin")
    t_edges = t0 + bin_ms * np.arange(nbin + 1)
    edges = np.searchsorted(tarr, t_edges * 1e-3)
    thin = np.flatnonzero(np.diff(edges) < min_samples)
    if thin.size:
        k = int(thin[0])
        raise ValueError(
            f"bin {k} ({t_edges[k]:g}-{t_edges[k + 1]:g} ms) holds "
            f"{int(edges[k + 1] - edges[k])} sample(s), below the {min_samples} "
            f"needed for a meaningful ratio; the record spans "
            f"{tarr[0] * 1e3:.2f}-{tarr[-1] * 1e3:.2f} ms at "
            f"{1e-6 / (tarr[1] - tarr[0]):.1f} MHz, so the window reaches outside "
            f"it or the bin is shorter than the sample interval.")
    return edges, t_edges[:-1] + bin_ms / 2


def binned_face_ratio(plus, minus, edges):
    """Per time bin from two ``(nshot, nt)`` stacks -> ``(ratio, counts, amplitude)``.

    Reduces over both shots and the bin's samples, keeping :func:`face_ratio`'s
    average-then-divide invariant: each face's pooled sum over the whole bin is
    formed before the division, never a mean of per-sample ratios.

    ``counts`` is the valid-sample count per bin, so an empty or thin bin is
    visible as a count rather than as a plausible NaN.

    ``amplitude`` is the mean of ``plus + minus`` over the same valid samples
    [A]. The ratio divides amplitude away, and that is exactly what makes a Mach
    number dangerous late in a shot: as both faces decay into noise, ``R`` stays
    finite and drifts smoothly, reading as a strengthening flow rather than as an
    empty machine. Returned from here rather than recomputed by callers so the
    two describe the *same* pooled sample -- a separate re-slice would silently
    average over samples this mask excluded.

    Vectorized with ``add.reduceat``, which is exact here (summation is
    associative, so pooling the shot axis before the bin segments gives the same
    sums). ``reduceat`` returns ``arr[i]`` instead of 0 for an empty segment;
    :func:`time_bin_edges` rules that out by rejecting any bin below
    ``min_samples``, so callers building edges by hand must do the same.
    """
    edges = np.asarray(edges)
    starts = edges[:-1]
    width = np.diff(edges)
    ok = valid_current_mask(plus, minus)
    with np.errstate(invalid="ignore", divide="ignore"):
        # sum/sum, not mean-of-ratios: n cancels, so no division per sample.
        sp = np.add.reduceat(np.where(ok, plus, 0.0).sum(axis=0), starts)
        sm = np.add.reduceat(np.where(ok, minus, 0.0).sum(axis=0), starts)
        counts = np.add.reduceat(ok.sum(axis=0), starts).astype(np.int32)
        ratio = np.where(counts > 0, sp / sm, np.nan)
        amplitude = (sp + sm) / (width * plus.shape[0])
    return ratio, counts, amplitude


def polar_components(vx, vy, pos_x, pos_y, centre):
    """In-plane flow resolved about ``centre`` -> ``(v_r, v_theta)``, same shape.

    ``v_r`` is positive outward; ``v_theta`` is positive counter-clockwise (the
    +z sense, with z out of the x-y plane along B). An azimuthal E x B flow
    reverses sign across the column in Cartesian components and averages to
    nearly zero; ``v_theta`` states it as one number per cell.

    ``centre`` is in the *same* coordinates as ``pos_x``/``pos_y`` and is used
    only by the projection -- nothing is translated, so a feature at x = 5 cm
    stays at x = 5 cm on every plot. Accepts ``(npos,)`` or ``(npos, nbin)``.
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
    """
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)
    px = np.asarray(pos_x, float)
    py = np.asarray(pos_y, float)
    finite = np.isfinite(vx) & np.isfinite(vy)
    grid = np.arange(-search_cm, search_cm + step_cm / 2, step_cm)
    best = None
    for cx in grid:
        for cy in grid:
            r = np.hypot(px - cx, py - cy)
            keep = (r > r_min) & (r < r_max) & finite
            if keep.sum() < min_cells:
                continue
            vr, vt = polar_components(vx[keep], vy[keep], px[keep], py[keep],
                                      (cx, cy))
            ratio = np.mean(np.abs(vr)) / (np.mean(np.abs(vt)) + 1e-12)
            if best is None or ratio < best[0]:
                best = (ratio, float(cx), float(cy))
    if best is None:
        raise ValueError(
            f"no candidate centre had {min_cells} valid cells in the annulus "
            f"{r_min}-{r_max} cm; {int(finite.sum())} of {finite.size} positions "
            f"are finite.")
    return best[1], best[2], best[0]


def combine_log(values):
    """Log-space average of ``values`` -> ``(log_mean, log_std, n_valid)``.

    Log space because kappa is multiplicative: ``ln R`` is the additive symmetric
    quantity, and the arithmetic mean of ratios sits above the geometric mean by a
    bias that grows with the spread. ``np.exp(log_mean)`` is the geometric mean.

    ``log_std`` is the **population** spread (``ddof=1``), not the standard error:
    callers divide by ``sqrt(n_valid)`` themselves, and some deliberately do not
    (a between-orientation spread is a systematic that more orientations do not
    shrink). Non-finite and non-positive entries are excluded and counted in
    ``n_valid``, so it always describes the sample the mean was taken over.
    """
    v = np.asarray(values, dtype=float).ravel()
    ok = np.isfinite(v) & (v > 0)
    n = int(ok.sum())
    if n == 0:
        return np.nan, np.nan, 0
    lg = np.log(v[ok])
    # ddof=1 needs >= 2 points; one point has a defined mean but no spread.
    return lg.mean(), lg.std(ddof=1) if n > 1 else np.nan, n


def fit_calibration(ln_R, sigma, design):
    """Weighted least squares for a joint kappa/flow calibration.

    Solves ``ln_R = design @ params`` with weights ``1/sigma``, for the linear
    model that one row per measurement expresses::

        ln R(axis, run) = ln kappa_axis + (2/K) * s * M_lab

    Returns ``(params, cov, residuals, chi2_dof)``: the fitted parameter vector in
    the caller's column order, its covariance, per-row residuals in ``ln R``, and
    the reduced chi-squared.

    **The design matrix is passed in, never inferred.** Which face looks upstream
    in which orientation is campaign geometry. A sign error there is invisible to
    the fit's own diagnostics -- it fits happily -- so callers cross-check against
    :func:`area_ratio`, which cannot be fooled by one.

    ``cov`` treats ``sigma`` as absolute (no ``lstsq`` residual rescaling), so a
    poor fit widens ``chi2_dof`` rather than hiding inside a comfortable error bar.
    """
    ln_R = np.asarray(ln_R, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    design = np.asarray(design, dtype=float)
    if design.shape[0] != ln_R.size:
        raise ValueError(f"design has {design.shape[0]} rows, {ln_R.size} measurements")

    w = 1.0 / sigma
    params, *_ = np.linalg.lstsq(design * w[:, None], ln_R * w, rcond=None)
    residuals = ln_R - design @ params

    dof = ln_R.size - design.shape[1]
    if dof <= 0:
        raise ValueError(f"{ln_R.size} measurements cannot constrain "
                         f"{design.shape[1]} parameters")
    chi2_dof = float(np.sum((residuals / sigma) ** 2) / dof)
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
    return params, cov, residuals, chi2_dof
