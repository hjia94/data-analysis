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
have these the other way round. Feeding flow-labelled data to these functions
returns the flow where kappa was asked for, silently and with a plausible
magnitude -- the caller cannot see it from the signature, hence this paragraph.

This module takes no time axis, no window, no run numbers and no orientation
table: selecting which samples enter a calibration is campaign policy, and a
default here would be the wrong default for the next campaign to inherit.
"""

from __future__ import annotations

import numpy as np

# Magnetised fluid model, Hutchinson, Phys. Fluids 30, 3777 (1987). The
# calibration constant relating ln(j_+/j_-) to the Mach number; unmagnetised and
# kinetic models give other values, so it is a parameter everywhere below.
K_HUTCHINSON = 0.45


def valid_current_mask(*currents):
    """``True`` where every current is finite and strictly positive.

    Isat is positive by construction on these probes, so a non-positive sample is
    missing data (dropped channel, baseline excursion), not a small measurement.
    Callers **count** what this excludes rather than letting it vanish into a
    ``nanmean``: a silently shrinking sample looks identical to a clean one.
    """
    mask = np.ones(np.broadcast(*currents).shape, dtype=bool)
    for c in currents:
        mask &= np.isfinite(c) & (c > 0)
    return mask


def face_ratio(j_plus, j_minus, axis=1, mask=None):
    """Face ratio of two current stacks, reduced along ``axis``.

    ``j_plus``/``j_minus`` are ``(nshot, nt_win)`` and **already windowed** -- the
    caller selects the samples. The reduction happens here, in the order that
    matters: average *first*, then divide.

    That order is not cosmetic. Dividing sample-by-sample divides by the
    instantaneous ``j_minus``, which is small and noisy at individual samples, so
    the ratio distribution grows a heavy tail that biases any later average.
    Pooling before dividing has the opposite failure: it collapses the sample to a
    single number, leaving no scatter to weight with. Both are easy to write by
    accident, which is why this function owns the reduction instead of taking
    pre-divided ratios.

    ``axis=1`` (default) averages over the window -> one ratio per shot, the
    calibration input. ``axis=0`` averages over shots -> one ratio per sample, the
    drift diagnostic. Same invariant either way, one implementation.

    ``mask`` overrides which samples count (default: :func:`valid_current_mask` of
    the two stacks). Pass a wider mask -- e.g. one requiring all four tips of a
    pairing valid -- to keep two ratios sample-aligned. Entries with nothing valid
    yield ``nan``.
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

    The **product**, because these are hardware labels (module docstring): the
    flow exponent enters ``R_a`` and ``R_b`` with opposite signs and cancels.
    Immune to a sign-convention error for the same reason, which is what makes it
    a usable cross-check on a fitted kappa.
    """
    return np.sqrt(np.asarray(R_a, dtype=float) * np.asarray(R_b, dtype=float))


def mach_number(R_a, R_b, K=K_HUTCHINSON):
    """Mach number from two opposing orientations -> ``M = (K/4) ln(R_a/R_b)``.

    The **ratio**: kappa cancels, leaving the flow along the lab axis this face
    pair faced in orientation ``a``. Sign follows ``a``; swapping the arguments
    flips it and leaves :func:`area_ratio` unchanged.
    """
    return (K / 4.0) * np.log(np.asarray(R_a, dtype=float)
                              / np.asarray(R_b, dtype=float))


def mach_single(R, kappa, K=K_HUTCHINSON):
    """Mach number from one orientation given a known kappa -> ``(K/2) ln(R/kappa)``.

    The point of calibrating: once kappa is known, a single measurement gives the
    flow. Accuracy is limited by kappa's systematic error, not by shot noise.
    """
    return (K / 2.0) * np.log(np.asarray(R, dtype=float)
                              / np.asarray(kappa, dtype=float))


def combine_log(values):
    """Log-space average of ``values`` -> ``(log_mean, log_std, n_valid)``.

    Returns the mean of ``ln values``, **not** the geometric mean: callers here
    work in log space (it is where the errors are symmetric and where the fit
    lives), so handing back ``exp`` of it would only be converted straight back.
    ``np.exp(log_mean)`` is the geometric mean when a caller wants one.

    ``log_std`` is the **population** spread (``ddof=1``), not the standard error
    -- callers divide by ``sqrt(n_valid)`` themselves when they want the error on
    the mean, and some deliberately do not (a between-orientation spread is a
    systematic, and averaging more orientations does not shrink it).

    Log space because kappa is multiplicative: ``ln R`` is the additive symmetric
    quantity, and the arithmetic mean of ratios sits above the geometric mean by a
    bias that grows with the spread. Non-finite and non-positive entries are
    excluded and counted in ``n_valid`` -- the one definition of a valid ratio, so
    that ``n_valid`` always describes the sample the mean was taken over.
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
    in which orientation is campaign geometry, and guessing it here would bury a
    sign convention inside shared code. A sign error is not visible in the fit's
    own diagnostics -- it fits happily -- so callers cross-check against
    :func:`area_ratio`, which cannot be fooled by one.

    ``cov`` uses the supplied ``sigma`` as absolute (``lstsq`` residual scaling is
    deliberately not applied): these sigmas are measured shot-to-shot errors, not
    relative weights, and rescaling them by the fit quality would hide a poor fit
    inside a comfortable-looking error bar. Read ``chi2_dof`` for fit quality
    instead.
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
