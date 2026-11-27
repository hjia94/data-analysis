"""Mach-probe estimator maths -- campaign-agnostic.

A Mach probe reads ion saturation current on two opposed faces of one axis. In
the magnetised fluid model (Hutchinson, *Phys. Fluids* **30**, 3777 (1987)),
the ratio of opposed-face currents obeys ``R = j_+/j_- = kappa * exp(s * M_lab / K)``
with ``K = M_c = 0.45``, i.e. per face:

    j_pm = j_sat * kappa_pm * exp(pm * s * M_lab / (2 * K))

where ``s = flow_sign`` maps face orientation to the lab axis: when the
hardware ``+`` tip physically faces ``+axis`` (e.g. ``X+`` faces ``+x``,
``Y+`` faces ``+y``, ``Z+`` faces ``+z``), plasma flowing *toward* ``+axis``
strikes the ``-`` face (``j_- > j_+``), so ``s = FLOW_SIGN_PLUS_FACING_AXIS = -1``
and ``M_lab = -K ln(R/kappa)`` is positive for flow along ``+axis``.
If a probe's ``+`` tip instead faces ``-axis`` (looking upstream into
``+axis`` flow), pass ``flow_sign=+1``.

**Every current here is hardware-labelled.** ``j_plus`` is a fixed physical tip
on a fixed DAQ channel -- *not* "the face pointing upstream". Under that
convention a 180 deg rotation inverts the flow exponent while leaving kappa
alone::

    R_a = kappa * exp(+s M_lab / K)        R_b = kappa * exp(-s M_lab / K)

so the **product** isolates the area ratio and the **ratio** isolates the flow.
Flow-referenced treatments (where ``J_+`` means the upstream face by definition)
have these the other way round. Feeding flow-labelled data in returns the flow
where kappa was asked for, silently and with a plausible magnitude.

Selecting which samples enter a calibration is campaign policy, so this module
takes no time axis, window, run numbers or orientation table. The reduction
helpers below (:func:`time_bin_edges`, :func:`binned_face_ratio`) are the
exception only in taking a time axis -- they are handed one, and still decide
nothing about which window is interesting.

Resolving a flow *map* about a rotation centre is not Mach-probe maths and lives
in :mod:`data_analysis.plasma.flow`.
"""

from __future__ import annotations

import numpy as np

from data_analysis.plasma.formulas import ion_sound_speed

# Magnetised fluid model, Hutchinson, Phys. Fluids 30, 3777 (1987). The
# calibration constant M_c relating the current ratio R = j_+/j_- to the Mach
# number via R = kappa * exp(s * M / K) <=> M = s * K * ln(R / kappa);
# unmagnetised and kinetic models give other values, so it is a parameter
# everywhere below.
K_HUTCHINSON = 0.45

#: Authoritative lab-axis sign convention when the ``+`` tip faces ``+axis``
#: (e.g. ``X+`` faces ``+x``, ``Y+`` faces ``+y``, ``Z+`` faces ``+z``):
#: flow moving along ``+axis`` strikes the ``-`` face (``j_- > j_+``), so
#: ``M_lab = FLOW_SIGN_PLUS_FACING_AXIS * K * ln(R/kappa)`` is positive for
#: flow along ``+axis``. Pass ``flow_sign=+1`` only if ``+`` faces ``-axis``.
FLOW_SIGN_PLUS_FACING_AXIS = -1


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


def mach_number(R_a, R_b, K=K_HUTCHINSON, flow_sign=FLOW_SIGN_PLUS_FACING_AXIS):
    """Lab-axis Mach number from two opposing orientations -> ``flow_sign * (K/2) ln(R_a/R_b)``.

    The **ratio**: kappa cancels, leaving the flow along the lab axis of
    orientation ``a`` (positive along ``+axis`` when ``flow_sign=-1`` and the
    ``+`` face points toward ``+axis``). Swapping ``R_a``/``R_b`` flips sign.
    """
    return float(flow_sign) * (K / 2.0) * np.log(
        np.asarray(R_a, dtype=float) / np.asarray(R_b, dtype=float)
    )


def mach_single(R, kappa, K=K_HUTCHINSON, flow_sign=FLOW_SIGN_PLUS_FACING_AXIS):
    """Lab-axis Mach number from one orientation given kappa -> ``flow_sign * K ln(R/kappa)``.

    With ``flow_sign=FLOW_SIGN_PLUS_FACING_AXIS`` (``-1``), positive ``M``
    denotes plasma flowing toward ``+axis`` when the ``+`` tip physically faces
    ``+axis`` (upstream-collecting ``-`` tip sees higher current).
    Accuracy is limited by kappa's systematic error, not by shot noise.
    """
    return float(flow_sign) * float(K) * np.log(
        np.asarray(R, dtype=float) / np.asarray(kappa, dtype=float)
    )


def flow_velocity(M, T_e_eV, mu, gamma=1.0, Z=1):
    """Mach number -> flow speed in **km/s**. ``mu`` is the ion mass in m_p (He = 4).

    Defaults to the isothermal ion sound speed ``c_s = sqrt(Z * T_e / m_i)``
    (``gamma = 1.0``), matching Hutchinson (1987) and the fluid/profile solvers
    in :mod:`data_analysis.plasma.profiles` and :mod:`data_analysis.plasma.stability`.
    A Mach probe does not measure ``T_e``, so every velocity carries that
    assumption (``v`` scales as ``sqrt(T_e)``). Callers state the T_e they used
    wherever the velocity is shown.
    """
    # ion_sound_speed returns cm/s; 1e-5 converts to km/s.
    return np.asarray(M, dtype=float) * ion_sound_speed(T_e_eV, mu, gamma, Z) * 1e-5


def rescale_mach_velocity(M, T_e_eV, mu, *, te_err_ev=None, M_err=None,
                          gamma=1.0, Z=1):
    """Rescale Mach number ``M`` with measured ``T_e(x, t)`` [eV] -> **km/s**.

    Defaults to the isothermal ion sound speed ``c_s = sqrt(Z * T_e / m_i)``
    (``gamma = 1.0``) matching :func:`flow_velocity`, Hutchinson (1987), and
    :func:`data_analysis.plasma.profiles.local_drift_scales`.
    Invalid or non-positive ``T_e_eV`` entries yield ``NaN`` cleanly (no
    ``RuntimeWarning``). When either ``te_err_ev`` or ``M_err`` is given,
    returns ``(v_km_s, v_err_km_s)`` propagating uncorrelated errors via::

        sigma_v^2 = (c_s * sigma_M)^2 + ((v / (2 * T_e)) * sigma_Te)^2

    Using ``c_s * sigma_M`` directly avoids a ``0 / 0`` singularity at ``M = 0``.
    """
    m_arr = np.asarray(M, dtype=float)
    te_arr = np.asarray(T_e_eV, dtype=float)
    valid_te = np.isfinite(te_arr) & (te_arr > 0.0)
    te_safe = np.where(valid_te, te_arr, 1.0)
    cs_km_s = np.where(valid_te, ion_sound_speed(te_safe, mu, gamma, Z) * 1e-5, np.nan)
    v = np.where(valid_te & np.isfinite(m_arr), m_arr * cs_km_s, np.nan)

    if te_err_ev is None and M_err is None:
        return v

    var_v = np.zeros_like(v, dtype=float)
    if M_err is not None:
        me = np.asarray(M_err, dtype=float)
        var_v = var_v + np.where(valid_te & np.isfinite(me), (cs_km_s * me) ** 2, np.nan)
    if te_err_ev is not None:
        te_e = np.asarray(te_err_ev, dtype=float)
        var_v = var_v + np.where(
            valid_te & np.isfinite(m_arr) & np.isfinite(te_e),
            (0.5 * v * te_e / te_safe) ** 2,
            np.nan,
        )
    v_err = np.where(np.isfinite(v), np.sqrt(var_v), np.nan)
    return v, v_err


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

    Uses cumulative sums padded with a leading zero so every bin ``k`` sums
    strictly over ``[edges[k], edges[k + 1])``: zero-width bins
    (``edges[k] == edges[k + 1]``, e.g. out-of-record sweep timestamps clamped
    to ``0`` or ``nt``) yield ``counts = 0``, ``ratio = nan``, and
    ``amplitude = nan``, and a final edge ``edges[-1] < nt`` never bleeds past
    the last bin into the rest of the trace.
    """
    edges = np.asarray(edges, dtype=np.intp)
    width = np.diff(edges)
    ok = valid_current_mask(plus, minus)

    def _bin_sums(arr_1d):
        cs = np.empty(arr_1d.size + 1, dtype=arr_1d.dtype)
        cs[0] = 0
        np.cumsum(arr_1d, out=cs[1:])
        return cs[edges[1:]] - cs[edges[:-1]]

    with np.errstate(invalid="ignore", divide="ignore"):
        # sum/sum, not mean-of-ratios: n cancels, so no division per sample.
        sp = _bin_sums(np.where(ok, plus, 0.0).sum(axis=0))
        sm = _bin_sums(np.where(ok, minus, 0.0).sum(axis=0))
        counts = _bin_sums(ok.sum(axis=0)).astype(np.int32)
        ratio = np.where(counts > 0, sp / sm, np.nan)
        amplitude = np.where(width > 0, (sp + sm) / (width * plus.shape[0]),
                             np.nan)
    return ratio, counts, amplitude


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

        ln R(axis, run) = ln kappa_axis + (1/K) * s * M_lab

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
