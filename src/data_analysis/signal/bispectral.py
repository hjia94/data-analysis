"""Bispectrum and bicoherence: three-wave (quadratic) phase coupling.

Second-order spectra (:func:`~data_analysis.signal.avg_cross_spectrum` and
friends) compare signals at one frequency at a time. They cannot tell whether a
peak at ``f1 + f2`` is *generated* by waves at ``f1`` and ``f2`` (a quadratic
interaction, which locks its phase to ``phi1 + phi2``) or is an independent wave
that happens to sit there. The bispectrum can::

    B(f1, f2)  = < X(f1) Y(f2) Z*(f1 + f2) >                      (ensemble mean)
    b2(f1, f2) = |B|^2 / ( <|X(f1) Y(f2)|^2> <|Z(f1 + f2)|^2> )   (Kim & Powers 1979)

If the biphase ``phi_X(f1) + phi_Y(f2) - phi_Z(f1+f2)`` is the same in every
realization, ``B`` survives the average and ``b2 -> 1``; if it is random,
``B`` averages away and ``b2 ~ 1/N`` for ``N`` realizations. ``b2`` is the
fraction of the power at ``f1 + f2`` that is phase-coupled to ``f1`` and ``f2``,
bounded to 0..1 by Cauchy-Schwarz. It is the magnitude-squared coherence between
the product ``X(f1) Y(f2)`` and ``Z(f1 + f2)``, so its no-coupling distribution
is the coherence one: :func:`bicoherence_threshold`.

:func:`bicoherence` is the auto form (``X = Y = Z``); :func:`cross_bicoherence`
takes three channels, e.g. a low-frequency flow on one probe, the pump wave on
another and a sideband on a third.

Realizations are Welch-style segments of every shot: rows of a ``(nshot, nt)``
stack, each cut into ``nperseg``-sample windows. Frequencies are evaluated on
the FFT grid inside the bands ``f1_band``/``f2_band`` only, so a narrow region
of a very long record (e.g. a few kHz around a 188 kHz pump at 40 ns sampling)
stays cheap. ``f2`` may be negative for difference-frequency triads
(``f1 - |f2|``), using ``X(-f) = X*(f)`` for real signals.

``b2`` is unchanged by any per-frequency gain of a channel, so raw B-dot
volts give the same bicoherence as integrated B.

**Triggered shots.** Realizations must differ in phase for ``b2`` to mean
coupling. If all three components are phase-locked to the trigger (a driven
pump, and anything reproducible shot to shot), the biphase is constant and
``b2 -> 1`` with no interaction at all. A triad with a locked pump is still a
valid test when the ``f1`` and ``f1 + f2`` components have random phase across
shots: then a high ``b2`` means the sideband's phase follows the
low-frequency one. Check each component's shot-to-shot phase consistency
``|<X/|X|>|`` first. Do not subtract the ensemble mean to fix this, because
that also removes a locked pump. Pool only realizations where the coupling
should have the same biphase, e.g. one probe position, or a fixed reference
probe across a scan.

Rewritten from the MIT-licensed HOSA port ``synergetics/spectrum``
(``bicoherence.py``, ``bicoherencex.py``). Changes: Python 3; physical
frequency axes; shot-ensemble averaging; band-limited evaluation; the bounded
Kim-Powers normalization instead of ``|B|^2 / (P1 P2 P12)``, which can exceed
1; no plotting.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import signal as sps

from data_analysis.signal.core import finite_row_mask

# Bins closer than this window correlation are treated as not independent.
_RHO_MIN = 0.1

# Elements in one block's temporaries -- (segments x nperseg) samples and
# (segments x n1 x n2) products -- which bounds memory per block (~64 MB each).
_BLOCK_ELEMENTS = 1 << 22


class BicoherenceResult(NamedTuple):
    """Output of :func:`bicoherence` / :func:`cross_bicoherence`.

    ``b2`` and ``bispectrum`` have shape ``(len(f1), len(f2))``; points where
    ``|f1 + f2|`` exceeds the Nyquist frequency are ``nan``. So are points
    where the three frequencies are not resolved from each other or from 0.
    That is, any of ``|f1|``, ``|f2|``, ``|f1+f2|``, ``||f1+f2| - |f1||`` or
    ``||f1+f2| - |f2||`` is within the window's main lobe (2 bins for Hann, 0
    for boxcar). There the "triad" is degenerate -- at ``f2 = 0`` it is a power
    spectrum times the mean -- or its bins share leakage, and ``b2`` loses its
    meaning and its chance level. ``angle(bispectrum)``
    is the biphase. ``bispectrum`` is in units of the input cubed: each
    segment spectrum is scaled so a bin-centred sinusoid of amplitude ``a`` has
    ``|X| = a``, hence ``|B| = a1 a2 a3`` for a fully coupled triad.
    ``n_avg`` is the number of segments averaged, ``threshold`` the ``b2`` that
    uncoupled signals exceed with probability ``p`` (:func:`bicoherence_threshold`),
    and ``1 / n_avg`` the mean no-coupling level.
    """

    f1: np.ndarray
    f2: np.ndarray
    b2: np.ndarray
    bispectrum: np.ndarray
    n_avg: int
    threshold: float


class PhaseConsistencyResult(NamedTuple):
    """Output of :func:`shot_phase_consistency`.

    ``consistency`` is the shot-to-shot phase locking ``|<X(f)/|X(f)|>|`` in
    ``[0, 1]`` (RMS across segment offsets when multiple segments fit per shot).
    Values significantly above ``threshold`` indicate trigger-locked phase
    consistency across shots, which can inflate bicoherence without genuine
    nonlinear phase coupling.
    """

    freq: np.ndarray
    consistency: np.ndarray
    n_shots: int
    threshold: float


def bicoherence_threshold(n_avg, p=0.05):
    """``b2`` level that uncoupled signals exceed with probability ``p``.

    With ``N`` independent realizations and no phase coupling, ``b2`` follows
    the magnitude-squared-coherence null distribution
    ``P(b2 > x) = (1 - x)**(N - 1)``, so the threshold is
    ``1 - p**(1/(N - 1))`` (about ``-ln(p) / N`` for large ``N``; mean ``1/N``).
    It holds per ``(f1, f2)`` point: over a grid of ``M`` points about ``p*M``
    exceed it by chance. Overlapping segments are not independent, so with
    ``noverlap > 0`` the chance rate is no longer exactly ``p``. For
    ``p = 0.05`` and 50 % overlap, Gaussian noise (20 trials of 200 segments)
    exceeded it 5.2 % of the time on average with a Hann window (3-7 % per
    trial, the same spread as without overlap) and 9 % with a boxcar
    (8-11 %). Use ``noverlap=0`` when the threshold matters.
    """
    n = int(n_avg)
    if n < 2:
        raise ValueError(f"need at least 2 realizations, got {n}")
    return 1.0 - p ** (1.0 / (n - 1))


def _band_bins(band, df, nyq, name, allow_negative):
    lo, hi = band
    if not allow_negative and lo < 0:
        raise ValueError(f"{name} must be non-negative, got {band}")
    k0 = max(int(np.ceil(lo / df - 1e-9)), -nyq)
    k1 = min(int(np.floor(hi / df + 1e-9)), nyq)
    if k1 < k0:
        raise ValueError(f"{name} {lo:g}-{hi:g} Hz selects no frequency bin "
                         f"(df={df:.4g} Hz, Nyquist={nyq * df:.4g} Hz)")
    return np.arange(k0, k1 + 1)


def _lobe_bins(win):
    """Largest bin offset at which ``win`` still correlates two bins of white noise > _RHO_MIN."""
    w2 = win ** 2
    rho = np.abs(np.fft.fft(w2)) / w2.sum()                   # |corr(X(k), X(k + dk))|, dk = 0..
    near = np.flatnonzero(rho[1:win.size // 2] > _RHO_MIN)
    return int(near[-1] + 1) if near.size else 0


def _at(F, k):
    """Spectrum at signed bins ``k`` from one-sided ``F`` (real input): X(-f) = X*(f)."""
    v = F[..., np.abs(k)]
    return np.where(k < 0, np.conj(v), v)


def _bispec(stacks, dt, nperseg, noverlap, window, detrend, f1_band, f2_band, p):
    arrs = []
    for s in stacks:
        a = np.asarray(s)
        if np.iscomplexobj(a):
            raise ValueError("inputs must be real signals")
        if not np.issubdtype(a.dtype, np.floating):
            a = a.astype(float)
        arrs.append(a[None, :] if a.ndim == 1 else a)     # no copy: blocks are cast to float64
    if any(a.ndim != 2 for a in arrs):
        raise ValueError("inputs must be 1-D records or 2-D (nshot, nt) stacks")
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError(f"input shapes differ: {[a.shape for a in arrs]}")
    rows_ok = np.flatnonzero(finite_row_mask(*arrs))       # index, don't copy, the good shots
    nshot, nt = rows_ok.size, arrs[0].shape[1]
    if nshot == 0:
        raise ValueError("no shot is finite in every input")

    nperseg = int(nperseg)
    if nperseg > nt:
        raise ValueError(f"nperseg={nperseg} is longer than the record ({nt} samples)")
    noverlap = 0 if noverlap is None else int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg); got {noverlap}")
    step = nperseg - noverlap
    starts = np.arange(0, nt - nperseg + 1, step)
    seg_shot = np.repeat(rows_ok, starts.size)
    seg_start = np.tile(starts, nshot)
    n_avg = seg_shot.size
    if n_avg < 2:
        raise ValueError("fewer than 2 segments to average; use more shots or a shorter nperseg")

    df = 1.0 / (nperseg * dt)
    nyq = nperseg // 2
    k1 = _band_bins(f1_band if f1_band is not None else (df, nyq * df), df, nyq, "f1_band", False)
    k2 = _band_bins(f2_band if f2_band is not None else (df, nyq * df), df, nyq, "f2_band", True)
    win = sps.get_window(window, nperseg)
    # The null distribution needs Z(f1+f2) independent of X(f1) Y(f2). A bin
    # within the window's main lobe of 0 (it leaks the mean) or of another
    # triad member breaks that. f2 = 0 and f1 + f2 = 0 are the exact cases.
    lobe = _lobe_bins(win)
    a1, a2 = np.abs(k1)[:, None], np.abs(k2)[None, :]
    k3 = k1[:, None] + k2[None, :]
    a3 = np.abs(k3)
    valid = ((a3 <= nyq) & (a1 > lobe) & (a2 > lobe) & (a3 > lobe)
             & (np.abs(a3 - a1) > lobe) & (np.abs(a3 - a2) > lobe))
    k3 = np.where(valid, k3, 0)

    # bin-centred sinusoid of amplitude a -> |X| = a; DC and Nyquist bins are not doubled
    scale = np.full(nyq + 1, 2.0 / win.sum())
    scale[0] = 1.0 / win.sum()
    if nperseg % 2 == 0:
        scale[nyq] = 1.0 / win.sum()
    idx = np.arange(nperseg)
    auto = len(arrs) == 1

    num = np.zeros(k3.shape, complex)
    pp = np.zeros(k3.shape)
    zz = np.zeros(k3.shape)
    block = max(1, min(_BLOCK_ELEMENTS // k3.size, _BLOCK_ELEMENTS // nperseg))
    for b0 in range(0, n_avg, block):
        rows = seg_shot[b0:b0 + block, None]
        cols = seg_start[b0:b0 + block, None] + idx
        spec = []
        for a in arrs:
            seg = a[rows, cols].astype(float)
            if detrend:
                seg = sps.detrend(seg, axis=-1, type=detrend)
            spec.append(np.fft.rfft(seg * win, axis=-1) * scale)
        Fx, Fy, Fz = (spec * 3) if auto else spec
        P = _at(Fx, k1)[:, :, None] * _at(Fy, k2)[:, None, :]
        Z = _at(Fz, k3)
        num += np.einsum("sij,sij->ij", P, np.conj(Z))
        pp += np.einsum("sij,sij->ij", P, np.conj(P)).real
        zz += np.einsum("sij,sij->ij", Z, np.conj(Z)).real

    B = num / n_avg
    denom = (pp / n_avg) * (zz / n_avg)
    with np.errstate(invalid="ignore", divide="ignore"):
        b2 = np.where(valid & (denom > 0), np.abs(B) ** 2 / denom, np.nan)
    B = np.where(valid, B, np.nan)
    return BicoherenceResult(k1 * df, k2 * df, b2, B, n_avg, bicoherence_threshold(n_avg, p))


def bicoherence(stack, dt, nperseg=1024, noverlap=None, *, f1_band=None,
                f2_band=None, window="hann", detrend="constant", p=0.05):
    """Auto-bicoherence and bispectrum of one channel, averaged over shots and segments.

    ``stack`` is ``(nshot, nt)`` (one row per shot) or a single 1-D record;
    rows with a non-finite sample are dropped. Each row is cut into segments of
    ``nperseg`` samples (``noverlap`` overlap, default 0 so the significance
    threshold is exact), detrended (``'constant'``, ``'linear'`` or ``None``),
    windowed and FFT'd; every segment is one realization.

    ``f1_band``/``f2_band`` are ``(f_lo, f_hi)`` in Hz; the default is
    ``(df, Nyquist)`` for both, ``df = 1/(nperseg*dt)``. Only bins inside the
    bands are evaluated, so for long records give the bands (and choose
    ``nperseg`` for the resolution ``df`` you need) instead of computing the
    full grid. ``f2`` may be negative. Points not resolved from 0 or from
    each other (within the window's main lobe, e.g. the lowest 2 bins for
    Hann) come back ``nan``; see :class:`BicoherenceResult`. For the auto-bispectrum
    ``B(f1, f2) = B(f2, f1)``, so ``f2 <= f1`` holds all the information.

    Returns a :class:`BicoherenceResult` ``(f1, f2, b2, bispectrum, n_avg,
    threshold)``; ``threshold`` is the chance level at probability ``p``.
    """
    return _bispec([stack], dt, nperseg, noverlap, window, detrend, f1_band, f2_band, p)


def cross_bicoherence(stack_x, stack_y, stack_z, dt, nperseg=1024, noverlap=None, *,
                      f1_band=None, f2_band=None, window="hann", detrend="constant",
                      p=0.05):
    """Cross-bicoherence ``<X(f1) Y(f2) Z*(f1+f2)>`` of three channels.

    Tests whether ``f1`` seen on ``x`` and ``f2`` seen on ``y`` are phase-coupled
    to ``f1 + f2`` seen on ``z``. The three stacks must share shape and time
    grid; a shot is dropped if it is non-finite in any of them. ``f2_band`` may
    be negative to test difference triads (``f1 - |f2|`` on ``z``). Pass the same
    stack twice for two-channel forms, e.g. ``(x, x, y)``. Arguments and
    result are as in :func:`bicoherence`.
    """
    return _bispec([stack_x, stack_y, stack_z], dt, nperseg, noverlap, window, detrend,
                   f1_band, f2_band, p)


def shot_phase_consistency(stack, dt, nperseg=1024, noverlap=None, *, f_band=None,
                           window="hann", detrend="constant", p=0.05):
    """Shot-to-shot phase consistency ``|<X(f) / |X(f)||>|`` at each frequency.

    Tests whether spectral components are phase-locked to the shot trigger (e.g.
    a reproducible bias turn-on transient or driven pump). Each finite shot in
    the 2-D ``(nshot, nt)`` stack is cut into segments matching
    :func:`bicoherence`; the unit-phasor mean is computed **across shots** at
    each fixed segment offset ``m`` so a trigger-locked tone never cancels across
    segment times, and the RMS across segment offsets is returned (identical to
    ``|<X/|X|>|`` when one segment fits per shot).

    ``threshold`` is the Rayleigh null-hypothesis critical value
    ``sqrt(-ln(p) / n_shots)`` (capped at ``1.0``) for uniform random shot phase.
    """
    a = np.asarray(stack)
    if np.iscomplexobj(a):
        raise ValueError("input must be a real signal")
    if a.ndim != 2:
        raise ValueError(f"stack must be 2-D (nshot, nt), got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(float)
    rows_ok = np.flatnonzero(finite_row_mask(a))
    n_shots, nt = rows_ok.size, a.shape[1]
    if n_shots < 2:
        raise ValueError(f"need at least 2 finite shots, got {n_shots}")

    nperseg = int(nperseg)
    if nperseg > nt:
        raise ValueError(f"nperseg={nperseg} is longer than the record ({nt} samples)")
    noverlap = 0 if noverlap is None else int(noverlap)
    if not 0 <= noverlap < nperseg:
        raise ValueError(f"noverlap must be in [0, nperseg); got {noverlap}")
    step = nperseg - noverlap
    starts = np.arange(0, nt - nperseg + 1, step)
    n_seg = starts.size

    df = 1.0 / (nperseg * dt)
    nyq = nperseg // 2
    k = _band_bins(f_band if f_band is not None else (df, nyq * df), df, nyq, "f_band", False)
    win = sps.get_window(window, nperseg)
    idx = np.arange(nperseg)

    phasor_sum = np.zeros((n_seg, k.size), dtype=complex)
    count = np.zeros((n_seg, k.size), dtype=int)
    block = max(1, _BLOCK_ELEMENTS // nperseg)
    for m, s0 in enumerate(starts):
        cols = s0 + idx
        for b0 in range(0, n_shots, block):
            rows = rows_ok[b0:b0 + block]
            seg = a[rows[:, None], cols[None, :]].astype(float)
            if detrend:
                seg = sps.detrend(seg, axis=-1, type=detrend)
            F = np.fft.rfft(seg * win, axis=-1)[:, k]
            mag = np.abs(F)
            nz = mag > 0
            u = np.zeros_like(F)
            u[nz] = F[nz] / mag[nz]
            phasor_sum[m] += u.sum(axis=0)
            count[m] += nz.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        R_m = np.where(count > 0, np.abs(phasor_sum) / count, np.nan)
    has_val = np.any(np.isfinite(R_m), axis=0)
    consistency = np.full(k.size, np.nan, dtype=float)
    if np.any(has_val):
        consistency[has_val] = np.sqrt(np.nanmean(R_m[:, has_val] ** 2, axis=0))
    threshold = min(1.0, float(np.sqrt(-np.log(p) / n_shots)))
    return PhaseConsistencyResult(k * df, consistency, int(n_shots), threshold)


def bicoherence_surrogate(stack_x, stack_y, stack_z, dt, nperseg=1024, noverlap=None, *,
                          f1_band=None, f2_band=None, window="hann", detrend="constant",
                          p=0.05, shift=None, rng=None):
    """Shot-shuffled surrogate bicoherence separating trigger locking from true coupling.

    Permutes the shot index of ``stack_z`` with zero fixed points before
    computing ``<X_s(f1) Y_s(f2) Z_{perm(s)}*(f1+f2)>``. When ``rng`` is
    provided (a :class:`numpy.random.Generator` or integer seed), a random
    Sattolo derangement is used; otherwise a deterministic cyclic roll by
    ``shift`` shots is used (default ``n_shots // 2`` to maximally separate
    paired shots and avoid leaking slow adjacent-shot drift into the surrogate).

    Any phase coincidence caused by shot-to-shot trigger locking survives the
    permutation, whereas genuine shot-by-shot quadratic phase coupling (where
    the fundamental phase varies across shots) averages away. Pass
    ``(x, x, x)`` for an auto-bicoherence surrogate.
    """
    arrs = [np.asarray(s) for s in (stack_x, stack_y, stack_z)]
    if any(a.ndim != 2 for a in arrs):
        raise ValueError("surrogate requires 2-D (nshot, nt) stacks")
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError(f"input shapes differ: {[a.shape for a in arrs]}")
    rows_ok = np.flatnonzero(finite_row_mask(*arrs))
    n = rows_ok.size
    if n < 2:
        raise ValueError(f"need at least 2 jointly finite shots for surrogate, got {n}")

    if rng is None:
        s_raw = (n // 2) if shift is None else int(shift)
        s = s_raw % n
        if s == 0:
            raise ValueError(f"shift={shift} is a multiple of n_shots={n} (produces fixed points)")
        perm = np.roll(np.arange(n), -s)
    else:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        perm = np.arange(n)
        for i in range(n - 1, 0, -1):
            j = int(gen.integers(0, i))
            perm[i], perm[j] = perm[j], perm[i]

    x_ok = arrs[0][rows_ok]
    y_ok = arrs[1][rows_ok]
    z_ok = arrs[2][rows_ok][perm]
    return cross_bicoherence(x_ok, y_ok, z_ok, dt, nperseg=nperseg, noverlap=noverlap,
                             f1_band=f1_band, f2_band=f2_band, window=window,
                             detrend=detrend, p=p)


def summed_bicoherence(result, *, symmetric=False):
    """Collapse a 2-D bicoherence map onto the target sum frequency ``f3 = f1 + f2``.

    For every positive ``f3`` that has at least one finite triad ``(f1, f2)`` in
    ``result.b2``, returns ``(f3_hz, b2_mean, b2_max, count)``. Set
    ``symmetric=True`` for auto-bicoherence so conjugate-symmetric pairs
    ``(f1, f2)`` and ``(f2, f1)`` are counted once (``f2 <= f1``).
    """
    f1 = np.asarray(result.f1, dtype=float)
    f2 = np.asarray(result.f2, dtype=float)
    b2 = np.asarray(result.b2, dtype=float)
    diffs = np.concatenate([np.diff(f1), np.diff(f2)])
    pos_diffs = diffs[diffs > 0]
    df = float(np.min(pos_diffs)) if pos_diffs.size else None

    f1g, f2g = np.meshgrid(f1, f2, indexing="ij")
    f3g = f1g + f2g
    mask = np.isfinite(b2) & (f3g > 0.0)
    if symmetric:
        tol = 1e-9 * (df if df is not None else 1.0)
        mask &= f2g <= f1g + tol

    if not np.any(mask):
        empty = np.array([], dtype=float)
        return empty, empty, empty, np.array([], dtype=int)

    f3_valid = f3g[mask]
    b2_valid = b2[mask]
    if df is not None:
        keys = np.rint(f3_valid / df).astype(np.int64)
        uniq, inv = np.unique(keys, return_inverse=True)
        f3_out = uniq * df
    else:
        uniq, inv = np.unique(np.round(f3_valid, 9), return_inverse=True)
        f3_out = uniq

    n_u = f3_out.size
    count = np.bincount(inv, minlength=n_u).astype(int)
    b2_sum = np.bincount(inv, weights=b2_valid, minlength=n_u)
    b2_mean = b2_sum / count
    b2_max = np.full(n_u, -np.inf, dtype=float)
    np.maximum.at(b2_max, inv, b2_valid)
    return f3_out, b2_mean, b2_max, count
