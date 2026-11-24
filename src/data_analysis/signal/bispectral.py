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

# Elements in one (segments x n1 x n2) complex temporary; bounds memory per block.
_BLOCK_ELEMENTS = 1 << 22


class BicoherenceResult(NamedTuple):
    """Output of :func:`bicoherence` / :func:`cross_bicoherence`.

    ``b2`` and ``bispectrum`` have shape ``(len(f1), len(f2))``; points where
    ``|f1 + f2|`` exceeds the Nyquist frequency are ``nan``. ``angle(bispectrum)``
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


def bicoherence_threshold(n_avg, p=0.05):
    """``b2`` level that uncoupled signals exceed with probability ``p``.

    With ``N`` independent realizations and no phase coupling, ``b2`` follows
    the magnitude-squared-coherence null distribution
    ``P(b2 > x) = (1 - x)**(N - 1)``, so the threshold is
    ``1 - p**(1/(N - 1))`` (about ``-ln(p) / N`` for large ``N``; mean ``1/N``).
    It holds per ``(f1, f2)`` point: over a grid of ``M`` points about ``p*M``
    exceed it by chance. Overlapping segments are not independent, so with
    ``noverlap > 0`` the threshold is optimistic (too low).
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


def _at(F, k):
    """Spectrum at signed bins ``k`` from one-sided ``F`` (real input): X(-f) = X*(f)."""
    v = F[..., np.abs(k)]
    return np.where(k < 0, np.conj(v), v)


def _bispec(stacks, dt, nperseg, noverlap, window, detrend, f1_band, f2_band, p):
    arrs = []
    for s in stacks:
        a = np.asarray(s, dtype=float)
        arrs.append(a[None, :] if a.ndim == 1 else a)
    if any(a.ndim != 2 for a in arrs):
        raise ValueError("inputs must be 1-D records or 2-D (nshot, nt) stacks")
    if any(a.shape != arrs[0].shape for a in arrs):
        raise ValueError(f"input shapes differ: {[a.shape for a in arrs]}")
    good = finite_row_mask(*arrs)
    arrs = [a[good] for a in arrs]
    nshot, nt = arrs[0].shape
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
    seg_shot = np.repeat(np.arange(nshot), starts.size)
    seg_start = np.tile(starts, nshot)
    n_avg = seg_shot.size
    if n_avg < 2:
        raise ValueError("fewer than 2 segments to average; use more shots or a shorter nperseg")

    df = 1.0 / (nperseg * dt)
    nyq = nperseg // 2
    k1 = _band_bins(f1_band if f1_band is not None else (df, nyq * df), df, nyq, "f1_band", False)
    k2 = _band_bins(f2_band if f2_band is not None else (df, nyq * df), df, nyq, "f2_band", True)
    k3 = k1[:, None] + k2[None, :]
    valid = np.abs(k3) <= nyq
    k3 = np.where(valid, k3, 0)

    win = sps.get_window(window, nperseg)
    scale = 2.0 / win.sum()           # bin-centred sinusoid of amplitude a -> |X| = a
    idx = np.arange(nperseg)
    auto = len(arrs) == 1

    num = np.zeros(k3.shape, complex)
    pp = np.zeros(k3.shape)
    zz = np.zeros(k3.shape)
    block = max(1, _BLOCK_ELEMENTS // k3.size)
    for b0 in range(0, n_avg, block):
        rows = seg_shot[b0:b0 + block, None]
        cols = seg_start[b0:b0 + block, None] + idx
        spec = []
        for a in arrs:
            seg = a[rows, cols]
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
    full grid. ``f2`` may be negative. For the auto-bispectrum
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
