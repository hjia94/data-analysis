"""B-dot probe maths -- probe voltage to magnetic field, campaign-agnostic.

A B-dot coil of effective area ``A`` (turns x loop area) behind an amplifier of
gain ``G`` reads ``V = G * A * dB/dt``. A three-axis probe has cross-talk between
its coils, so the general form is a calibration matrix::

    [V1]       [A11 A12 A13] [dB1/dt]                    [dB1/dt]               [V1]
    [V2] = G * [A21 A22 A23] [dB2/dt]        hence       [dB2/dt] = (1/G) A^-1  [V2]
    [V3]       [A31 A32 A33] [dB3/dt]                    [dB3/dt]               [V3]

:func:`inverse_area_matrix` builds ``(1/G) A^-1`` (or ``1/(G A)`` for a single
coil), :func:`voltage_to_bdot` applies it, and :func:`integrate_bdot` integrates
dB/dt in time and removes the unknown constant / drift of integration.

Everything is SI: areas in m^2, dB/dt in T/s, B in tesla (x1e4 for gauss).
Components stay in the **probe frame**; rotating them into chamber coordinates
depends on how the probe enters the machine and is left to the campaign.

Ported from the 2019 process-plasma ``Bdot_analysis/integrate_Bi.py`` and the
probe-area branches of ``read_field_3D.read_fields`` (Patrick Pribyl / J. Han).
The old 1-axis integrator returned tesla while the 3-axis one returned gauss;
here both return tesla.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, signal


def inverse_area_matrix(area, gain=1.0):
    """Volts -> dB/dt conversion: ``1/(gain*area)`` or ``inv(area)/gain``.

    ``area`` is the effective coil area in **m^2** (multiply mm^2 by 1e-6), as

    - a scalar, for a single coil -> returns a scalar;
    - a length-3 sequence, for three independent coils -> diagonal 3x3;
    - a 3x3 calibration matrix ``A`` with ``V = gain * A @ dB/dt``, whose
      off-diagonal terms are the coil cross-talk -> ``inv(A) / gain``.

    ``gain`` is the total voltage gain between coil and digitizer, including
    attenuators (e.g. ``100 / 10**(10/20)`` for x100 behind 10 dB pads). A
    negative area or gain flips the sign of that axis, which is how a reversed
    coil winding is expressed.
    """
    a = np.asarray(area, dtype=float)
    if a.ndim == 0:
        return 1.0 / (gain * float(a))
    if a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3):
        raise ValueError(f"area must be a scalar, length 3 or 3x3; got shape {a.shape}")
    return np.linalg.inv(a) / gain


def voltage_to_bdot(v, inv_area):
    """Apply :func:`inverse_area_matrix` to digitized coil voltages -> dB/dt (T/s).

    For a scalar ``inv_area``, ``v`` has any shape and is simply scaled. For a
    3x3 ``inv_area``, ``v`` carries the three coil channels on **axis 0**, shape
    ``(3, ...)`` (e.g. ``(3, npos, nt)``, or ``np.stack([V1, V2, V3])``), and
    the result has the same shape with the three probe-frame dB/dt components
    on axis 0.

    Any DC amplifier offset is still in the result; :func:`integrate_bdot`
    removes it (``baseline='pretrigger'`` or the periodic fits).
    """
    m = np.asarray(inv_area, dtype=float)
    v = np.asarray(v, dtype=float)
    if m.ndim == 0:
        return v * m
    if v.shape[0] != 3:
        raise ValueError(f"3x3 inv_area needs the 3 coil channels on axis 0; got shape {v.shape}")
    return np.einsum("ij,j...->i...", m, v)


def _smooth_last_axis(x, sigma):
    """Gaussian smoothing along the last axis; FFT convolution, reflected edges."""
    half = int(math.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * (x.ndim - 1) + [(half, half)]
    # Reflect padding needs pad < length; fall back to edge padding for short traces.
    mode = "reflect" if half < x.shape[-1] else "edge"
    xp = np.pad(x, pad, mode=mode)
    out = signal.fftconvolve(xp, k.reshape((1,) * (x.ndim - 1) + (-1,)), mode="same", axes=-1)
    return out[..., half:half + x.shape[-1]]


def integrate_bdot(bdot, dt, baseline="pretrigger", *, pretrigger=None,
                   period=None, smooth_periods=4.0):
    """Time-integrate dB/dt (T/s) along the **last axis** -> B (tesla).

    Integration is cumulative trapezoid from ``B = 0`` at the first sample, so
    the result is defined only up to a constant (the field before the record
    started) plus a linear drift from any DC offset in ``bdot``. ``baseline``
    chooses how to remove them:

    ``'pretrigger'`` (default; pulsed shots)
        Subtract the mean of ``bdot`` over the quiet samples ``pretrigger``
        (an ``int`` n -> the first n samples, or a ``slice``) before
        integrating. B is then zero before the event and carries no offset
        drift. Requires ``pretrigger``.
    ``'periodic'`` (steady RF drive)
        Fit and subtract a straight line over an odd number of half cycles,
        then subtract the mean over an integer number of cycles, so a
        stationary sinusoid ends up centred on zero. Requires ``period``
        (samples per cycle, may be fractional, e.g. from
        :func:`data_analysis.signal.first_and_last_zerocrossings`).
    ``'smooth'`` (RF drive whose amplitude changes)
        Subtract a Gaussian-smoothed copy of B. The smoothing width follows
        the original ``gsmooth`` interval of ``period * smooth_periods / 2``
        samples (sigma = interval / sqrt 2); use ``smooth_periods >= 4`` so the
        drive itself is not removed. Requires ``period``.
    ``None``
        Raw cumulative integral.

    ``bdot`` may have any leading shape, e.g. ``(npos, nt)`` or the
    ``(3, npos, nt)`` output of :func:`voltage_to_bdot`; every trace is
    treated independently.
    """
    b = np.asarray(bdot, dtype=float)
    nt = b.shape[-1]

    if baseline == "pretrigger":
        if pretrigger is None:
            raise ValueError("baseline='pretrigger' needs pretrigger= (sample count or slice)")
        sl = slice(0, pretrigger) if isinstance(pretrigger, (int, np.integer)) else pretrigger
        quiet = b[..., sl]
        if quiet.shape[-1] == 0:
            raise ValueError(f"pretrigger={pretrigger!r} selects no samples")
        b = b - quiet.mean(axis=-1, keepdims=True)
    elif baseline in ("periodic", "smooth"):
        if period is None or period <= 1:
            raise ValueError(f"baseline={baseline!r} needs period= (samples per cycle, > 1)")
    elif baseline is not None:
        raise ValueError(f"unknown baseline {baseline!r}; "
                         "expected 'pretrigger', 'periodic', 'smooth' or None")

    B = integrate.cumulative_trapezoid(b, dx=dt, axis=-1, initial=0)

    if baseline == "periodic":
        n_cycles = math.floor(nt / period)
        if n_cycles < 1:
            raise ValueError(f"record of {nt} samples is shorter than one period ({period:g})")
        # A line fitted over ^v^v^v (whole cycles) is biased by the last half
        # cycle's slope; an odd number of half cycles (^v^v^) is not.
        n_fit = int(n_cycles * period - period / 2)
        t = np.arange(nt, dtype=float)
        tf = t[:n_fit]
        flat = B.reshape(-1, nt)
        slope, intercept = np.polynomial.polynomial.polyfit(tf, flat[:, :n_fit].T, 1)[::-1]
        flat = flat - (slope[:, None] * t + intercept[:, None])
        # ...but ^v^v^ has the wrong mean, so recentre over whole cycles.
        n_mean = int(n_cycles * period)
        flat = flat - flat[:, :n_mean].mean(axis=1, keepdims=True)
        B = flat.reshape(B.shape)
    elif baseline == "smooth":
        sigma = period * smooth_periods / 2 / math.sqrt(2)
        B = B - _smooth_last_axis(B, sigma)

    return B
