"""Electromagnetic field maths on gridded measurements -- diagnostic-agnostic.

Operators for E, B and J sampled on a regular spatial grid (B-dot probe planes
and volumes, simulations, ...):

- :func:`curl`, :func:`current_density` (J = curl B / mu_0), :func:`poynting`
  (S = E x B / mu_0), :func:`power_density` (J . E);
- :func:`faraday_ephi` -- azimuthal E from dBz/dt, assuming the field is symmetric about its axis;
- :func:`vector_potential` -- A from J by Biot-Savart (FFT convolution);
- :func:`cycle_average`, :func:`cycle_means`, :func:`axisymmetric_integral`
  -- averages over drive cycles and integrals over volumes symmetric about the axis;
- :func:`phasor`, :func:`ac_conductivity` -- complex sigma = J/E at a drive frequency.

**Units are SI throughout**: positions in m, B in T, E in V/m, J in A/m^2.
Convert cm grids (x 0.01) and gauss (x 1e-4) before calling. The linear
operators (curl, J, E_phi, A, integrals) accept complex arrays, so they apply
directly to phasor maps from :func:`phasor`; :func:`poynting` and
:func:`power_density` are instantaneous products of real fields (for phasors
the time average is ``0.5 * Re(E x conj(B)) / mu_0``).

**Grid layout.** Spatial operators take each vector component as a separate
array and an ``axes=(x_axis, y_axis, z_axis)`` argument naming which array
axis runs along x, y and z. The default ``(-1, -2, -3)`` matches arrays laid
out ``(..., nz, ny, nx)`` (the imshow-friendly order of the 2019 code); a
trailing time axis is handled by passing e.g. ``axes=(-2, -3, -4)``. Any other
axes are carried along untouched.

Ported from the 2019 process-plasma ``Bdot_analysis`` (``general.curl``,
``JdotE``, ``vector_potential``). Sign fixes relative to that code: E_phi now
includes Faraday's minus sign, and J is divided by mu_0 inside
:func:`current_density`.
"""

from __future__ import annotations

import numpy as np
from scipy import fft, integrate
from scipy.constants import mu_0


# ---------------------------------------------------------------------------
# differential operators and products
# ---------------------------------------------------------------------------

def _inexact(x):
    """``x`` as a float array, or complex if it already is (phasor maps stay complex)."""
    x = np.asarray(x)
    return x.astype(np.result_type(x, float), copy=False)


def _ddx(f, spacing, axis):
    """d f / d(coordinate) along ``axis``; ``spacing`` scalar or coordinate array.

    ``spacing=None`` means the field does not vary along that direction (a
    single plane, or a direction with no array axis) -> zero derivative.
    """
    f = _inexact(f)
    if spacing is None:
        return np.zeros_like(f)
    edge = 2 if f.shape[axis] >= 3 else 1
    return np.gradient(f, spacing, axis=axis, edge_order=edge)


def curl(Fx, Fy, Fz, dx, dy, dz, axes=(-1, -2, -3)):
    """Curl of a vector field on a grid -> ``(cx, cy, cz)``.

    ``dx, dy, dz`` are the grid spacings (scalar) or 1-D coordinate arrays
    (non-uniform grids), in m. Pass ``None`` for a direction the data do not
    resolve -- e.g. ``dy=None`` for an x-z plane -- and derivatives along it are
    taken as zero (its entry in ``axes`` is then ignored). A component that is
    identically zero may be given as a scalar ``0``.

    Second-order central differences inside, second-order one-sided at the
    edges (``np.gradient``); edge values are the least reliable, so mask them
    where the probe drive or chamber wall cuts the grid.
    """
    ax, ay, az = axes
    shape = np.broadcast_shapes(*(np.shape(F) for F in (Fx, Fy, Fz)))
    Fx, Fy, Fz = (np.broadcast_to(_inexact(F), shape) for F in (Fx, Fy, Fz))
    cx = _ddx(Fz, dy, ay) - _ddx(Fy, dz, az)
    cy = _ddx(Fx, dz, az) - _ddx(Fz, dx, ax)
    cz = _ddx(Fy, dx, ax) - _ddx(Fx, dy, ay)
    return cx, cy, cz


def current_density(Bx, By, Bz, dx, dy, dz, axes=(-1, -2, -3)):
    """J = curl(B) / mu_0 (A/m^2) from B in tesla on a grid in m; see :func:`curl`."""
    return tuple(c / mu_0 for c in curl(Bx, By, Bz, dx, dy, dz, axes))


def poynting(Ex, Ey, Ez, Bx, By, Bz):
    """Poynting vector S = E x B / mu_0 (W/m^2). Zero components may be scalars."""
    Sx = (Ey * Bz - Ez * By) / mu_0
    Sy = (Ez * Bx - Ex * Bz) / mu_0
    Sz = (Ex * By - Ey * Bx) / mu_0
    return Sx, Sy, Sz


def power_density(Jx, Jy, Jz, Ex, Ey, Ez):
    """Power transferred from field to charges, J . E (W/m^3). Zero components may be scalars."""
    return Jx * Ex + Jy * Ey + Jz * Ez


# ---------------------------------------------------------------------------
# induced electric field, assuming symmetry about the axis
# ---------------------------------------------------------------------------

def _radial_cumint(g, r):
    """int_0^r r' g dr' along the last axis for ascending r >= 0 (from the axis)."""
    # The integrand r' g vanishes on the axis, so prepending (r=0, 0) makes the
    # integral start at the axis whether or not the grid includes r = 0.
    r0 = np.concatenate(([0.0], r))
    f = np.concatenate((np.zeros(g.shape[:-1] + (1,)), r * g), axis=-1)
    return integrate.cumulative_trapezoid(f, r0, axis=-1)


def faraday_ephi(dbz_dt, r, axis=-1):
    """Azimuthal E from dBz/dt, assuming symmetry about the axis.

    Faraday's law around a circle of radius r centred on the axis::

        2 pi r E_phi(r) = - d/dt int_0^r Bz 2 pi r' dr'
        =>  E_phi(r) = -(1/r) int_0^r r' dBz/dt dr'

    ``r`` (m) is the radial coordinate along ``axis`` of ``dbz_dt`` (T/s). It
    must contain the axis position or extend to it from one side; a line
    crossing the axis may carry signed radii (r < 0 on the far side), and each
    side is integrated outward from r = 0 separately. Returns E_phi (V/m), the
    component along phi-hat at each point, same shape as ``dbz_dt``; 0 on the
    axis. A uniform dBz/dt gives E_phi = -r dBz/dt / 2.
    """
    g = np.moveaxis(_inexact(dbz_dt), axis, -1)
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size != g.shape[-1]:
        raise ValueError(f"r must be 1-D with {g.shape[-1]} points along axis {axis}")
    E = np.zeros_like(g)
    for side in (r >= 0, r < 0):
        idx = np.flatnonzero(side)
        if idx.size == 0:
            continue
        rr = np.abs(r[idx])
        order = np.argsort(rr)
        idx, rr = idx[order], rr[order]
        flux = _radial_cumint(g[..., idx], rr)
        with np.errstate(divide="ignore", invalid="ignore"):
            E[..., idx] = np.where(rr > 0, -flux / rr, 0.0)
    return np.moveaxis(E, -1, axis)


# ---------------------------------------------------------------------------
# vector potential
# ---------------------------------------------------------------------------

def vector_potential(Jx, Jy, Jz, dx, dy, dz, axes=(-1, -2, -3)):
    """A = (mu_0 / 4 pi) int J(r') / |r - r'| dV'  (T m) on the same grid as J.

    Biot-Savart sum over the measured grid only: current outside the grid
    (e.g. in coils, or beyond the probe's reach) is not included, so A near
    the grid edge is incomplete. ``dx, dy, dz`` are uniform spacings (m).

    Evaluated as an FFT convolution with the 1/r kernel (O(N log N), seconds
    rather than hours for the old point-by-point sum). The kernel's
    singular self-cell term uses the mean of 1/r over a sphere of the cell's
    volume, 1.5 / R_eq, instead of dropping the cell (within ~2% of the exact
    cube value). Leading (non-spatial) axes such as time are looped over with
    one shared kernel transform.
    """
    comps = [_inexact(J) for J in (Jx, Jy, Jz)]
    shape = np.broadcast_shapes(*(c.shape for c in comps))
    comps = [np.broadcast_to(c, shape) for c in comps]
    ndim = len(shape)
    sp_axes = [a % ndim for a in axes]           # array axes of x, y, z
    if len(set(sp_axes)) != 3:
        raise ValueError(f"axes must name three distinct array axes; got {axes}")
    # spatial axes last, in (x, y, z) order
    comps = [np.moveaxis(c, sp_axes, (-3, -2, -1)) for c in comps]
    grid = comps[0].shape[-3:]
    lead = comps[0].shape[:-3]

    # 1/r kernel on offsets -(n-1)..(n-1) per axis
    d = (dx, dy, dz)
    offs = [np.arange(-(n - 1), n) * h for n, h in zip(grid, d)]
    X, Y, Z = np.meshgrid(*offs, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    dV = dx * dy * dz
    r_eq = (3 * dV / (4 * np.pi)) ** (1 / 3)
    with np.errstate(divide="ignore"):
        K = np.where(R > 0, 1.0 / R, 1.5 / r_eq)
    K *= mu_0 / (4 * np.pi) * dV

    fshape = [fft.next_fast_len(n + k - 1, real=True) for n, k in zip(grid, K.shape)]
    Kf = fft.rfftn(K, fshape)
    crop = tuple(slice(n - 1, 2 * n - 1) for n in grid)

    def conv(v):
        return fft.irfftn(fft.rfftn(v, fshape) * Kf, fshape)[crop]

    out = []
    for c in comps:
        flat = c.reshape((-1,) + grid)
        A = np.empty_like(flat)
        for i in range(flat.shape[0]):
            # the kernel is real, so complex (phasor) J convolves part by part
            A[i] = (conv(flat[i].real) + 1j * conv(flat[i].imag)) if np.iscomplexobj(flat) \
                else conv(flat[i])
        out.append(np.moveaxis(A.reshape(lead + grid), (-3, -2, -1), sp_axes))
    return tuple(out)


# ---------------------------------------------------------------------------
# averages over drive cycles and integrals over axisymmetric volumes
# ---------------------------------------------------------------------------

def cycle_means(x, period, start=0, n_cycles=None, axis=-1):
    """Mean of ``x`` over each successive drive cycle along ``axis``.

    ``period`` is samples per cycle (may be fractional); cycle k spans samples
    ``round(start + k*period)`` to ``round(start + (k+1)*period)``.
    ``n_cycles`` defaults to as many whole cycles as fit after ``start``.
    Returns an array with ``axis`` replaced by the cycle index -- e.g. the
    cycle-by-cycle J.E or |E| history of a drive that changes from cycle to cycle.
    """
    x = np.moveaxis(np.asarray(x), axis, -1)
    nt = x.shape[-1]
    if n_cycles is None:
        n_cycles = int((nt - start) // period)
    edges = np.round(start + np.arange(n_cycles + 1) * period).astype(int)
    if n_cycles < 1 or edges[0] < 0 or edges[-1] > nt:
        raise ValueError(f"{n_cycles} cycles of {period:g} samples from {start} "
                         f"do not fit in {nt} samples")
    means = np.stack([x[..., a:b].mean(axis=-1) for a, b in zip(edges[:-1], edges[1:])], axis=-1)
    return np.moveaxis(means, -1, axis)


def cycle_average(x, period, start=0, n_cycles=1, axis=-1):
    """Mean of ``x`` over ``n_cycles`` whole drive cycles from ``start`` (``axis`` removed)."""
    x = np.moveaxis(np.asarray(x), axis, -1)
    stop = int(round(start + n_cycles * period))
    if n_cycles <= 0 or start < 0 or stop > x.shape[-1]:
        raise ValueError(f"{n_cycles} cycles of {period:g} samples from {start} "
                         f"do not fit in {x.shape[-1]} samples")
    return x[..., int(start):stop].mean(axis=-1)


def axisymmetric_integral(f, r, z=None, r_axis=-1, z_axis=-2):
    """Integrate over the volume, assuming symmetry about the axis: int f 2 pi r dr [dz] (trapezoid).

    ``r`` (m, all >= 0) runs along ``r_axis``. With ``z`` (m) along ``z_axis``
    the result is the volume integral -- e.g. total absorbed power from a J.E
    map in an r-z plane; with ``z=None`` it is the integral over the cross-section
    at each remaining index. Data sampled on both sides of the axis must be
    reduced to one half (or averaged) first -- integrating both halves counts
    the volume twice.
    """
    f = _inexact(f)
    r = np.asarray(r, dtype=float)
    if np.any(r < 0):
        raise ValueError("r must be >= 0; select one side of the axis first")
    if z is None:
        g = np.moveaxis(f, r_axis, -1)
        return integrate.trapezoid(g * (2 * np.pi * r), r, axis=-1)
    g = np.moveaxis(f, (z_axis, r_axis), (-2, -1))
    radial = integrate.trapezoid(g * (2 * np.pi * r), r, axis=-1)
    return integrate.trapezoid(radial, np.asarray(z, dtype=float), axis=-1)


# ---------------------------------------------------------------------------
# complex response at the drive frequency
# ---------------------------------------------------------------------------

def phasor(x, dt, freq, axis=-1, window="hann"):
    """Complex amplitude of ``x`` at ``freq`` (Hz): x(t) ~ Re[X exp(i 2 pi f t)].

    A single-frequency DFT evaluated exactly at ``freq`` (no FFT-bin rounding),
    with t = 0 at the first sample. ``window`` ('hann', None, or any
    ``scipy.signal.get_window`` name) suppresses leakage when the record is not
    a whole number of cycles; the amplitude is normalised by the window sum,
    so a pure cosine of amplitude a returns |X| = a.
    """
    from scipy.signal import get_window

    x = np.moveaxis(_inexact(x), axis, -1)
    n = x.shape[-1]
    w = np.ones(n) if window is None else get_window(window, n, fftbins=False)
    e = w * np.exp(-2j * np.pi * freq * dt * np.arange(n))
    return 2 * (x @ e) / w.sum()


def ac_conductivity(J, E, dt, freq, axis=-1, min_E=None, window="hann"):
    """Complex conductivity sigma = J(f) / E(f) (S/m) at the drive frequency.

    ``J`` (A/m^2) and ``E`` (V/m) are matching components sampled on the same
    time axis ``axis``. ``abs(sigma)`` is the magnitude and
    ``np.angle(sigma, deg=True)`` the phase by which J leads E (a collisional
    plasma sigma = eps0 wpe^2 / (nu + i w) gives a negative phase -- the
    electrons' inertia makes J lag E). Where |E(f)| < ``min_E`` (V/m), the
    ratio is noise and NaN is returned.
    """
    Jf = phasor(J, dt, freq, axis, window)
    Ef = phasor(E, dt, freq, axis, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = Jf / Ef
    if min_E is not None:
        sigma = np.where(np.abs(Ef) < min_E, np.nan + 0j, sigma)
    return sigma
