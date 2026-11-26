"""Diagnostic-agnostic broadband spectrum, intermittency, and shear-transport helpers.

Provides:
- :func:`fit_broadband_spectrum`: harmonic-power fraction and exponential
  (Pace / Maggs-Morales Lorentzian-pulse ``P(f) ~ exp(-f / f_e)``) vs.
  power-law (``P(f) ~ f^-alpha``) continuum comparison strictly above the
  background noise floor.
- :func:`fluctuation_moments`: per-shot linearly detrended relative RMS,
  skewness, and excess kurtosis (Fisher convention: pure sinusoid = ``-1.5``,
  Gaussian = ``0.0``).
- :func:`reynolds_shear_production`: cylindrical rotational shear
  ``gamma_E = r * d(v_theta / r) / dr``, local Reynolds-stress energy
  production ``P_shear = -<dv_r dv_theta> * gamma_E`` (positive = extraction
  from mean flow to waves; negative = wave feeds mean flow), and radially
  integrated ``integral(P_shear * r dr)``.
"""

from typing import NamedTuple

import numpy as np
from scipy import signal, stats

# np.trapz was renamed np.trapezoid in numpy 2.0; fall back for numpy < 2.0.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


class SpectrumModelFit(NamedTuple):
    """Harmonic power fraction and exponential vs. power-law broadband continuum fits."""
    coh_harmonic_frac: float
    n_fit_bins: int
    f_fit_lo_hz: float
    f_fit_hi_hz: float
    r2_exp: float
    f_e_hz: float
    tau_lorentz_s: float
    r2_pow: float
    alpha: float
    preferred_model: str  # "power_law", "exponential", "tie", or "insufficient_data"


class FluctuationMoments(NamedTuple):
    """Ensemble summary of per-shot linearly detrended relative fluctuation PDFs."""
    rel_rms_med: float
    skewness_med: float
    skewness_iqr: tuple[float, float]
    excess_kurtosis_med: float
    excess_kurtosis_iqr: tuple[float, float]
    n_shots: int


class ReynoldsProduction(NamedTuple):
    """Cylindrical shear-flow energy transfer from Reynolds stress."""
    r_cm: np.ndarray
    omega_per_cm: np.ndarray
    shear_per_cm: np.ndarray       # gamma_E = r * d(v_theta / r) / dr
    p_shear_local: np.ndarray      # -reynolds_stress * gamma_E (>0: mean->wave; <0: wave->mean)
    p_shear_integrated: float      # integral(p_shear_local * r dr)


def fit_broadband_spectrum(
    freqs_hz,
    psd,
    *,
    bg_psd=None,
    total_band_hz=(1000.0, 80000.0),
    f_min_hz=10000.0,
    f_max_hz=70000.0,
    snr_min=5.0,
    harmonic_f0_hz=None,
    n_harmonics=4,
    harm_halfwidth_hz=1200.0,
    min_points=6,
    r2_tie_tol=0.01,
):
    """Compare exponential vs. power-law broadband continuum fits outside harmonic lines.

    Args:
        freqs_hz: 1-D frequency axis [Hz].
        psd: 1-D power spectral density (same length as ``freqs_hz``).
        bg_psd: Optional reference/background PSD on the same grid (e.g.
            pre-bias ambient plasma turbulence, not necessarily instrument dark
            noise); when given, only bins with ``psd > snr_min * bg_psd`` enter
            the continuum fit.
        total_band_hz: ``(f_lo, f_hi)`` [Hz] used for ``coh_harmonic_frac``.
        f_min_hz, f_max_hz: Continuum fit bounds [Hz]. When ``harmonic_f0_hz``
            is finite, the lower bound is raised to at least
            ``max(f_min_hz, (n_harmonics - 0.5) * harmonic_f0_hz)`` so discrete
            low-order harmonics do not bias the broadband tail.
        snr_min: Minimum ``psd / bg_psd`` ratio required per fit bin.
        harmonic_f0_hz: Fundamental mode frequency [Hz] (optional).
        n_harmonics: Number of harmonics ``1..n_harmonics`` masked as coherent.
        harm_halfwidth_hz: Half-width [Hz] around each ``n * f0``.
        min_points: Minimum finite bins required to perform the continuum fit.
        r2_tie_tol: Minimum ``|R2_pow - R2_exp|`` to declare a preferred model.

    Returns:
        :class:`SpectrumModelFit`.
    """
    f = np.asarray(freqs_hz, dtype=float)
    p = np.asarray(psd, dtype=float)
    if f.ndim != 1 or p.ndim != 1 or f.size != p.size or f.size < 3:
        raise ValueError("freqs_hz and psd must be matching 1-D arrays with len >= 3")

    m_harm = np.zeros_like(f, dtype=bool)
    f0 = float(harmonic_f0_hz) if harmonic_f0_hz is not None else np.nan
    if np.isfinite(f0) and f0 > 0.0 and int(n_harmonics) > 0:
        for nh in range(1, int(n_harmonics) + 1):
            m_harm |= np.abs(f - nh * f0) <= float(harm_halfwidth_hz)

    m_tot = np.isfinite(p) & (p > 0.0) & (f >= float(total_band_hz[0])) & (f <= float(total_band_hz[1]))
    p_tot = float(np.sum(p[m_tot]))
    coh_frac = float(np.sum(p[m_tot & m_harm]) / p_tot) if p_tot > 0.0 else np.nan

    f_lo_eff = float(f_min_hz)
    if np.isfinite(f0) and f0 > 0.0 and int(n_harmonics) > 0:
        f_lo_eff = max(f_lo_eff, (float(n_harmonics) - 0.5) * f0)

    m_fit = np.isfinite(p) & (p > 0.0) & (f >= f_lo_eff) & (f <= float(f_max_hz)) & (~m_harm)
    if bg_psd is not None:
        bg = np.asarray(bg_psd, dtype=float)
        if bg.shape != p.shape:
            raise ValueError("bg_psd must have the same shape as psd")
        m_fit &= np.isfinite(bg) & (p > float(snr_min) * np.maximum(bg, 0.0))

    n_fit = int(np.sum(m_fit))
    if n_fit < int(min_points):
        return SpectrumModelFit(
            coh_harmonic_frac=coh_frac,
            n_fit_bins=n_fit,
            f_fit_lo_hz=np.nan,
            f_fit_hi_hz=np.nan,
            r2_exp=np.nan,
            f_e_hz=np.nan,
            tau_lorentz_s=np.nan,
            r2_pow=np.nan,
            alpha=np.nan,
            preferred_model="insufficient_data",
        )

    ff = f[m_fit]
    lp = np.log(p[m_fit])
    ss_tot = float(np.sum((lp - float(np.mean(lp))) ** 2))
    if ss_tot <= 0.0:
        return SpectrumModelFit(
            coh_harmonic_frac=coh_frac,
            n_fit_bins=n_fit,
            f_fit_lo_hz=float(ff[0]),
            f_fit_hi_hz=float(ff[-1]),
            r2_exp=np.nan,
            f_e_hz=np.nan,
            tau_lorentz_s=np.nan,
            r2_pow=np.nan,
            alpha=np.nan,
            preferred_model="insufficient_data",
        )

    # Exponential: ln P = a - f / f_e
    ce = np.polyfit(ff, lp, 1)
    r2_exp = float(1.0 - np.sum((lp - np.polyval(ce, ff)) ** 2) / ss_tot)
    f_e_hz = float(-1.0 / ce[0]) if ce[0] < 0.0 else np.nan
    tau_lorentz_s = float(1.0 / (4.0 * np.pi * f_e_hz)) if np.isfinite(f_e_hz) and f_e_hz > 0.0 else np.nan

    # Power-law: ln P = b - alpha * ln(f)
    lf = np.log(ff)
    cp = np.polyfit(lf, lp, 1)
    r2_pow = float(1.0 - np.sum((lp - np.polyval(cp, lf)) ** 2) / ss_tot)
    alpha = float(-cp[0])

    if not (np.isfinite(r2_exp) and np.isfinite(r2_pow)):
        pref = "insufficient_data"
    elif r2_pow - r2_exp > float(r2_tie_tol):
        pref = "power_law"
    elif r2_exp - r2_pow > float(r2_tie_tol):
        pref = "exponential"
    else:
        pref = "tie"

    return SpectrumModelFit(
        coh_harmonic_frac=coh_frac,
        n_fit_bins=n_fit,
        f_fit_lo_hz=float(ff[0]),
        f_fit_hi_hz=float(ff[-1]),
        r2_exp=r2_exp,
        f_e_hz=f_e_hz,
        tau_lorentz_s=tau_lorentz_s,
        r2_pow=r2_pow,
        alpha=alpha,
        preferred_model=pref,
    )


def fluctuation_moments(stack, *, normalize_by_mean=True, detrend=True):
    """Per-shot relative RMS, skewness, and excess kurtosis (Fisher convention).

    Args:
        stack: 1-D ``(nt,)`` or 2-D ``(nshot, nt)`` time-series array.
        normalize_by_mean: Divide detrended fluctuations by ``|shot_mean|`` when
            computing ``rel_rms_med`` (has no effect on scale-invariant skewness
            and kurtosis).
        detrend: Remove a linear ramp per shot before computing moments.

    Returns:
        :class:`FluctuationMoments`.
    """
    arr = np.asarray(stack, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"stack must be 1-D or 2-D, got shape {arr.shape}")
    if arr.shape[1] < 4:
        raise ValueError(f"need at least 4 time samples, got {arr.shape[1]}")

    t_lin = np.linspace(-1.0, 1.0, arr.shape[1])
    denom_t = float(np.dot(t_lin, t_lin))

    rms_l, skew_l, kurt_l = [], [], []
    for k in range(arr.shape[0]):
        row = arr[k]
        if not np.all(np.isfinite(row)):
            continue
        mu = float(np.mean(row))
        if detrend:
            sl = float(np.dot(t_lin, row - mu) / denom_t)
            resid = row - (mu + sl * t_lin)
        else:
            resid = row - mu
        std = float(np.std(resid))
        if std <= 0.0:
            continue
        if normalize_by_mean:
            if abs(mu) <= 0.0:
                continue
            rms_l.append(std / abs(mu))
        else:
            rms_l.append(std)
        skew_l.append(float(stats.skew(resid)))
        kurt_l.append(float(stats.kurtosis(resid, fisher=True)))

    if not rms_l:
        return FluctuationMoments(
            rel_rms_med=np.nan,
            skewness_med=np.nan,
            skewness_iqr=(np.nan, np.nan),
            excess_kurtosis_med=np.nan,
            excess_kurtosis_iqr=(np.nan, np.nan),
            n_shots=0,
        )

    sk = np.asarray(skew_l, dtype=float)
    ku = np.asarray(kurt_l, dtype=float)
    return FluctuationMoments(
        rel_rms_med=float(np.median(rms_l)),
        skewness_med=float(np.median(sk)),
        skewness_iqr=(float(np.percentile(sk, 25)), float(np.percentile(sk, 75))),
        excess_kurtosis_med=float(np.median(ku)),
        excess_kurtosis_iqr=(float(np.percentile(ku, 25)), float(np.percentile(ku, 75))),
        n_shots=int(len(rms_l)),
    )


def reynolds_shear_production(r_cm, reynolds_stress, mean_v_theta):
    """Cylindrical shear-flow energy transfer rate from turbulent Reynolds stress.

    Computes::

        Omega(r) = mean_v_theta(r) / r
        gamma_E(r) = r * dOmega / dr
        P_shear(r) = -reynolds_stress(r) * gamma_E(r)

    By cylindrical integration by parts of the mean azimuthal kinetic-energy
    equation (vanishing boundary stress), ``P_shear > 0`` transfers energy
    **from mean sheared flow to fluctuations** (Kelvin-Helmholtz drive), while
    ``P_shear < 0`` transfers energy **from fluctuations into mean sheared
    flow** (stabilizing / zonal-flow drive).

    Args:
        r_cm: Strictly increasing 1-D positive radial coordinates [cm] (len >= 2).
        reynolds_stress: 1-D radial profile of ``<dv_r * dv_theta>`` (same units
            squared as ``mean_v_theta``, e.g. Mach^2 or (m/s)^2).
        mean_v_theta: 1-D radial profile of mean azimuthal velocity ``v_theta``
            (or ``M_theta``).

    Returns:
        :class:`ReynoldsProduction`.
    """
    r = np.asarray(r_cm, dtype=float)
    rs = np.asarray(reynolds_stress, dtype=float)
    vt = np.asarray(mean_v_theta, dtype=float)
    if r.ndim != 1 or rs.shape != r.shape or vt.shape != r.shape or r.size < 2:
        raise ValueError("r_cm, reynolds_stress, and mean_v_theta must be matching 1-D arrays (len >= 2)")
    if not np.all(np.diff(r) > 0.0) or np.any(r <= 0.0):
        raise ValueError("r_cm must be strictly increasing and strictly positive")

    omega = vt / r
    domega_dr = np.gradient(omega, r)
    shear = r * domega_dr
    p_local = -rs * shear
    ok = np.isfinite(p_local)
    p_int = float(trapezoid(p_local[ok] * r[ok], r[ok])) if int(np.sum(ok)) >= 2 else np.nan
    return ReynoldsProduction(
        r_cm=r,
        omega_per_cm=omega,
        shear_per_cm=shear,
        p_shear_local=p_local,
        p_shear_integrated=p_int,
    )


def amplitude_normalized_cross_phase(
    sig_a,
    sig_b,
    fs_hz,
    f_band_hz,
    *,
    mask=None,
    min_amp_frac=0.05,
    filter_order=2,
):
    """Amplitude-normalized cross-phase and phase-locking value (PLV) in a frequency band.

    Unlike standard Welch CSD (which weights every time sample by ``|A(t)|^2``
    and is therefore dominated by the saturated end of a growing ramp), this
    truncates each shot at its last masked sample and applies an identical
    causal Butterworth bandpass (``sosfilt``) to both channels before forming
    the unit-phasor average of the analytic signals::

        Z_unit = mean( conj(z_a(t)) * z_b(t) / (|z_a(t)| * |z_b(t)|) )

    Because both channels share the identical causal filter phase response
    (which cancels in ``conj(z_a) * z_b``) and the trace is truncated at the
    end of ``mask``, future saturated power can never leak backward into an
    earlier weak linear-growth ramp. The sign convention
    ``angle(conj(z_a) * z_b)`` matches ``scipy.signal.csd(sig_a, sig_b)``
    (``phi_b - phi_a``).

    Args:
        sig_a, sig_b: Matching 1-D ``(nt,)`` or 2-D ``(nshot, nt)`` signals.
        fs_hz: Sampling rate [Hz].
        f_band_hz: ``(f_lo, f_hi)`` bandpass corners [Hz] (must satisfy
            ``0 < f_lo < f_hi < fs_hz / 2``).
        mask: Optional boolean array of shape ``(nt,)`` or ``(nshot, nt)``
            selecting samples to include (e.g. a per-shot linear growth ramp).
        min_amp_frac: Exclude samples where ``|z_a| < min_amp_frac * peak_a``
            or ``|z_b| < min_amp_frac * peak_b`` inside the masked region so
            near-zero noise does not dilute phase locking.
        filter_order: Causal Butterworth bandpass order (applied identically
            to both channels via ``sosfilt``).

    Returns:
        ``(plv, phi_deg, n_samples)`` where ``plv = |Z_unit| in [0, 1]`` and
        ``phi_deg = degrees(angle(Z_unit))`` in ``(-180, 180]``. Returns
        ``(NaN, NaN, 0)`` if no valid samples survive masking.
    """
    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)
    if a.shape != b.shape or a.ndim not in (1, 2):
        raise ValueError(f"sig_a and sig_b must have matching 1-D or 2-D shapes, got {a.shape} vs {b.shape}")
    if a.ndim == 1:
        a = a[None, :]
        b = b[None, :]
    nshot, nt = a.shape
    if nt < 16:
        return np.nan, np.nan, 0

    sos = signal.butter(int(filter_order), [float(f_band_hz[0]), float(f_band_hz[1])],
                        btype="bandpass", fs=float(fs_hz), output="sos")
    fin = np.isfinite(a) & np.isfinite(b)

    if mask is None:
        sel = fin.copy()
    else:
        m = np.asarray(mask, dtype=bool)
        if m.ndim == 1 and m.size == nt:
            sel = fin & m[None, :]
        elif m.shape == (nshot, nt):
            sel = fin & m
        else:
            raise ValueError(f"mask shape {m.shape} incompatible with signal shape {(nshot, nt)}")

    unit_phasors = []
    for k in range(nshot):
        mk = sel[k]
        idx_k = np.flatnonzero(mk)
        if idx_k.size == 0:
            continue
        # Truncate strictly at the last masked sample so no future saturated samples enter
        # either the bandpass filter or the Hilbert transform.
        end_k = int(idx_k[-1]) + 1
        if end_k < 16:
            continue
        fk = fin[k, :end_k]
        ak = a[k, :end_k]
        bk = b[k, :end_k]
        ak0 = np.where(fk, ak - float(np.mean(ak[fk])), 0.0)
        bk0 = np.where(fk, bk - float(np.mean(bk[fk])), 0.0)
        af = signal.sosfilt(sos, ak0)
        bf = signal.sosfilt(sos, bk0)
        za = signal.hilbert(af)
        zb = signal.hilbert(bf)
        aa = np.abs(za)
        ab = np.abs(zb)

        mk_sub = mk[:end_k]
        pa = float(np.max(aa[mk_sub]))
        pb = float(np.max(ab[mk_sub]))
        if pa <= 0.0 or pb <= 0.0:
            continue
        keep = mk_sub & (aa >= float(min_amp_frac) * pa) & (ab >= float(min_amp_frac) * pb)
        if not np.any(keep):
            continue
        z_k = (np.conj(za[keep]) * zb[keep]) / (aa[keep] * ab[keep])
        unit_phasors.append(z_k)

    if not unit_phasors:
        return np.nan, np.nan, 0
    all_z = np.concatenate(unit_phasors)
    z_mean = complex(np.mean(all_z))
    return float(abs(z_mean)), float(np.degrees(np.angle(z_mean))), int(all_z.size)
