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
- :func:`amplitude_normalized_cross_phase`: causal unit-phasor cross-phase
  that prevents saturated tail power from dominating linear growth ramps.
- :func:`harmonic_steepening`: bispectral biphase ``beta = phi_2 - 2*phi_1``
  decomposition separating symmetric wall-curvature peakedness
  ``(A_2/A_1)*cos(beta)`` from true antisymmetric sawtooth steepening
  ``(A_2/A_1)*sin(beta)``, alongside Elgar-Guza Hilbert asymmetry,
  rise-vs-fall slope ratio, rise-time fraction, and a phase-folded cycle
  waveform normalized to a unit-amplitude fundamental ``cos(theta)``.
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


class HarmonicSteepening(NamedTuple):
    """Bispectral biphase and waveform-asymmetry decomposition of a rotating mode."""
    a2_a1_med: float
    a3_a1_med: float
    biphase_lock: float            # |<B>| / <|B|> in [0, 1], B = X(2f1) * conj(X(f1))^2
    biphase_lock_crit: float       # Rayleigh random-phase threshold (~lock_sigma * sqrt(pi/(4N)))
    biphase_locked: bool           # True when n_shots >= 4 and biphase_lock >= biphase_lock_crit
    biphase_deg: float             # beta = phi_2 - 2*phi_1 in (-180, 180] (NaN if gated unlocked)
    curvature_comp: float          # (A2/A1) * cos(beta): symmetric peakedness (NaN if gated unlocked)
    steepening_comp: float         # (A2/A1) * sin(beta): sawtooth steepening (NaN if gated unlocked)
    skewness_med: float            # harmonic-band waveform skewness S
    hilbert_asymmetry_med: float   # Elgar-Guza skew(Im(hilbert(x))) (<0 steep rise, >0 steep fall)
    slope_ratio_med: float         # p95(dx>0) / p95(-dx[dx<0]) (>1 steep rise, <1 steep fall)
    rise_frac_med: float           # fraction of samples with dx > 0 (<0.5 steep rise)
    phase_rad: np.ndarray          # fundamental-phase bin centers in (-pi, pi)
    folded_wave: np.ndarray        # phase-folded waveform normalized so fundamental = cos(phase_rad)
    folded_std: np.ndarray         # across-shot std of folded waveform in each phase bin
    n_shots: int


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


def harmonic_steepening(
    stack,
    fs_hz,
    f0_hz,
    *,
    n_harmonics=4,
    f0_search_hz=600.0,
    slope_pct=95.0,
    n_phase_bins=64,
    edge_trim_frac=0.10,
    lock_sigma=3.0,
    gate_unlocked=True,
):
    """Quantify nonlinear sawtooth steepening vs. symmetric peakedness of a harmonic mode.

    For a harmonic waveform ``A_1 cos(wt + phi_1) + A_2 cos(2wt + phi_2) + ...``,
    the complex bispectrum ``B = X(2 f_1) * conj(X(f_1))^2`` yields the biphase
    ``beta = phi_2 - 2*phi_1`` (invariant under per-shot time/phase shifts and
    symmetric windowing):

    - ``beta = 0 deg`` or ``180 deg`` (``cos(beta) = +-1``, ``sin(beta) = 0``):
      symmetric peaked crests or deep troughs (e.g. curved background density
      wall ``n_0''(r)``), giving skewness ``S != 0`` but equal rise/fall slopes
      and zero Hilbert asymmetry.
    - ``beta = +90 deg`` (``sin(beta) = +1``): steep rising front / slow falling
      tail (``slope_ratio > 1``, ``rise_frac < 0.5``, ``hilbert_asymmetry < 0``).
    - ``beta = -90 deg`` (``sin(beta) = -1``): slow rise / steep falling back
      (``slope_ratio < 1``, ``rise_frac > 0.5``, ``hilbert_asymmetry > 0``).

    Multi-shot phase-locking significance is checked against the Rayleigh
    random-phase expectation ``E[R] = sqrt(pi / (4 N))``: when ``n_shots >= 4``,
    ``biphase_lock_crit = min(0.95, lock_sigma * sqrt(pi / (4 N)))`` (at
    ``lock_sigma = 3.0``, ``~2.66 / sqrt(N)``, Rayleigh ``p < 1e-3``). When
    ``gate_unlocked=True`` and ``biphase_lock < biphase_lock_crit`` (for
    ``n_shots >= 4``), ``biphase_deg``, ``curvature_comp``, and
    ``steepening_comp`` are set to ``NaN`` so random-phase noise is never
    plotted or tabulated as locked steepening.

    Caveat: if ``f_0`` chirps appreciably across a long analysis window, the
    ``2 f_0`` spectral line spreads twice as fast as ``f_0``, pulling the
    single-bin peak ratio ``a2_a1_med`` slightly low.

    Also constructs a fundamental-phase-folded cycle waveform normalized per
    shot by ``sqrt(2) * std(x_1)`` so a pure sinusoid maps to ``cos(phase_rad)``
    of exact unit amplitude.

    Args:
        stack: 1-D ``(nt,)`` or 2-D ``(nshot, nt)`` time-series array.
        fs_hz: Sampling rate [Hz].
        f0_hz: Nominal fundamental frequency [Hz].
        n_harmonics: Upper harmonic index retained in the zero-phase harmonic
            bandpass ``[0.5*f0, (n_harmonics + 0.5)*f0]``.
        f0_search_hz: Half-width [Hz] around ``f0_hz`` searched per shot for
            the exact spectral peak ``f_1`` (zero-padded FFT).
        slope_pct: Percentile of positive vs. negative ``dx`` used for the
            robust rise/fall slope ratio (default 95).
        n_phase_bins: Number of azimuthal/phase bins on ``[-pi, pi)`` for the
            phase-folded cycle waveform.
        edge_trim_frac: Fraction trimmed from each end after ``sosfiltfilt``
            to eliminate filter boundary transients.
        lock_sigma: Multiple of the Rayleigh random-phase mean
            ``sqrt(pi / (4 N))`` required to declare ``biphase_locked=True``
            (default ``3.0`` -> ``~2.66 / sqrt(N)``).
        gate_unlocked: When ``True`` (default) and ``n_shots >= 4``, blank
            ``biphase_deg``, ``curvature_comp``, and ``steepening_comp`` to
            ``NaN`` whenever ``biphase_lock < biphase_lock_crit``.

    Returns:
        :class:`HarmonicSteepening`.
    """
    arr = np.asarray(stack, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim != 2:
        raise ValueError(f"stack must be 1-D or 2-D, got shape {arr.shape}")

    nb = int(n_phase_bins)
    if nb < 8:
        raise ValueError(f"n_phase_bins must be >= 8, got {n_phase_bins}")
    phase_edges = np.linspace(-np.pi, np.pi, nb + 1)
    phase_centers = 0.5 * (phase_edges[:-1] + phase_edges[1:])

    def _empty():
        nan_vec = np.full(nb, np.nan, dtype=float)
        return HarmonicSteepening(
            a2_a1_med=np.nan,
            a3_a1_med=np.nan,
            biphase_lock=np.nan,
            biphase_lock_crit=np.nan,
            biphase_locked=False,
            biphase_deg=np.nan,
            curvature_comp=np.nan,
            steepening_comp=np.nan,
            skewness_med=np.nan,
            hilbert_asymmetry_med=np.nan,
            slope_ratio_med=np.nan,
            rise_frac_med=np.nan,
            phase_rad=phase_centers,
            folded_wave=nan_vec.copy(),
            folded_std=nan_vec.copy(),
            n_shots=0,
        )

    fs = float(fs_hz)
    f0 = float(f0_hz) if f0_hz is not None else np.nan
    nsh, nt = arr.shape
    if nt < 24 or not np.isfinite(fs) or fs <= 0.0 or not np.isfinite(f0) or f0 <= 0.0:
        return _empty()
    if 2.0 * f0 >= 0.48 * fs:
        return _empty()

    f_lo_h = max(0.5 * f0, min(500.0, 0.25 * f0))
    f_hi_h = min((float(n_harmonics) + 0.5) * f0, 0.45 * fs)
    f_lo_1 = max(0.65 * f0, f_lo_h)
    f_hi_1 = min(1.35 * f0, 0.45 * fs)
    if not (0.0 < f_lo_h < f_hi_h < 0.5 * fs and 0.0 < f_lo_1 < f_hi_1 < 0.5 * fs):
        return _empty()

    sos_harm = signal.butter(4, [f_lo_h, f_hi_h], btype="bandpass", fs=fs, output="sos")
    sos_f1 = signal.butter(4, [f_lo_1, f_hi_1], btype="bandpass", fs=fs, output="sos")

    t_lin = np.linspace(-1.0, 1.0, nt)
    denom_t = float(np.dot(t_lin, t_lin))
    win = np.hanning(nt)
    nfft = max(2048, 1 << int(np.ceil(np.log2(max(4 * nt, 2048)))))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    search_hw = max(float(f0_search_hz), 0.15 * f0)
    idx1_cand = np.flatnonzero(np.abs(freqs - f0) <= search_hw)
    if idx1_cand.size == 0:
        return _empty()

    trim = min(max(8, int(float(edge_trim_frac) * nt)), nt // 4)
    B_list, a21_l, a31_l = [], [], []
    S_l, A_l, sr_l, rf_l = [], [], [], []
    folded_rows = []

    for k in range(nsh):
        row = arr[k]
        if not np.all(np.isfinite(row)):
            continue
        mu = float(np.mean(row))
        sl = float(np.dot(t_lin, row - mu) / denom_t)
        det = row - (mu + sl * t_lin)

        X = np.fft.rfft(det * win, n=nfft)
        kp1 = int(idx1_cand[int(np.argmax(np.abs(X[idx1_cand])))])
        kp2 = 2 * kp1
        kp3 = 3 * kp1
        if kp2 >= X.size or abs(X[kp1]) <= 0.0:
            continue
        x1, x2 = X[kp1], X[kp2]
        x3 = X[kp3] if kp3 < X.size else 0.0
        B_list.append(x2 * np.conj(x1) ** 2)
        a21_l.append(float(abs(x2) / abs(x1)))
        a31_l.append(float(abs(x3) / abs(x1)))

        xh = signal.sosfiltfilt(sos_harm, det)
        x1_t = signal.sosfiltfilt(sos_f1, det)
        xc = xh[trim:-trim] if nt > 2 * trim + 8 else xh
        x1c = x1_t[trim:-trim] if nt > 2 * trim + 8 else x1_t
        if xc.size < 8 or float(np.std(xc)) <= 0.0:
            continue

        S_l.append(float(stats.skew(xc)))
        A_l.append(float(stats.skew(np.imag(signal.hilbert(xc)))))
        dx = np.diff(xc)
        pos_dx = dx[dx > 0.0]
        neg_dx = -dx[dx < 0.0]
        if pos_dx.size and neg_dx.size:
            pos_s = float(np.percentile(pos_dx, float(slope_pct)))
            neg_s = float(np.percentile(neg_dx, float(slope_pct)))
            if neg_s > 0.0:
                sr_l.append(pos_s / neg_s)
        rf_l.append(float(np.mean(dx > 0.0)))

        # Phase-fold onto fundamental phase theta in [-pi, pi), normalized so fundamental ~ cos(theta).
        amp1 = float(np.sqrt(2.0) * np.std(x1c))
        if amp1 > 0.0:
            theta = np.angle(signal.hilbert(x1c))
            b_idx = np.clip(np.digitize(theta, phase_edges) - 1, 0, nb - 1)
            counts = np.bincount(b_idx, minlength=nb)
            sums = np.bincount(b_idx, weights=xc / amp1, minlength=nb)
            row_fold = np.where(counts > 0, sums / np.where(counts > 0, counts, 1), np.nan)
            folded_rows.append(row_fold)

    if not B_list or not S_l:
        return _empty()

    B_arr = np.asarray(B_list, dtype=complex)
    n_b = int(B_arr.size)
    b_denom = float(np.mean(np.abs(B_arr)))
    b_lock = float(abs(np.mean(B_arr)) / b_denom) if b_denom > 0.0 else np.nan
    b_crit = float(min(0.95, float(lock_sigma) * np.sqrt(np.pi / (4.0 * n_b)))) if n_b >= 4 else np.nan
    b_locked = bool(n_b >= 4 and np.isfinite(b_lock) and b_lock >= b_crit)
    beta_deg = float(np.degrees(np.angle(np.mean(B_arr)))) if b_denom > 0.0 else np.nan
    a21_med = float(np.median(a21_l))
    a31_med = float(np.median(a31_l))
    curv = a21_med * float(np.cos(np.radians(beta_deg))) if np.isfinite(beta_deg) else np.nan
    steep = a21_med * float(np.sin(np.radians(beta_deg))) if np.isfinite(beta_deg) else np.nan
    if gate_unlocked and n_b >= 4 and not b_locked:
        beta_deg = np.nan
        curv = np.nan
        steep = np.nan

    if folded_rows:
        fm = np.vstack(folded_rows)
        cnt = np.sum(np.isfinite(fm), axis=0)
        f_mean = np.where(cnt > 0, np.nansum(fm, axis=0) / np.where(cnt > 0, cnt, 1), np.nan)
        diff2 = np.where(np.isfinite(fm), (fm - f_mean[None, :]) ** 2, 0.0)
        f_std = np.where(cnt > 1, np.sqrt(np.sum(diff2, axis=0) / np.where(cnt > 1, cnt - 1, 1)), 0.0)
    else:
        f_mean = np.full(nb, np.nan, dtype=float)
        f_std = np.full(nb, np.nan, dtype=float)

    return HarmonicSteepening(
        a2_a1_med=a21_med,
        a3_a1_med=a31_med,
        biphase_lock=b_lock,
        biphase_lock_crit=b_crit,
        biphase_locked=b_locked,
        biphase_deg=beta_deg,
        curvature_comp=curv,
        steepening_comp=steep,
        skewness_med=float(np.median(S_l)),
        hilbert_asymmetry_med=float(np.median(A_l)),
        slope_ratio_med=float(np.median(sr_l)) if sr_l else np.nan,
        rise_frac_med=float(np.median(rf_l)) if rf_l else np.nan,
        phase_rad=phase_centers,
        folded_wave=f_mean,
        folded_std=f_std,
        n_shots=int(len(S_l)),
    )
