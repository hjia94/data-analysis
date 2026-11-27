"""Azimuthal/axial mode decomposition and plasma-frame dispersion helpers.

Campaign-agnostic spectral-geometry and linear-dispersion helpers operating on
coherence/cross-phase arrays (from :func:`data_analysis.signal.avg_cross_spectrum`)
and local plasma profile parameters (from :mod:`data_analysis.plasma.profiles`).

Phase & sign conventions (verified against :func:`scipy.signal.csd`):
- A wave ``cos(m * theta + k_par * z - omega * t)`` with ``omega > 0`` rotates
  counter-clockwise (``+theta``) when ``m > 0`` and clockwise (``-theta``) when
  ``m < 0``.
- ``avg_cross_spectrum(ch_a, ch_b)`` returns ``phase = angle(<P_ab>)``, which
  equals ``+(m * theta)`` when ``ch_a`` is the moving probe (``moving_is_a=True``)
  and ``-(m * theta)`` when ``ch_b`` is the moving probe (``moving_is_a=False``).
- All returned mode numbers ``m`` and phase differences in this module are in
  the **physical** ``exp(i * (m * theta - omega * t))`` convention.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import constants as const

from data_analysis.plasma.profiles import local_drift_scales

M_ELECTRON = const.electron_mass
M_PROTON = const.proton_mass


class AzimuthalModeResult(NamedTuple):
    """Best-fit physical azimuthal mode number ``m`` over a 2-D scan plane."""
    m: int
    purity: float
    center_x_cm: float
    center_y_cm: float
    n_points: int
    purities: dict[int, float]


class AxialWavenumberResult(NamedTuple):
    """Circular-averaged axial phase difference and parallel wavenumber."""
    mean_dphi_rad: float
    mean_dphi_deg: float
    resultant_R: float
    k_par_rad_m: float
    lambda_par_m: float
    n_points: int


class BdotDipoleResult(NamedTuple):
    """Cartesian ``(Bx, By)`` phase quadrature test for a rotating ``J_z`` filament."""
    dphi_by_minus_bx_deg: float
    resultant_R: float
    inferred_m_j: int
    n_points: int


def _physical_phase(phase_rad, moving_is_a=True):
    """Convert ``avg_cross_spectrum`` phase to physical ``+phi_moving`` sign."""
    ph = np.asarray(phase_rad, dtype=float)
    return ph if moving_is_a else -ph


def fit_azimuthal_mode(pos_x_cm, pos_y_cm, gamma2, phase_rad, *,
                       m_values=(-4, -3, -2, -1, 0, 1, 2, 3, 4),
                       center=None,
                       search_x_cm=np.linspace(-2.0, 4.0, 13),
                       search_y_cm=np.linspace(-3.0, 3.0, 13),
                       r_min_cm=1.5, r_max_cm=8.0, gamma2_min=0.5,
                       moving_is_a=True, min_points=10):
    """Fit physical azimuthal mode number ``m`` from 2-D coherence/phase maps.

    Maximizes the coherence-weighted circular purity::

        P(m) = |sum_j w_j exp(i * (phi_phys_j - m * theta_j))| / sum_j w_j

    over candidate centers (or a fixed ``center=(cx, cy)``) and annulus
    ``r in [r_min_cm, r_max_cm]`` restricted to ``gamma2 >= gamma2_min``.
    Returns :class:`AzimuthalModeResult` with ``m=0, purity=NaN, n_points=0``
    if fewer than ``min_points`` valid samples exist.
    """
    px = np.asarray(pos_x_cm, dtype=float)
    py = np.asarray(pos_y_cm, dtype=float)
    g2 = np.asarray(gamma2, dtype=float)
    phi = _physical_phase(phase_rad, moving_is_a=moving_is_a)

    hi = np.isfinite(px) & np.isfinite(py) & np.isfinite(g2) & np.isfinite(phi) & (g2 >= gamma2_min)
    if int(np.sum(hi)) < min_points:
        return AzimuthalModeResult(0, np.nan, np.nan, np.nan, int(np.sum(hi)), {})

    px_h, py_h, g2_h, phi_h = px[hi], py[hi], g2[hi], phi[hi]
    if center is not None:
        centers = [(float(center[0]), float(center[1]))]
    else:
        centers = [(float(cx), float(cy)) for cx in search_x_cm for cy in search_y_cm]

    best = None
    for cx, cy in centers:
        r = np.hypot(px_h - cx, py_h - cy)
        ring = (r >= r_min_cm) & (r <= r_max_cm)
        n_ring = int(np.sum(ring))
        if n_ring < min_points:
            continue
        th = np.arctan2(py_h[ring] - cy, px_h[ring] - cx)
        w = g2_h[ring]
        w_sum = float(np.sum(w))
        if w_sum <= 0.0:
            continue
        ph_r = phi_h[ring]
        purities = {}
        for m in m_values:
            purities[int(m)] = float(np.abs(np.sum(w * np.exp(1j * (ph_r - m * th)))) / w_sum)
        top_m = max(purities, key=purities.get)
        score = purities[top_m]
        if best is None or score > best.purity:
            best = AzimuthalModeResult(top_m, score, cx, cy, n_ring, purities)

    if best is None:
        return AzimuthalModeResult(0, np.nan, np.nan, np.nan, 0, {})
    return best


def axial_wavenumber(phase_diff_rad, dz_m, gamma2_a=None, gamma2_b=None, *,
                     gamma2_min=0.6):
    """Estimate axial wavenumber ``k_par = dphi / dz`` via circular averaging.

    ``phase_diff_rad`` is the phase difference ``phi(z_a) - phi(z_b)`` [rad]
    across spatial points separated axially by ``dz_m = z_a - z_b`` [m].
    The sign of ``k_par`` follows the caller's ``+z`` convention (reversing
    ``dz_m`` flips ``k_par`` while leaving ``|k_par|`` and ``lambda_par``
    unchanged). Note that axial wavenumbers with ``|k_par * dz_m| > pi``
    alias into ``[-pi, pi]``. Circular statistics handle ``+-pi`` wrapping
    cleanly and report the resultant length ``R in [0, 1]`` (close to 1 means
    spatially uniform axial phase shift).
    """
    dz_m = float(dz_m)
    if dz_m == 0.0:
        raise ValueError("dz_m must be non-zero")
    dphi = np.asarray(phase_diff_rad, dtype=float)
    mask = np.isfinite(dphi)
    if gamma2_a is not None:
        ga = np.asarray(gamma2_a, dtype=float)
        mask &= np.isfinite(ga) & (ga >= gamma2_min)
    if gamma2_b is not None:
        gb = np.asarray(gamma2_b, dtype=float)
        mask &= np.isfinite(gb) & (gb >= gamma2_min)

    n = int(np.sum(mask))
    if n == 0:
        return AxialWavenumberResult(np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    z = np.mean(np.exp(1j * dphi[mask]))
    mean_rad = float(np.angle(z))
    R = float(np.abs(z))
    k_par = mean_rad / dz_m
    lam_par = (2.0 * np.pi / abs(k_par)) if abs(k_par) > 1e-12 else np.inf
    return AxialWavenumberResult(
        mean_dphi_rad=mean_rad,
        mean_dphi_deg=float(np.degrees(mean_rad)),
        resultant_R=R,
        k_par_rad_m=float(k_par),
        lambda_par_m=float(lam_par),
        n_points=n,
    )


def bdot_cartesian_dipole_check(phase_bx_rad, phase_by_rad,
                                gamma2_bx, gamma2_by, *,
                                gamma2_min=0.7, moving_is_a=False):
    """Check ``(Bx, By)`` quadrature phase for a rotating ``m_J = +-1`` current filament.

    Outside a rotating axial current perturbation ``J_z ~ cos(m_J * theta - w*t)``
    with ``|m_J| = 1``, the Biot-Savart magnetic field has Cartesian components
    rotating with azimuthal number ``m_B = 2 * m_J`` and constant phase lag:
    - ``m_J = +1`` (CCW): ``phi_phys(By) - phi_phys(Bx) = -90 deg``
    - ``m_J = -1`` (CW):  ``phi_phys(By) - phi_phys(Bx) = +90 deg``

    **Coil-polarity caveat:** inverting one coil's winding sign flips
    ``dphi_by_minus_bx_deg`` by 180 deg and reverses ``inferred_m_j``. By
    contrast, the single-coil azimuthal fit (:func:`fit_azimuthal_mode` on
    ``Bx`` or ``By`` alone) gives ``m_B = 2 * m_J`` independently of coil
    sign; verify the two agree to confirm right-handed ``(Bx, By)`` polarity.

    Returns :class:`BdotDipoleResult` with ``inferred_m_j in {-1, 0, +1}``
    (``0`` if the circular mean is not within 35 deg of ``+-90 deg`` or
    ``R < 0.5``).
    """
    pbx = _physical_phase(phase_bx_rad, moving_is_a=moving_is_a)
    pby = _physical_phase(phase_by_rad, moving_is_a=moving_is_a)
    gx = np.asarray(gamma2_bx, dtype=float)
    gy = np.asarray(gamma2_by, dtype=float)
    good = (
        np.isfinite(pbx) & np.isfinite(pby) & np.isfinite(gx) & np.isfinite(gy)
        & (gx >= gamma2_min) & (gy >= gamma2_min)
    )
    n = int(np.sum(good))
    if n == 0:
        return BdotDipoleResult(np.nan, np.nan, 0, 0)

    z = np.mean(np.exp(1j * (pby[good] - pbx[good])))
    dphi_deg = float(np.degrees(np.angle(z)))
    R = float(np.abs(z))
    if R >= 0.5 and abs(dphi_deg - 90.0) <= 35.0:
        inferred = -1
    elif R >= 0.5 and abs(dphi_deg + 90.0) <= 35.0:
        inferred = +1
    else:
        inferred = 0
    return BdotDipoleResult(dphi_deg, R, inferred, n)


def plasma_frame_dispersion(f_lab_hz, m, r_cm, v_rot_m_s, *,
                            b0_g, ne_cm3, te_ev,
                            dln_ne_dr_cm, dln_te_dr_cm,
                            k_par_m=None, mu=4.0, Z=1,
                            b_sign_z=+1):
    """Evaluate plasma-frame frequency and linear-mode scales at ``(r_cm, m)``.

    Sign conventions:
    - ``f_lab_hz > 0`` is the measured lab-frame mode frequency [Hz].
    - ``m`` is the signed physical azimuthal mode number (``m > 0`` CCW,
      ``m < 0`` CW) for ``exp(i * (m * theta - omega * t))``.
    - ``v_rot_m_s`` is the signed azimuthal plasma velocity ``v_theta`` [m/s]
      (positive CCW). The Doppler shift is ``f_doppler = m * v_rot / (2*pi*r)``,
      so ``f_plasma = f_lab - f_doppler``: when ``m`` and ``v_rot`` share a
      sign, the plasma rotates WITH the mode and ``f_plasma < f_lab``.
    - ``b_sign_z`` (``+1`` if ``B_0`` points along ``+z``, ``-1`` along ``-z``)
      sets the lab-frame sign of the electron diamagnetic drift
      ``v_*e = +(T_e / (e B)) (d ln p_e / dr) (r_hat x b_hat)`` where
      ``r_hat x (+z_hat) = -theta_hat``. Thus ``f_* = m * v_* / (2*pi*r)``
      in the same signed convention as ``f_plasma``.

    Returns a dict of frequencies [Hz] and dimensionless regime parameters
    (``k_perp_rho_s``, ``beta_e_over_me_mi``, ``f_alfven_hz``,
    ``nu_ei_over_omega_lab``). Passing ``f_lab_hz=np.nan`` (e.g., for stable
    runs with no coherent mode) leaves ``f_plasma_hz`` and
    ``nu_ei_over_omega_lab`` as ``NaN`` while still computing all
    frequency-independent Doppler, diamagnetic, and Alfvén scales.
    """
    r_m = float(r_cm) * 1e-2
    if r_m <= 0.0:
        raise ValueError(f"r_cm must be positive, got {r_cm}")
    k_theta_cm = abs(int(m)) / float(r_cm)
    k_theta_m = k_theta_cm * 1e2

    scales = local_drift_scales(
        ne_cm3, te_ev, b0_g, mu=mu, Z=Z,
        k_perp_cm=k_theta_cm,
        dln_ne_dr_cm=dln_ne_dr_cm,
        dln_te_dr_cm=dln_te_dr_cm,
    )

    f_rot_hz = float(v_rot_m_s) / (2.0 * np.pi * r_m)
    f_doppler_hz = int(m) * f_rot_hz
    f_plasma_hz = float(f_lab_hz) - f_doppler_hz

    # Signed diamagnetic velocities along +theta_hat:
    # r_hat x b_hat = -b_sign_z * theta_hat, so
    # v_*n,theta = -b_sign_z * (Te [eV] / B [T]) * d(ln ne)/dr [m^-1].
    B_T = float(b0_g) * 1e-4
    pref = -int(np.sign(b_sign_z or 1)) * (float(te_ev) / B_T)
    v_star_n_m_s = pref * (float(dln_ne_dr_cm) * 1e2)
    v_star_T_m_s = pref * (float(dln_te_dr_cm) * 1e2)
    v_star_p_m_s = v_star_n_m_s + v_star_T_m_s

    # Signed mode diamagnetic frequencies f_* = m * v_* / (2*pi*r):
    # positive means the mode propagates in the electron diamagnetic direction
    # for that gradient term.
    f_star_n_hz = int(m) * v_star_n_m_s / (2.0 * np.pi * r_m)
    f_star_T_hz = int(m) * v_star_T_m_s / (2.0 * np.pi * r_m)
    f_star_p_hz = f_star_n_hz + f_star_T_hz

    me_over_mi = M_ELECTRON / (float(mu) * M_PROTON)
    beta_e = float(scales["beta_e"])
    rho_s_cm = float(scales["rho_s_cm"])
    v_A = float(scales["v_alfven_m_s"])
    nu_ei = float(scales["nu_ei_s"])
    omega_lab = 2.0 * np.pi * float(f_lab_hz)

    if k_par_m is not None and np.isfinite(k_par_m):
        f_alfven_hz = abs(float(k_par_m)) * v_A / (2.0 * np.pi)
    else:
        f_alfven_hz = np.nan

    return {
        "f_lab_hz": float(f_lab_hz),
        "m": int(m),
        "r_cm": float(r_cm),
        "k_theta_m": float(k_theta_m),
        "f_rot_hz": float(f_rot_hz),
        "f_doppler_hz": float(f_doppler_hz),
        "f_plasma_hz": float(f_plasma_hz),
        "f_star_n_hz": float(f_star_n_hz),
        "f_star_T_hz": float(f_star_T_hz),
        "f_star_p_hz": float(f_star_p_hz),
        "f_alfven_hz": float(f_alfven_hz),
        "k_perp_rho_s": float(k_theta_cm * rho_s_cm),
        "beta_e": beta_e,
        "beta_e_over_me_mi": float(beta_e / me_over_mi) if np.isfinite(beta_e) else np.nan,
        "v_alfven_m_s": v_A,
        "nu_ei_s": nu_ei,
        "nu_ei_over_omega_lab": float(nu_ei / omega_lab) if omega_lab > 0 else np.nan,
    }
