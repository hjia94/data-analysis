"""Diagnostic-agnostic radial plasma profile, gradient, and scale helpers.

Computes electron pressure, signed spatial/radial logarithmic gradients with
analytic error propagation, floating potential and sheath factor from I-V
sweeps, and local magnetized-plasma scales (``rho_s``, ``nu_ei``, ``beta_e``,
``v_ExB``, and diamagnetic drift frequencies).

Unit conventions (kept explicit in parameter names and docstrings):
- positions: ``cm``
- density ``ne``: ``cm^-3`` (converted to ``m^-3`` by ``* 1e6`` internally)
- temperature ``Te``: ``eV``
- potentials ``Vp``, ``Vf``: ``V``
- electric field ``Er``: ``V/cm``
- magnetic field ``B0``: ``G`` (converted to ``T`` by ``* 1e-4`` internally)
- electron pressure ``pe``: ``Pa``
- inverse gradient scale lengths ``d ln(f) / dr``: ``cm^-1``
"""

from typing import NamedTuple

import numpy as np
from scipy import constants as const
from scipy.signal import savgol_coeffs, savgol_filter

# Exact CODATA elementary charge [J/eV = C]
E_CHARGE_J_PER_EV = const.e  # 1.602176634e-19
MU_0 = const.mu_0
M_PROTON = const.proton_mass


class ProfileGradients(NamedTuple):
    """Signed 1-D spatial gradients along ``x`` (and outward radius ``r`` if centered).

    All derivative arrays match the input shape. When ``x_center_cm`` is given,
    ``sign_r = sign(x - x_center_cm)`` (with ``0`` at exact center) converts
    signed ``d/dx`` into signed outward ``d/dr`` so left and right walls of a
    cylindrical column share the same radial sign convention.
    """
    pe_pa: np.ndarray
    pe_err_pa: np.ndarray | None
    dne_dx: np.ndarray           # [cm^-3 / cm]
    dte_dx: np.ndarray           # [eV / cm]
    dpe_dx: np.ndarray           # [Pa / cm]
    dln_ne_dx: np.ndarray        # [cm^-1], signed
    dln_te_dx: np.ndarray        # [cm^-1], signed
    dln_pe_dx: np.ndarray        # [cm^-1], signed (= dln_ne_dx + dln_te_dx)
    eta_e: np.ndarray            # dimensionless (d ln Te / dx) / (d ln ne / dx)
    er_v_per_cm: np.ndarray | None  # -dVp/dx [V/cm] (or -dVp/dr when centered)
    dln_ne_err: np.ndarray | None
    dln_te_err: np.ndarray | None
    dln_pe_err: np.ndarray | None
    er_err_v_per_cm: np.ndarray | None


def electron_pressure(ne_cm3, te_ev, ne_err_cm3=None, te_err_ev=None):
    """Electron pressure ``p_e [Pa] = e * (n_e [cm^-3] * 1e6) * T_e [eV]``.

    Non-positive ``ne`` or ``Te`` entries yield ``NaN``. When either
    ``ne_err_cm3`` or ``te_err_ev`` is provided, returns ``(pe_pa, pe_err_pa)``
    propagating uncorrelated uncertainties in quadrature; otherwise returns
    ``pe_pa`` alone.
    """
    ne = np.asarray(ne_cm3, dtype=float)
    te = np.asarray(te_ev, dtype=float)
    valid = np.isfinite(ne) & np.isfinite(te) & (ne > 0.0) & (te > 0.0)
    pe = np.where(valid, E_CHARGE_J_PER_EV * (ne * 1e6) * te, np.nan)

    if ne_err_cm3 is None and te_err_ev is None:
        return pe

    rel2 = np.zeros_like(pe, dtype=float)
    if ne_err_cm3 is not None:
        ne_err = np.asarray(ne_err_cm3, dtype=float)
        rel2 = rel2 + np.where(valid, (ne_err / ne) ** 2, np.nan)
    if te_err_ev is not None:
        te_err = np.asarray(te_err_ev, dtype=float)
        rel2 = rel2 + np.where(valid, (te_err / te) ** 2, np.nan)
    pe_err = np.where(valid, pe * np.sqrt(rel2), np.nan)
    return pe, pe_err


def _diff_and_error(x, y, y_err=None, *, smooth_window=1, polyorder=2, axis=0):
    """Differentiate ``y`` along ``axis`` on uniform ``x``, propagating ``y_err``.

    ``smooth_window <= 1`` uses 2nd-order central differences (``np.gradient``);
    odd ``smooth_window >= 3`` uses Savitzky-Golay 1st-derivative filtering.
    Error propagation uses the exact linear filter weights ``c_j`` so
    ``sigma(dy/dx) = sqrt(sum_j c_j^2 sigma_j^2)`` for interior points.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a 1-D array with at least 3 points")
    if y.shape[axis] != x.size:
        raise ValueError(f"x length ({x.size}) does not match y.shape[{axis}] ({y.shape[axis]})")

    dx_all = np.diff(x)
    if not np.allclose(dx_all, dx_all[0], rtol=1e-3, atol=1e-9):
        if smooth_window > 1:
            raise ValueError("Savitzky-Golay differentiation requires a uniform x grid")
    dx = float(dx_all[0])
    if dx == 0.0:
        raise ValueError("x grid spacing must be non-zero")

    ym = np.moveaxis(y, axis, 0)  # (nx, ...)
    nx = ym.shape[0]
    em = np.moveaxis(np.asarray(y_err, dtype=float), axis, 0) if y_err is not None else None

    if smooth_window <= 1:
        dy = np.gradient(ym, x, axis=0)
        if em is None:
            return np.moveaxis(dy, 0, axis), None
        # Analytic variance propagation for np.gradient (central interior, 1-sided ends).
        var = np.full_like(ym, np.nan, dtype=float)
        # Interior: (y[i+1] - y[i-1]) / (x[i+1] - x[i-1]) on uniform grid
        d2 = (x[2:] - x[:-2]).reshape((-1,) + (1,) * (ym.ndim - 1))
        var[1:-1] = (em[2:] ** 2 + em[:-2] ** 2) / (d2 ** 2)
        var[0] = (em[1] ** 2 + em[0] ** 2) / ((x[1] - x[0]) ** 2)
        var[-1] = (em[-1] ** 2 + em[-2] ** 2) / ((x[-1] - x[-2]) ** 2)
        return np.moveaxis(dy, 0, axis), np.moveaxis(np.sqrt(var), 0, axis)

    if smooth_window % 2 == 0 or smooth_window < 3:
        raise ValueError(f"smooth_window must be <= 1 or an odd integer >= 3, got {smooth_window}")
    if smooth_window > nx:
        raise ValueError(f"smooth_window ({smooth_window}) exceeds grid length ({nx})")
    if polyorder >= smooth_window:
        raise ValueError(f"polyorder ({polyorder}) must be < smooth_window ({smooth_window})")

    # Handle NaNs cleanly column-by-column by operating on each 1-D profile:
    # interior isolated NaNs cannot silently poison the entire trace if we
    # evaluate savgol_filter on finite slices, while leaving NaN locations NaN.
    dy = np.full_like(ym, np.nan, dtype=float)
    derr = np.full_like(ym, np.nan, dtype=float) if em is not None else None
    # use="dot" aligns coeffs[k] directly with forward window samples y[i0 + k].
    coeffs = savgol_coeffs(smooth_window, polyorder, deriv=1, delta=dx, use="dot")
    c2 = coeffs ** 2
    half = smooth_window // 2

    flat_y = ym.reshape(nx, -1)
    flat_dy = dy.reshape(nx, -1)
    flat_e = em.reshape(nx, -1) if em is not None else None
    flat_de = derr.reshape(nx, -1) if derr is not None else None

    for j in range(flat_y.shape[1]):
        col = flat_y[:, j]
        if np.all(np.isfinite(col)):
            flat_dy[:, j] = savgol_filter(col, smooth_window, polyorder, deriv=1, delta=dx)
        else:
            # Interpolate strictly inside finite bounds for the filter, then mask original NaNs.
            fin = np.flatnonzero(np.isfinite(col))
            if fin.size >= smooth_window:
                i0, i1 = int(fin[0]), int(fin[-1])
                if i1 - i0 + 1 >= smooth_window:
                    sub = np.interp(np.arange(i0, i1 + 1), fin, col[fin])
                    sub_d = savgol_filter(sub, smooth_window, polyorder, deriv=1, delta=dx)
                    out = np.full(nx, np.nan)
                    out[i0:i1 + 1] = sub_d
                    out[~np.isfinite(col)] = np.nan
                    flat_dy[:, j] = out

        if flat_e is not None:
            ecol = flat_e[:, j]
            e2 = np.where(np.isfinite(ecol), ecol ** 2, np.nan)
            # Exact dot-product variance propagation over interior and boundary
            # Savitzky-Golay polynomial fits (use="dot" matches forward slices).
            out_var = np.full(nx, np.nan)
            if nx >= smooth_window:
                # Interior: i in [half, nx - half - 1]
                win_e2 = np.lib.stride_tricks.sliding_window_view(e2, smooth_window)
                out_var[half:nx - half] = np.sum(win_e2 * c2[None, :], axis=1)
                # Left and right boundaries: exact pos-shifted dot-product coeffs
                for pos in range(half):
                    cl = savgol_coeffs(
                        smooth_window, polyorder, deriv=1, delta=dx, pos=pos, use="dot"
                    )
                    out_var[pos] = np.sum((cl ** 2) * e2[:smooth_window])
                    cr = savgol_coeffs(
                        smooth_window, polyorder, deriv=1, delta=dx,
                        pos=smooth_window - 1 - pos, use="dot",
                    )
                    out_var[nx - 1 - pos] = np.sum((cr ** 2) * e2[nx - smooth_window:])
            flat_de[:, j] = np.sqrt(out_var)

    return np.moveaxis(dy, 0, axis), (np.moveaxis(derr, 0, axis) if derr is not None else None)


def radial_profile_gradients(x_cm, ne_cm3, te_ev, vp_v=None, *,
                             ne_err=None, te_err=None, vp_err=None,
                             x_center_cm=None, smooth_window=1, polyorder=2,
                             axis=0, min_inv_len=1e-4):
    """Compute signed gradients of ``ne``, ``Te``, ``pe``, and ``Vp`` along ``x``.

    Preserves signs before any magnitude is taken, and computes
    ``d(ln p_e)/dr = d(ln n_e)/dr + d(ln T_e)/dr`` directly from the signed
    logarithmic derivatives so the exact thermodynamic identity holds to
    machine precision.

    When ``x_center_cm`` is provided, all derivatives and ``Er`` are multiplied
    by ``sign(x_cm - x_center_cm)`` (yielding outward radial derivatives
    ``d/dr`` on both sides of the column, with ``0`` at exact ``x = x_center_cm``).
    """
    x_cm = np.asarray(x_cm, dtype=float)
    ne = np.asarray(ne_cm3, dtype=float)
    te = np.asarray(te_ev, dtype=float)

    if ne_err is not None or te_err is not None:
        pe, pe_err = electron_pressure(ne, te, ne_err, te_err)
    else:
        pe = electron_pressure(ne, te)
        pe_err = None

    # Differentiate ln(ne) and ln(Te) directly so d(ln f)/dx is scale-invariant
    # and d(ln pe)/dx = d(ln ne)/dx + d(ln Te)/dx holds identically.
    valid_ne = np.isfinite(ne) & (ne > 0.0)
    valid_te = np.isfinite(te) & (te > 0.0)
    ln_ne = np.where(valid_ne, np.log(np.where(valid_ne, ne, 1.0)), np.nan)
    ln_te = np.where(valid_te, np.log(np.where(valid_te, te, 1.0)), np.nan)
    ln_ne_err = (np.where(valid_ne, np.asarray(ne_err, dtype=float) / ne, np.nan)
                 if ne_err is not None else None)
    ln_te_err = (np.where(valid_te, np.asarray(te_err, dtype=float) / te, np.nan)
                 if te_err is not None else None)

    dne_dx, _ = _diff_and_error(x_cm, ne, None, smooth_window=smooth_window, polyorder=polyorder, axis=axis)
    dte_dx, _ = _diff_and_error(x_cm, te, None, smooth_window=smooth_window, polyorder=polyorder, axis=axis)
    dpe_dx, _ = _diff_and_error(x_cm, pe, None, smooth_window=smooth_window, polyorder=polyorder, axis=axis)

    dln_ne, dln_ne_err = _diff_and_error(
        x_cm, ln_ne, ln_ne_err, smooth_window=smooth_window, polyorder=polyorder, axis=axis
    )
    dln_te, dln_te_err = _diff_and_error(
        x_cm, ln_te, ln_te_err, smooth_window=smooth_window, polyorder=polyorder, axis=axis
    )
    dln_pe = dln_ne + dln_te

    if dln_ne_err is not None or dln_te_err is not None:
        e_n2 = dln_ne_err ** 2 if dln_ne_err is not None else 0.0
        e_t2 = dln_te_err ** 2 if dln_te_err is not None else 0.0
        dln_pe_err = np.sqrt(e_n2 + e_t2)
    else:
        dln_pe_err = None

    er = er_err = None
    if vp_v is not None:
        dvp, dvp_err = _diff_and_error(
            x_cm, vp_v, vp_err, smooth_window=smooth_window, polyorder=polyorder, axis=axis
        )
        er = -dvp
        er_err = dvp_err

    if x_center_cm is not None:
        sgn = np.sign(x_cm - float(x_center_cm))
        shape = [1] * ne.ndim
        shape[axis] = x_cm.size
        sgn = sgn.reshape(shape)
        dne_dx = dne_dx * sgn
        dte_dx = dte_dx * sgn
        dpe_dx = dpe_dx * sgn
        dln_ne = dln_ne * sgn
        dln_te = dln_te * sgn
        dln_pe = dln_pe * sgn
        if er is not None:
            er = er * sgn

    eta_e = np.where(
        np.isfinite(dln_ne) & (np.abs(dln_ne) >= min_inv_len),
        dln_te / np.where(np.abs(dln_ne) >= min_inv_len, dln_ne, 1.0),
        np.nan,
    )

    return ProfileGradients(
        pe_pa=pe,
        pe_err_pa=pe_err,
        dne_dx=dne_dx,
        dte_dx=dte_dx,
        dpe_dx=dpe_dx,
        dln_ne_dx=dln_ne,
        dln_te_dx=dln_te,
        dln_pe_dx=dln_pe,
        eta_e=eta_e,
        er_v_per_cm=er,
        dln_ne_err=dln_ne_err,
        dln_te_err=dln_te_err,
        dln_pe_err=dln_pe_err,
        er_err_v_per_cm=er_err,
    )


def floating_potential_from_iv(V, I, *, smooth_samples=51):
    """Extract floating potential ``V_f [V]`` (where ``I(V) = 0``) from I-V sweeps.

    Accepts 1-D ``(nt,)`` or N-D ``(..., nt)`` arrays (``V`` may broadcast
    against ``I``, e.g. ``V`` of shape ``(npos, nswp, nt)`` and ``I`` of shape
    ``(npos, nshot, nswp, nt)`` after inserting the shot axis). Each trace is
    sorted by ascending voltage, optionally boxcar-smoothed along the sweep by
    ``smooth_samples`` to suppress digitizer/RF ripple near ``I = 0``, and
    linearly interpolated at the zero crossing. When multiple crossings remain,
    the median crossing voltage is returned; if ``I`` never changes sign,
    returns ``NaN``.
    """
    V_arr = np.asarray(V, dtype=float)
    I_arr = np.asarray(I, dtype=float)
    V_b, I_b = np.broadcast_arrays(V_arr, I_arr)
    lead_shape = I_b.shape[:-1]
    nt = I_b.shape[-1]
    if nt < 2:
        raise ValueError(f"I-V sweep must have at least 2 samples, got {nt}")

    V_flat = V_b.reshape(-1, nt)
    I_flat = I_b.reshape(-1, nt)
    out = np.full(V_flat.shape[0], np.nan, dtype=float)

    klen = int(smooth_samples)
    if klen > 1:
        if klen % 2 == 0:
            klen += 1
        klen = min(klen, nt if nt % 2 == 1 else nt - 1)
        kernel = np.ones(klen, dtype=float) / klen
    else:
        kernel = None

    for idx in range(V_flat.shape[0]):
        v = V_flat[idx]
        cur = I_flat[idx]
        fin = np.isfinite(v) & np.isfinite(cur)
        if np.sum(fin) < 2:
            continue
        if not np.all(fin):
            v, cur = v[fin], cur[fin]
        # Sort by ascending V (handles both up- and down-sweeps cleanly).
        if v[0] > v[-1]:
            v = v[::-1]
            cur = cur[::-1]
        if not np.all(np.diff(v) >= 0.0):
            order = np.argsort(v)
            v = v[order]
            cur = cur[order]

        if kernel is not None and cur.size >= kernel.size:
            pad = kernel.size // 2
            cur_s = np.convolve(np.pad(cur, pad, mode="edge"), kernel, mode="valid")
        else:
            cur_s = cur

        cross = np.flatnonzero(cur_s[:-1] * cur_s[1:] <= 0.0)
        if cross.size == 0:
            continue
        vf_candidates = []
        for c in cross:
            i0, i1 = cur_s[c], cur_s[c + 1]
            if i0 == i1:
                vf_candidates.append(0.5 * (v[c] + v[c + 1]))
            else:
                vf_candidates.append(v[c] - i0 * (v[c + 1] - v[c]) / (i1 - i0))
        out[idx] = float(np.median(vf_candidates))

    return float(out[0]) if lead_shape == () else out.reshape(lead_shape)


def sheath_factor(vp_v, vf_v, te_ev, vp_err=None, vf_err=None, te_err=None):
    """Floating-sheath potential drop factor ``Lambda = (V_p - V_f) / T_e``.

    For a thermal Maxwellian He plasma without primary electrons,
    ``Lambda_th = 0.5 * ln(m_i / (2*pi*m_e)) ~ 3.53`` (typically ~3.1-4.6 in
    magnetized LAPD). Values well above ~5-6 indicate fast/primary electron
    contamination depressing ``V_f``.
    """
    vp = np.asarray(vp_v, dtype=float)
    vf = np.asarray(vf_v, dtype=float)
    te = np.asarray(te_ev, dtype=float)
    valid = np.isfinite(vp) & np.isfinite(vf) & np.isfinite(te) & (te > 0.0)
    lam = np.where(valid, (vp - vf) / np.where(valid, te, 1.0), np.nan)

    if vp_err is None and vf_err is None and te_err is None:
        return lam

    var_num = np.zeros_like(lam, dtype=float)
    if vp_err is not None:
        var_num = var_num + np.asarray(vp_err, dtype=float) ** 2
    if vf_err is not None:
        var_num = var_num + np.asarray(vf_err, dtype=float) ** 2
    var_lam = np.where(valid, var_num / np.where(valid, te, 1.0) ** 2, np.nan)
    if te_err is not None:
        te_e = np.asarray(te_err, dtype=float)
        var_lam = var_lam + np.where(valid, (lam * te_e / np.where(valid, te, 1.0)) ** 2, np.nan)
    return lam, np.sqrt(var_lam)


def local_drift_scales(ne_cm3, te_ev, b0_g, *, mu=4.0, Z=1,
                       k_perp_cm=None, dln_ne_dr_cm=None, dln_te_dr_cm=None,
                       er_v_per_cm=None):
    """Local magnetized-plasma scales, collisionality, and drift frequencies.

    Args:
        ne_cm3: Electron density [cm^-3] (scalar or array).
        te_ev: Electron temperature [eV].
        b0_g: Axial magnetic field [G].
        mu: Ion mass ratio ``m_i / m_p`` (default ``4.0`` for He+).
        Z: Ion charge state (default ``1``).
        k_perp_cm: Optional perpendicular wavenumber ``k_perp`` [cm^-1] for
            diamagnetic frequencies (e.g. ``m / r_cm``).
        dln_ne_dr_cm: Optional signed ``d(ln n_e)/dr`` [cm^-1].
        dln_te_dr_cm: Optional signed ``d(ln T_e)/dr`` [cm^-1].
        er_v_per_cm: Optional signed radial electric field ``E_r`` [V/cm].

    Returns:
        Dict of arrays/scalars with explicit unit-suffixed keys:
        ``pe_pa``, ``cs_m_s``, ``omega_ci_rad_s``, ``rho_s_cm``, ``beta_e``,
        ``coulomb_log``, ``nu_ei_s``, ``v_alfven_m_s``, plus optional
        ``v_ExB_m_s``, ``omega_star_n_rad_s``, ``omega_star_T_rad_s``,
        ``omega_star_p_rad_s``.
    """
    ne = np.asarray(ne_cm3, dtype=float)
    te = np.asarray(te_ev, dtype=float)
    B_T = np.asarray(b0_g, dtype=float) * 1e-4  # G -> T
    mi_kg = float(mu) * M_PROTON

    B_valid = np.isfinite(B_T) & (B_T > 0.0)
    B_pos = np.where(B_valid, B_T, 1.0)
    valid = np.isfinite(ne) & np.isfinite(te) & (ne > 0.0) & (te > 0.0) & B_valid
    ne_safe = np.where(valid, ne, 1.0)
    te_safe = np.where(valid, te, 1.0)
    B_safe = np.where(valid, B_T, 1.0)

    pe_pa = electron_pressure(ne, te)
    cs_m_s = np.where(valid, np.sqrt(Z * E_CHARGE_J_PER_EV * te_safe / mi_kg), np.nan)
    omega_ci = np.where(valid, Z * E_CHARGE_J_PER_EV * B_safe / mi_kg, np.nan)
    rho_s_cm = np.where(valid, (cs_m_s / omega_ci) * 1e2, np.nan)
    beta_e = np.where(valid, pe_pa / (B_safe ** 2 / (2.0 * MU_0)), np.nan)

    # NRL Plasma Formulary thermal e-i Coulomb logarithm (Te < 10 eV branch,
    # smooth transition to 24 - ln(sqrt(ne)/Te) above 10 eV), clamped >= 2.
    ln_lam_lo = 23.0 - np.log(np.sqrt(ne_safe) * (Z * te_safe ** (-1.5)))
    ln_lam_hi = 24.0 - np.log(np.sqrt(ne_safe) / te_safe)
    coulomb_log = np.where(valid, np.maximum(np.where(te_safe < 10.0, ln_lam_lo, ln_lam_hi), 2.0), np.nan)
    nu_ei_s = np.where(valid, 2.91e-6 * Z * ne_safe * coulomb_log * (te_safe ** -1.5), np.nan)

    ni_m3 = (ne_safe * 1e6) / float(Z)
    v_alfven_m_s = np.where(valid, B_safe / np.sqrt(MU_0 * ni_m3 * mi_kg), np.nan)

    out = {
        "pe_pa": pe_pa,
        "cs_m_s": cs_m_s,
        "omega_ci_rad_s": omega_ci,
        "rho_s_cm": rho_s_cm,
        "beta_e": beta_e,
        "coulomb_log": coulomb_log,
        "nu_ei_s": nu_ei_s,
        "v_alfven_m_s": v_alfven_m_s,
    }

    if er_v_per_cm is not None:
        er = np.asarray(er_v_per_cm, dtype=float)
        # E_r [V/m] = er [V/cm] * 1e2; v_ExB = E_r / B [m/s] (depends only on Er & B,
        # never on ne/Te validity, so use B_pos rather than ne/Te-gated B_safe).
        out["v_ExB_m_s"] = np.where(np.isfinite(er) & B_valid, (er * 1e2) / B_pos, np.nan)

    if k_perp_cm is not None:
        kp_m = np.asarray(k_perp_cm, dtype=float) * 1e2  # cm^-1 -> m^-1
        # omega_* = -k_perp [m^-1] * (Te [eV] / B [T]) * d(ln f)/dr [m^-1]
        # (since k_B T_e / e = Te [eV] numerically).
        if dln_ne_dr_cm is not None:
            dn_m = np.asarray(dln_ne_dr_cm, dtype=float) * 1e2
            out["omega_star_n_rad_s"] = np.where(valid, -kp_m * (te_safe / B_safe) * dn_m, np.nan)
        if dln_te_dr_cm is not None:
            dt_m = np.asarray(dln_te_dr_cm, dtype=float) * 1e2
            out["omega_star_T_rad_s"] = np.where(valid, -kp_m * (te_safe / B_safe) * dt_m, np.nan)
        if "omega_star_n_rad_s" in out and "omega_star_T_rad_s" in out:
            out["omega_star_p_rad_s"] = out["omega_star_n_rad_s"] + out["omega_star_T_rad_s"]

    return out


class HollowProfileMetrics(NamedTuple):
    """Summary metrics for a hollow radial density or ion-saturation profile.

    Attributes:
        core_val: Median profile value in ``r <= core_r_max_cm``.
        ring_val: Median profile value in ``ring_r_cm[0] <= r <= ring_r_cm[1]``.
        hollowness: ``ring_val / core_val`` (``NaN`` when ``core_val <= 0``,
            ``ring_val <= 0``, or either band has no finite samples).
        wall_inv_scale_cm: Linear least-squares slope ``d(ln f)/dr`` [``cm^-1``]
            across positive finite samples in ``wall_r_cm[0] <= r <= wall_r_cm[1]``
            (``NaN`` when fewer than 2 distinct valid radii exist).
        radial_edges_cm: Optional 1-D radial bin edges [``cm``], or ``None``.
        radial_profile: Optional bin-median radial profile of shape
            ``(nbins,)`` or ``(nbins, nt)``, or ``None``.
    """
    core_val: np.ndarray | float
    ring_val: np.ndarray | float
    hollowness: np.ndarray | float
    wall_inv_scale_cm: np.ndarray | float
    radial_edges_cm: np.ndarray | None
    radial_profile: np.ndarray | None


def _safe_median_1d(vals):
    fin = vals[np.isfinite(vals)]
    return float(np.median(fin)) if fin.size else np.nan


def hollow_profile_metrics(
    r_cm,
    prof,
    *,
    core_r_max_cm=2.0,
    ring_r_cm=(4.5, 7.0),
    wall_r_cm=(1.5, 4.5),
    radial_edges_cm=None,
):
    """Quantify radial hollowness and inner-wall log gradient of a 1-D or 2-D profile.

    Works identically for 1-D line cuts (``r_cm = |x - x_c|``) and scattered
    2-D plane scans (``r_cm = hypot(x - x_c, y - y_c)``).

    Args:
        r_cm: 1-D non-negative radial coordinate array of shape ``(npos,)`` [cm].
        prof: Positive plasma profile (``ne``, ``Isat``, etc.) of shape
            ``(npos,)`` or ``(npos, nt)``.
        core_r_max_cm: Outer radius of the hollow core band [cm].
        ring_r_cm: ``(r_lo, r_hi)`` [cm] of the surrounding dense ring.
        wall_r_cm: ``(r_lo, r_hi)`` [cm] of the inner-wall gradient band.
        radial_edges_cm: Optional monotonic 1-D radial bin edges [cm] for
            azimuthally averaged radial profiles.

    Returns:
        :class:`HollowProfileMetrics`. Scalars are returned when ``prof`` is
        1-D; 1-D ``(nt,)`` arrays are returned when ``prof`` is 2-D.
    """
    r = np.asarray(r_cm, dtype=float)
    p = np.asarray(prof, dtype=float)
    if r.ndim != 1 or r.size == 0:
        raise ValueError("r_cm must be a non-empty 1-D array")
    if p.ndim not in (1, 2) or p.shape[0] != r.size:
        raise ValueError(
            f"prof must have shape ({r.size},) or ({r.size}, nt), got {p.shape}"
        )

    is_1d = p.ndim == 1
    p2 = p[:, None] if is_1d else p
    nt = p2.shape[1]

    m_core = np.isfinite(r) & (r >= 0.0) & (r <= float(core_r_max_cm))
    m_ring = np.isfinite(r) & (r >= float(ring_r_cm[0])) & (r <= float(ring_r_cm[1]))
    m_wall = np.isfinite(r) & (r >= float(wall_r_cm[0])) & (r <= float(wall_r_cm[1]))

    core_arr = np.full(nt, np.nan, dtype=float)
    ring_arr = np.full(nt, np.nan, dtype=float)
    slope_arr = np.full(nt, np.nan, dtype=float)

    r_wall = r[m_wall]
    p_wall = p2[m_wall]

    for j in range(nt):
        if np.any(m_core):
            core_arr[j] = _safe_median_1d(p2[m_core, j])
        if np.any(m_ring):
            ring_arr[j] = _safe_median_1d(p2[m_ring, j])

        if r_wall.size >= 2:
            col = p_wall[:, j]
            ok = np.isfinite(col) & (col > 0.0)
            if int(np.sum(ok)) >= 2:
                rv = r_wall[ok]
                r_centered = rv - float(np.mean(rv))
                denom = float(np.dot(r_centered, r_centered))
                if denom > 0.0:
                    yv = np.log(col[ok])
                    slope_arr[j] = float(np.dot(r_centered, yv - float(np.mean(yv))) / denom)

    valid_h = np.isfinite(core_arr) & np.isfinite(ring_arr) & (core_arr > 0.0) & (ring_arr > 0.0)
    h_arr = np.where(valid_h, ring_arr / np.where(valid_h, core_arr, 1.0), np.nan)

    r_edges_out = None
    r_prof_out = None
    if radial_edges_cm is not None:
        r_edges_out = np.asarray(radial_edges_cm, dtype=float)
        if r_edges_out.ndim != 1 or r_edges_out.size < 2 or not np.all(np.diff(r_edges_out) > 0.0):
            raise ValueError("radial_edges_cm must be a strictly increasing 1-D array with len >= 2")
        nbins = r_edges_out.size - 1
        r_prof = np.full((nbins, nt), np.nan, dtype=float)
        for b in range(nbins):
            lo, hi = float(r_edges_out[b]), float(r_edges_out[b + 1])
            mb = np.isfinite(r) & (r >= lo) & (r <= hi if b == nbins - 1 else r < hi)
            if not np.any(mb):
                continue
            for j in range(nt):
                r_prof[b, j] = _safe_median_1d(p2[mb, j])
        r_prof_out = r_prof[:, 0] if is_1d else r_prof

    if is_1d:
        return HollowProfileMetrics(
            core_val=float(core_arr[0]),
            ring_val=float(ring_arr[0]),
            hollowness=float(h_arr[0]),
            wall_inv_scale_cm=float(slope_arr[0]),
            radial_edges_cm=r_edges_out,
            radial_profile=r_prof_out,
        )
    return HollowProfileMetrics(
        core_val=core_arr,
        ring_val=ring_arr,
        hollowness=h_arr,
        wall_inv_scale_cm=slope_arr,
        radial_edges_cm=r_edges_out,
        radial_profile=r_prof_out,
    )
