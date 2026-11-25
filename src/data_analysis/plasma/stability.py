"""Linear magnetized-plasma stability solvers (CWM, Braginskii EM drift-Alfvén, KH).

Diagnostic-agnostic linear fluid eigenvalue solvers for straight magnetized
cylindrical plasmas (e.g., LAPD, linear divertor simulators), using the exact
same sign and unit conventions as :mod:`data_analysis.plasma.profiles` and
:mod:`data_analysis.plasma.modes`:

- Perturbations follow ``exp(i * (m * theta + k_par * z - omega * t))`` with
  ``omega = omega_r + i * gamma`` (``gamma > 0`` is unstable).
- ``b_sign_z = +1`` means ``B_0`` points along ``+z`` so outward ``E_r > 0``
  drives clockwise ``E x B`` rotation (``v_E0,theta = -b_sign_z * E_r / B_0``).
- Electron diamagnetic drift frequencies ``omega_*n, omega_*T`` carry the
  signed mode convention ``omega_* = -b_sign_z * (m / r) * (T_e / B_0) * d(ln f)/dr``
  so ``omega_* > 0`` rotates with ``m`` in the lab frame.
- Centrifugal gravity from cylindrical ``E x B`` rotation ``Omega_E = v_E0 / r``
  produces ion drift ``omega_g = -b_sign_z * m * Omega_E^2 / Omega_ci``. For a
  hollow density wall (``d ln n_e / dr > 0``), ``omega_g * omega_*n > 0`` is
  centrifugally stabilizing; for a peaked edge (``d ln n_e / dr < 0``), it
  drives the rotational interchange instability.

Implemented models:
1. :func:`solve_local_cwm` — 2-field local Conducting Wall Mode (Berk, Ryutov &
   Tsidulko 1991; Friedman 2013 Sec. 4.2.2) with optional series parallel
   column resistance and parallel electron thermal conduction.
2. :func:`solve_local_braginskii` — 5-field local electromagnetic resistive
   Braginskii drift-Alfvén dispersion (``n, phi, T_e, J_par, v_{i,par}``) with
   electron inertia, ``0.71 grad_par T_e`` thermal force, parallel thermal
   conduction, centrifugal gravity, and optional lumped sheath closure.
3. :func:`solve_axial_braginskii_bvp` — 1-D axial Chebyshev pseudospectral
   boundary-value problem for the full 5-field Braginskii PDEs on
   ``z in [0, L_par]`` with exact conducting-plate sheath or insulating end
   boundary conditions (self-consistently resolving axial localization, end
   current closure, and two-probe axial phase lag).
4. :func:`solve_radial_cylindrical_modes` — 1-D radial cylindrical eigenmode
   solver on ``r in (0, R_max]`` for pure Kelvin-Helmholtz (Rayleigh) and
   ``E x B``-sheared Conducting Wall Modes with exact cylindrical polarization
   operator ``-(1/(r n_0)) d/dr(r n_0 dphi/dr) + (m^2/r^2) phi``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import constants as const
from scipy.interpolate import make_interp_spline
from scipy.linalg import eig
from scipy.optimize import curve_fit

from data_analysis.plasma.profiles import local_drift_scales

E_CHARGE = const.e
M_ELECTRON = const.electron_mass
M_PROTON = const.proton_mass
MU_0 = const.mu_0


class LocalPlasmaState(NamedTuple):
    """Consistent SI local plasma parameters for linear dispersion solvers."""
    ne_m3: float
    te_ev: float
    b0_t: float
    mu: float
    Z: int
    r_m: float
    m: int
    b_sign_z: int
    cs_m_s: float
    vte_m_s: float
    v_alfven_m_s: float
    omega_ci_rad_s: float
    rho_s_m: float
    beta_e: float
    nu_ei_s: float
    eta_par_ohm_m: float
    chi_par_m2_s: float
    k_theta_m: float
    omega_star_n_rad_s: float
    omega_star_T_rad_s: float
    omega_E_rad_s: float
    omega_g_rad_s: float


class AxialModeSummary(NamedTuple):
    """Summary of one eigenmode from :func:`solve_axial_braginskii_bvp`."""
    omega_rad_s: complex
    f_hz: float
    gamma_s: float
    k_par_rms_m: float
    j_end_rel: float
    phi_end_rel: float
    dphi_ab_deg: float
    phi_over_n: float
    T_over_n: float


def chebyshev_diff_matrix(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Chebyshev Gauss-Lobatto nodes on ``[-1, 1]`` (ascending) and 1st-derivative matrix.

    Returns ``(x, D)`` with ``x.shape == (n + 1,)``, ``x[0] == -1.0``,
    ``x[-1] == +1.0``, and ``(D @ f)[j]`` approximating ``df/dx`` at ``x[j]``.
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"Chebyshev order n must be >= 2, got {n}")
    x_desc = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.ones(n + 1, dtype=float)
    c[0] = 2.0
    c[n] = 2.0
    c *= (-1.0) ** np.arange(n + 1)
    X = np.tile(x_desc, (n + 1, 1)).T
    dX = X - X.T
    D_desc = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D_desc -= np.diag(np.sum(D_desc, axis=1))
    return x_desc[::-1].copy(), D_desc[::-1, ::-1].copy()


def local_braginskii_state(ne_cm3, te_ev, b0_g, dln_ne_dr_cm, dln_te_dr_cm, *,
                           er_v_per_cm=0.0, r_cm=3.0, m=-1, mu=4.0, Z=1,
                           b_sign_z=+1) -> LocalPlasmaState:
    """Build a :class:`LocalPlasmaState` from profile quantities in standard lab units."""
    r_m = float(r_cm) * 1e-2
    if r_m <= 0.0:
        raise ValueError(f"r_cm must be positive, got {r_cm}")
    m_int = int(m)
    sgn = int(np.sign(b_sign_z or 1))
    ne_c = float(ne_cm3)
    te = float(te_ev)
    b0_t = float(b0_g) * 1e-4
    if ne_c <= 0.0 or te <= 0.0 or b0_t <= 0.0:
        raise ValueError("ne_cm3, te_ev, and b0_g must be strictly positive")

    scales = local_drift_scales(ne_c, te, float(b0_g), mu=mu, Z=Z)
    ne_m3 = ne_c * 1e6
    cs = float(scales["cs_m_s"])
    vte = float(np.sqrt(te * E_CHARGE / M_ELECTRON))
    vA = float(scales["v_alfven_m_s"])
    oci = float(scales["omega_ci_rad_s"])
    rhos = float(scales["rho_s_cm"]) * 1e-2
    beta_e = float(scales["beta_e"])
    nu_ei = float(scales["nu_ei_s"])

    # Braginskii (1965) Z=1 parallel resistivity & electron thermal diffusivity.
    eta_par = 0.51 * M_ELECTRON * nu_ei / (ne_m3 * E_CHARGE ** 2)
    chi_par = 3.16 * (vte ** 2) / nu_ei

    k_theta = abs(m_int) / r_m
    pref = -sgn * (te / b0_t)
    w_star_n = m_int * pref * (float(dln_ne_dr_cm) * 1e2) / r_m
    w_star_T = m_int * pref * (float(dln_te_dr_cm) * 1e2) / r_m

    # Signed E x B azimuthal velocity v_E0 = -b_sign_z * Er / B0
    v_E0 = -sgn * (float(er_v_per_cm) * 1e2) / b0_t
    omega_E = v_E0 / r_m
    # Ion centrifugal drift frequency: g_eff = r * Omega_E^2 outward
    omega_g = -sgn * m_int * (omega_E ** 2) / oci

    return LocalPlasmaState(
        ne_m3=ne_m3, te_ev=te, b0_t=b0_t, mu=float(mu), Z=int(Z),
        r_m=r_m, m=m_int, b_sign_z=sgn,
        cs_m_s=cs, vte_m_s=vte, v_alfven_m_s=vA,
        omega_ci_rad_s=oci, rho_s_m=rhos, beta_e=beta_e,
        nu_ei_s=nu_ei, eta_par_ohm_m=eta_par, chi_par_m2_s=chi_par,
        k_theta_m=k_theta,
        omega_star_n_rad_s=float(w_star_n),
        omega_star_T_rad_s=float(w_star_T),
        omega_E_rad_s=float(omega_E),
        omega_g_rad_s=float(omega_g),
    )


def solve_local_cwm(state: LocalPlasmaState, k_perp_m: float, *,
                    l_par_m: float = 14.0, sheath_lambda: float = 3.7,
                    n_ends: float = 1.0, nu_in_s: float = 0.0,
                    include_res: bool = True, include_chi: bool = True,
                    k_par_eff_m: float | None = None) -> tuple[complex, np.ndarray]:
    """Solve the 2-field local Conducting Wall Mode dispersion in the plasma frame.

    Based on Berk, Ryutov & Tsidulko (1991) and Friedman (2013, Sec. 4.2.2):

    .. math::
        (-i\\bar{\\omega} + \\nu_{in}) b \\hat{\\phi}
            + \\nu_{\\mathrm{sh,eff}}(\\hat{\\phi} - \\Lambda \\hat{T}_e) = 0,
        \\qquad
        (-i\\bar{\\omega} + \\nu_T) \\hat{T}_e + i\\omega_{*T} \\hat{\\phi} = 0,

    where ``b = (k_perp * rho_s)^2``, ``nu_sh = n_ends * c_s / L_par``, and
    optional series column resistance reduces ``nu_sh`` by
    ``1 / (1 + R_col / R_sh)``.

    Returns ``(most_unstable_omega, all_eigenvalues)`` [rad/s] in the plasma frame.
    """
    k_perp_m = float(k_perp_m)
    if k_perp_m <= 0.0 or l_par_m <= 0.0:
        raise ValueError("k_perp_m and l_par_m must be strictly positive")
    b = (k_perp_m * state.rho_s_m) ** 2
    mi = state.mu * M_PROTON
    nu_sh = float(n_ends) * state.cs_m_s / float(l_par_m)
    if include_res:
        l_eff = float(l_par_m) / max(2.0 * float(n_ends), 1e-12)
        res_ratio = 0.51 * (M_ELECTRON / mi) * (state.nu_ei_s * l_eff / state.cs_m_s)
        nu_sh_eff = nu_sh / (1.0 + res_ratio)
    else:
        nu_sh_eff = nu_sh
    nu_C = nu_sh_eff / b

    if include_chi:
        kp = (np.pi / float(l_par_m)) if k_par_eff_m is None else float(k_par_eff_m)
        nu_T = (2.0 / 3.0) * state.chi_par_m2_s * (kp ** 2)
    else:
        nu_T = 0.0

    mat = np.array([
        [-1j * (float(nu_in_s) + nu_C), 1j * nu_C * float(sheath_lambda)],
        [state.omega_star_T_rad_s,      -1j * nu_T],
    ], dtype=complex)
    evals = np.linalg.eigvals(mat)
    idx = int(np.argmax(evals.imag))
    return complex(evals[idx]), evals


def solve_local_braginskii(state: LocalPlasmaState, k_perp_m: float, k_par_m: float, *,
                           nu_in_s: float = 0.0, em: bool = True,
                           thermal_force: bool = True, chi_par_on: bool = True,
                           include_centrifugal: bool = True,
                           sheath_l_par_m: float | None = None,
                           sheath_lambda: float = 3.7,
                           n_ends: float = 1.0) -> tuple[complex, np.ndarray, np.ndarray]:
    """Solve the 5-field local Braginskii EM resistive drift-Alfvén dispersion.

    State vector ``y = [n_hat, phi_hat, T_hat, j_hat, vi_hat]`` in the plasma frame:
    1. Electron continuity
    2. Ion vorticity (+ optional centrifugal gravity ``omega_g`` & sheath closure)
    3. Electron thermal balance (``1.71`` Braginskii thermal-drift flux & ``chi_par``)
    4. Parallel Ohm's law (electron inertia, EM ``A_par``, ``0.71`` thermal force, ``eta_par``)
    5. Parallel ion momentum (cold ions, acoustic waves, ``nu_in`` drag)

    Returns ``(most_unstable_omega, eigenvector, all_eigenvalues)`` [rad/s].
    """
    k_perp_m = float(k_perp_m)
    if k_perp_m <= 0.0:
        raise ValueError("k_perp_m must be strictly positive")
    b = (k_perp_m * state.rho_s_m) ** 2
    mi = state.mu * M_PROTON
    mu_e = M_ELECTRON / mi
    c_th = 0.71 if thermal_force else 0.0
    alpha_em = (state.beta_e / (2.0 * b)) if em else 0.0
    nu_res = 0.51 * mu_e * state.nu_ei_s
    nu_T = ((2.0 / 3.0) * state.chi_par_m2_s * (float(k_par_m) ** 2)) if chi_par_on else 0.0
    nu_sh = (float(n_ends) * state.cs_m_s / float(sheath_l_par_m)) if sheath_l_par_m is not None else 0.0
    w_g = state.omega_g_rad_s if include_centrifugal else 0.0
    kcs = float(k_par_m) * state.cs_m_s

    A = np.zeros((5, 5), dtype=complex)
    B = np.zeros((5, 5), dtype=complex)

    # 1) Electron continuity
    B[0, 0] = 1.0
    A[0, 1] = state.omega_star_n_rad_s
    A[0, 3] = -kcs
    A[0, 4] = +kcs

    # 2) Vorticity
    B[1, 1] = b
    A[1, 0] = w_g
    A[1, 1] = -1j * (float(nu_in_s) * b + nu_sh)
    A[1, 2] = +1j * nu_sh * float(sheath_lambda)
    A[1, 3] = +kcs

    # 3) Electron temperature
    B[2, 2] = 1.0
    A[2, 1] = state.omega_star_T_rad_s
    A[2, 2] = -1j * nu_T
    A[2, 3] = -(2.0 / 3.0) * (1.0 + c_th) * kcs
    A[2, 4] = +(2.0 / 3.0) * kcs

    # 4) Parallel Ohm's law
    B[3, 3] = mu_e + alpha_em
    B[3, 4] = -mu_e
    A[3, 0] = -kcs
    A[3, 1] = +kcs
    A[3, 2] = -(1.0 + c_th) * kcs
    A[3, 3] = -1j * nu_res

    # 5) Ion parallel momentum
    B[4, 4] = 1.0
    A[4, 0] = +kcs
    A[4, 2] = +kcs
    A[4, 4] = -1j * float(nu_in_s)

    evals, evecs = eig(A, B)
    fin = np.isfinite(evals)
    evals = evals[fin]
    evecs = evecs[:, fin]
    idx = int(np.argmax(evals.imag))
    return complex(evals[idx]), evecs[:, idx], evals


def solve_axial_braginskii_bvp(state: LocalPlasmaState, k_perp_m: float, *,
                               l_par_m: float = 14.0, sheath_lambda: float = 3.7,
                               bc: str = "conducting_1", nu_in_s: float = 0.0,
                               em: bool = True, thermal_force: bool = True,
                               chi_par_on: bool = True,
                               include_centrifugal: bool = True,
                               sheath_heat_gamma: float = 0.0,
                               n_nodes: int = 24,
                               z_probe_a_m: float | None = None,
                               z_probe_b_m: float | None = None,
                               cwm_j_end_min: float = 0.85,
                               cwm_phi_end_min: float = 0.65,
                               cwm_gamma_min_s: float = 0.0) -> dict:
    """Solve the 1-D axial 5-field Braginskii BVP on ``z in [0, l_par_m]``.

    Uses Chebyshev Gauss-Lobatto collocation (``n_nodes + 1`` nodes) in the
    **plasma frame** with **axially uniform** equilibrium coefficients along
    ``z`` (no axial profile gradients or axial ``E x B`` shear).

    Boundary conditions at ``z = 0`` (cathode/far end) and ``z = l_par_m``
    (biased conducting end plate):

    - ``"conducting_1"``: insulating ``j = 0, v_i = 0`` at ``z = 0``;
      conducting sheath ``j = +(phi - Lambda * T)``, ``v_i = +0.5 * T`` at ``z = L``.
    - ``"conducting_2"``: conducting sheath at both ``z = 0`` and ``z = L``.
    - ``"insulating"``: ``j = 0, v_i = 0`` at both ends (zero axial particle
      flux / reflecting ends; note that physical insulating end plates still
      draw ions at the Bohm speed with only ``j = 0`` forced).

    Unstable Conducting Wall Mode classification (``out["cwm"]``) is based on
    conducting-end sheath anchoring rather than a hard-coded frequency sign:
    when ``bc != "insulating"``, selects the fastest-growing mode with
    ``gamma_s > cwm_gamma_min_s`` (default ``> 0``), ``j_end_rel >= cwm_j_end_min``
    (normalized ``|j|`` at the active conducting end(s)), and
    ``phi_end_rel >= cwm_phi_end_min``. Returns ``None`` when no unstable
    sheath-anchored CWM exists (or for ``bc == "insulating"``).

    When ``z_probe_a_m`` and ``z_probe_b_m`` are given (defaulting to
    ``l_par_m - 1.92`` and ``l_par_m - 3.20``, corresponding to LAPD Ports 29
    and 25 relative to Port 35), reports the predicted density phase difference
    ``angle(n(z_a) / n(z_b))`` in degrees.

    Returns a dict with ``"top"`` (:class:`AxialModeSummary`), ``"cwm"`` (most
    unstable sheath-anchored CWM or ``None``), ``"bulk_pos"`` (most unstable
    ``f > 0`` mode, or ``None``), ``"modes"`` (list of top summaries), and
    axial profiles (``z_m``, ``n_z``, ``phi_z``, ``T_z``, ``j_z``) for the
    primary CWM (or top) mode.
    """
    if bc not in ("conducting_1", "conducting_2", "insulating"):
        raise ValueError(f"bc must be 'conducting_1', 'conducting_2', or 'insulating', got {bc!r}")
    if k_perp_m <= 0.0 or l_par_m <= 0.0:
        raise ValueError("k_perp_m and l_par_m must be strictly positive")

    x_asc, D_asc = chebyshev_diff_matrix(n_nodes)
    M = n_nodes + 1
    z = 0.5 * float(l_par_m) * (x_asc + 1.0)
    Dz = (2.0 / float(l_par_m)) * D_asc
    Dzz = Dz @ Dz

    b = (float(k_perp_m) * state.rho_s_m) ** 2
    mi = state.mu * M_PROTON
    mu_e = M_ELECTRON / mi
    c_th = 0.71 if thermal_force else 0.0
    alpha_em = (state.beta_e / (2.0 * b)) if em else 0.0
    nu_res = 0.51 * mu_e * state.nu_ei_s
    cs = state.cs_m_s
    chi = state.chi_par_m2_s if chi_par_on else 0.0
    w_g = state.omega_g_rad_s if include_centrifugal else 0.0

    I = np.eye(M, dtype=complex)
    A = np.zeros((5 * M, 5 * M), dtype=complex)
    B = np.zeros((5 * M, 5 * M), dtype=complex)

    def sl(k):
        return slice(k * M, (k + 1) * M)

    # 1) Electron continuity: w*n = w_*n*phi - i*cs*Dz*(vi - j)
    B[sl(0), sl(0)] = I
    A[sl(0), sl(1)] = state.omega_star_n_rad_s * I
    A[sl(0), sl(3)] = +1j * cs * Dz
    A[sl(0), sl(4)] = -1j * cs * Dz

    # 2) Vorticity: w*b*phi = w_g*n - i*nu_in*b*phi - i*cs*Dz*j
    B[sl(1), sl(1)] = b * I
    A[sl(1), sl(0)] = w_g * I
    A[sl(1), sl(1)] = -1j * float(nu_in_s) * b * I
    A[sl(1), sl(3)] = -1j * cs * Dz

    # 3) Electron temperature
    B[sl(2), sl(2)] = I
    A[sl(2), sl(1)] = state.omega_star_T_rad_s * I
    A[sl(2), sl(2)] = +1j * (2.0 / 3.0) * chi * Dzz
    A[sl(2), sl(3)] = +1j * (2.0 / 3.0) * (1.0 + c_th) * cs * Dz
    A[sl(2), sl(4)] = -1j * (2.0 / 3.0) * cs * Dz

    # 4) Parallel Ohm's law
    B[sl(3), sl(3)] = (mu_e + alpha_em) * I
    B[sl(3), sl(4)] = -mu_e * I
    A[sl(3), sl(0)] = +1j * cs * Dz
    A[sl(3), sl(1)] = -1j * cs * Dz
    A[sl(3), sl(2)] = +1j * (1.0 + c_th) * cs * Dz
    A[sl(3), sl(3)] = -1j * nu_res * I

    # 5) Ion parallel momentum
    B[sl(4), sl(4)] = I
    A[sl(4), sl(0)] = -1j * cs * Dz
    A[sl(4), sl(2)] = -1j * cs * Dz
    A[sl(4), sl(4)] = -1j * float(nu_in_s) * I

    def set_bc_row(field_idx, node_idx, terms):
        row = field_idx * M + node_idx
        B[row, :] = 0.0
        A[row, :] = 0.0
        for f_j, val in terms.items():
            if np.ndim(val) == 0:
                A[row, f_j * M + node_idx] = val
            else:
                A[row, sl(f_j)] = val

    lam = float(sheath_lambda)
    if bc == "conducting_2":
        set_bc_row(3, 0, {3: 1.0, 1: +1.0, 2: -lam})
        set_bc_row(3, n_nodes, {3: 1.0, 1: -1.0, 2: +lam})
        set_bc_row(4, 0, {4: 1.0, 2: +0.5})
        set_bc_row(4, n_nodes, {4: 1.0, 2: -0.5})
    elif bc == "conducting_1":
        set_bc_row(3, 0, {3: 1.0})
        set_bc_row(3, n_nodes, {3: 1.0, 1: -1.0, 2: +lam})
        set_bc_row(4, 0, {4: 1.0})
        set_bc_row(4, n_nodes, {4: 1.0, 2: -0.5})
    else:
        set_bc_row(3, 0, {3: 1.0})
        set_bc_row(3, n_nodes, {3: 1.0})
        set_bc_row(4, 0, {4: 1.0})
        set_bc_row(4, n_nodes, {4: 1.0})

    if chi_par_on:
        g0 = float(sheath_heat_gamma) if bc == "conducting_2" else 0.0
        gL = float(sheath_heat_gamma) if bc in ("conducting_1", "conducting_2") else 0.0
        row0 = chi * Dz[0, :].copy()
        row0[0] -= g0 * cs
        rowL = chi * Dz[n_nodes, :].copy()
        rowL[n_nodes] += gL * cs
        set_bc_row(2, 0, {2: row0})
        set_bc_row(2, n_nodes, {2: rowL})

    evals, evecs = eig(A, B)
    good = np.isfinite(evals) & (np.abs(evals) < 1e7)
    evals = evals[good]
    evecs = evecs[:, good]
    order = np.argsort(-evals.imag)

    za = (float(l_par_m) - 1.92) if z_probe_a_m is None else float(z_probe_a_m)
    zb = (float(l_par_m) - 3.20) if z_probe_b_m is None else float(z_probe_b_m)
    use_probes = (0.0 <= za <= float(l_par_m)) and (0.0 <= zb <= float(l_par_m))

    summaries = []
    vectors = []
    for idx in order:
        v = evecs[:, idx]
        n_z, phi_z, T_z, j_z, _ = [v[sl(k)] for k in range(5)]
        p_norm = float(np.linalg.norm(phi_z))
        if p_norm < 1e-12:
            continue
        # Reject spurious Nyquist/checkerboard numerical modes
        if float(np.max(np.abs(np.diff(phi_z)))) / float(np.max(np.abs(phi_z))) > 1.8:
            continue
        k_rms = float(np.linalg.norm(Dz @ phi_z) / p_norm)
        j_max = max(float(np.max(np.abs(j_z))), 1e-30)
        phi_max = max(float(np.max(np.abs(phi_z))), 1e-30)
        if bc == "conducting_2":
            j_end = float(max(abs(j_z[0]), abs(j_z[-1])) / j_max)
            phi_end = float(max(abs(phi_z[0]), abs(phi_z[-1])) / phi_max)
        elif bc == "conducting_1":
            j_end = float(abs(j_z[-1]) / j_max)
            phi_end = float(abs(phi_z[-1]) / phi_max)
        else:
            j_end = 0.0
            phi_end = 0.0
        if use_probes:
            na = np.interp(za, z, n_z.real) + 1j * np.interp(za, z, n_z.imag)
            nb = np.interp(zb, z, n_z.real) + 1j * np.interp(zb, z, n_z.imag)
            dphi_deg = float(np.degrees(np.angle(na / nb))) if abs(nb) > 1e-30 else np.nan
        else:
            dphi_deg = np.nan
        n_norm = max(float(np.linalg.norm(n_z)), 1e-30)
        s = AxialModeSummary(
            omega_rad_s=complex(evals[idx]),
            f_hz=float(evals[idx].real / (2.0 * np.pi)),
            gamma_s=float(evals[idx].imag),
            k_par_rms_m=k_rms,
            j_end_rel=j_end,
            phi_end_rel=phi_end,
            dphi_ab_deg=dphi_deg,
            phi_over_n=float(p_norm / n_norm),
            T_over_n=float(np.linalg.norm(T_z) / n_norm),
        )
        summaries.append(s)
        vectors.append((n_z, phi_z, T_z, j_z))

    top = summaries[0] if summaries else None
    cwm = None
    cwm_vec = None
    bulk_pos = None
    for s, vec in zip(summaries, vectors):
        if (
            cwm is None
            and bc != "insulating"
            and s.gamma_s > float(cwm_gamma_min_s)
            and s.j_end_rel >= float(cwm_j_end_min)
            and s.phi_end_rel >= float(cwm_phi_end_min)
        ):
            cwm = s
            cwm_vec = vec
        if bulk_pos is None and s.f_hz > 200.0:
            bulk_pos = s

    prof_vec = cwm_vec if cwm_vec is not None else (vectors[0] if vectors else None)
    out = {
        "top": top,
        "cwm": cwm,
        "bulk_pos": bulk_pos,
        "modes": summaries[:12],
        "z_m": z,
    }
    if prof_vec is not None:
        out["n_z"], out["phi_z"], out["T_z"], out["j_z"] = prof_vec
    return out


def fit_smooth_radial_profile(x_cm, x_center_cm, y_x, r_grid_cm, *,
                              r_max_fit_cm: float = 15.0,
                              min_width_cm: float = 2.2) -> tuple[np.ndarray, float]:
    """Symmetrize a 1-D chord profile ``y(x)`` about ``x_center_cm`` and fit a smooth 2-Gaussian model.

    Enforcing ``min_width_cm >= 2.2`` guarantees smooth 1st/2nd/3rd radial
    derivatives near ``r = 0`` without overfitting grid-scale noise on 1 cm
    probe scans. Note that inviscid Kelvin-Helmholtz growth depends on the
    third derivative ``V_p'''(r)``, so imposing a minimum width can smooth away
    unresolved narrow shear layers; pair with a ``min_width_cm`` / smoothing-
    spline sensitivity check when evaluating KH stability.

    Returns ``(y_fit_on_r_grid, rms_residual)``.
    """
    x_cm = np.asarray(x_cm, dtype=float)
    y_x = np.asarray(y_x, dtype=float)
    r_grid = np.asarray(r_grid_cm, dtype=float)
    r_abs = np.abs(x_cm - float(x_center_cm))

    # Bin +/- sides by unique rounded radii up to r_max_fit_cm
    r_round = np.round(r_abs, 3)
    r_uniq = np.unique(r_round[r_round <= float(r_max_fit_cm) + 1e-6])
    y_sym = np.array([
        np.nanmean(y_x[r_round == rv]) if np.any(np.isfinite(y_x[r_round == rv])) else np.nan
        for rv in r_uniq
    ], dtype=float)
    fin = np.isfinite(y_sym)
    r_f, y_f = r_uniq[fin], y_sym[fin]
    if r_f.size < 5:
        raise ValueError(f"need at least 5 finite radial points, got {r_f.size}")

    def _model(r, c0, a1, w1, a2, w2):
        return c0 + a1 * np.exp(-(r / w1) ** 2) + a2 * np.exp(-(r / w2) ** 2)

    p0 = [float(y_f[-1]), float(y_f[0] - y_f[-1]), max(3.5, float(min_width_cm)), 0.0, max(8.0, 2.0 * float(min_width_cm))]
    lo = [-np.inf, -np.inf, float(min_width_cm), -np.inf, max(4.0, 1.8 * float(min_width_cm))]
    hi = [np.inf, np.inf, 12.0, np.inf, 25.0]
    popt, _ = curve_fit(_model, r_f, y_f, p0=p0, bounds=(lo, hi), maxfev=15000)
    y_fit = _model(r_grid, *popt)
    resid = float(np.sqrt(np.mean((_model(r_f, *popt) - y_f) ** 2)))
    return y_fit, resid


def solve_radial_cylindrical_modes(r_cm, ne_cm3, te_ev, vp_v, *,
                                   b0_g: float = 800.0, m: int = -1,
                                   model: str = "cwm_sheath",
                                   l_par_m: float = 14.0,
                                   sheath_lambda: float = 3.7,
                                   n_ends: float = 1.0,
                                   k_par_m: float = 0.45,
                                   nu_in_s: float = 0.0,
                                   include_kh: bool = True,
                                   include_centrifugal: bool = True,
                                   include_exb_shear: bool = True,
                                   mu: float = 4.0, Z: int = 1,
                                   b_sign_z: int = +1) -> dict:
    """1-D cylindrical radial eigenmode solver on a uniform grid ``r_cm > 0``.

    Supports:
    - ``model="pure_kh"``: 2-D incompressible cylindrical Rayleigh equation
      ``(w - m*Omega_E + i*nu_in) (-nabla_perp^2 phi) = (m/r) (dW_0/dr) phi``
      with ``W_0 = (1/r) d/dr(r v_E0)``. Because ``dW_0/dr`` uses the third
      radial derivative ``V_p'''(r)``, KH results are sensitive to profile
      smoothing/fitting and cannot exclude shear layers narrower than the
      probe scan spacing.
    - ``model="cwm_sheath"``: 3-field ``(n_hat, phi, T_hat)`` radial Conducting
      Wall Mode with exact non-uniform-density polarization operator
      ``-(1/(r n_0)) d/dr(r n_0 dphi/dr) + (m^2/r^2) phi``, radial ``E x B``
      shear ``Omega_E(r)``, optional Kelvin-Helmholtz vorticity-gradient term,
      optional centrifugal stabilization ``omega_g(r)``, series column
      resistivity, parallel thermal conduction ``(2/3) chi_par k_par^2``, and
      ambipolar Bohm end-plate particle loss ``nu_sh * (n_hat + 0.5 * T_hat)``
      in the continuity equation alongside the sheath current closure
      ``nu_sh_eff * (phi / T_e - Lambda * T_hat)`` (floating-sheath ``J_0 = 0``
      linearization for the current difference; finite DC bias current would
      add a ``J_0 * (n_hat + 0.5 * T_hat)`` contribution to vorticity).

    Returns a dict with ``f_lab_hz``, ``f_plasma_at_peak_hz``, ``gamma_s``,
    ``r_peak_cm``, ``omega_rad_s``, ``phi_r``, and ``r_cm``.
    """
    if model not in ("pure_kh", "cwm_sheath"):
        raise ValueError(f"model must be 'pure_kh' or 'cwm_sheath', got {model!r}")
    r = np.asarray(r_cm, dtype=float) * 1e-2
    if r.ndim != 1 or r.size < 6 or np.any(r <= 0.0):
        raise ValueError("r_cm must be a 1-D strictly positive radial grid with >= 6 points")
    dr_all = np.diff(r)
    if not np.allclose(dr_all, dr_all[0], rtol=1e-3, atol=1e-9):
        raise ValueError("r_cm must be uniformly spaced")
    dr = float(dr_all[0])
    Nr = r.size

    m_int = int(m)
    sgn = int(np.sign(b_sign_z or 1))
    B = float(b0_g) * 1e-4
    mi = float(mu) * M_PROTON
    oci = float(Z) * E_CHARGE * B / mi

    ne = np.asarray(ne_cm3, dtype=float) * 1e6
    te = np.asarray(te_ev, dtype=float)
    vp = np.asarray(vp_v, dtype=float)
    if np.any(ne <= 0.0) or np.any(te <= 0.0):
        raise ValueError("ne_cm3 and te_ev must be strictly positive everywhere")

    # High-order cylindrical radial derivatives via quintic B-spline on an
    # even-mirrored grid across r = 0 (ne, Te, Vp are even in r; v_E0 is odd).
    # Computing W_0 = sgn/B * (Vp'' + Vp'/r) and W_0' = sgn/B * (Vp''' + Vp''/r - Vp'/r^2)
    # directly from the smooth spline avoids 1st-difference stencil broadening
    # and preserves discrete Rayleigh stability to machine precision.
    def _even_spline_derivs(y, max_order=1):
        if r[0] <= 1.5 * dr:
            n_mir = min(3, Nr)
            r_ext = np.concatenate((-r[n_mir - 1::-1], r))
            y_ext = np.concatenate((y[n_mir - 1::-1], y))
        else:
            r_ext, y_ext = r, y
        k_spl = min(5, r_ext.size - 1)
        if k_spl % 2 == 0:
            k_spl -= 1
        spl = make_interp_spline(r_ext, y_ext, k=max(3, k_spl))
        return [spl.derivative(order)(r) for order in range(1, max_order + 1)]

    (dln_n,) = _even_spline_derivs(np.log(ne), max_order=1)
    (dln_T,) = _even_spline_derivs(np.log(te), max_order=1)
    dvp, d2vp, d3vp = _even_spline_derivs(vp, max_order=3)
    v_E0 = sgn * dvp / B
    Omega_E = v_E0 / r
    W_0 = sgn * (d2vp + dvp / r) / B
    dW0_dr = sgn * (d3vp + d2vp / r - dvp / (r ** 2)) / B

    # Cylindrical -nabla_perp^2 (Dirichlet at r=0 for |m|>=1 and r=R_max)
    L_perp = np.zeros((Nr, Nr), dtype=float)
    L_n = np.zeros((Nr, Nr), dtype=float)
    for i in range(Nr):
        ri = r[i]
        c_p = 1.0 / ri
        c_n = 1.0 / ri + dln_n[i]
        L_perp[i, i] = 2.0 / (dr ** 2) + (m_int ** 2) / (ri ** 2)
        L_n[i, i] = L_perp[i, i]
        if i > 0:
            L_perp[i, i - 1] = -1.0 / (dr ** 2) + c_p / (2.0 * dr)
            L_n[i, i - 1] = -1.0 / (dr ** 2) + c_n / (2.0 * dr)
        if i < Nr - 1:
            L_perp[i, i + 1] = -1.0 / (dr ** 2) - c_p / (2.0 * dr)
            L_n[i, i + 1] = -1.0 / (dr ** 2) - c_n / (2.0 * dr)

    if model == "pure_kh":
        D_mOm = np.diag(m_int * Omega_E)
        Bm = L_perp.astype(complex)
        Am = D_mOm @ L_perp - 1j * float(nu_in_s) * L_perp + np.diag((m_int / r) * dW0_dr)
        evals, evecs = eig(Am, Bm)
        good = np.isfinite(evals) & (np.abs(evals) < 1e7)
        evals, evecs = evals[good], evecs[:, good]
        idx = int(np.argmax(evals.imag))
        w = complex(evals[idx])
        phi_r = evecs[:, idx]
        ipk = int(np.argmax(np.abs(phi_r)))
        f_dop_pk = float(m_int * Omega_E[ipk] / (2.0 * np.pi))
        return {
            "omega_rad_s": w,
            "f_lab_hz": float(w.real / (2.0 * np.pi)),
            "f_plasma_at_peak_hz": float(w.real / (2.0 * np.pi) - f_dop_pk),
            "gamma_s": float(w.imag),
            "r_peak_cm": float(r[ipk] * 1e2),
            "phi_r": phi_r,
            "r_cm": np.asarray(r_cm, dtype=float),
        }

    # model == "cwm_sheath"
    scales = local_drift_scales(ne * 1e-6, te, float(b0_g), mu=mu, Z=Z)
    cs = np.asarray(scales["cs_m_s"], dtype=float)
    nu_ei = np.asarray(scales["nu_ei_s"], dtype=float)
    vte = np.sqrt(te * E_CHARGE / M_ELECTRON)
    chi_par = 3.16 * (vte ** 2) / nu_ei

    nu_sh = float(n_ends) * cs / float(l_par_m)
    l_eff = float(l_par_m) / max(2.0 * float(n_ends), 1e-12)
    res_ratio = 0.51 * (M_ELECTRON / mi) * (nu_ei * l_eff / cs)
    nu_sh_eff = nu_sh / (1.0 + res_ratio)
    nu_T = (2.0 / 3.0) * chi_par * (float(k_par_m) ** 2)

    if include_exb_shear:
        Om_eff = Omega_E
    else:
        band = (r >= 0.02) & (r <= 0.04)
        Om_ref = float(np.median(Omega_E[band])) if np.any(band) else float(np.median(Omega_E))
        Om_eff = np.full(Nr, Om_ref, dtype=float)

    w_g = (-sgn * m_int * (Omega_E ** 2) / oci) if include_centrifugal else np.zeros(Nr, dtype=float)

    D_mOm = np.diag(m_int * Om_eff)
    I = np.eye(Nr, dtype=complex)
    phi_scale = float(np.median(te))
    coeff_sh = (B ** 2 / (ne * mi)) * (E_CHARGE * ne) * nu_sh_eff / phi_scale

    Am = np.zeros((3 * Nr, 3 * Nr), dtype=complex)
    Bm = np.zeros((3 * Nr, 3 * Nr), dtype=complex)
    s0, s1, s2 = slice(0, Nr), slice(Nr, 2 * Nr), slice(2 * Nr, 3 * Nr)

    # 1) Electron continuity with ambipolar Bohm end-loss + sheath current closure:
    #    -i*(w - m*Om)*n + i*w_*n*phi + nu_sh*(n + 0.5*T) - nu_sh_eff*(phi/Te - Lambda*T) = 0
    #    => w*n = m*Om*n + w_*n*phi - i*nu_sh*(n + 0.5*T) + i*nu_sh_eff*(phi/Te - Lambda*T)
    Bm[s0, s0] = I
    Am[s0, s0] = D_mOm - 1j * np.diag(nu_sh)
    Am[s0, s1] = np.diag(-sgn * (m_int / (r * B)) * dln_n * phi_scale) + 1j * np.diag(nu_sh_eff * phi_scale / te)
    Am[s0, s2] = -1j * np.diag(nu_sh_eff * float(sheath_lambda) + 0.5 * nu_sh)

    # 2) Vorticity
    Bm[s1, s1] = L_n
    Am[s1, s0] = np.diag((B * oci / phi_scale) * w_g)
    Am[s1, s1] = (
        D_mOm @ L_n
        - 1j * float(nu_in_s) * L_n
        + (np.diag((m_int / r) * dW0_dr) if (include_kh and include_exb_shear) else 0.0)
        - 1j * np.diag(coeff_sh * phi_scale / te)
    )
    Am[s1, s2] = +1j * np.diag(coeff_sh * float(sheath_lambda))

    # 3) Electron temperature
    Bm[s2, s2] = I
    Am[s2, s1] = np.diag(-sgn * (m_int / (r * B)) * dln_T * phi_scale)
    Am[s2, s2] = D_mOm - 1j * np.diag(nu_T)

    evals, evecs = eig(Am, Bm)
    good = np.isfinite(evals) & (np.abs(evals) < 1e7)
    evals, evecs = evals[good], evecs[:, good]
    idx = int(np.argmax(evals.imag))
    w = complex(evals[idx])
    phi_r = evecs[s1, idx]
    ipk = int(np.argmax(np.abs(phi_r)))
    f_dop_pk = float(m_int * Om_eff[ipk] / (2.0 * np.pi))
    return {
        "omega_rad_s": w,
        "f_lab_hz": float(w.real / (2.0 * np.pi)),
        "f_plasma_at_peak_hz": float(w.real / (2.0 * np.pi) - f_dop_pk),
        "gamma_s": float(w.imag),
        "r_peak_cm": float(r[ipk] * 1e2),
        "phi_r": phi_r,
        "r_cm": np.asarray(r_cm, dtype=float),
    }
