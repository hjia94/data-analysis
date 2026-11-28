"""Jun-2026 linear stability benchmark: CWM vs. Braginskii drift-Alfvén vs. KH.

Tests four linear magnetized-plasma instability models directly on the measured
Jun-2026 Day-1 gas-puff scan profiles (Runs 06, 00–05; 23–45 ms at 800 G) from
``profiles_summary_jun2026.npz``, ``onset_summary_jun2026.npz``, and
``modes_dispersion_jun2026.npz``:

1. **Pure cylindrical Kelvin-Helmholtz (Rayleigh)** on smooth radial ``Vp(r)``
   fits (testing whether the ``m = -1`` inner shear layer is KH-unstable).
2. **1-D Axial 5-field Braginskii BVP with insulating ends (``j_par = 0``)**
   (pure bulk resistive-EM drift-Alfvén + centrifugal gravity, no sheath closure).
3. **1-D Axial 5-field Braginskii BVP with conducting Port-35 bias plate**
   (``j_par = +(phi - Lambda * T_e)`` at ``z = L_par``) — self-consistently
   predicts plasma-frame frequency, growth rate, axial localization near Port 35,
   and P29–P25 axial phase lag, with sensitivity scans over sheath heat flux
   ``gamma_sh``, ion-neutral drag ``nu_in``, perpendicular wavenumber ``k_perp``,
   near-onset vs. reference sweeps, and ``B0 = 600 / 800 / 1000 G``.
4. **1-D Radial cylindrical ``E x B``-sheared CWM** on smooth ``(ne, Te, Vp)``
   radial profiles, isolating radial ``E x B`` flow-shear stabilization.

Saves ``stability_linear_jun2026.npz`` in ``DATA_DIR`` and prints a concise
synthesis comparison table.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_stability.py
"""

import datetime
import glob
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from data_analysis.io import open_lapd, position_shots
from data_analysis.plasma.fields import reference_locked_phasor
from data_analysis.plasma.stability import (
    E_CHARGE,
    biased_sheath_parameters,
    fit_smooth_radial_profile,
    local_braginskii_state,
    solve_axial_braginskii_bvp,
    solve_local_cwm,
    solve_radial_cylindrical_modes,
)
from data_analysis.signal import clip_time_window
from data_analysis.utils import merge_save_npz

import Jun2026_IV as jiv
import Jun2026_modes as jmo
import Jun2026_onset as jon
import Jun2026_profiles as jpr

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "stability_linear_jun2026.npz"

DEFAULT_M = -1
DEFAULT_R_CM = 3.0
DEFAULT_R_HALF_CM = 1.0
DEFAULT_WALL_WIDTH_CM = 4.0       # inner-wall radial scale -> k_r = pi / 4 cm
DEFAULT_L_PAR_M = 14.0            # effective axial column length [m]
DEFAULT_K_PAR_M = 0.45            # representative 800 G inner-wall |k_par| [rad/m]
DEFAULT_B_SIGN_Z = +1
DEFAULT_N_NODES = 24
DEFAULT_R_GRID_CM = np.linspace(0.5, 12.0, 24)
DEFAULT_MIN_WIDTH_CM = 2.4        # smooth radial fit lower bound (prevents r->0 spikes)
KH_SENSITIVITY_MIN_WIDTHS_CM = (1.5, 2.0, 2.4, 3.0)
KH_SPLINE_SMOOTHING = 0.5         # quartic spline smoothing factor s on symmetrized Vp(r)

# Port-35 concentric disc electrode geometry & biasscope current-shunt factor
# (confirmed from HDF5 run descriptions: diameters 2", 4", 6", 8", 10"; shunts 1 V / 100 A).
BIASSCOPE_SHUNT_A_PER_V = 100.0
DISC_AREAS_M2 = {
    "2in": float(np.pi * (0.0254 ** 2)),
    "4in": float(np.pi * ((0.0508 ** 2) - (0.0254 ** 2))),
    "6in": float(np.pi * ((0.0762 ** 2) - (0.0508 ** 2))),
}
BIASSCOPE_PRE_MS = (8.0, 11.5)    # pre-bias baseline in discharge-start scope time
BIASSCOPE_DC_MS = (13.0, 16.5)    # bias-on DC plateau (1.0-4.5 ms bias-relative)
BIASSCOPE_AC_MS = (14.4, 16.8)    # saturated AC window (2.4-4.8 ms bias-relative)
L_P29_TO_P35_M = 1.92             # axial distance from P29 LP to Port-35 bias plate [m]
L_P28_TO_P35_M = 2.24             # axial distance from P28 B-dot to Port-35 bias plate [m]

# Calibrated P28 B-dot / P29-R LP sheath-closure benchmark defaults
SHEATH_PROFILE_RUN = "03"         # matched 35 ms puff, 800 G Day-1 IV-sweep run
SHEATH_BDOT_RUNS = (("26", 800.0), ("31", 600.0))
SHEATH_LP_MOTION_GROUP = "<Hermes>    p29_LP"
SHEATH_LP_SCOPE = "lpscope"
SHEATH_LP_CHAN = "C2"
SHEATH_BDOT_SCOPE = "bdotscope"
SHEATH_TMIN_MS = 2.4              # matches Jun2026_bdot_2d saturated window
SHEATH_TMAX_MS = 4.8
SHEATH_FALLBACK_J0_4IN = -0.436475
SHEATH_FALLBACK_REFF_4IN = 1.436583


def select_near_onset_sweep(t_start_ms, t_mid_ms, regimes, t_onset_med_ms, ref_sweep_idx):
    """Derive the sweep closest to instability onset from sweep timestamps and ``t_onset``.

    For stable runs (``t_onset_med_ms`` non-finite), returns ``ref_sweep_idx``.
    For unstable runs, selects the latest bias-on sweep (``t_mid_ms > 0``) that
    starts and has its midpoint before ``t_onset_med_ms`` (allowing ``"straddling"``
    sweeps such as Run 01 at 2.10 ms and Run 02 at 1.60 ms where gradients have
    finished developing, while returning the regime label explicitly so callers
    can distinguish strictly ``"pre_onset"`` from ``"straddling"``).
    """
    t0 = np.asarray(t_start_ms, dtype=float)
    tm = np.asarray(t_mid_ms, dtype=float)
    regs = [str(r) for r in regimes]
    ref_idx = int(ref_sweep_idx)
    if not np.isfinite(t_onset_med_ms):
        return ref_idx, regs[ref_idx]

    cands = [
        i for i in range(tm.size)
        if tm[i] > 0.0
        and t0[i] < float(t_onset_med_ms)
        and tm[i] <= float(t_onset_med_ms)
        and regs[i] in ("pre_onset", "straddling")
    ]
    if not cands:
        return ref_idx, regs[ref_idx]
    best = max(cands, key=lambda i: float(tm[i]))
    return int(best), regs[best]


def _kh_spline_sensitivity(x_cm, x_center_cm, vp_x, r_grid_cm, ne_g, te_g, *,
                           b0_g=800.0, m=DEFAULT_M, smoothing=KH_SPLINE_SMOOTHING):
    """Evaluate pure-KH growth on an even-mirrored quartic smoothing spline of ``Vp(r)``."""
    r_abs = np.round(np.abs(np.asarray(x_cm, dtype=float) - float(x_center_cm)), 3)
    ru = np.unique(r_abs[r_abs <= 15.0 + 1e-6])
    vp_sym = np.array([
        np.nanmean(vp_x[r_abs == rv]) if np.any(np.isfinite(vp_x[r_abs == rv])) else np.nan
        for rv in ru
    ], dtype=float)
    fin = np.isfinite(vp_sym)
    ru, vp_sym = ru[fin], vp_sym[fin]
    # Even mirror across r = 0 so spline enforces Vp'(0) = 0 cleanly
    pos = ru > 0.0
    r_mir = np.concatenate((-ru[pos][::-1], ru))
    v_mir = np.concatenate((vp_sym[pos][::-1], vp_sym))
    spl = UnivariateSpline(r_mir, v_mir, k=4, s=float(smoothing))
    vp_sp = spl(r_grid_cm)
    res = solve_radial_cylindrical_modes(
        r_grid_cm, ne_g, te_g, vp_sp, b0_g=b0_g, m=m, model="pure_kh",
    )
    return float(res["gamma_s"])


def _extract_wall_state(pf, run_num, swp_idx, *, b0_g=800.0,
                        r_cm=DEFAULT_R_CM, r_half_cm=DEFAULT_R_HALF_CM,
                        m=DEFAULT_M, b_sign_z=DEFAULT_B_SIGN_Z):
    """Extract matched inner-wall median plasma parameters and sheath Lambda."""
    x = pf[f"{run_num}__x_cm"]
    xc = float(pf[f"{run_num}__x_center_cm"])
    r = np.abs(x - xc)
    band = (r >= r_cm - r_half_cm) & (r <= r_cm + r_half_cm)

    ne = float(np.nanmedian(pf[f"{run_num}__ne"][band, swp_idx]))
    te = float(np.nanmedian(pf[f"{run_num}__Te"][band, swp_idx]))
    dn = float(np.nanmedian(pf[f"{run_num}__dln_ne_dr"][band, swp_idx]))
    dt = float(np.nanmedian(pf[f"{run_num}__dln_te_dr"][band, swp_idx]))
    er = float(np.nanmedian(pf[f"{run_num}__er_r_v_cm"][band, swp_idx]))
    lam = float(np.nanmedian(pf[f"{run_num}__lambda"][band, swp_idx]))

    state = local_braginskii_state(
        ne, te, b0_g, dn, dt, er_v_per_cm=er,
        r_cm=r_cm, m=m, mu=4.0, Z=1, b_sign_z=b_sign_z,
    )
    return state, lam


def _pack_axial(mode_summary):
    if mode_summary is None:
        return {
            "f_hz": np.nan, "gamma_s": np.nan, "k_par_rms_m": np.nan,
            "j_end_rel": np.nan, "phi_end_rel": np.nan, "dphi_ab_deg": np.nan,
            "phi_over_n": np.nan, "T_over_n": np.nan,
        }
    return {
        "f_hz": mode_summary.f_hz,
        "gamma_s": mode_summary.gamma_s,
        "k_par_rms_m": mode_summary.k_par_rms_m,
        "j_end_rel": mode_summary.j_end_rel,
        "phi_end_rel": mode_summary.phi_end_rel,
        "dphi_ab_deg": mode_summary.dphi_ab_deg,
        "phi_over_n": mode_summary.phi_over_n,
        "T_over_n": mode_summary.T_over_n,
    }


def analyze_puff_scan(data_dir=DATA_DIR, *, l_par_m=DEFAULT_L_PAR_M,
                      wall_width_cm=DEFAULT_WALL_WIDTH_CM,
                      k_par_m=DEFAULT_K_PAR_M, n_nodes=DEFAULT_N_NODES,
                      r_grid_cm=DEFAULT_R_GRID_CM):
    """Run all linear stability models across the Day-1 23–45 ms puff scan."""
    prof_p = os.path.join(data_dir, jpr.OUT_NAME)
    onset_p = os.path.join(data_dir, jon.OUT_NAME)
    modes_p = os.path.join(data_dir, jmo.OUT_NAME)

    results = {}
    with np.load(prof_p, allow_pickle=True) as pf, \
         np.load(onset_p, allow_pickle=True) as of, \
         np.load(modes_p, allow_pickle=True) as mf:

        k_r = np.pi / (float(wall_width_cm) * 1e-2)

        for spec in jpr.PROFILE_RUNS:
            rnum = spec.run_num
            ref_swp = int(pf[f"{rnum}__ref_sweep_idx"])
            t_mid = pf[f"{rnum}__t_mid_ms"]
            t_start = pf[f"{rnum}__t_start_ms"]
            regs = pf[f"{rnum}__regimes"]
            t_on_med = float(of[f"{rnum}__t_onset_med_ms"])
            near_swp, near_regime = select_near_onset_sweep(
                t_start, t_mid, regs, t_on_med, ref_swp
            )

            st_ref, lam_ref = _extract_wall_state(pf, rnum, ref_swp, b0_g=spec.b0_g)
            st_near, lam_near = _extract_wall_state(pf, rnum, near_swp, b0_g=spec.b0_g)
            k_perp = float(np.hypot(st_ref.k_theta_m, k_r))

            # 1. Local 2-field CWM (ref sweep, with resistivity & chi_par)
            w_loc_cwm, _ = solve_local_cwm(
                st_ref, k_perp, l_par_m=l_par_m, sheath_lambda=lam_ref,
                n_ends=1.0, include_res=True, include_chi=True, k_par_eff_m=k_par_m,
            )

            # 2. 1-D Axial Braginskii BVP — insulating ends (pure bulk drift-Alfven)
            ax_ins = solve_axial_braginskii_bvp(
                st_ref, k_perp, l_par_m=l_par_m, sheath_lambda=lam_ref,
                bc="insulating", include_centrifugal=True, n_nodes=n_nodes,
            )

            # 3. 1-D Axial Braginskii BVP — conducting Port-35 bias plate (ref & near-onset)
            ax_cwm_ref = solve_axial_braginskii_bvp(
                st_ref, k_perp, l_par_m=l_par_m, sheath_lambda=lam_ref,
                bc="conducting_1", include_centrifugal=True, n_nodes=n_nodes,
            )
            ax_cwm_ref_nocent = solve_axial_braginskii_bvp(
                st_ref, k_perp, l_par_m=l_par_m, sheath_lambda=lam_ref,
                bc="conducting_1", include_centrifugal=False, n_nodes=n_nodes,
            )
            ax_cwm_near = solve_axial_braginskii_bvp(
                st_near, k_perp, l_par_m=l_par_m, sheath_lambda=lam_near,
                bc="conducting_1", include_centrifugal=True, n_nodes=n_nodes,
            )
            ax_cwm_near_gsh1 = solve_axial_braginskii_bvp(
                st_near, k_perp, l_par_m=l_par_m, sheath_lambda=lam_near,
                bc="conducting_1", include_centrifugal=True,
                sheath_heat_gamma=1.0, n_nodes=n_nodes,
            )
            ax_cwm_near_nu15 = solve_axial_braginskii_bvp(
                st_near, k_perp, l_par_m=l_par_m, sheath_lambda=lam_near,
                bc="conducting_1", include_centrifugal=True,
                nu_in_s=1500.0, n_nodes=n_nodes,
            )

            # 4. 1-D Radial cylindrical modes on smooth near-onset profiles
            x = pf[f"{rnum}__x_cm"]
            xc = float(pf[f"{rnum}__x_center_cm"])
            ne_g, _ = fit_smooth_radial_profile(
                x, xc, pf[f"{rnum}__ne"][:, near_swp], r_grid_cm,
                min_width_cm=DEFAULT_MIN_WIDTH_CM,
            )
            te_g, _ = fit_smooth_radial_profile(
                x, xc, pf[f"{rnum}__Te"][:, near_swp], r_grid_cm,
                min_width_cm=DEFAULT_MIN_WIDTH_CM,
            )
            vp_g, _ = fit_smooth_radial_profile(
                x, xc, pf[f"{rnum}__Vp"][:, near_swp], r_grid_cm,
                min_width_cm=DEFAULT_MIN_WIDTH_CM,
            )
            ne_g = np.maximum(ne_g, 1e10)
            te_g = np.maximum(te_g, 0.1)

            rad_kh = solve_radial_cylindrical_modes(
                r_grid_cm, ne_g, te_g, vp_g, b0_g=spec.b0_g, m=DEFAULT_M,
                model="pure_kh",
            )
            # KH profile-fit sensitivity (varying 2-Gaussian min_width_cm + quartic smoothing spline)
            kh_sens_gammas = []
            for mw in KH_SENSITIVITY_MIN_WIDTHS_CM:
                vp_mw, _ = fit_smooth_radial_profile(
                    x, xc, pf[f"{rnum}__Vp"][:, near_swp], r_grid_cm, min_width_cm=mw,
                )
                kh_mw = solve_radial_cylindrical_modes(
                    r_grid_cm, ne_g, te_g, vp_mw, b0_g=spec.b0_g, m=DEFAULT_M, model="pure_kh",
                )
                kh_sens_gammas.append(float(kh_mw["gamma_s"]))
            kh_spline_gamma = _kh_spline_sensitivity(
                x, xc, pf[f"{rnum}__Vp"][:, near_swp], r_grid_cm, ne_g, te_g,
                b0_g=spec.b0_g, m=DEFAULT_M,
            )
            kh_max_gamma = float(max(kh_sens_gammas + [kh_spline_gamma]))
            rad_cwm_nosh = solve_radial_cylindrical_modes(
                r_grid_cm, ne_g, te_g, vp_g, b0_g=spec.b0_g, m=DEFAULT_M,
                model="cwm_sheath", l_par_m=l_par_m, sheath_lambda=lam_near,
                k_par_m=k_par_m, include_centrifugal=False, include_exb_shear=False,
            )
            rad_cwm_sh = solve_radial_cylindrical_modes(
                r_grid_cm, ne_g, te_g, vp_g, b0_g=spec.b0_g, m=DEFAULT_M,
                model="cwm_sheath", l_par_m=l_par_m, sheath_lambda=lam_near,
                k_par_m=k_par_m, include_centrifugal=False, include_exb_shear=True,
            )
            rad_cwm_sh_cent = solve_radial_cylindrical_modes(
                r_grid_cm, ne_g, te_g, vp_g, b0_g=spec.b0_g, m=DEFAULT_M,
                model="cwm_sheath", l_par_m=l_par_m, sheath_lambda=lam_near,
                k_par_m=k_par_m, include_centrifugal=True, include_exb_shear=True,
            )

            # Experimental reference values
            f_lab_exp = float(of[f"{rnum}__f_peak_hz"])
            gamma_exp = float(of[f"{rnum}__gamma_med_s"])
            unstable = bool(np.isfinite(of[f"{rnum}__t_onset_med_ms"]))
            f_pl_exp = float(mf[f"disp__{rnum}__f_plasma_exb_hz"])

            row = {
                "gas_puff_ms": float(spec.gas_puff_ms),
                "b0_g": float(spec.b0_g),
                "ref_sweep_idx": ref_swp,
                "ref_sweep_t_ms": float(t_mid[ref_swp]),
                "near_sweep_idx": near_swp,
                "near_sweep_t_ms": float(t_mid[near_swp]),
                "near_sweep_regime": str(near_regime),
                "k_perp_m": k_perp,
                "lambda_ref": lam_ref,
                "lambda_near": lam_near,
                "exp_unstable": unstable,
                "exp_f_lab_hz": f_lab_exp,
                "exp_f_plasma_exb_hz": f_pl_exp,
                "exp_gamma_s": gamma_exp,
                "local_cwm_f_hz": float(w_loc_cwm.real / (2.0 * np.pi)),
                "local_cwm_gamma_s": float(w_loc_cwm.imag),
                "rad_kh_f_lab_hz": rad_kh["f_lab_hz"],
                "rad_kh_gamma_s": rad_kh["gamma_s"],
                "rad_kh_spline_gamma_s": kh_spline_gamma,
                "rad_kh_sens_max_gamma_s": kh_max_gamma,
                "rad_cwm_nosh_f_lab_hz": rad_cwm_nosh["f_lab_hz"],
                "rad_cwm_nosh_gamma_s": rad_cwm_nosh["gamma_s"],
                "rad_cwm_nosh_r_pk_cm": rad_cwm_nosh["r_peak_cm"],
                "rad_cwm_sh_f_lab_hz": rad_cwm_sh["f_lab_hz"],
                "rad_cwm_sh_gamma_s": rad_cwm_sh["gamma_s"],
                "rad_cwm_sh_r_pk_cm": rad_cwm_sh["r_peak_cm"],
                "rad_cwm_sh_cent_f_lab_hz": rad_cwm_sh_cent["f_lab_hz"],
                "rad_cwm_sh_cent_gamma_s": rad_cwm_sh_cent["gamma_s"],
                "rad_cwm_sh_cent_r_pk_cm": rad_cwm_sh_cent["r_peak_cm"],
                "ax_cwm_near_z_m": ax_cwm_near["z_m"],
                "ax_cwm_near_phi_abs": (
                    np.abs(ax_cwm_near["phi_z"]) / max(float(np.max(np.abs(ax_cwm_near["phi_z"]))), 1e-30)
                    if "phi_z" in ax_cwm_near else np.full_like(ax_cwm_near["z_m"], np.nan)
                ),
                "ax_cwm_near_j_abs": (
                    np.abs(ax_cwm_near["j_z"]) / max(float(np.max(np.abs(ax_cwm_near["j_z"]))), 1e-30)
                    if "j_z" in ax_cwm_near else np.full_like(ax_cwm_near["z_m"], np.nan)
                ),
            }
            for pfx, summary in [
                ("ax_ins_top", ax_ins["top"]),
                ("ax_cwm_ref", ax_cwm_ref["cwm"]),
                ("ax_cwm_ref_nocent", ax_cwm_ref_nocent["cwm"]),
                ("ax_cwm_near", ax_cwm_near["cwm"]),
                ("ax_cwm_near_gsh1", ax_cwm_near_gsh1["cwm"]),
                ("ax_cwm_near_nu15", ax_cwm_near_nu15["cwm"]),
            ]:
                for k, v in _pack_axial(summary).items():
                    row[f"{pfx}_{k}"] = v
            results[rnum] = row

    return results


def analyze_b0_scaling(data_dir=DATA_DIR, *, ref_run="02", b0_list=(600.0, 800.0, 1000.0),
                       l_par_m=DEFAULT_L_PAR_M, wall_width_cm=DEFAULT_WALL_WIDTH_CM,
                       n_nodes=DEFAULT_N_NODES):
    """Evaluate 1-D Axial CWM BVP across ``B0 = 600, 800, 1000 G`` using Run 02 profile."""
    prof_p = os.path.join(data_dir, jpr.OUT_NAME)
    onset_p = os.path.join(data_dir, jon.OUT_NAME)
    k_r = np.pi / (float(wall_width_cm) * 1e-2)
    out = {}
    with np.load(prof_p, allow_pickle=True) as pf, np.load(onset_p, allow_pickle=True) as of:
        swp, _ = select_near_onset_sweep(
            pf[f"{ref_run}__t_start_ms"], pf[f"{ref_run}__t_mid_ms"],
            pf[f"{ref_run}__regimes"], float(of[f"{ref_run}__t_onset_med_ms"]),
            int(pf[f"{ref_run}__ref_sweep_idx"]),
        )
        for b0_g in b0_list:
            st, lam = _extract_wall_state(pf, ref_run, swp, b0_g=b0_g)
            k_perp = float(np.hypot(st.k_theta_m, k_r))
            res = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam,
                bc="conducting_1", include_centrifugal=True, n_nodes=n_nodes,
            )
            cwm = res["cwm"] if res["cwm"] is not None else res["top"]
            f_dop = float(DEFAULT_M * st.omega_E_rad_s / (2.0 * np.pi))
            out[f"{int(b0_g)}G"] = {
                "b0_g": float(b0_g),
                "f_dop_const_er_hz": f_dop,
                "f_plasma_hz": cwm.f_hz if cwm else np.nan,
                "f_lab_const_er_hz": (f_dop + cwm.f_hz) if cwm else np.nan,
                "gamma_s": cwm.gamma_s if cwm else np.nan,
                "dphi_ab_deg": cwm.dphi_ab_deg if cwm else np.nan,
                "k_par_rms_m": cwm.k_par_rms_m if cwm else np.nan,
            }
    return out


def extract_biasscope_dc_impedance(data_dir=DATA_DIR, runs=None, onset_table=None, *,
                                   max_shots=60):
    """Extract Port-35 disc DC voltages, DC currents [A], and AC circuit impedance ``Z_ext(f0)``.

    Reads ``biasscope`` channels ``C1-C3`` (2"/4"/6" currents via ``1 V / 100 A`` shunts)
    and ``C4-C7`` (2"/4"/6"/8" bias voltages, already x200-scaled in scope headers).
    Returns ``{run_num: {...}}``. Raises ``ImportError`` / ``FileNotFoundError`` if
    raw HDF5 files or ``scope_io`` are unavailable.
    """
    specs = jpr.PROFILE_RUNS if runs is None else tuple(s for s in jpr.PROFILE_RUNS if s.run_num in runs)
    out = {}
    for spec in specs:
        rnum = spec.run_num
        hits = sorted(glob.glob(os.path.join(data_dir, f"{rnum}-*.hdf5")))
        if not hits:
            raise FileNotFoundError(f"no HDF5 for run {rnum} in {data_dir}")
        run = open_lapd(hits[0])
        tb = run.time_array(scope_name="biasscope")
        dt_b = float(tb[1] - tb[0])
        ip0, ip1 = clip_time_window(tb, *BIASSCOPE_PRE_MS)
        id0, id1 = clip_time_window(tb, *BIASSCOPE_DC_MS)
        ia0, ia1 = clip_time_window(tb, *BIASSCOPE_AC_MS)

        all_shots = np.asarray(run.shots(scope_name="biasscope"), dtype=int)
        idx = np.unique(np.linspace(0, all_shots.size - 1, min(all_shots.size, max_shots), dtype=int))
        sel = all_shots[idx].tolist()

        f0 = (float(onset_table[rnum]["f_peak_hz"])
              if (onset_table and rnum in onset_table and np.isfinite(onset_table[rnum]["f_peak_hz"]))
              else 3500.0)

        rec = {"gas_puff_ms": float(spec.gas_puff_ms), "f0_hz": float(f0)}
        ac_segs = {}
        for ch, key, scale in (
            ("C1", "I_2in", BIASSCOPE_SHUNT_A_PER_V),
            ("C2", "I_4in", BIASSCOPE_SHUNT_A_PER_V),
            ("C3", "I_6in", BIASSCOPE_SHUNT_A_PER_V),
            ("C4", "V_2in", 1.0),
            ("C5", "V_4in", 1.0),
            ("C6", "V_6in", 1.0),
            ("C7", "V_8in", 1.0),
        ):
            st, _ = run.channel(ch, scope_name="biasscope", shots=sel)
            arr = np.asarray(st, dtype=float) * scale
            pre = float(np.nanmean(arr[:, ip0:ip1]))
            dc = float(np.nanmean(arr[:, id0:id1]))
            rec[f"{key}_pre"] = pre
            rec[f"{key}_dc"] = dc
            rec[f"{key}_delta"] = dc - pre
            seg = arr[:, ia0:ia1]
            ac_segs[key] = seg - np.nanmean(seg, axis=1, keepdims=True)

        # AC external circuit impedance Z_ext(f0) = dV / dI locked to disc current.
        # Physical passive supply sag has angle(Z_ext) ~ +-180 deg -> R_ext = -Re(Z_ext) > 0.
        for lbl in ("2in", "4in", "6in"):
            dv_c, g2_vi = reference_locked_phasor(ac_segs[f"V_{lbl}"], ac_segs[f"I_{lbl}"], dt_b, f0)
            di_c, _ = reference_locked_phasor(ac_segs[f"I_{lbl}"], ac_segs[f"I_{lbl}"], dt_b, f0)
            z = complex(dv_c / di_c) if abs(di_c) > 0.0 else (np.nan + 0j)
            rec[f"Z_{lbl}_abs_ohm"] = float(abs(z))
            rec[f"Z_{lbl}_phase_deg"] = float(np.degrees(np.angle(z)))
            rec[f"R_ext_{lbl}_ohm"] = float(-z.real) if np.isfinite(z.real) else np.nan
            rec[f"g2_VI_{lbl}"] = float(g2_vi)
            rec[f"dV_{lbl}_v"] = float(abs(dv_c))
            rec[f"dI_{lbl}_a"] = float(abs(di_c))

        out[rnum] = rec
    return out


def _disc_mode_coupling(bdot_2d_npz=None):
    """Compute off-center ``m = -1`` disc current overlap ``c_m = |int Jz dA| / int |Jz| dA``.

    For a centered ``m != 0`` mode on a continuous disc/annulus, ``c_m = 0`` and
    an external series resistor ``R_ext`` does not see the mode. When the mode
    is off-center (``~2 cm`` in Jun-2026), ``c_m > 0`` and the mode-projected
    effective external impedance scales as ``c_m**2 * R_ext / R_sh``.
    Reads Run 26 ``h1`` 2-D ``Jz`` from ``bdot_2d_jun2026.npz`` when available;
    otherwise falls back to the measured 800 G values (``0.561`` / ``0.297``).

    Caveats: (1) derived from Run 26 (800 G, 35 ms puff) and applied across all
    puff runs; (2) assumes the Port-35 disc stack is centered on the P28 B-dot
    probe origin ``(0, 0)``; (3) uses ``Jz`` measured at Port 28 (2.24 m upstream
    of Port 35) rather than at the plate; and (4) incoherent noise in ``|Jz|``
    inflates the denominator, slightly biasing ``c_m`` low.
    """
    if bdot_2d_npz and os.path.isfile(bdot_2d_npz):
        with np.load(bdot_2d_npz) as bd:
            if "26__h1__Jz" in bd.files:
                x_cm = bd["26__x_cm"]
                y_cm = bd["26__y_cm"]
                Jz = bd["26__h1__Jz"]
                X, Y = np.meshgrid(x_cm, y_cm)
                R = np.hypot(X, Y)
                out = {}
                for lbl, mask in (("2in", R <= 2.54), ("4in", (R > 2.54) & (R <= 5.08))):
                    denom = float(np.nansum(np.abs(Jz[mask])))
                    out[lbl] = float(abs(np.nansum(Jz[mask])) / denom) if denom > 0.0 else 0.0
                return out["2in"], out["4in"], "bdot_2d_run26_h1"
    return 0.561, 0.297, "fallback_run26_h1"


def analyze_biased_sheath(data_dir=DATA_DIR, *, profiles_npz=None, onset_npz=None,
                          bdot_2d_npz=None, l_par_m=DEFAULT_L_PAR_M,
                          wall_width_cm=DEFAULT_WALL_WIDTH_CM,
                          n_nodes=DEFAULT_N_NODES, max_shots=60, biasscope_data=None):
    """Benchmark biased-sheath, mode-coupled external-impedance, electron-saturation, and CC BVPs.

    All BVP inputs are matched to the **4" annulus** (``r = 2.54-5.08 cm``,
    covering the steep inner wall ``r = 3.0 +- 1.0 cm``), using baseline-corrected
    ``I_delta = I_dc - I_pre`` (verified against ``V_bias > V_p``) and the
    off-center ``m = -1`` disc overlap ``c_4in`` from ``bdot_2d_jun2026.npz``
    (``r_ext_eff_4in = c_4in**2 * R_ext_4in / R_sh_4in``). Matched 2" disc
    parameters (``r = 0-2.54 cm``, evaluated on the hollow-core plasma footprint
    ``ne_2in, Te_2in, lambda_2in``) are stored alongside without mixing.
    """
    prof_path = profiles_npz or os.path.join(data_dir, jpr.OUT_NAME)
    onset_path = onset_npz or os.path.join(data_dir, jon.OUT_NAME)
    bdot_path = bdot_2d_npz or os.path.join(data_dir, "bdot_2d_jun2026.npz")
    onset_table = jon.load_onset_table(onset_path) if os.path.isfile(onset_path) else {}
    c_2in, c_4in, c_src = _disc_mode_coupling(bdot_path)

    if biasscope_data is None:
        biasscope_data = extract_biasscope_dc_impedance(
            data_dir=data_dir, onset_table=onset_table, max_shots=max_shots,
        )

    k_r = np.pi / (float(wall_width_cm) * 1e-2)
    out = {}
    with np.load(prof_path, allow_pickle=True) as pf:
        for spec in jpr.PROFILE_RUNS:
            r = spec.run_num
            if r not in biasscope_data or f"{r}__ref_sweep_idx" not in pf.files:
                continue
            bs = biasscope_data[r]
            ref_idx = int(pf[f"{r}__ref_sweep_idx"])
            st, lam = _extract_wall_state(pf, r, ref_idx, b0_g=800.0)
            k_perp = float(np.hypot(st.k_theta_m, k_r))

            # Matched footprint Vp, ne, Te, lambda on 2" core disc (r <= 2.54 cm)
            # and Vp on 4" wall annulus (2.54-5.08 cm)
            x_cm = pf[f"{r}__x_cm"]
            xc = float(pf[f"{r}__x_center_cm"])
            rr = np.abs(x_cm - xc)
            vp_prof = pf[f"{r}__Vp"][:, ref_idx]
            ne_prof = pf[f"{r}__ne"][:, ref_idx]
            te_prof = pf[f"{r}__Te"][:, ref_idx]
            lam_prof = pf[f"{r}__lambda"][:, ref_idx]
            m2 = (rr <= 2.54) & np.isfinite(vp_prof)
            m4 = (rr >= 2.54) & (rr < 5.08) & np.isfinite(vp_prof)
            m2_ne = (rr <= 2.54) & np.isfinite(ne_prof) & (ne_prof > 0.0)
            m2_te = (rr <= 2.54) & np.isfinite(te_prof) & (te_prof > 0.0)
            m2_lam = (rr <= 2.54) & np.isfinite(lam_prof) & (lam_prof > 0.0)
            vp_2in = float(np.nanmedian(vp_prof[m2])) if np.any(m2) else float(pf[f"{r}__ref_vp_core_v"])
            vp_4in = float(np.nanmedian(vp_prof[m4])) if np.any(m4) else float(pf[f"{r}__ref_vp_ring_v"])
            ne_2in_m3 = (
                float(np.nanmedian(ne_prof[m2_ne])) if np.any(m2_ne) else float(pf[f"{r}__ref_ne_core_cm3"])
            ) * 1e6
            te_2in_ev = float(np.nanmedian(te_prof[m2_te])) if np.any(m2_te) else float(pf[f"{r}__ref_te_core_ev"])
            lam_2in = float(np.nanmedian(lam_prof[m2_lam])) if np.any(m2_lam) else lam

            # Use baseline-corrected currents (I_dc - I_pre) and verify shunt sign against V_bias > V_p
            i2_net = float(bs["I_2in_delta"])
            i4_net = float(bs["I_4in_delta"])
            sign_status = {}
            for lbl, vb, vp_d, inet in (
                ("2in", float(bs["V_2in_dc"]), vp_2in, i2_net),
                ("4in", float(bs["V_4in_dc"]), vp_4in, i4_net),
            ):
                if vb > vp_d + 1.0:
                    if inet <= 0.0:
                        raise ValueError(
                            f"Run {r} {lbl}: V_bias={vb:.1f} V > V_p={vp_d:.1f} V expects positive "
                            f"electron-collection shunt reading, got I_delta={inet:+.2f} A; check shunt polarity"
                        )
                    sign_status[lbl] = "verified"
                else:
                    sign_status[lbl] = "not_checked"
            if all(v == "verified" for v in sign_status.values()):
                overall_sign_status = "verified"
            elif any(v == "verified" for v in sign_status.values()):
                overall_sign_status = "partial"
            else:
                overall_sign_status = "not_checked"

            # Fallback R_ext (when Run 06/05 has low AC g2_VI) uses typical passive pulser sag 0.15 / 0.18 Ohm
            r_ext_2 = float(bs["R_ext_2in_ohm"]) if (bs["g2_VI_2in"] >= 0.5 and bs["R_ext_2in_ohm"] > 0.0) else 0.15
            r_ext_4 = float(bs["R_ext_4in_ohm"]) if (bs["g2_VI_4in"] >= 0.5 and bs["R_ext_4in_ohm"] > 0.0) else 0.18
            p2 = biased_sheath_parameters(
                i2_net, DISC_AREAS_M2["2in"], r_ext_2,
                ne_2in_m3, te_2in_ev, sheath_lambda=lam_2in, l_ohm_m=L_P29_TO_P35_M,
                mode_coupling=c_2in,
            )
            p4 = biased_sheath_parameters(
                i4_net, DISC_AREAS_M2["4in"], r_ext_4,
                st.ne_m3, st.te_ev, sheath_lambda=lam, l_ohm_m=L_P29_TO_P35_M,
                mode_coupling=c_4in,
            )

            # Matched 4" annulus parameters (spans r=2.54-5.08 cm over the inner wall)
            j0_w = p4["j0_norm"]
            r_eff_4 = p4["r_ext_eff_hat"]

            res_flt = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="conducting_1", n_nodes=n_nodes,
            )
            res_flt_r = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="conducting_1",
                r_ext_hat=r_eff_4, n_nodes=n_nodes,
            )
            res_bsh_r = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="conducting_1",
                j0_norm=j0_w, r_ext_hat=r_eff_4, n_nodes=n_nodes,
            )
            res_bsh_cc = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="conducting_1",
                j0_norm=j0_w, r_ext_hat=r_eff_4, include_current_convective=True, n_nodes=n_nodes,
            )
            res_esat_cc = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="conducting_1",
                j0_norm=j0_w, r_ext_hat=0.0, sheath_conductance=0.0, sheath_lambda_eff=0.0,
                include_current_convective=True, n_nodes=n_nodes,
            )
            res_ins_cc = solve_axial_braginskii_bvp(
                st, k_perp, l_par_m=l_par_m, sheath_lambda=lam, bc="insulating",
                j0_norm=j0_w, include_current_convective=True, n_nodes=n_nodes,
            )

            row = {
                **bs,
                "disc_coupling_source": c_src,
                "mode_coupling_2in": float(c_2in),
                "mode_coupling_4in": float(c_4in),
                "vp_2in_v": vp_2in,
                "vp_4in_v": vp_4in,
                "ne_2in_m3": ne_2in_m3,
                "te_2in_ev": te_2in_ev,
                "lambda_2in": lam_2in,
                "vp_minus_v2_v": vp_2in - float(bs["V_2in_dc"]),
                "vp_minus_v4_v": vp_4in - float(bs["V_4in_dc"]),
                "shunt_sign_status": overall_sign_status,
                "shunt_sign_2in_status": sign_status["2in"],
                "shunt_sign_4in_status": sign_status["4in"],
                "j0_2in_a_m2": p2["j0_a_m2"],
                "j0_2in_norm": p2["j0_norm"],
                "r_sh_2in_ohm": p2["r_sh_ohm"],
                "r_ext_hat_2in": p2["r_ext_hat"],
                "r_ext_eff_hat_2in": p2["r_ext_eff_hat"],
                "electron_sat_2in": p2["electron_saturated"],
                "electron_sat_frac_2in": p2["electron_sat_fraction"],
                "near_electron_sat_2in": p2["near_electron_sat"],
                "dv_ohm_2in_v": p2["dv_ohm_v"],
                "j0_4in_a_m2": p4["j0_a_m2"],
                "j0_4in_norm": p4["j0_norm"],
                "r_sh_4in_ohm": p4["r_sh_ohm"],
                "r_ext_hat_4in": p4["r_ext_hat"],
                "r_ext_eff_hat_4in": p4["r_ext_eff_hat"],
                "lambda_eff_4in": p4["lambda_eff"],
                "electron_sat_4in": p4["electron_saturated"],
                "electron_sat_frac_4in": p4["electron_sat_fraction"],
                "near_electron_sat_4in": p4["near_electron_sat"],
                "dv_ohm_4in_v": p4["dv_ohm_v"],
            }
            for tag, res, use_cwm in (
                ("flt", res_flt, True),
                ("flt_rext", res_flt_r, True),
                ("bsh_rext", res_bsh_r, True),
                ("bsh_cc", res_bsh_cc, True),
                ("esat_cc_top", res_esat_cc, False),
                ("ins_cc_top", res_ins_cc, False),
            ):
                mode = res["cwm"] if use_cwm else res["top"]
                for k, v in _pack_axial(mode).items():
                    row[f"{tag}_{k}"] = v
            out[r] = row
    return out


def _wrap_deg(deg):
    """Wrap angle(s) in degrees to ``(-180, +180]``."""
    return float(np.degrees(np.angle(np.exp(1j * np.radians(float(deg))))))


def _interp_complex(z_grid, arr, z_target):
    """Linearly interpolate a complex 1-D axial eigenfunction at ``z_target``."""
    z_g = np.asarray(z_grid, dtype=float)
    a = np.asarray(arr, dtype=complex)
    return complex(np.interp(float(z_target), z_g, a.real) + 1j * np.interp(float(z_target), z_g, a.imag))


def extract_lp_2d_dc_baseline(data_dir=DATA_DIR, runs=("26", "31"), *,
                              tmin_ms=SHEATH_TMIN_MS, tmax_ms=SHEATH_TMAX_MS,
                              grid_step_cm=1.0):
    """Extract 2-D DC ``I_sat`` baseline [V] for moving P29-R LP (``lpscope/C2``).

    Snaps ``<Hermes>    p29_LP`` positions onto the same ``grid_step_cm`` 2-D
    ``(ny, nx)`` grid and ``[tmin_ms, tmax_ms]`` post-onset window used by
    ``Jun2026_bdot_2d.py`` so ``|dI_sat / I_sat|`` is exact on the same probe.
    Also verifies the polarity sign of ``LP_dc`` (``+1`` when ``LP_dc > 0`` so
    ``+dV_LP`` corresponds to ``+dI_sat``).
    """
    out = {}
    for rnum in runs:
        hits = sorted(glob.glob(os.path.join(data_dir, f"{rnum}-*.hdf5")))
        if not hits:
            raise FileNotFoundError(f"no HDF5 for run {rnum} in {data_dir}")
        ifn = hits[0]
        run = open_lapd(ifn)
        pos_lp = jiv.read_lp_positions(ifn, motion_group_name=SHEATH_LP_MOTION_GROUP)
        tl = run.time_array(scope_name=SHEATH_LP_SCOPE)
        il0, il1 = clip_time_window(tl, float(tmin_ms), float(tmax_ms))

        all_shots = np.asarray(run.shots(scope_name=SHEATH_BDOT_SCOPE), dtype=int)
        shot_to_idx = {int(s): i for i, s in enumerate(all_shots)}
        st_lp, _ = run.channel(SHEATH_LP_CHAN, scope_name=SHEATH_LP_SCOPE, shots=all_shots.tolist())
        seg = np.asarray(st_lp[:, il0:il1], dtype=float)

        px = np.round(pos_lp.pos_array["x"][::pos_lp.nshot][:pos_lp.npos] / float(grid_step_cm)) * float(grid_step_cm)
        py = np.round(pos_lp.pos_array["y"][::pos_lp.nshot][:pos_lp.npos] / float(grid_step_cm)) * float(grid_step_cm)
        ux = np.unique(px)
        uy = np.unique(py)
        lp_dc = np.full((uy.size, ux.size), np.nan, dtype=float)
        for p_i in range(pos_lp.npos):
            sh = position_shots(pos_lp.shot_nums, p_i, pos_lp.nshot)
            idx = [shot_to_idx[int(s)] for s in sh if int(s) in shot_to_idx]
            if not idx:
                continue
            ix = int(np.searchsorted(ux, px[p_i]))
            iy = int(np.searchsorted(uy, py[p_i]))
            lp_dc[iy, ix] = float(np.nanmean(seg[idx]))

        fin = lp_dc[np.isfinite(lp_dc)]
        if fin.size == 0 or float(np.median(np.abs(fin))) == 0.0:
            raise ValueError(f"Run {rnum}: empty or zero P29-R LP DC baseline on {SHEATH_LP_SCOPE}/{SHEATH_LP_CHAN}")
        pol = +1 if float(np.median(fin)) > 0.0 else -1
        if np.any(fin * pol <= 0.0):
            raise ValueError(f"Run {rnum}: mixed-sign P29-R LP DC baseline (min={np.min(fin):.4f}, max={np.max(fin):.4f})")

        out[str(rnum)] = {
            "x_cm": ux,
            "y_cm": uy,
            "lp_dc": lp_dc,
            "lp_polarity_sign": int(pol),
            "source": f"hdf5_{SHEATH_LP_SCOPE}_{SHEATH_LP_CHAN}",
        }
    return out


def evaluate_axial_sheath_closure(res, *, z_p28_m, z_p29_m=None, l_par_m=None):
    """Evaluate 1-D axial CWM BVP eigenmode sheath-closure ratios and absolute phases.

    Uses the linearized ion-saturation current fluctuation
    ``i_sat_hat = n_hat + 0.5 * T_hat`` (crucial when ``eta_e < 0`` makes
    ``dT_e/T_e`` and ``dn/n`` nearly 180 deg out of phase).

    Sign & phase convention
    -----------------------
    In :func:`solve_axial_braginskii_bvp`, ``z = L_par`` is the Port-35 bias
    plate (lab ``+z``, North), ``j_z = J_par / (e n_0 c_s)`` is conventional
    parallel current density along ``+z`` (negative for electrons collected into
    Port 35, matching ``bdot_2d_jun2026.npz`` under ``DEFAULT_CHANNEL_SIGNS =
    (-1, -1, +1)``), and perturbations follow ``exp(i * (m * theta - omega * t))``.
    Under the Doppler shift ``theta' = theta - Omega_E * t``,
    ``m*theta' - omega_plasma*t == m*theta - omega_lab*t`` with ``omega_lab > 0``
    (no complex conjugation), so ``angle(j_z(z_j) / i_sat_z(z_i))`` compares
    directly (sign ``+1``, ``0 deg`` offset) to experimental
    ``dphi(J_par - I_sat) = angle(Jz * conj(LP))`` from
    :func:`reference_locked_phasor` with ``convention="exp_minus_iwt"``.
    """
    mode = res.get("cwm")
    if mode is None or "j_z" not in res:
        return {
            "has_cwm": False,
            "f_hz": np.nan, "gamma_s": np.nan, "k_par_rms_m": np.nan,
            "j_end_rel": np.nan, "phi_end_rel": np.nan,
            "j_over_isat_p28": np.nan, "dphi_j_isat_p28_deg": np.nan,
            "j_over_n_p28": np.nan, "dphi_j_n_p28_deg": np.nan,
            "T_over_n_p28": np.nan, "dphi_T_n_p28_deg": np.nan,
            "isat_over_n_p28": np.nan,
            "j_over_isat_p28_p29": np.nan, "dphi_j_isat_p28_p29_deg": np.nan,
            "j_over_n_p28_p29": np.nan, "dphi_j_n_p28_p29_deg": np.nan,
            "j_over_isat_sheath": np.nan, "dphi_j_isat_sheath_deg": np.nan,
            "j_over_n_sheath": np.nan, "dphi_j_n_sheath_deg": np.nan,
            "T_over_n_sheath": np.nan, "dphi_T_n_sheath_deg": np.nan,
            "isat_over_n_sheath": np.nan,
        }

    z = np.asarray(res["z_m"], dtype=float)
    z_end = float(z[-1]) if l_par_m is None else float(l_par_m)
    z_28 = float(z_p28_m)
    z_29 = z_28 if z_p29_m is None else float(z_p29_m)

    n_z = np.asarray(res["n_z"], dtype=complex)
    T_z = np.asarray(res["T_z"], dtype=complex)
    j_z = np.asarray(res["j_z"], dtype=complex)
    isat_z = n_z + 0.5 * T_z

    def _ratio_phase(num, den):
        if abs(den) <= 1e-30:
            return np.nan, np.nan
        q = num / den
        return float(abs(q)), float(np.degrees(np.angle(q)))

    j_28 = _interp_complex(z, j_z, z_28)
    n_28 = _interp_complex(z, n_z, z_28)
    T_28 = _interp_complex(z, T_z, z_28)
    i_28 = _interp_complex(z, isat_z, z_28)
    n_29 = _interp_complex(z, n_z, z_29)
    i_29 = _interp_complex(z, isat_z, z_29)

    j_L = _interp_complex(z, j_z, z_end)
    n_L = _interp_complex(z, n_z, z_end)
    T_L = _interp_complex(z, T_z, z_end)
    i_L = _interp_complex(z, isat_z, z_end)

    ji_28, dp_ji_28 = _ratio_phase(j_28, i_28)
    jn_28, dp_jn_28 = _ratio_phase(j_28, n_28)
    Tn_28, dp_Tn_28 = _ratio_phase(T_28, n_28)
    in_28, _ = _ratio_phase(i_28, n_28)

    ji_2829, dp_ji_2829 = _ratio_phase(j_28, i_29)
    jn_2829, dp_jn_2829 = _ratio_phase(j_28, n_29)

    ji_L, dp_ji_L = _ratio_phase(j_L, i_L)
    jn_L, dp_jn_L = _ratio_phase(j_L, n_L)
    Tn_L, dp_Tn_L = _ratio_phase(T_L, n_L)
    in_L, _ = _ratio_phase(i_L, n_L)

    return {
        "has_cwm": True,
        "f_hz": float(mode.f_hz),
        "gamma_s": float(mode.gamma_s),
        "k_par_rms_m": float(mode.k_par_rms_m),
        "j_end_rel": float(mode.j_end_rel),
        "phi_end_rel": float(mode.phi_end_rel),
        "j_over_isat_p28": ji_28,
        "dphi_j_isat_p28_deg": dp_ji_28,
        "j_over_n_p28": jn_28,
        "dphi_j_n_p28_deg": dp_jn_28,
        "T_over_n_p28": Tn_28,
        "dphi_T_n_p28_deg": dp_Tn_28,
        "isat_over_n_p28": in_28,
        "j_over_isat_p28_p29": ji_2829,
        "dphi_j_isat_p28_p29_deg": dp_ji_2829,
        "j_over_n_p28_p29": jn_2829,
        "dphi_j_n_p28_p29_deg": dp_jn_2829,
        "j_over_isat_sheath": ji_L,
        "dphi_j_isat_sheath_deg": dp_ji_L,
        "j_over_n_sheath": jn_L,
        "dphi_j_n_sheath_deg": dp_jn_L,
        "T_over_n_sheath": Tn_L,
        "dphi_T_n_sheath_deg": dp_Tn_L,
        "isat_over_n_sheath": in_L,
    }


def _extract_profile_zone_jsat(pf, run_num, swp_idx, mask, *, b0_g=800.0):
    """Extract matched zone median ``(ne_cm3, te_ev, cs_m_s, j_sat_a_m2, dln_ne, dln_te, eta_e, lam)``."""
    ne = float(np.nanmedian(pf[f"{run_num}__ne"][mask, swp_idx]))
    te = float(np.nanmedian(pf[f"{run_num}__Te"][mask, swp_idx]))
    dn = float(np.nanmedian(pf[f"{run_num}__dln_ne_dr"][mask, swp_idx]))
    dt = float(np.nanmedian(pf[f"{run_num}__dln_te_dr"][mask, swp_idx]))
    lam = float(np.nanmedian(pf[f"{run_num}__lambda"][mask, swp_idx]))
    st = local_braginskii_state(
        ne, te, b0_g, dn, dt, er_v_per_cm=0.0, r_cm=DEFAULT_R_CM, m=DEFAULT_M,
    )
    jsat = float(E_CHARGE * st.ne_m3 * st.cs_m_s)
    eta_e = float(dt / dn) if abs(dn) > 1e-12 else np.nan
    return {
        "ne_cm3": ne,
        "te_ev": te,
        "cs_m_s": float(st.cs_m_s),
        "j_sat_a_m2": jsat,
        "dln_ne_dr_cm": dn,
        "dln_te_dr_cm": dt,
        "eta_e": eta_e,
        "lambda": lam,
    }


def analyze_sheath_closure(data_dir=DATA_DIR, *, profiles_npz=None, bdot_2d_npz=None,
                           profile_run=SHEATH_PROFILE_RUN,
                           bdot_runs=SHEATH_BDOT_RUNS,
                           l_par_m=DEFAULT_L_PAR_M,
                           wall_width_cm=DEFAULT_WALL_WIDTH_CM,
                           n_nodes=DEFAULT_N_NODES,
                           biased_res=None, lp_dc_data=None):
    """Benchmark calibrated 2-D ``J_par`` amplitude & absolute phase against 1-D CWM BVP.

    Compares BOTH:
    1. Dimensionless sheath-current ratio
       ``(|dJ_par| / (e n_0 c_s)) / (|dI_sat| / I_sat)``, using exact peak-vs-RMS
       accounting (:func:`reference_locked_phasor` returns peak complex amplitude
       ``|X| = sqrt(2) * X_rms``; dividing ``|dJ_par| / j_sat`` by
       ``|dI_sat| / I_sat_DC`` on the same P29-R probe cancels ``sqrt(2)``
       identically) and linearized ``dI_sat / I_sat = dn/n + 0.5 * dT_e / T_e``.
    2. Absolute phase ``dphi(J_par - I_sat)`` [deg] under verified ``C16``
       coil polarity (``DEFAULT_CHANNEL_SIGNS = (-1, -1, +1)``) and positive
       ``lpscope/C2`` DC baseline.

    Evaluates ``conducting_1``, ``flt+Reff4``, and ``biased_sheath+Reff4`` on
    the matched Run ``03`` (35 ms puff) inner-wall state at ``800 G`` (Run 26)
    and ``600 G`` (Run 31) at ``z_P28`` (``2.24 m`` upstream of Port 35),
    two-port ``P28/P29`` (``0.32 m`` probe separation), and the Port-35 sheath
    (``z = L_par``). Also tests how ``eta_e < 0`` (hollow column) flips
    ``dphi(j_par - I_sat)`` by ``~180 deg`` relative to a peaked-density
    (``eta_e > 0``) plasma.
    """
    prof_path = profiles_npz or os.path.join(data_dir, jpr.OUT_NAME)
    bdot_path = bdot_2d_npz or os.path.join(data_dir, "bdot_2d_jun2026.npz")

    out = {}

    # --- 1. Matched Run 03 equilibrium reference profiles (core, wall, BVP wall) ---
    with np.load(prof_path, allow_pickle=True) as pf:
        ref_swp = int(pf[f"{profile_run}__ref_sweep_idx"])
        x_cm = pf[f"{profile_run}__x_cm"]
        xc_cm = float(pf[f"{profile_run}__x_center_cm"])
        rr_p = np.abs(x_cm - xc_cm)

        prof_core = _extract_profile_zone_jsat(pf, profile_run, ref_swp, rr_p <= 2.5, b0_g=800.0)
        prof_wall = _extract_profile_zone_jsat(pf, profile_run, ref_swp, (rr_p > 2.5) & (rr_p <= 4.5), b0_g=800.0)
        prof_ring = _extract_profile_zone_jsat(
            pf, profile_run, ref_swp,
            (rr_p >= DEFAULT_R_CM - DEFAULT_R_HALF_CM) & (rr_p <= DEFAULT_R_CM + DEFAULT_R_HALF_CM),
            b0_g=800.0,
        )

        st_800, lam_800 = _extract_wall_state(pf, profile_run, ref_swp, b0_g=800.0)
        st_600, lam_600 = _extract_wall_state(pf, profile_run, ref_swp, b0_g=600.0)

        # Raw wall medians for the eta_e sign-flip test
        band_w = (rr_p >= DEFAULT_R_CM - DEFAULT_R_HALF_CM) & (rr_p <= DEFAULT_R_CM + DEFAULT_R_HALF_CM)
        ne_w = float(np.nanmedian(pf[f"{profile_run}__ne"][band_w, ref_swp]))
        te_w = float(np.nanmedian(pf[f"{profile_run}__Te"][band_w, ref_swp]))
        dn_w = float(np.nanmedian(pf[f"{profile_run}__dln_ne_dr"][band_w, ref_swp]))
        dt_w = float(np.nanmedian(pf[f"{profile_run}__dln_te_dr"][band_w, ref_swp]))
        er_w = float(np.nanmedian(pf[f"{profile_run}__er_r_v_cm"][band_w, ref_swp]))

    out["ref_profile"] = {
        "profile_run": str(profile_run),
        "ref_sweep_idx": int(ref_swp),
        **{f"core_{k}": v for k, v in prof_core.items()},
        **{f"wall_{k}": v for k, v in prof_wall.items()},
        **{f"wall_bvp_{k}": v for k, v in prof_ring.items()},
    }

    # --- 2. Experimental calibrated J_par & P29-R I_sat on Runs 26 & 31 ---
    run_nums = tuple(r for r, _ in bdot_runs)
    if lp_dc_data is None:
        try:
            lp_dc_map = extract_lp_2d_dc_baseline(data_dir=data_dir, runs=run_nums)
        except (FileNotFoundError, ImportError) as exc:
            print(f"  [note] P29-R LP DC baseline unavailable ({exc}); dI_sat/I_sat will be NaN")
            lp_dc_map = {}
    else:
        lp_dc_map = lp_dc_data

    exp_dc_ref_by_zone = {}
    with np.load(bdot_path, allow_pickle=True) as bd:
        for rnum, b0_g in bdot_runs:
            if f"{rnum}__h1__Jz" not in bd.files:
                continue
            cal_ok = bool(bd[f"{rnum}__bdot_calibrated"]) if f"{rnum}__bdot_calibrated" in bd.files else False
            pol_ok = (
                bool(bd[f"{rnum}__abs_coil_polarity_verified"])
                if f"{rnum}__abs_coil_polarity_verified" in bd.files else False
            )
            if not cal_ok:
                raise ValueError(f"Run {rnum}: bdot_2d_jun2026.npz is not calibrated (bdot_calibrated=False)")
            if not pol_ok:
                raise ValueError(f"Run {rnum}: absolute B-dot coil polarity is not verified")

            ux = np.asarray(bd[f"{rnum}__x_cm"], dtype=float)
            uy = np.asarray(bd[f"{rnum}__y_cm"], dtype=float)
            Jz = np.asarray(bd[f"{rnum}__h1__Jz"], dtype=complex)
            LP = np.asarray(bd[f"{rnum}__h1__LP"], dtype=complex)
            g2_LP = np.asarray(bd[f"{rnum}__h1__g2_LP"], dtype=float)
            cx = float(bd[f"{rnum}__h1__center_x_cm"])
            cy = float(bd[f"{rnum}__h1__center_y_cm"])
            f0_hz = float(bd[f"{rnum}__f0_hz"])

            XX, YY = np.meshgrid(ux, uy)
            RR = np.hypot(XX - cx, YY - cy)
            j_int_max = float(np.nanmax(np.abs(Jz[1:-1, 1:-1])))

            dc_rec = lp_dc_map.get(str(rnum))
            if dc_rec is not None:
                lp_dc = np.asarray(dc_rec["lp_dc"], dtype=float)
                if lp_dc.shape != LP.shape:
                    raise ValueError(
                        f"Run {rnum}: LP DC grid shape {lp_dc.shape} != bdot_2d LP shape {LP.shape}"
                    )
                if "x_cm" in dc_rec and "y_cm" in dc_rec:
                    if not (np.allclose(dc_rec["x_cm"], ux, atol=1e-6) and np.allclose(dc_rec["y_cm"], uy, atol=1e-6)):
                        raise ValueError(f"Run {rnum}: LP DC coordinate grid differs from bdot_2d_jun2026.npz")
                lp_pol = int(dc_rec.get("lp_polarity_sign", +1))
                dc_src = str(dc_rec.get("source", "supplied"))
            else:
                lp_dc = np.full_like(LP.real, np.nan)
                lp_pol = +1
                dc_src = "unavailable"

            LP_signed = float(lp_pol) * LP

            zones = (
                ("core", (RR <= 2.5), prof_core["j_sat_a_m2"]),
                ("wall", (RR > 2.5) & (RR <= 4.5), prof_wall["j_sat_a_m2"]),
                (
                    "ring_coh",
                    (RR >= 2.0) & (RR <= 5.0)
                    & np.isfinite(Jz) & np.isfinite(LP_signed) & np.isfinite(g2_LP)
                    & (np.abs(Jz) >= 0.3 * j_int_max) & (g2_LP >= 0.5),
                    prof_ring["j_sat_a_m2"],
                ),
            )

            rec = {
                "run_num": str(rnum),
                "b0_g": float(b0_g),
                "f0_hz": f0_hz,
                "bdot_calibrated": True,
                "abs_coil_polarity_verified": True,
                "lp_polarity_sign": int(lp_pol),
                "lp_dc_source": dc_src,
                "phasor_normalization": "peak_complex_amplitude_rms_eq_peak_over_sqrt2",
                "dphi_j_lp_stored_deg": (
                    float(bd[f"{rnum}__h1__dphi_J_minus_LP_deg"])
                    if f"{rnum}__h1__dphi_J_minus_LP_deg" in bd.files else np.nan
                ),
                "R_j_lp_stored": (
                    float(bd[f"{rnum}__h1__R_J_LP"])
                    if f"{rnum}__h1__R_J_LP" in bd.files else np.nan
                ),
            }

            for zname, zmask, jsat_ref in zones:
                valid = zmask & np.isfinite(Jz) & np.isfinite(LP_signed)
                n_pts = int(np.sum(valid))
                if n_pts == 0:
                    for suf in (
                        "n_pts", "j_par_peak_rms_a_m2", "j_par_time_rms_a_m2",
                        "j_sat_ref_a_m2", "j_par_norm_peak", "j_par_norm_rms",
                        "lp_peak_rms_v", "lp_dc_med_v",
                        "disat_over_isat_peak", "disat_over_isat_rms",
                        "j_over_isat_ratio", "dphi_j_isat_deg", "R_j_isat",
                    ):
                        rec[f"{zname}_{suf}"] = 0 if suf == "n_pts" else np.nan
                    continue

                jz_pk = float(np.sqrt(np.nanmean(np.abs(Jz[valid]) ** 2)))
                jz_rms = jz_pk / np.sqrt(2.0)
                j_norm_pk = jz_pk / float(jsat_ref)
                j_norm_rms = jz_rms / float(jsat_ref)

                lp_pk = float(np.sqrt(np.nanmean(np.abs(LP_signed[valid]) ** 2)))
                dc_vals = np.abs(lp_dc[valid])
                dc_med = float(np.nanmedian(dc_vals)) if np.any(np.isfinite(dc_vals)) else np.nan
                if np.isfinite(dc_med) and dc_med > 0.0:
                    di_pk = lp_pk / dc_med
                    di_rms = di_pk / np.sqrt(2.0)
                    ratio = j_norm_pk / di_pk if di_pk > 0.0 else np.nan
                else:
                    di_pk = di_rms = ratio = np.nan

                z_jl = np.mean(np.exp(1j * np.angle(Jz[valid] * np.conj(LP_signed[valid]))))
                dphi_deg = float(np.degrees(np.angle(z_jl)))
                R_jl = float(np.abs(z_jl))

                if np.isclose(b0_g, 800.0) and np.isfinite(dc_med) and dc_med > 0.0:
                    exp_dc_ref_by_zone[zname] = dc_med

                rec.update({
                    f"{zname}_n_pts": n_pts,
                    f"{zname}_j_par_peak_rms_a_m2": jz_pk,
                    f"{zname}_j_par_time_rms_a_m2": jz_rms,
                    f"{zname}_j_sat_ref_a_m2": float(jsat_ref),
                    f"{zname}_j_par_norm_peak": j_norm_pk,
                    f"{zname}_j_par_norm_rms": j_norm_rms,
                    f"{zname}_lp_peak_rms_v": lp_pk,
                    f"{zname}_lp_dc_med_v": dc_med,
                    f"{zname}_disat_over_isat_peak": di_pk,
                    f"{zname}_disat_over_isat_rms": di_rms,
                    f"{zname}_j_over_isat_ratio": ratio,
                    f"{zname}_dphi_j_isat_deg": dphi_deg,
                    f"{zname}_R_j_isat": R_jl,
                })

            out[f"exp_{rnum}"] = rec

    # For non-800 G runs (Run 31 at 600 G), add LP_dc-scaled j_sat & ratio using the
    # same P29-R probe's DC I_sat ratio relative to the 800 G run (Run 26).
    for rnum, b0_g in bdot_runs:
        key = f"exp_{rnum}"
        if key not in out:
            continue
        rec = out[key]
        for zname in ("core", "wall", "ring_coh"):
            dc_med = rec.get(f"{zname}_lp_dc_med_v", np.nan)
            dc_800 = exp_dc_ref_by_zone.get(zname, np.nan)
            ratio_raw = rec.get(f"{zname}_j_over_isat_ratio", np.nan)
            if (not np.isclose(b0_g, 800.0)) and np.isfinite(dc_med) and np.isfinite(dc_800) and dc_800 > 0.0:
                scale = float(dc_med / dc_800)
                rec[f"{zname}_lp_dc_scale_vs_800G"] = scale
                rec[f"{zname}_j_sat_dc_scaled_a_m2"] = float(rec[f"{zname}_j_sat_ref_a_m2"] * scale)
                rec[f"{zname}_j_over_isat_ratio_dc_scaled"] = float(ratio_raw / scale) if np.isfinite(ratio_raw) else np.nan
            else:
                rec[f"{zname}_lp_dc_scale_vs_800G"] = 1.0
                rec[f"{zname}_j_sat_dc_scaled_a_m2"] = float(rec.get(f"{zname}_j_sat_ref_a_m2", np.nan))
                rec[f"{zname}_j_over_isat_ratio_dc_scaled"] = float(ratio_raw) if np.isfinite(ratio_raw) else np.nan

    # --- 3. 1-D Axial CWM BVP predictions at 800 G and 600 G ---
    if biased_res and profile_run in biased_res:
        j0_4 = float(biased_res[profile_run]["j0_4in_norm"])
        r_eff_4 = float(biased_res[profile_run]["r_ext_eff_hat_4in"])
        bias_src = f"biased_res_{profile_run}"
    else:
        j0_4 = float(SHEATH_FALLBACK_J0_4IN)
        r_eff_4 = float(SHEATH_FALLBACK_REFF_4IN)
        bias_src = f"fallback_{profile_run}"

    k_r = np.pi / (float(wall_width_cm) * 1e-2)
    z_p28 = float(l_par_m) - float(L_P28_TO_P35_M)
    z_p29 = float(l_par_m) - float(L_P29_TO_P35_M)

    model_specs = (
        ("conducting_1", dict(bc="conducting_1")),
        ("flt_rext", dict(bc="conducting_1", r_ext_hat=r_eff_4)),
        ("bsh_rext", dict(bc="conducting_1", j0_norm=j0_4, r_ext_hat=r_eff_4)),
    )

    for lbl, st_b, lam_b, b0_g in (("bvp_800G", st_800, lam_800, 800.0), ("bvp_600G", st_600, lam_600, 600.0)):
        kp = float(np.hypot(st_b.k_theta_m, k_r))
        row = {
            "b0_g": float(b0_g),
            "l_par_m": float(l_par_m),
            "z_p28_m": z_p28,
            "z_p29_m": z_p29,
            "k_perp_m": kp,
            "sheath_lambda": float(lam_b),
            "j0_4in_norm": j0_4,
            "r_ext_eff_hat_4in": r_eff_4,
            "biased_params_source": bias_src,
            "eta_e": float(st_b.omega_star_T_rad_s / st_b.omega_star_n_rad_s),
        }
        for mtag, mkw in model_specs:
            res = solve_axial_braginskii_bvp(
                st_b, kp, l_par_m=l_par_m, sheath_lambda=lam_b,
                include_centrifugal=True, n_nodes=n_nodes, **mkw,
            )
            ev = evaluate_axial_sheath_closure(res, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=l_par_m)
            for k, v in ev.items():
                row[f"{mtag}_{k}"] = v
        out[lbl] = row

    # --- 4. Hollow (eta_e < 0) vs Peaked (eta_e > 0) ~180 deg phase-flip test ---
    # Disable centrifugal gravity (include_centrifugal=False) so rotational
    # interchange (unstable for dn/dr < 0) does not mask the CWM eigenmode.
    st_hollow = local_braginskii_state(
        ne_w, te_w, 800.0, +abs(dn_w), -abs(dt_w),
        er_v_per_cm=er_w, r_cm=DEFAULT_R_CM, m=DEFAULT_M, b_sign_z=DEFAULT_B_SIGN_Z,
    )
    st_peaked = local_braginskii_state(
        ne_w, te_w, 800.0, -abs(dn_w), -abs(dt_w),
        er_v_per_cm=er_w, r_cm=DEFAULT_R_CM, m=DEFAULT_M, b_sign_z=DEFAULT_B_SIGN_Z,
    )
    kp_flip = float(np.hypot(st_hollow.k_theta_m, k_r))
    flip_row = {
        "eta_e_hollow": float(st_hollow.omega_star_T_rad_s / st_hollow.omega_star_n_rad_s),
        "eta_e_peaked": float(st_peaked.omega_star_T_rad_s / st_peaked.omega_star_n_rad_s),
        "include_centrifugal": False,
    }
    for mtag, mkw in model_specs:
        res_h = solve_axial_braginskii_bvp(
            st_hollow, kp_flip, l_par_m=l_par_m, sheath_lambda=lam_800,
            include_centrifugal=False, n_nodes=n_nodes, **mkw,
        )
        res_p = solve_axial_braginskii_bvp(
            st_peaked, kp_flip, l_par_m=l_par_m, sheath_lambda=lam_800,
            include_centrifugal=False, n_nodes=n_nodes, **mkw,
        )
        ev_h = evaluate_axial_sheath_closure(res_h, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=l_par_m)
        ev_p = evaluate_axial_sheath_closure(res_p, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=l_par_m)
        for pfx, ev in (("hollow", ev_h), ("peaked", ev_p)):
            for k in (
                "has_cwm", "f_hz", "gamma_s",
                "j_over_isat_p28", "dphi_j_isat_p28_deg",
                "j_over_isat_sheath", "dphi_j_isat_sheath_deg",
                "j_over_n_sheath", "dphi_j_n_sheath_deg",
                "dphi_T_n_sheath_deg", "isat_over_n_sheath",
            ):
                flip_row[f"{mtag}_{pfx}_{k}"] = ev[k]
        if np.isfinite(ev_h["dphi_j_isat_sheath_deg"]) and np.isfinite(ev_p["dphi_j_isat_sheath_deg"]):
            delta = _wrap_deg(ev_h["dphi_j_isat_sheath_deg"] - ev_p["dphi_j_isat_sheath_deg"])
            flip_row[f"{mtag}_delta_dphi_sheath_deg"] = delta
            flip_row[f"{mtag}_abs_delta_dphi_sheath_deg"] = float(abs(delta))
        else:
            flip_row[f"{mtag}_delta_dphi_sheath_deg"] = np.nan
            flip_row[f"{mtag}_abs_delta_dphi_sheath_deg"] = np.nan
    out["eta_flip"] = flip_row

    return out


def batch_all(data_dir=DATA_DIR, out_path=None, *, l_par_m=DEFAULT_L_PAR_M,
              wall_width_cm=DEFAULT_WALL_WIDTH_CM, k_par_m=DEFAULT_K_PAR_M,
              n_nodes=DEFAULT_N_NODES, include_biased_sheath=True,
              include_sheath_closure=True, overwrite=False):
    """Run all linear stability comparisons and save ``stability_linear_jun2026.npz``."""
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    if os.path.isfile(out_path):
        if not overwrite:
            print(f"  {os.path.basename(out_path)} exists, skipping (pass overwrite=True to regenerate)")
            return out_path
        os.remove(out_path)

    puff_res = analyze_puff_scan(
        data_dir=data_dir, l_par_m=l_par_m,
        wall_width_cm=wall_width_cm, k_par_m=k_par_m, n_nodes=n_nodes,
    )
    b0_res = analyze_b0_scaling(
        data_dir=data_dir, l_par_m=l_par_m,
        wall_width_cm=wall_width_cm, n_nodes=n_nodes,
    )
    biased_res = {}
    if include_biased_sheath:
        try:
            biased_res = analyze_biased_sheath(
                data_dir=data_dir, l_par_m=l_par_m,
                wall_width_cm=wall_width_cm, n_nodes=n_nodes,
            )
        except (FileNotFoundError, ImportError) as exc:
            print(f"  [note] skipping biased-sheath biasscope extraction: {exc}")

    sheath_res = {}
    if include_sheath_closure:
        try:
            sheath_res = analyze_sheath_closure(
                data_dir=data_dir, l_par_m=l_par_m,
                wall_width_cm=wall_width_cm, n_nodes=n_nodes,
                biased_res=biased_res,
            )
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"  [note] skipping sheath-closure B-dot/BVP benchmark: {exc}")

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
        "l_par_m": np.float64(l_par_m),
        "wall_width_cm": np.float64(wall_width_cm),
        "k_par_m": np.float64(k_par_m),
        "n_nodes": np.int64(n_nodes),
    }

    def _pack(prefix, nested):
        for key_id, fields in nested.items():
            for k, v in fields.items():
                name = f"{prefix}__{key_id}__{k}"
                if isinstance(v, str):
                    payload[name] = np.str_(v)
                elif isinstance(v, (bool, np.bool_)):
                    payload[name] = np.bool_(v)
                elif isinstance(v, (int, np.integer)):
                    payload[name] = np.int64(v)
                elif isinstance(v, (float, np.floating)):
                    payload[name] = np.float64(v)
                else:
                    payload[name] = np.asarray(v)

    _pack("puff", puff_res)
    _pack("b0", b0_res)
    if biased_res:
        _pack("biased", biased_res)
    if sheath_res:
        _pack("sheath", sheath_res)

    merge_save_npz(out_path, payload)
    print(f"\nWrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print the linear stability model comparison across the Jun-2026 puff & B0 scans."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path, allow_pickle=True) as d:
        def _fmt_cwm(f_hz, g_s, dp_deg, width=22):
            if not np.isfinite(g_s):
                return f"{'no unstable CWM':>{width}}"
            return f"{f_hz*1e-3:+6.2f} {g_s*1e-3:+6.2f} {dp_deg:+7.1f}d".rjust(width)

        print(
            f"\n=== 1. Linear Stability Model Comparison Across 23–45 ms Puff Scan "
            f"(L_par={float(d['l_par_m']):.0f} m, wall_w={float(d['wall_width_cm']):.0f} cm) ==="
        )
        print(
            f"{'Run':>3} {'Puff':>4} {'Near':>10} | {'EXP f_pl':>8} {'EXP g':>6} | "
            f"{'KH max':>7} | {'Insul-BVP (f, g)':>16} | "
            f"{'CWM-BVP ref (f, g, dp)':>22} | {'CWM-BVP near (f, g, dp)':>23}"
        )
        for spec in jpr.PROFILE_RUNS:
            r = spec.run_num
            if f"puff__{r}__gas_puff_ms" not in d.files:
                continue
            g_exp = float(d[f"puff__{r}__exp_gamma_s"]) * 1e-3
            g_s = f"{g_exp:6.2f}" if bool(d[f"puff__{r}__exp_unstable"]) else "  stbl"
            n_t = float(d[f"puff__{r}__near_sweep_t_ms"])
            n_reg = str(d[f"puff__{r}__near_sweep_regime"])
            n_tag = f"{n_t:4.2f}({'S' if n_reg == 'straddling' else 'P'})"
            kh_max = float(d[f"puff__{r}__rad_kh_sens_max_gamma_s"]) * 1e-3
            c_ref = _fmt_cwm(
                float(d[f"puff__{r}__ax_cwm_ref_f_hz"]),
                float(d[f"puff__{r}__ax_cwm_ref_gamma_s"]),
                float(d[f"puff__{r}__ax_cwm_ref_dphi_ab_deg"]),
                width=22,
            )
            c_near = _fmt_cwm(
                float(d[f"puff__{r}__ax_cwm_near_f_hz"]),
                float(d[f"puff__{r}__ax_cwm_near_gamma_s"]),
                float(d[f"puff__{r}__ax_cwm_near_dphi_ab_deg"]),
                width=23,
            )
            f_pl_exp = float(d[f"puff__{r}__exp_f_plasma_exb_hz"]) * 1e-3
            f_pl_s = f"{f_pl_exp:+8.2f}" if np.isfinite(f_pl_exp) else f"{'---':>8}"
            print(
                f"{r:>3} {float(d[f'puff__{r}__gas_puff_ms']):4.0f} {n_tag:>10} | "
                f"{f_pl_s} {g_s} | "
                f"{kh_max:+7.3f} | "
                f"{float(d[f'puff__{r}__ax_ins_top_f_hz'])*1e-3:+6.2f} "
                f"{float(d[f'puff__{r}__ax_ins_top_gamma_s'])*1e-3:+9.2f} | "
                f"{c_ref} | {c_near}"
            )
        print("  Near tag: (P)=pre_onset, (S)=straddling; KH max = max gamma over min_width=1.5..3.0 cm & spline(s=0.5)")
        print("  Note: KH depends on Vp'''(r) and cannot exclude unresolved shear layers narrower than the 1 cm scan spacing.")

        print("\n=== 2. Damping & Radial E x B Flow-Shear Stabilization ===")
        print(
            f"{'Run':>3} {'Puff':>4} | {'EXP f_lab':>9} {'EXP g':>6} | "
            f"{'Ax CWM g_sh=1':>13} | {'Ax CWM nu=1.5':>13} | "
            f"{'Rad CWM noSh':>14} | {'Rad CWM +Shear':>15} | {'Rad +Sh+Cent':>14}"
        )
        for spec in jpr.PROFILE_RUNS:
            r = spec.run_num
            if f"puff__{r}__gas_puff_ms" not in d.files:
                continue
            g_exp = float(d[f"puff__{r}__exp_gamma_s"]) * 1e-3
            g_s = f"{g_exp:6.2f}" if bool(d[f"puff__{r}__exp_unstable"]) else "  stbl"
            g_gsh = float(d[f"puff__{r}__ax_cwm_near_gsh1_gamma_s"]) * 1e-3
            g_nu = float(d[f"puff__{r}__ax_cwm_near_nu15_gamma_s"]) * 1e-3
            s_gsh = f"{g_gsh:+13.2f}" if np.isfinite(g_gsh) else f"{'no CWM':>13}"
            s_nu = f"{g_nu:+13.2f}" if np.isfinite(g_nu) else f"{'no CWM':>13}"
            f_lab_exp = float(d[f"puff__{r}__exp_f_lab_hz"]) * 1e-3
            f_lab_s = f"{f_lab_exp:9.2f}" if np.isfinite(f_lab_exp) else f"{'---':>9}"
            print(
                f"{r:>3} {float(d[f'puff__{r}__gas_puff_ms']):4.0f} | "
                f"{f_lab_s} {g_s} | "
                f"{s_gsh} | {s_nu} | "
                f"{float(d[f'puff__{r}__rad_cwm_nosh_f_lab_hz'])*1e-3:+6.2f} "
                f"{float(d[f'puff__{r}__rad_cwm_nosh_gamma_s'])*1e-3:+7.2f} | "
                f"{float(d[f'puff__{r}__rad_cwm_sh_f_lab_hz'])*1e-3:+6.2f} "
                f"{float(d[f'puff__{r}__rad_cwm_sh_gamma_s'])*1e-3:+8.2f} | "
                f"{float(d[f'puff__{r}__rad_cwm_sh_cent_f_lab_hz'])*1e-3:+6.2f} "
                f"{float(d[f'puff__{r}__rad_cwm_sh_cent_gamma_s'])*1e-3:+7.2f}"
            )

        print("\n=== 3. B0 Scaling (1-D Axial CWM-BVP on Run 02 near-onset profile) ===")
        for lbl in ("600G", "800G", "1000G"):
            if f"b0__{lbl}__b0_g" not in d.files:
                continue
            print(
                f"  {lbl:>5}: f_pl = {float(d[f'b0__{lbl}__f_plasma_hz'])*1e-3:+6.2f} kHz, "
                f"f_dop(const Er) = {float(d[f'b0__{lbl}__f_dop_const_er_hz'])*1e-3:+5.2f} -> "
                f"f_lab = {float(d[f'b0__{lbl}__f_lab_const_er_hz'])*1e-3:+5.2f} kHz | "
                f"gamma = {float(d[f'b0__{lbl}__gamma_s'])*1e-3:+5.2f} ms^-1, "
                f"dphi(P29-P25) = {float(d[f'b0__{lbl}__dphi_ab_deg']):+5.1f} deg"
            )

        if any(k.startswith("biased__") for k in d.files):
            print("\n=== 4. Port-35 Biased-Sheath, Mode-Coupled External Impedance & CC Benchmark (Matched 4\" Annulus) ===")
            print(
                f"{'Run':>3} {'Puff':>4} | {'dI2/4[A]':>9} {'R2/R4[Ohm]':>11} {'dV_ohm':>6} "
                f"{'j0_4':>6} {'Reff4':>5} | {'Flt(0R)':>11} | {'Flt+Reff4':>11} | "
                f"{'BiasSh+R4':>11} | {'Bsh+R4+CC':>11} | {'ESat+CC(t)':>11} | {'Ins+CC(t)':>11}"
            )
            def _fg(pfx, tag):
                fk, gk = f"{pfx}{tag}_f_hz", f"{pfx}{tag}_gamma_s"
                if fk not in d.files or not np.isfinite(float(d[gk])):
                    return "   ---   ---"
                return f"{float(d[fk])*1e-3:+5.2f} {float(d[gk])*1e-3:+5.2f}"

            for spec in jpr.PROFILE_RUNS:
                r = spec.run_num
                pfx = f"biased__{r}__"
                if f"{pfx}I_2in_delta" not in d.files:
                    continue
                print(
                    f"{r:>3} {float(d[pfx+'gas_puff_ms']):4.0f} | "
                    f"{float(d[pfx+'I_2in_delta']):4.1f}/{float(d[pfx+'I_4in_delta']):4.1f} "
                    f"{float(d[pfx+'R_ext_2in_ohm']):5.2f}/{float(d[pfx+'R_ext_4in_ohm']):5.2f} "
                    f"{float(d[pfx+'dv_ohm_2in_v']):5.1f}V "
                    f"{float(d[pfx+'j0_4in_norm']):+6.2f} {float(d[pfx+'r_ext_eff_hat_4in']):5.2f} | "
                    f"{_fg(pfx, 'flt')} | {_fg(pfx, 'flt_rext')} | "
                    f"{_fg(pfx, 'bsh_rext')} | {_fg(pfx, 'bsh_cc')} | "
                    f"{_fg(pfx, 'esat_cc_top')} | {_fg(pfx, 'ins_cc_top')}"
                )

        if any(k.startswith("sheath__") for k in d.files):
            print("\n=== 5. Calibrated Sheath-Closure Benchmark: |J_par/(e n0 cs)| / |dI_sat/I_sat| & Absolute Phase ===")
            print(
                "  Experimental (P28 C16 B-dot vs P29-R Isat; peak-phasor RMS / sqrt(2) = time-RMS; "
                "matched Run 03 j_sat):"
            )
            print(
                f"  {'Run(B0)':>11} {'Zone':>11} | {'|Jz|_rms':>8} {'|j|_rms':>7} "
                f"{'|dI/I|_rms':>10} | {'|j|/|dI/I|':>10} {'(DC-scl)':>8} | {'dphi(J-I)':>10} {'R':>5}"
            )
            for rnum, b0_g in SHEATH_BDOT_RUNS:
                pfx = f"sheath__exp_{rnum}__"
                if f"{pfx}b0_g" not in d.files:
                    continue
                for zname, zlbl in (("core", "core<=2.5"), ("wall", "wall2.5-4.5"), ("ring_coh", "ring_coh")):
                    if f"{pfx}{zname}_j_par_time_rms_a_m2" not in d.files:
                        continue
                    jz_rms = float(d[f"{pfx}{zname}_j_par_time_rms_a_m2"])
                    jn_rms = float(d[f"{pfx}{zname}_j_par_norm_rms"])
                    di_rms = float(d[f"{pfx}{zname}_disat_over_isat_rms"])
                    rat = float(d[f"{pfx}{zname}_j_over_isat_ratio"])
                    rat_sc = float(d[f"{pfx}{zname}_j_over_isat_ratio_dc_scaled"])
                    dphi = float(d[f"{pfx}{zname}_dphi_j_isat_deg"])
                    rval = float(d[f"{pfx}{zname}_R_j_isat"])
                    sc_s = f"{rat_sc:8.3f}" if (not np.isclose(b0_g, 800.0) and np.isfinite(rat_sc)) else "     ---"
                    print(
                        f"  {f'{rnum}({int(b0_g)}G)':>11} {zlbl:>11} | "
                        f"{jz_rms:8.1f} {jn_rms:7.3f} {di_rms:10.3f} | "
                        f"{rat:10.3f} {sc_s} | {dphi:+9.1f}d {rval:5.2f}"
                    )
            print(
                "  Note: ring_coh (2-5 cm, |Jz|>=0.3 max, g2_LP>=0.5) isolates the active CWM channel; "
                "unfiltered wall2.5-4.5 mixes in the weak outer density-turnover zone (3.5-4.5 cm) where dphi rotates."
            )

            print(
                "\n  1-D Axial CWM BVP (Run 03 wall, dI_sat/I_sat = dn/n + 0.5*dTe/Te; "
                "z_P28=L-2.24m, z_P29=L-1.92m, sheath z=L):"
            )
            print(
                f"  {'B0':>5} {'Model':>13} | {'P28 |j|/|I|':>11} {'dphi(P28)':>9} | "
                f"{'P28/29 |j|/|I|':>14} {'dphi(28/29)':>11} | "
                f"{'Sheath |j|/|I|':>14} {'dphi(L)':>8} | {'|I|/|n|(L)':>10} {'dp(T-n)':>8}"
            )
            for b_lbl, b_tag in (("800G", "bvp_800G"), ("600G", "bvp_600G")):
                pfx = f"sheath__{b_tag}__"
                if f"{pfx}b0_g" not in d.files:
                    continue
                for mtag, mlbl in (
                    ("conducting_1", "conducting_1"),
                    ("flt_rext", "flt+Reff4"),
                    ("bsh_rext", "bsh+Reff4"),
                ):
                    if f"{pfx}{mtag}_j_over_isat_p28" not in d.files:
                        continue
                    print(
                        f"  {b_lbl:>5} {mlbl:>13} | "
                        f"{float(d[f'{pfx}{mtag}_j_over_isat_p28']):11.3f} "
                        f"{float(d[f'{pfx}{mtag}_dphi_j_isat_p28_deg']):+8.1f}d | "
                        f"{float(d[f'{pfx}{mtag}_j_over_isat_p28_p29']):14.3f} "
                        f"{float(d[f'{pfx}{mtag}_dphi_j_isat_p28_p29_deg']):+10.1f}d | "
                        f"{float(d[f'{pfx}{mtag}_j_over_isat_sheath']):14.3f} "
                        f"{float(d[f'{pfx}{mtag}_dphi_j_isat_sheath_deg']):+7.1f}d | "
                        f"{float(d[f'{pfx}{mtag}_isat_over_n_sheath']):10.2f} "
                        f"{float(d[f'{pfx}{mtag}_dphi_T_n_sheath_deg']):+7.1f}d"
                    )

            epfx = "sheath__eta_flip__"
            if f"{epfx}eta_e_hollow" in d.files:
                print(
                    f"\n  Hollow (eta_e={float(d[epfx+'eta_e_hollow']):+.2f}) vs. Peaked "
                    f"(eta_e={float(d[epfx+'eta_e_peaked']):+.2f}) Sheath Phase Flip (centrifugal off):"
                )
                for mtag, mlbl in (
                    ("conducting_1", "conducting_1"),
                    ("flt_rext", "flt+Reff4"),
                    ("bsh_rext", "bsh+Reff4"),
                ):
                    if f"{epfx}{mtag}_delta_dphi_sheath_deg" not in d.files:
                        continue
                    print(
                        f"    {mlbl:>13}: hollow dphi(L)={float(d[f'{epfx}{mtag}_hollow_dphi_j_isat_sheath_deg']):+6.1f}d "
                        f"(|I|/|n|={float(d[f'{epfx}{mtag}_hollow_isat_over_n_sheath']):.2f}, "
                        f"dp(T-n)={float(d[f'{epfx}{mtag}_hollow_dphi_T_n_sheath_deg']):+6.1f}d) vs. "
                        f"peaked dphi(L)={float(d[f'{epfx}{mtag}_peaked_dphi_j_isat_sheath_deg']):+6.1f}d "
                        f"(|I|/|n|={float(d[f'{epfx}{mtag}_peaked_isat_over_n_sheath']):.2f}, "
                        f"dp(T-n)={float(d[f'{epfx}{mtag}_peaked_dphi_T_n_sheath_deg']):+5.1f}d) "
                        f"-> |Delta dphi|={float(d[f'{epfx}{mtag}_abs_delta_dphi_sheath_deg']):5.1f} deg"
                    )


if __name__ == "__main__":
    out = batch_all(overwrite=True)
    print_summary(out)
