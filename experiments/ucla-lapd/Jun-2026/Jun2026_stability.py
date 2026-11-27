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

from data_analysis.io import open_lapd
from data_analysis.plasma.fields import reference_locked_phasor
from data_analysis.plasma.stability import (
    biased_sheath_parameters,
    fit_smooth_radial_profile,
    local_braginskii_state,
    solve_axial_braginskii_bvp,
    solve_local_cwm,
    solve_radial_cylindrical_modes,
)
from data_analysis.signal import clip_time_window
from data_analysis.utils import merge_save_npz

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


def batch_all(data_dir=DATA_DIR, out_path=None, *, l_par_m=DEFAULT_L_PAR_M,
              wall_width_cm=DEFAULT_WALL_WIDTH_CM, k_par_m=DEFAULT_K_PAR_M,
              n_nodes=DEFAULT_N_NODES, include_biased_sheath=True, overwrite=False):
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
            print(
                f"{r:>3} {float(d[f'puff__{r}__gas_puff_ms']):4.0f} {n_tag:>10} | "
                f"{float(d[f'puff__{r}__exp_f_plasma_exb_hz'])*1e-3:+8.2f} {g_s} | "
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
            print(
                f"{r:>3} {float(d[f'puff__{r}__gas_puff_ms']):4.0f} | "
                f"{float(d[f'puff__{r}__exp_f_lab_hz'])*1e-3:9.2f} {g_s} | "
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


if __name__ == "__main__":
    out = batch_all(overwrite=True)
    print_summary(out)
