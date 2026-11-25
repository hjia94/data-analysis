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
import os

import numpy as np
from scipy.interpolate import UnivariateSpline

from data_analysis.plasma.stability import (
    fit_smooth_radial_profile,
    local_braginskii_state,
    solve_axial_braginskii_bvp,
    solve_local_cwm,
    solve_radial_cylindrical_modes,
)
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


def batch_all(data_dir=DATA_DIR, out_path=None, *, l_par_m=DEFAULT_L_PAR_M,
              wall_width_cm=DEFAULT_WALL_WIDTH_CM, k_par_m=DEFAULT_K_PAR_M,
              n_nodes=DEFAULT_N_NODES, overwrite=False):
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


if __name__ == "__main__":
    out = batch_all(overwrite=True)
    print_summary(out)
