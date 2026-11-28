"""Jun-2026 post-bias Mach spin-down, Pedersen closure bounds, and neutral-drag test.

Measures high-resolution post-bias azimuthal flow relaxation after Port-35 bias
turn-off (pulser 90->10% fall time ``~0.2 us`` on ``biasscope``) across all
seven 800 G gas-puff runs (Mach line Runs ``08–14``, 23–45 ms, plus 3-D Mach
plane Run ``32`` at 35 ms), separating two distinct physical timescales:

1. **Fast spin-down (``5.012–5.085 ms``, ``tau_fast = 13 -> 37 us``):** Scales
   with the Alfvén transit time ``L_par / v_A ~ sqrt(n_e)``
   (``tau_fast / (L_par / v_A) ~ 0.55–0.61`` on Runs 11–14), with a visible
   ``~80 us`` (``2 L_par / v_A``) Alfvén bounce echo on low-puff Runs 08–10.
   This is torsional Alfvén unwinding once Port-35 bias drops to ground, *not*
   ion-neutral drag.
2. **Slow post-overshoot relaxation (``~5.2–7.9 ms``, ``nu_slow = 0.55–2.0 ms^-1``):**
   Exponential relaxation (``R^2 = 0.99–1.00`` on Runs 09–14, matching Run 32's
   2-D plane) of the small post-turnoff reversal tail. Because radial refilling
   of the hollow profile also contributes after turn-off, ``nu_slow`` is an
   **upper bound** on ion-neutral charge-exchange drag ``nu_in`` (Run 08 at
   23 ms has a flat slow tail ``R^2 < 0.8`` and is left ``NaN``, never guessed).
3. **Rigorous bound-direction physics tests:**
   Because ``nu_in <= nu_slow`` is an *upper* bound:
   - ``sigma_P,in <= sigma_P,in(nu_slow)`` is an **upper bound**, which comes
     out **5x–30x below** the radial conductivity ``sigma_P,req`` needed to
     close the 39–57 A bias current even over the full 14 m column — proving
     classical ion-neutral Pedersen drag *cannot* carry the DC bias current.
   - ``L_pen = r_wall * sqrt(sigma_par / sigma_P,in)`` is a **lower bound**
     (``L_pen >= 7.6–51 m``); it does *not* prove ``L_pen < 14 m``.
   - Cross-field ion-neutral Pedersen vs. parallel shorting
     ``(L_par / (pi * L_pen))^2 <= 0.008–0.34`` and ``nu_in / Omega_ci <= 1e-3``
     are **upper bounds** showing parallel shorting dominates ion-neutral
     Pedersen charge separation across Runs 09–14.
   - Stability growth rates at ``nu_in = nu_slow`` represent **maximum possible
     neutral damping** (lower-bound ``gamma``): even at this maximum damping,
     1-D bulk current-convective (`Ax Bsh+CC`) stays strongly unstable at
     40–45 ms (`+10.9 to +11.9 ms^-1`), while `Rad+Sh+C` at 24 ms remains
     unstable (`+1.10 ms^-1`); quenching of `Rad+Sh+C` / `Ax Bsh+R4` at 45 ms
     occurs only near the upper-bound damping limit.

Saves ``spindown_pedersen_jun2026.npz`` in ``DATA_DIR`` and prints a summary.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_spindown.py
"""

import datetime
import os

import numpy as np

from data_analysis.io import open_lapd, parse_shunts, position_shots
from data_analysis.plasma.flow import (
    find_flow_centre,
    fit_exponential_relaxation,
    polar_components,
)
from data_analysis.plasma.mach import binned_face_ratio, mach_single, time_bin_edges
from data_analysis.plasma.stability import (
    fit_smooth_radial_profile,
    pedersen_closure_diagnostics,
    solve_axial_braginskii_bvp,
    solve_radial_cylindrical_modes,
)
from data_analysis.signal import clip_time_window
from data_analysis.utils import (
    find_run_hdf5,
    merge_save_npz,
    pack_nested_npz,
    pack_npz_entries,
)

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
import Jun2026_mach_line as jml
import Jun2026_modes as jmo
import Jun2026_profiles as jpr
import Jun2026_stability as jst

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jst.DATA_DIR
OUT_NAME = "spindown_pedersen_jun2026.npz"

PUFF_PAIRS = (
    ("08", "06", 23.0),
    ("09", "00", 24.0),
    ("10", "01", 25.0),
    ("11", "02", 30.0),
    ("12", "03", 35.0),
    ("13", "04", 40.0),
    ("14", "05", 45.0),
)

DEFAULT_FAST_BIN_MS = 0.005       # 5 us bins resolve 13-37 us Alfven unwinding
DEFAULT_SLOW_BIN_MS = 0.05        # 50 us bins resolve 0.5-1.8 ms slow tail
DEFAULT_FAST_FIT_MS = (5.012, 5.085)  # past 5.006 ms pulser switching glitch, before ~5.093 ms echo
DEFAULT_X_CENTER_CM = 1.0         # Day-2 P33 Mach line rotation center
DEFAULT_R_BAND_CM = (2.5, 5.5)    # inner-wall radial band [cm]
DEFAULT_R_WALL_M = 0.035          # representative radial closure radius [m]
DEFAULT_L_PAR_M = 14.0
DEFAULT_B0_T = 0.08               # 800 G
DEFAULT_SLOW_R2_MIN = 0.8         # require strong exponential fit before using nu_slow as nu_in bound


def _find_excursion_peak(ts, y, baseline, t_lo, t_hi):
    """Find the index of the largest excursion ``|y - baseline|`` inside ``[t_lo, t_hi]``."""
    mask = (ts >= float(t_lo)) & (ts <= float(t_hi)) & np.isfinite(y)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    k = int(idx[int(np.argmax(np.abs(y[idx] - float(baseline))))])
    return k


def extract_biasscope_turnoff(data_dir=DATA_DIR, runs=None, max_shots=20,
                              biasscope_offset_ms=jmo.BIASSCOPE_OFFSET_MS):
    """Measure Port-35 ``V_2in`` and ``I_2in`` 90->10% turn-off time [ms] and fall time [us].

    ``biasscope`` triggers at discharge start (1 kA), ``biasscope_offset_ms``
    (12.0 ms in Jun-2026) before bias start (which triggers ``machscope``).
    Measures across all requested puff-scan runs (default ``06, 00-05``) and
    returns per-run values plus median summary keys.
    """
    run_nums = tuple(riv for _, riv, _ in PUFF_PAIRS) if runs is None else tuple(runs)
    t50_v, fall_v, t50_i, fall_i = [], [], [], []
    per_run = {}
    t_win = (biasscope_offset_ms + 4.8, biasscope_offset_ms + 5.4)

    for rnum in run_nums:
        run = open_lapd(find_run_hdf5(data_dir, rnum))
        tb = run.time_array(scope_name="biasscope")
        shots = np.asarray(run.shots(scope_name="biasscope"), dtype=int)[:max_shots].tolist()
        v2, _ = run.channel("C4", scope_name="biasscope", shots=shots)
        i2, _ = run.channel("C1", scope_name="biasscope", shots=shots)
        ib0, ib1 = clip_time_window(tb, *t_win)
        t_rel_ms = tb[ib0:ib1] * 1e3 - float(biasscope_offset_ms)

        def _fall(sig):
            s = np.nanmean(np.asarray(sig, dtype=float), axis=0)[ib0:ib1]
            hi = float(np.mean(s[:20]))
            lo = float(np.mean(s[-20:]))
            t90 = float(t_rel_ms[np.argmax(s < lo + 0.9 * (hi - lo))])
            t50 = float(t_rel_ms[np.argmax(s < lo + 0.5 * (hi - lo))])
            t10 = float(t_rel_ms[np.argmax(s < lo + 0.1 * (hi - lo))])
            return hi, lo, t50, (t10 - t90) * 1e3

        v_hi, v_lo, vt50, vfall = _fall(v2)
        i_hi, i_lo, it50, ifall = _fall(i2 * jst.BIASSCOPE_SHUNT_A_PER_V)
        t50_v.append(vt50)
        fall_v.append(vfall)
        t50_i.append(it50)
        fall_i.append(ifall)
        per_run[f"r{rnum}_v2_t50_ms"] = vt50
        per_run[f"r{rnum}_v2_fall_us"] = vfall
        per_run[f"r{rnum}_i2_fall_us"] = ifall

    return {
        "runs": ",".join(run_nums),
        "biasscope_offset_ms": float(biasscope_offset_ms),
        "v2_t50_med_ms": float(np.median(t50_v)),
        "v2_t50_ptp_us": float(np.ptp(t50_v) * 1e3),
        "v2_fall_90_10_med_us": float(np.median(fall_v)),
        "v2_fall_90_10_max_us": float(np.max(fall_v)),
        "i2_t50_med_ms": float(np.median(t50_i)),
        "i2_fall_90_10_med_us": float(np.median(fall_i)),
        "i2_fall_90_10_max_us": float(np.max(fall_i)),
        **per_run,
    }


def analyze_spindown_runs(data_dir=DATA_DIR, *, cal_npz=None,
                          fast_bin_ms=DEFAULT_FAST_BIN_MS,
                          slow_bin_ms=DEFAULT_SLOW_BIN_MS,
                          fast_fit_ms=DEFAULT_FAST_FIT_MS,
                          x_center_cm=DEFAULT_X_CENTER_CM,
                          r_band_cm=DEFAULT_R_BAND_CM):
    """Extract high-resolution fast & slow post-bias ``M_rot(t)`` relaxation on Runs 08-14."""
    cal_path = cal_npz or os.path.join(data_dir, os.path.basename(jmf.CAL_NPZ))
    kappa, _, _ = jmf.load_calibration(cal_path)
    out = {}
    for rm, riv, puff in PUFF_PAIRS:
        ifn = find_run_hdf5(data_dir, rm)
        run = open_lapd(ifn)
        chans = jml.tip_channels(ifn)
        shunts = parse_shunts(run.description().raw)
        scope = chans["Y+"][0]
        pos = jiv.read_lp_positions(ifn)
        pos_1d, _ = jml.scanned_axis(pos)
        tarr = run.time_array(scope_name=scope)

        edges_f, tf = time_bin_edges(tarr, (4.90, 5.30), fast_bin_ms)
        edges_s, ts = time_bin_edges(tarr, (5.10, 7.95), slow_bin_ms)
        r0, r1 = float(r_band_cm[0]), float(r_band_cm[1])
        idx_p = np.flatnonzero((pos_1d - x_center_cm >= r0) & (pos_1d - x_center_cm <= r1))
        idx_n = np.flatnonzero((pos_1d - x_center_cm <= -r0) & (pos_1d - x_center_cm >= -r1))

        def _read_rot(edges):
            def _side(indices):
                rows = []
                for p in indices:
                    sh = position_shots(pos.shot_nums, int(p), pos.nshot)
                    sp, _ = run.channel(chans["Y+"][1], scope_name=chans["Y+"][0], shots=sh)
                    sm, _ = run.channel(chans["Y-"][1], scope_name=chans["Y-"][0], shots=sh)
                    if sp is None or sm is None:
                        continue
                    R, _, _ = binned_face_ratio(sp / shunts["Y+"], sm / shunts["Y-"], edges)
                    rows.append(mach_single(R, kappa["Y"]))
                return np.nanmean(rows, axis=0)
            return 0.5 * (_side(idx_p) - _side(idx_n))

        mf = _read_rot(edges_f)
        ms = _read_rot(edges_s)
        m_on = float(np.nanmean(mf[(tf >= 4.91) & (tf <= 4.995)]))
        m_end = float(np.nanmean(ms[(ts >= 7.20) & (ts <= 7.90)]))

        # 1) Fast Alfven unwinding over [5.012, 5.085] ms
        fit_f = fit_exponential_relaxation(
            tf, mf, t_start=fast_fit_ms[0], t_end=fast_fit_ms[1],
            tau_bounds=(0.003, 0.5), offset_init=m_end,
        )

        # Check for Alfven bounce dip in [5.080, 5.115] ms relative to [5.050, 5.075] ms
        m_pre_dip = float(np.nanmax(mf[(tf >= 5.050) & (tf <= 5.075)]))
        m_dip = float(np.nanmin(mf[(tf >= 5.080) & (tf <= 5.115)]))
        echo_dip_amp = max(m_pre_dip - m_dip, 0.0) if (np.isfinite(m_pre_dip) and np.isfinite(m_dip)) else 0.0

        # 2) Slow post-overshoot relaxation over [t_pk_ov, 7.90] ms (sign-agnostic excursion peak)
        k_pk = _find_excursion_peak(ts, ms, m_end, 5.15, 5.70)
        if k_pk is not None:
            t_pk = float(ts[k_pk])
            m_pk = float(ms[k_pk])
            fit_s = fit_exponential_relaxation(
                ts, ms, t_start=t_pk, t_end=7.90,
                tau_bounds=(0.05, 10.0), offset_init=m_end,
            )
        else:
            t_pk, m_pk = np.nan, np.nan
            fit_s = {"tau": np.nan, "rate": np.nan, "r2": np.nan}

        out[riv] = {
            "mach_run": rm,
            "iv_run": riv,
            "gas_puff_ms": float(puff),
            "t_fast_ms": tf,
            "m_rot_fast": mf,
            "t_slow_ms": ts,
            "m_rot_slow": ms,
            "m_on": m_on,
            "m_late": m_end,
            "tau_fast_us": float(fit_f["tau"] * 1e3),
            "nu_fast_s": float(fit_f["rate"] * 1e3),
            "r2_fast": float(fit_f["r2"]),
            "echo_dip_amp": float(echo_dip_amp),
            "t_ov_peak_ms": t_pk,
            "m_ov_excess": float(m_pk - m_end) if np.isfinite(m_pk) else np.nan,
            "tau_slow_ms": float(fit_s["tau"]),
            "nu_slow_s": float(fit_s["rate"] * 1e3),
            "r2_slow": float(fit_s["r2"]),
        }
    return out


def analyze_run32_slow_spindown(data_dir=DATA_DIR, *, r_band_cm=DEFAULT_R_BAND_CM):
    """Cross-check slow post-overshoot relaxation on 3-D Mach plane Run 32 (35 ms puff)."""
    p32 = os.path.join(data_dir, "32-mach-flow-data.npz")
    if not os.path.isfile(p32):
        return {}
    with np.load(p32) as d:
        t32 = d["t_ms"]
        px, py = d["pos_x"], d["pos_y"]
        vx, vy = d["v_X"], d["v_Y"]
        mx, my = d["M_X"], d["M_Y"]
    m_on = (t32 >= 0.5) & (t32 <= 4.5)
    m_pre = t32 < -0.5
    with np.errstate(invalid="ignore"):
        vx_m = np.nanmean(vx[:, m_on], axis=1)
        vy_m = np.nanmean(vy[:, m_on], axis=1)
        if np.any(m_pre):
            vx_m = vx_m - np.nanmean(vx[:, m_pre], axis=1)
            vy_m = vy_m - np.nanmean(vy[:, m_pre], axis=1)
    cx, cy, ratio = find_flow_centre(vx_m, vy_m, px, py)
    rr = np.hypot(px - cx, py - cy)
    ring = (rr >= r_band_cm[0]) & (rr <= r_band_cm[1])
    _, mt_all = polar_components(mx, my, px, py, (cx, cy))
    mt_t = np.nanmean(mt_all[ring], axis=0)
    m_end = float(np.nanmean(mt_t[(t32 >= 7.2) & (t32 <= 7.95)]))
    k_pk = _find_excursion_peak(t32, mt_t, m_end, 5.15, 5.60)
    if k_pk is None:
        return {}
    t_pk = float(t32[k_pk])
    fit_s = fit_exponential_relaxation(
        t32, mt_t, t_start=t_pk, t_end=7.95, tau_bounds=(0.05, 10.0), offset_init=m_end,
    )
    return {
        "center_x_cm": float(cx),
        "center_y_cm": float(cy),
        "center_ratio": float(ratio),
        "t_ov_peak_ms": t_pk,
        "m_ov_excess": float(mt_t[k_pk] - m_end),
        "tau_slow_ms": float(fit_s["tau"]),
        "nu_slow_s": float(fit_s["rate"] * 1e3),
        "r2_slow": float(fit_s["r2"]),
    }


def analyze_pedersen_and_damping(spindown_runs, data_dir=DATA_DIR, *,
                                 profiles_npz=None, stability_npz=None,
                                 r_wall_m=DEFAULT_R_WALL_M, l_par_m=DEFAULT_L_PAR_M,
                                 b0_t=DEFAULT_B0_T, n_nodes=jst.DEFAULT_N_NODES,
                                 slow_r2_min=DEFAULT_SLOW_R2_MIN):
    """Evaluate Pedersen-closure bounds, parallel-shorting ratio, and max-damping stability.

    Never substitutes silent hand-set fallbacks:
    - When ``r2_slow < slow_r2_min`` (e.g., flat slow tail on Run 08 / 23 ms),
      ``nu_in_upper_s`` and every quantity derived from it stay ``NaN``.
    - Missing biased-sheath keys in ``stability_linear_jun2026.npz`` stay
      ``NaN`` rather than defaulting ``I_tot`` or ``(j0, r_ext)``.
    """
    prof_path = profiles_npz or os.path.join(data_dir, jpr.OUT_NAME)
    stab_path = stability_npz or os.path.join(data_dir, jst.OUT_NAME)
    r_g = jst.DEFAULT_R_GRID_CM
    kp = float(np.hypot(-1.0 / 0.03, np.pi / (jst.DEFAULT_WALL_WIDTH_CM * 1e-2)))
    out = {}

    with np.load(prof_path, allow_pickle=True) as pf, np.load(stab_path, allow_pickle=True) as stn:
        for _, riv, _ in PUFF_PAIRS:
            if riv not in spindown_runs or f"{riv}__ref_sweep_idx" not in pf.files:
                continue
            sd = spindown_runs[riv]
            isw = int(pf[f"{riv}__ref_sweep_idx"])
            x = pf[f"{riv}__x_cm"]
            xc = float(pf[f"{riv}__x_center_cm"])
            band = (np.abs(x - xc) >= 2.0) & (np.abs(x - xc) <= 4.0)
            ne_m3 = float(np.nanmedian(pf[f"{riv}__ne"][band, isw])) * 1e6
            te_ev = float(np.nanmedian(pf[f"{riv}__Te"][band, isw]))
            er_v_m = float(pf[f"{riv}__ref_er_wall_v_cm"]) * 100.0

            has_bias = (
                f"biased__{riv}__I_2in_delta" in stn.files
                and f"biased__{riv}__I_4in_delta" in stn.files
            )
            i_tot = (
                float(stn[f"biased__{riv}__I_2in_delta"]) + float(stn[f"biased__{riv}__I_4in_delta"])
                if has_bias else np.nan
            )

            # Use measured nu_slow strictly as an upper bound when R^2 >= slow_r2_min; else NaN.
            has_nu = bool(
                np.isfinite(sd["r2_slow"])
                and sd["r2_slow"] >= float(slow_r2_min)
                and np.isfinite(sd["nu_slow_s"])
                and sd["nu_slow_s"] > 0.0
            )
            nu_upper = float(sd["nu_slow_s"]) if has_nu else np.nan

            ped = pedersen_closure_diagnostics(
                i_tot, er_v_m, ne_m3, te_ev, b0_t,
                r_wall_m=r_wall_m, l_par_m=l_par_m,
                nu_in_s=(nu_upper if has_nu else None),
            )

            # Maximum-damping (nu_in = nu_upper) stability check; NaN if nu_upper or biased keys missing.
            rad_nu_g = np.nan
            ax_bsh_nu_g = np.nan
            ax_cc_nu_g = np.nan
            has_bvp_bias = (
                f"biased__{riv}__j0_4in_norm" in stn.files
                and f"biased__{riv}__r_ext_eff_hat_4in" in stn.files
            )
            if has_nu:
                ne_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__ne"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
                te_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__Te"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
                vp_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__Vp"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
                rad_nu_g = float(solve_radial_cylindrical_modes(
                    r_g, ne_g, te_g, vp_g, b0_g=b0_t * 1e4, m=jst.DEFAULT_M,
                    model="cwm_sheath", include_centrifugal=True, include_exb_shear=True,
                    nu_in_s=nu_upper,
                )["gamma_s"])
                if has_bvp_bias:
                    st_loc, lam = jst._extract_wall_state(pf, riv, isw, b0_g=b0_t * 1e4)
                    j0 = float(stn[f"biased__{riv}__j0_4in_norm"])
                    reff = float(stn[f"biased__{riv}__r_ext_eff_hat_4in"])
                    c_bsh = solve_axial_braginskii_bvp(
                        st_loc, kp, l_par_m=l_par_m, sheath_lambda=lam,
                        j0_norm=j0, r_ext_hat=reff, nu_in_s=nu_upper, n_nodes=n_nodes,
                    )["cwm"]
                    c_cc = solve_axial_braginskii_bvp(
                        st_loc, kp, l_par_m=l_par_m, sheath_lambda=lam,
                        j0_norm=j0, r_ext_hat=reff, include_current_convective=True,
                        nu_in_s=nu_upper, n_nodes=n_nodes,
                    )["cwm"]
                    ax_bsh_nu_g = float(c_bsh.gamma_s) if c_bsh else np.nan
                    ax_cc_nu_g = float(c_cc.gamma_s) if c_cc else np.nan

            out[riv] = {
                **sd,
                "v_alfven_m_s": ped["v_alfven_m_s"],
                "tau_alfven_s": ped["tau_alfven_s"],
                "omega_ci_rad_s": ped["omega_ci_rad_s"],
                "j_r_a_m2": ped["j_r_a_m2"],
                "sigma_par_s_m": ped["sigma_par_s_m"],
                "sigma_p_req_s_m": ped["sigma_p_req_s_m"],
                "nu_in_ped_req_s": ped["nu_in_ped_req_s"],
                "nu_in_upper_s": float(nu_upper),
                "nu_in_over_omega_ci_upper": ped["nu_in_over_omega_ci"],
                "sigma_p_in_upper_s_m": ped["sigma_p_in_s_m"],
                "sigma_p_in_upper_over_req": ped["sigma_p_in_over_req"],
                "l_pen_poulos_lower_m": ped["l_pen_poulos_m"],
                "pedersen_to_parallel_upper": ped["pedersen_to_parallel"],
                "i_tot_a": float(i_tot),
                "er_wall_v_cm": float(er_v_m * 1e-2),
                "ne_wall_cm3": float(ne_m3 * 1e-6),
                "te_wall_ev": float(te_ev),
                "tau_fast_over_tau_alfven": float((sd["tau_fast_us"] * 1e-6) / ped["tau_alfven_s"]),
                "rad_cwm_sh_cent_nu0_gamma_s": float(stn[f"puff__{riv}__rad_cwm_sh_cent_gamma_s"]),
                "rad_cwm_sh_cent_nu_max_gamma_s": rad_nu_g,
                "ax_bsh_r4_nu0_gamma_s": float(stn[f"biased__{riv}__bsh_rext_gamma_s"]) if f"biased__{riv}__bsh_rext_gamma_s" in stn.files else np.nan,
                "ax_bsh_r4_nu_max_gamma_s": ax_bsh_nu_g,
                "ax_bsh_cc_nu0_gamma_s": float(stn[f"biased__{riv}__bsh_cc_gamma_s"]) if f"biased__{riv}__bsh_cc_gamma_s" in stn.files else np.nan,
                "ax_bsh_cc_nu_max_gamma_s": ax_cc_nu_g,
            }
    return out


def batch_all(data_dir=DATA_DIR, out_path=None, *, overwrite=False):
    """Run Task #2 spin-down, Pedersen-bound, and max-damping stability analysis -> npz."""
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    if os.path.isfile(out_path):
        if not overwrite:
            print(f"  {os.path.basename(out_path)} exists, skipping (pass overwrite=True to regenerate)")
            return out_path
        os.remove(out_path)

    turnoff = extract_biasscope_turnoff(data_dir=data_dir)
    sd_runs = analyze_spindown_runs(data_dir=data_dir)
    r32 = analyze_run32_slow_spindown(data_dir=data_dir)
    full = analyze_pedersen_and_damping(sd_runs, data_dir=data_dir)

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
        "r_wall_m": np.float64(DEFAULT_R_WALL_M),
        "l_par_m": np.float64(DEFAULT_L_PAR_M),
        "b0_t": np.float64(DEFAULT_B0_T),
    }

    payload.update(pack_npz_entries("turnoff", turnoff))
    if r32:
        payload.update(pack_npz_entries("run32", r32))
    payload.update(pack_nested_npz("puff", full))

    merge_save_npz(out_path, payload)
    print(f"\nWrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print post-bias Mach spin-down, rigorous Pedersen bounds, and max-damping summary."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path, allow_pickle=True) as d:
        print(
            f"\n=== 1. Post-Bias Mach Spin-Down: Fast Alfven Unwinding vs. Slow Relaxation ==="
            f"\n  Biasscope turn-off ({str(d['turnoff__runs'])}, offset={float(d['turnoff__biasscope_offset_ms']):.1f} ms): "
            f"V2 t50={float(d['turnoff__v2_t50_med_ms']):.3f} ms (ptp={float(d['turnoff__v2_t50_ptp_us']):.1f} us), "
            f"90->10% fall med={float(d['turnoff__v2_fall_90_10_med_us']):.1f} us (max={float(d['turnoff__v2_fall_90_10_max_us']):.1f} us)"
        )
        if "run32__tau_slow_ms" in d.files:
            print(
                f"  Run 32 (35 ms, 3-D plane) slow tail cross-check: "
                f"tau_slow={float(d['run32__tau_slow_ms']):.2f} ms -> "
                f"nu_slow={float(d['run32__nu_slow_s'])*1e-3:.2f} ms^-1 (R^2={float(d['run32__r2_slow']):.2f})"
            )
        print(
            f"\n{'Mach/IV':>7} {'Puff':>4} | {'L/vA[us]':>8} {'tau_f[us]':>9} {'t_f/t_A':>7} {'R2_f':>5} {'echo':>5} | "
            f"{'tau_s[ms]':>9} {'nu_up[/m]':>9} {'R2_s':>5} {'nu/Oci<=':>8}"
        )
        def _fmt(val, fmt_spec, width):
            return format(val, fmt_spec).rjust(width) if np.isfinite(val) else "---".rjust(width)

        for rm, riv, _ in PUFF_PAIRS:
            p = f"puff__{riv}__"
            if f"{p}tau_fast_us" not in d.files:
                continue
            print(
                f"  {rm}/{riv} {float(d[p+'gas_puff_ms']):4.0f} | "
                f"{float(d[p+'tau_alfven_s'])*1e6:8.1f} {float(d[p+'tau_fast_us']):9.1f} "
                f"{float(d[p+'tau_fast_over_tau_alfven']):7.2f} {float(d[p+'r2_fast']):5.2f} "
                f"{float(d[p+'echo_dip_amp']):5.3f} | "
                f"{_fmt(float(d[p+'tau_slow_ms']), '.2f', 9)} "
                f"{_fmt(float(d[p+'nu_in_upper_s'])*1e-3, '.2f', 9)} "
                f"{_fmt(float(d[p+'r2_slow']), '.2f', 5)} "
                f"{_fmt(float(d[p+'nu_in_over_omega_ci_upper']), '.5f', 8)}"
            )

        print("\n=== 2. Rigorous Pedersen-Closure Bounds & Max-Damping (nu_in <= nu_slow) Stability ===")
        print(
            f"{'Run':>3} {'Puff':>4} | {'I_tot':>5} {'sigP_rq':>7} {'sigPin<=':>8} {'frac<=':>7} "
            f"{'Lpen>=':>6} {'Ped/Par<=':>9} | {'Rad+Sh+C 0->max':>15} | "
            f"{'Ax Bsh+R4 0->max':>16} | {'Ax Bsh+CC 0->max':>16}"
        )
        def _pair(g0, g1):
            s0 = f"{g0*1e-3:+6.2f}" if np.isfinite(g0) else "   ---"
            s1 = f"{g1*1e-3:+6.2f}" if np.isfinite(g1) else "   ---"
            return f"{s0} -> {s1}"

        for _, riv, _ in PUFF_PAIRS:
            p = f"puff__{riv}__"
            if f"{p}sigma_p_req_s_m" not in d.files:
                continue
            print(
                f"{riv:>3} {float(d[p+'gas_puff_ms']):4.0f} | "
                f"{_fmt(float(d[p+'i_tot_a']), '.1f', 5)} "
                f"{_fmt(float(d[p+'sigma_p_req_s_m']), '.3f', 7)} "
                f"{_fmt(float(d[p+'sigma_p_in_upper_s_m']), '.3f', 8)} "
                f"{_fmt(float(d[p+'sigma_p_in_upper_over_req']), '.3f', 7)} "
                f"{_fmt(float(d[p+'l_pen_poulos_lower_m']), '.1f', 6)} "
                f"{_fmt(float(d[p+'pedersen_to_parallel_upper']), '.3f', 9)} | "
                f"{_pair(float(d[p+'rad_cwm_sh_cent_nu0_gamma_s']), float(d[p+'rad_cwm_sh_cent_nu_max_gamma_s'])):>15} | "
                f"{_pair(float(d[p+'ax_bsh_r4_nu0_gamma_s']), float(d[p+'ax_bsh_r4_nu_max_gamma_s'])):>16} | "
                f"{_pair(float(d[p+'ax_bsh_cc_nu0_gamma_s']), float(d[p+'ax_bsh_cc_nu_max_gamma_s'])):>16}"
            )


if __name__ == "__main__":
    out = batch_all(overwrite=True)
    print_summary(out)
