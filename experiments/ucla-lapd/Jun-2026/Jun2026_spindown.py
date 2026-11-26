"""Jun-2026 post-bias Mach spin-down, Poulos Pedersen closure, and neutral-drag stability test.

Measures high-resolution post-bias azimuthal flow relaxation after Port-35 bias
turn-off (``t = 5.006 ms``, pulser 90->10% fall time ``~0.2 us`` on ``biasscope``)
across all seven 800 G gas-puff runs (Mach line Runs ``08–14``, 23–45 ms, plus
3-D Mach plane Run ``32`` at 35 ms), separating two distinct physical timescales:

1. **Fast spin-down (``5.012–5.085 ms``, ``tau_fast = 13 -> 37 us``):** Scales
   proportionally to the Alfvén transit time ``L_par / v_A ~ sqrt(n_e)``
   (``tau_fast / (L_par / v_A) ~ 0.55–0.61`` on Runs 11–14), with a visible
   ``~80 us`` (``2 L_par / v_A``) Alfvén bounce echo on low-puff Runs 08–10.
   This is torsional Alfvén unwinding once Port-35 bias drops to ground, *not*
   ion-neutral drag.
2. **Slow post-overshoot relaxation (``~5.2–7.9 ms``, ``nu_slow = 0.55–2.0 ms^-1``):**
   Exponential relaxation (``R^2 = 0.99–1.00`` on Runs 09–14, matching Run 32's
   2-D plane) of the small post-turnoff reversal tail. Because radial refilling
   of the hollow profile also contributes after turn-off, ``nu_slow`` serves as
   a conservative upper bound on ion-neutral charge-exchange drag ``nu_in``.
3. **Poulos (2019) Pedersen closure & Simon-Hoh test:** Evaluates the classical
   Pedersen conductivity ``sigma_P`` and axial current-penetration length
   ``L_pen = r_wall * sqrt(sigma_par / sigma_P)`` (transitioning from
   ``L_pen >> 14 m`` end-to-end closure at 23 ms to ``L_pen ~ 6–8 m < 14 m``
   bulk cross-field closure at 35–45 ms) and checks ``nu_in / Omega_ci <= 1e-3``
   (quantitatively ruling out collisional Simon-Hoh as a co-driver).
4. **Neutral-drag damping benchmark:** Re-evaluates radial sheared CWM and axial
   biased-sheath CWM vs. current-convective modes at ``nu_in = nu_slow``, testing
   whether realistic damping reproduces the 45 ms quench and discriminates CWM
   from bulk current-convective drive.

Saves ``spindown_pedersen_jun2026.npz`` in ``DATA_DIR`` and prints a summary.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_spindown.py
"""

import datetime
import glob
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
from data_analysis.utils import merge_save_npz

import Jun2026_IV as jiv
import Jun2026_mach_flow as jmf
import Jun2026_mach_line as jml
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
DEFAULT_FALLBACK_NU_23MS_S = 500.0  # conservative 23 ms nu_in upper bound (Run 08 slow tail flat)


def extract_biasscope_turnoff(data_dir=DATA_DIR, run_num="02", max_shots=20):
    """Measure Port-35 ``V_2in`` and ``I_2in`` 90->10% turn-off time [ms] and fall time [us]."""
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    run = open_lapd(hits[0])
    tb = run.time_array(scope_name="biasscope")
    shots = np.asarray(run.shots(scope_name="biasscope"), dtype=int)[:max_shots].tolist()
    v2, _ = run.channel("C4", scope_name="biasscope", shots=shots)
    i2, _ = run.channel("C1", scope_name="biasscope", shots=shots)
    ib0, ib1 = clip_time_window(tb, 16.8, 17.4)
    t_rel_ms = tb[ib0:ib1] * 1e3 - 12.0

    def _fall(sig):
        s = np.nanmean(np.asarray(sig, dtype=float), axis=0)[ib0:ib1]
        hi = float(np.mean(s[:20]))
        lo = float(np.mean(s[-20:]))
        t90 = float(t_rel_ms[np.argmax(s < lo + 0.9 * (hi - lo))])
        t50 = float(t_rel_ms[np.argmax(s < lo + 0.5 * (hi - lo))])
        t10 = float(t_rel_ms[np.argmax(s < lo + 0.1 * (hi - lo))])
        return hi, lo, t90, t50, t10, (t10 - t90) * 1e3

    v_hi, v_lo, vt90, vt50, vt10, vfall = _fall(v2)
    i_hi, i_lo, it90, it50, it10, ifall = _fall(i2 * jst.BIASSCOPE_SHUNT_A_PER_V)
    return {
        "run_num": str(run_num),
        "v2_hi_v": v_hi, "v2_lo_v": v_lo,
        "v2_t50_ms": vt50, "v2_fall_90_10_us": vfall,
        "i2_hi_a": i_hi, "i2_lo_a": i_lo,
        "i2_t50_ms": it50, "i2_fall_90_10_us": ifall,
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
        hits = sorted(glob.glob(os.path.join(data_dir, f"{rm}-*.hdf5")))
        if not hits:
            raise FileNotFoundError(f"no HDF5 for Mach run {rm} in {data_dir}")
        ifn = hits[0]
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

        # Check for Alfven bounce dip in [5.075, 5.125] ms relative to [5.055, 5.075] ms
        m_pre_dip = float(np.nanmax(mf[(tf >= 5.050) & (tf <= 5.075)]))
        m_dip = float(np.nanmin(mf[(tf >= 5.080) & (tf <= 5.115)]))
        echo_dip_amp = max(m_pre_dip - m_dip, 0.0) if (np.isfinite(m_pre_dip) and np.isfinite(m_dip)) else 0.0

        # 2) Slow post-overshoot relaxation over [t_pk_ov, 7.90] ms
        ov_mask = (ts >= 5.15) & (ts <= 5.70)
        ov_idx = np.flatnonzero(ov_mask)
        k_pk = int(ov_idx[int(np.nanargmax(ms[ov_idx]))])
        t_pk = float(ts[k_pk])
        m_pk = float(ms[k_pk])
        fit_s = fit_exponential_relaxation(
            ts, ms, t_start=t_pk, t_end=7.90,
            tau_bounds=(0.05, 10.0), offset_init=m_end,
        )

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
            "m_ov_excess": float(m_pk - m_end),
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
    with np.errstate(invalid="ignore"):
        vx_m = np.nanmean(vx[:, m_on], axis=1)
        vy_m = np.nanmean(vy[:, m_on], axis=1)
    cx, cy, ratio = find_flow_centre(vx_m, vy_m, px, py)
    rr = np.hypot(px - cx, py - cy)
    ring = (rr >= r_band_cm[0]) & (rr <= r_band_cm[1])
    _, mt_all = polar_components(mx, my, px, py, (cx, cy))
    mt_t = np.nanmean(mt_all[ring], axis=0)
    m_end = float(np.nanmean(mt_t[(t32 >= 7.2) & (t32 <= 7.95)]))
    ov = (t32 >= 5.15) & (t32 <= 5.60)
    k_pk = int(np.flatnonzero(ov)[int(np.nanargmax(mt_t[ov]))])
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
                                 b0_t=DEFAULT_B0_T, n_nodes=jst.DEFAULT_N_NODES):
    """Evaluate Poulos Pedersen closure, Simon-Hoh ratio, and nu_slow-damped stability."""
    prof_path = profiles_npz or os.path.join(data_dir, jpr.OUT_NAME)
    stab_path = stability_npz or os.path.join(data_dir, jst.OUT_NAME)
    r_g = jst.DEFAULT_R_GRID_CM
    kp = float(np.hypot(-1.0 / 0.03, np.pi / (jst.DEFAULT_WALL_WIDTH_CM * 1e-2)))
    out = {}

    with np.load(prof_path, allow_pickle=True) as pf, np.load(stab_path, allow_pickle=True) as stn:
        for _, riv, puff in PUFF_PAIRS:
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

            # Total baseline-corrected inner disc + 4" annulus current from Task #1 archive
            i_tot = (
                float(stn[f"biased__{riv}__I_2in_delta"]) + float(stn[f"biased__{riv}__I_4in_delta"])
                if f"biased__{riv}__I_2in_delta" in stn.files else np.nan
            )
            # Use measured nu_slow when R^2 >= 0.8 (Runs 09-14); fallback 500 s^-1 on flat Run 08 (23 ms)
            nu_used = (
                float(sd["nu_slow_s"])
                if (np.isfinite(sd["r2_slow"]) and sd["r2_slow"] >= 0.8 and np.isfinite(sd["nu_slow_s"]))
                else float(DEFAULT_FALLBACK_NU_23MS_S)
            )
            ped = pedersen_closure_diagnostics(
                i_tot if np.isfinite(i_tot) else 50.0, er_v_m, ne_m3, te_ev, b0_t,
                r_wall_m=r_wall_m, l_par_m=l_par_m, nu_in_s=nu_used,
            )

            # Re-evaluate radial & axial stability at nu_in = nu_used
            ne_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__ne"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            te_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__Te"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            vp_g, _ = fit_smooth_radial_profile(x, xc, pf[f"{riv}__Vp"][:, isw], r_g, min_width_cm=jst.DEFAULT_MIN_WIDTH_CM)
            rad_nu = solve_radial_cylindrical_modes(
                r_g, ne_g, te_g, vp_g, b0_g=b0_t * 1e4, m=jst.DEFAULT_M,
                model="cwm_sheath", include_centrifugal=True, include_exb_shear=True,
                nu_in_s=nu_used,
            )

            st_loc, lam = jst._extract_wall_state(pf, riv, isw, b0_g=b0_t * 1e4)
            j0 = float(stn[f"biased__{riv}__j0_4in_norm"]) if f"biased__{riv}__j0_4in_norm" in stn.files else 0.0
            reff = float(stn[f"biased__{riv}__r_ext_eff_hat_4in"]) if f"biased__{riv}__r_ext_eff_hat_4in" in stn.files else 0.0

            c_bsh = solve_axial_braginskii_bvp(
                st_loc, kp, l_par_m=l_par_m, sheath_lambda=lam,
                j0_norm=j0, r_ext_hat=reff, nu_in_s=nu_used, n_nodes=n_nodes,
            )["cwm"]
            c_cc = solve_axial_braginskii_bvp(
                st_loc, kp, l_par_m=l_par_m, sheath_lambda=lam,
                j0_norm=j0, r_ext_hat=reff, include_current_convective=True,
                nu_in_s=nu_used, n_nodes=n_nodes,
            )["cwm"]

            out[riv] = {
                **sd,
                **ped,
                "i_tot_a": float(i_tot),
                "er_wall_v_cm": float(er_v_m * 1e-2),
                "ne_wall_cm3": float(ne_m3 * 1e-6),
                "te_wall_ev": float(te_ev),
                "tau_fast_over_tau_alfven": float((sd["tau_fast_us"] * 1e-6) / ped["tau_alfven_s"]),
                "nu_in_used_s": float(nu_used),
                "nu_in_from_fit": bool(np.isfinite(sd["r2_slow"]) and sd["r2_slow"] >= 0.8),
                "rad_cwm_sh_cent_nu0_gamma_s": float(stn[f"puff__{riv}__rad_cwm_sh_cent_gamma_s"]),
                "rad_cwm_sh_cent_nu_gamma_s": float(rad_nu["gamma_s"]),
                "ax_bsh_r4_nu0_gamma_s": float(stn[f"biased__{riv}__bsh_rext_gamma_s"]) if f"biased__{riv}__bsh_rext_gamma_s" in stn.files else np.nan,
                "ax_bsh_r4_nu_gamma_s": float(c_bsh.gamma_s) if c_bsh else np.nan,
                "ax_bsh_cc_nu0_gamma_s": float(stn[f"biased__{riv}__bsh_cc_gamma_s"]) if f"biased__{riv}__bsh_cc_gamma_s" in stn.files else np.nan,
                "ax_bsh_cc_nu_gamma_s": float(c_cc.gamma_s) if c_cc else np.nan,
            }
    return out


def batch_all(data_dir=DATA_DIR, out_path=None, *, overwrite=False):
    """Run Task #2 spin-down, Poulos Pedersen, and neutral-drag stability analysis -> npz."""
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

    def _pack(prefix, d_in):
        for k, v in d_in.items():
            name = f"{prefix}__{k}"
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

    _pack("turnoff", turnoff)
    if r32:
        _pack("run32", r32)
    for riv, fields in full.items():
        _pack(f"puff__{riv}", fields)

    merge_save_npz(out_path, payload)
    print(f"\nWrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print post-bias Mach spin-down, Poulos Pedersen closure, and neutral-drag summary."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path, allow_pickle=True) as d:
        print(
            f"\n=== 1. Post-Bias Mach Spin-Down: Fast Alfven Unwinding vs. Slow Relaxation ==="
            f"\n  Biasscope Run {str(d['turnoff__run_num'])} turn-off: "
            f"V2={float(d['turnoff__v2_hi_v']):.1f}->{float(d['turnoff__v2_lo_v']):.1f} V "
            f"(t50={float(d['turnoff__v2_t50_ms']):.3f} ms, 90->10% fall={float(d['turnoff__v2_fall_90_10_us']):.1f} us)"
        )
        if "run32__tau_slow_ms" in d.files:
            print(
                f"  Run 32 (35 ms, 3-D plane) slow tail cross-check: "
                f"tau_slow={float(d['run32__tau_slow_ms']):.2f} ms -> "
                f"nu_slow={float(d['run32__nu_slow_s'])*1e-3:.2f} ms^-1 (R^2={float(d['run32__r2_slow']):.2f})"
            )
        print(
            f"\n{'Mach/IV':>7} {'Puff':>4} | {'L/vA[us]':>8} {'tau_f[us]':>9} {'t_f/t_A':>7} {'R2_f':>5} {'echo':>5} | "
            f"{'tau_s[ms]':>9} {'nu_s[/ms]':>9} {'R2_s':>5} {'nu/Oci':>8}"
        )
        for rm, riv, _ in PUFF_PAIRS:
            p = f"puff__{riv}__"
            if f"{p}tau_fast_us" not in d.files:
                continue
            print(
                f"  {rm}/{riv} {float(d[p+'gas_puff_ms']):4.0f} | "
                f"{float(d[p+'tau_alfven_s'])*1e6:8.1f} {float(d[p+'tau_fast_us']):9.1f} "
                f"{float(d[p+'tau_fast_over_tau_alfven']):7.2f} {float(d[p+'r2_fast']):5.2f} "
                f"{float(d[p+'echo_dip_amp']):5.3f} | "
                f"{float(d[p+'tau_slow_ms']):9.2f} {float(d[p+'nu_slow_s'])*1e-3:9.2f} "
                f"{float(d[p+'r2_slow']):5.2f} {float(d[p+'nu_in_over_omega_ci']):8.5f}"
            )

        print("\n=== 2. Poulos (2019) Pedersen Closure & Neutral-Drag Damping Test ===")
        print(
            f"{'Run':>3} {'Puff':>4} | {'I_tot':>5} {'Er':>5} {'sigP_rq':>7} {'sigP_in':>7} {'L_pen[m]':>8} | "
            f"{'nu_u':>4} | {'Rad+Sh+C 0->nu':>15} | {'Ax Bsh+R4 0->nu':>16} | {'Ax Bsh+CC 0->nu':>16}"
        )
        def _pair(g0, g1):
            s0 = f"{g0*1e-3:+6.2f}" if np.isfinite(g0) else "   ---"
            s1 = f"{g1*1e-3:+6.2f}" if np.isfinite(g1) else "   ---"
            return f"{s0} -> {s1}"

        for _, riv, _ in PUFF_PAIRS:
            p = f"puff__{riv}__"
            if f"{p}l_pen_poulos_m" not in d.files:
                continue
            print(
                f"{riv:>3} {float(d[p+'gas_puff_ms']):4.0f} | "
                f"{float(d[p+'i_tot_a']):5.1f} {float(d[p+'er_wall_v_cm']):5.2f} "
                f"{float(d[p+'sigma_p_req_s_m']):7.3f} {float(d[p+'sigma_p_in_s_m']):7.3f} "
                f"{float(d[p+'l_pen_poulos_m']):8.1f} | "
                f"{float(d[p+'nu_in_used_s'])*1e-3:4.2f} | "
                f"{_pair(float(d[p+'rad_cwm_sh_cent_nu0_gamma_s']), float(d[p+'rad_cwm_sh_cent_nu_gamma_s'])):>15} | "
                f"{_pair(float(d[p+'ax_bsh_r4_nu0_gamma_s']), float(d[p+'ax_bsh_r4_nu_gamma_s'])):>16} | "
                f"{_pair(float(d[p+'ax_bsh_cc_nu0_gamma_s']), float(d[p+'ax_bsh_cc_nu_gamma_s'])):>16}"
            )


if __name__ == "__main__":
    out = batch_all(overwrite=True)
    print_summary(out)
