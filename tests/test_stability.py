"""Unit tests for linear magnetized-plasma stability solvers (CWM, Braginskii, KH)."""

import os
import sys
import tempfile
import unittest

import numpy as np

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

from data_analysis.plasma.stability import (
    M_ELECTRON,
    M_PROTON,
    biased_sheath_parameters,
    chebyshev_diff_matrix,
    fit_smooth_radial_profile,
    local_braginskii_state,
    solve_axial_braginskii_bvp,
    solve_local_braginskii,
    solve_local_cwm,
    solve_radial_cylindrical_modes,
)
import Jun2026_stability as jst


class TestLinearStabilitySolvers(unittest.TestCase):
    def test_chebyshev_diff_matrix_exact_on_polynomials_and_sine(self):
        x, D = chebyshev_diff_matrix(16)
        self.assertAlmostEqual(x[0], -1.0, places=12)
        self.assertAlmostEqual(x[-1], +1.0, places=12)
        self.assertTrue(np.all(np.diff(x) > 0.0))

        # Exact derivative of f(x) = x^3 - 2x^2 + 5
        f = x ** 3 - 2.0 * x ** 2 + 5.0
        df_exact = 3.0 * x ** 2 - 4.0 * x
        self.assertLess(float(np.max(np.abs(D @ f - df_exact))), 1e-11)

        # Spectral accuracy on smooth trigonometric function (n=16 -> ~1e-9; n=20 -> ~1e-13)
        g = np.sin(np.pi * x)
        dg_exact = np.pi * np.cos(np.pi * x)
        self.assertLess(float(np.max(np.abs(D @ g - dg_exact))), 1e-9)

    def test_braginskii_recovers_alfven_and_acoustic_waves(self):
        # Uniform plasma (zero gradients, zero resistivity/chi/nu_in) -> exact
        # inertial/kinetic Alfven wave w^2 = k_par^2 v_A^2 / (1 + k_perp^2 d_e^2)
        # and ion acoustic wave w^2 = (5/3) k_par^2 c_s^2 (with adiabatic T_e)!
        st = local_braginskii_state(
            ne_cm3=5e12, te_ev=2.0, b0_g=800.0,
            dln_ne_dr_cm=0.0, dln_te_dr_cm=0.0, er_v_per_cm=0.0,
        )
        # Zero resistivity by disabling collisions via high Te or testing directly on a custom state
        st_ideal = st._replace(nu_ei_s=0.0, eta_par_ohm_m=0.0, chi_par_m2_s=0.0)
        k_perp = 80.0
        k_par = 0.5
        _, _, evals = solve_local_braginskii(
            st_ideal, k_perp, k_par,
            nu_in_s=0.0, em=True, thermal_force=False, chi_par_on=False,
            include_centrifugal=False,
        )
        # All modes must be neutrally stable (real frequencies)
        self.assertLess(float(np.max(np.abs(evals.imag))), 1e-3)

        b = (k_perp * st_ideal.rho_s_m) ** 2
        mu_e = M_ELECTRON / (st_ideal.mu * M_PROTON)
        # Exact kinetic + inertial Alfven wave (with adiabatic electron gamma_e = 5/3):
        # w^2 = k_par^2 v_A^2 * (1 + (5/3)*k_perp^2*rho_s^2) / (1 + k_perp^2*d_e^2)
        w_A_exact = (
            k_par * st_ideal.v_alfven_m_s
            * np.sqrt((1.0 + (5.0 / 3.0) * b) / (1.0 + 2.0 * b * mu_e / st_ideal.beta_e))
        )
        # Adiabatic oblique ion-acoustic / slow wave (gamma_e = 5/3 when chi_par = 0):
        # w_s = sqrt(5/3) * k_par * c_s / sqrt(1 + (5/3)*k_perp^2*rho_s^2)
        w_s_exact = np.sqrt(5.0 / 3.0) * k_par * st_ideal.cs_m_s / np.sqrt(1.0 + (5.0 / 3.0) * b)

        re_pos = np.sort(evals.real[evals.real > 1.0])
        self.assertAlmostEqual(re_pos[-1] / w_A_exact, 1.0, delta=0.01)
        self.assertAlmostEqual(re_pos[0] / w_s_exact, 1.0, delta=0.01)

    def test_centrifugal_stability_sign_hollow_vs_peaked(self):
        # Flute limit (k_par = 0): w^2 = (k_theta / k_perp)^2 * r * Omega_E^2 * (d ln n / dr)
        k_perp = 100.0
        st_hollow = local_braginskii_state(
            ne_cm3=5e12, te_ev=2.0, b0_g=800.0,
            dln_ne_dr_cm=+0.35, dln_te_dr_cm=0.0, er_v_per_cm=+1.2, r_cm=3.0, m=-1,
        )
        w_h, _, evals_h = solve_local_braginskii(
            st_hollow, k_perp, k_par_m=0.0,
            em=False, thermal_force=False, chi_par_on=False, include_centrifugal=True,
        )
        # Hollow wall (dn/dr > 0): strictly stable (gamma = 0) with real Brunt-Vaisala frequency
        self.assertLess(abs(w_h.imag), 1e-4)
        w_bv = (st_hollow.k_theta_m / k_perp) * np.sqrt(
            st_hollow.r_m * (st_hollow.omega_E_rad_s ** 2) * (0.35 * 1e2)
        )
        self.assertAlmostEqual(float(np.max(np.abs(evals_h.real))) / w_bv, 1.0, delta=1e-4)

        # Peaked edge (dn/dr < 0): rotational interchange unstable with exact gamma = w_bv
        st_peaked = local_braginskii_state(
            ne_cm3=5e12, te_ev=2.0, b0_g=800.0,
            dln_ne_dr_cm=-0.35, dln_te_dr_cm=0.0, er_v_per_cm=+1.2, r_cm=3.0, m=-1,
        )
        w_p, _, _ = solve_local_braginskii(
            st_peaked, k_perp, k_par_m=0.0,
            em=False, thermal_force=False, chi_par_on=False, include_centrifugal=True,
        )
        self.assertAlmostEqual(w_p.imag / w_bv, 1.0, delta=1e-4)
        self.assertLess(abs(w_p.real), 1e-4)

    def test_local_cwm_matches_analytic_asymptote_and_axial_bvp_short_limit(self):
        # Asymptotic CWM limit (nu_C >> |Lambda * w_*T|, no resistance/chi):
        # w_r ~ Lambda * w_*T,  gamma ~ (Lambda * w_*T)^2 / nu_C > 0
        st = local_braginskii_state(
            ne_cm3=5e12, te_ev=2.0, b0_g=800.0,
            dln_ne_dr_cm=0.0, dln_te_dr_cm=-0.04, er_v_per_cm=0.0, r_cm=3.0, m=-1,
        )
        k_perp = 40.0
        L = 5.0
        lam = 3.7
        w_loc, _ = solve_local_cwm(
            st, k_perp, l_par_m=L, sheath_lambda=lam, n_ends=2.0,
            include_res=False, include_chi=False,
        )
        b = (k_perp * st.rho_s_m) ** 2
        nu_C = (2.0 * st.cs_m_s / L) / b
        w_asym_re = lam * st.omega_star_T_rad_s
        w_asym_im = (w_asym_re ** 2) / nu_C
        # Check asymptotic regime holds and matches within 5%
        self.assertGreater(nu_C, 5.0 * abs(w_asym_re))
        self.assertAlmostEqual(w_loc.real / w_asym_re, 1.0, delta=0.05)
        self.assertAlmostEqual(w_loc.imag / w_asym_im, 1.0, delta=0.08)

        # At k_par = 0, the 5-field local Braginskii solver with lumped sheath closure
        # reduces identically to the 2-field local CWM dispersion.
        w_brag_cwm, _, _ = solve_local_braginskii(
            st, k_perp, k_par_m=0.0, em=False, thermal_force=False,
            chi_par_on=False, include_centrifugal=False,
            sheath_l_par_m=L, sheath_lambda=lam, n_ends=2.0,
        )
        self.assertAlmostEqual(w_brag_cwm.real / w_loc.real, 1.0, places=6)
        self.assertAlmostEqual(w_brag_cwm.imag / w_loc.imag, 1.0, places=6)

        # In the ideal (zero-resistivity, zero-chi, ES) short-column limit, the 5-field
        # 1-D axial BVP recovers the same-sign sheath-anchored CWM branch (with O(1)
        # shift from parallel electron compressional work (2/3) div_par(v_e) and Bohm vi).
        st_ideal = st._replace(nu_ei_s=0.0, eta_par_ohm_m=0.0, chi_par_m2_s=0.0)
        res_ax = solve_axial_braginskii_bvp(
            st_ideal, k_perp, l_par_m=L, sheath_lambda=lam, bc="conducting_2",
            em=False, thermal_force=False, chi_par_on=False,
            include_centrifugal=False, n_nodes=14,
        )
        self.assertIsNotNone(res_ax["top"])
        self.assertGreater(res_ax["top"].omega_rad_s.real * w_loc.real, 0.0)
        self.assertGreater(res_ax["top"].omega_rad_s.imag, 0.0)
        self.assertGreater(res_ax["top"].j_end_rel, 0.99)
        self.assertAlmostEqual(res_ax["top"].omega_rad_s.real / w_loc.real, 1.0, delta=0.25)

    def test_resistive_drift_wave_has_positive_growth_along_omega_star_n(self):
        # Hasegawa-Wakatani limit (grad T_e = 0, finite nu_ei & k_par):
        # unstable with Re(w) same sign as omega_*n!
        st = local_braginskii_state(
            ne_cm3=6e12, te_ev=2.5, b0_g=800.0,
            dln_ne_dr_cm=+0.35, dln_te_dr_cm=0.0, er_v_per_cm=0.0, r_cm=3.0, m=-1,
        )
        w, _, _ = solve_local_braginskii(
            st, k_perp_m=80.0, k_par_m=1.0,
            em=False, thermal_force=False, chi_par_on=False, include_centrifugal=False,
        )
        self.assertGreater(w.imag, 0.0)
        self.assertGreater(w.real * st.omega_star_n_rad_s, 0.0)

    def test_radial_pure_kh_stable_on_gaussian_well_and_shear_suppresses_cwm(self):
        r_cm = np.linspace(0.5, 12.0, 24)
        # Smooth hollow density, peaked core Te, smooth Gaussian Vp well
        ne = 9e12 - 6e12 * np.exp(-(r_cm / 3.5) ** 2)
        te = 1.2 + 3.0 * np.exp(-(r_cm / 3.5) ** 2)
        vp = 20.0 + 6.0 * np.exp(-(r_cm / 4.0) ** 2)

        kh = solve_radial_cylindrical_modes(r_cm, ne, te, vp, model="pure_kh")
        self.assertLessEqual(kh["gamma_s"], 1e-2)

        cwm_nosh = solve_radial_cylindrical_modes(
            r_cm, ne, te, vp, model="cwm_sheath",
            include_centrifugal=False, include_exb_shear=False,
        )
        cwm_sh = solve_radial_cylindrical_modes(
            r_cm, ne, te, vp, model="cwm_sheath",
            include_centrifugal=False, include_exb_shear=True,
        )
        self.assertGreater(cwm_nosh["gamma_s"], 1e3)
        self.assertLess(cwm_sh["gamma_s"], cwm_nosh["gamma_s"])

    def test_fit_smooth_radial_profile_recovers_gaussian(self):
        x = np.arange(-15.0, 16.0, 1.0)
        xc = 1.0
        y = 20.0 + 6.0 * np.exp(-((x - xc) / 4.0) ** 2)
        r_grid = np.linspace(0.5, 12.0, 24)
        y_fit, resid = fit_smooth_radial_profile(x, xc, y, r_grid)
        self.assertLess(resid, 1e-6)
        self.assertLess(float(np.max(np.abs(y_fit - (20.0 + 6.0 * np.exp(-(r_grid / 4.0) ** 2))))), 1e-6)

    def test_axial_cwm_classifier_sign_agnostic_and_requires_positive_gamma(self):
        # Repro with outward-rising Te (d ln T/dr > 0 -> omega_*T > 0 for m = -1):
        # CWM propagates at positive frequency (+9.66 kHz, gamma ~ +3.86 ms^-1),
        # while the bulk drift mode sits at +16.3 kHz (j_end ~ 0.15). Must select
        # the positive-f unstable CWM consistently across N = 24, 32, 48 (not a
        # damped negative-f numerical mode).
        st = local_braginskii_state(
            ne_cm3=1e12, te_ev=5.0, b0_g=800.0,
            dln_ne_dr_cm=+0.5, dln_te_dr_cm=+0.3,
            er_v_per_cm=3.0, r_cm=3.0, m=-1,
        )
        kp = float(np.hypot(st.k_theta_m, np.pi / 0.04))
        cwm_freqs, cwm_gammas = [], []
        for n_nodes in (24, 32, 48):
            res = solve_axial_braginskii_bvp(st, kp, bc="conducting_1", n_nodes=n_nodes)
            self.assertIsNotNone(res["cwm"])
            self.assertGreater(res["cwm"].gamma_s, 1e3)
            self.assertGreater(res["cwm"].f_hz, 0.0)
            self.assertGreaterEqual(res["cwm"].j_end_rel, 0.85)
            self.assertGreaterEqual(res["cwm"].phi_end_rel, 0.65)
            cwm_freqs.append(res["cwm"].f_hz)
            cwm_gammas.append(res["cwm"].gamma_s)
        # Converged across node counts (within 1%)
        self.assertLess(np.ptp(cwm_freqs) / np.mean(cwm_freqs), 0.01)
        self.assertLess(np.ptp(cwm_gammas) / np.mean(cwm_gammas), 0.02)

        # Insulating ends must never report a CWM
        res_ins = solve_axial_braginskii_bvp(st, kp, bc="insulating", n_nodes=24)
        self.assertIsNone(res_ins["cwm"])

        # Damped plasma (zero gradients, finite nu_in): no unstable CWM -> None
        st_flat = local_braginskii_state(
            ne_cm3=5e12, te_ev=1.5, b0_g=800.0,
            dln_ne_dr_cm=0.0, dln_te_dr_cm=0.0, er_v_per_cm=0.0,
        )
        res_damped = solve_axial_braginskii_bvp(
            st_flat, kp, bc="conducting_1", nu_in_s=500.0, n_nodes=24,
        )
        self.assertIsNone(res_damped["cwm"])

    def test_select_near_onset_sweep_derives_from_timestamps_and_flags_straddling(self):
        t_start = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
        t_mid = np.array([-0.25, 0.25, 0.75, 1.25, 1.60, 2.25])
        regs = np.array(["pre_bias", "pre_onset", "pre_onset", "pre_onset", "straddling", "post_onset"])

        # Stable run -> uses ref_sweep_idx
        idx_s, reg_s = jst.select_near_onset_sweep(t_start, t_mid, regs, np.nan, ref_sweep_idx=3)
        self.assertEqual((idx_s, reg_s), (3, "pre_onset"))

        # Unstable run with t_onset = 1.65 ms -> picks straddling sweep at 1.60 ms (idx 4)
        idx_u, reg_u = jst.select_near_onset_sweep(t_start, t_mid, regs, 1.65, ref_sweep_idx=3)
        self.assertEqual((idx_u, reg_u), (4, "straddling"))

        # Unstable run with t_onset = 1.40 ms -> picks pre_onset sweep at 1.25 ms (idx 3)
        idx_p, reg_p = jst.select_near_onset_sweep(t_start, t_mid, regs, 1.40, ref_sweep_idx=3)
        self.assertEqual((idx_p, reg_p), (3, "pre_onset"))

    def test_biased_sheath_parameters_and_generalized_axial_bvp(self):
        # 1. Exact biased_sheath_parameters scaling, mode_coupling, and electron saturation auto-clamp
        area = np.pi * (0.0254 ** 2)
        p = biased_sheath_parameters(
            i_disc_a=20.0, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, sheath_lambda=3.7, l_ohm_m=1.92,
            mode_coupling=0.3,
        )
        self.assertLess(p["j0_norm"], 0.0)
        self.assertGreater(p["g_sh"], 1.0)
        self.assertLess(p["lambda_eff"], 3.7)
        self.assertFalse(p["electron_saturated"])
        self.assertAlmostEqual(p["r_ext_hat"], 0.20 / p["r_sh_ohm"], places=12)
        self.assertAlmostEqual(p["r_ext_eff_hat"], 0.09 * p["r_ext_hat"], places=12)
        self.assertGreater(p["dv_ohm_v"], 0.0)

        # Centered m != 0 mode (mode_coupling=0) sees zero external impedance
        p_ctr = biased_sheath_parameters(
            i_disc_a=20.0, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, mode_coupling=0.0,
        )
        self.assertEqual(p_ctr["r_ext_eff_hat"], 0.0)

        # Electron saturation knee tapers g_sh smoothly without a jump at u = 1
        lam_k = 3.5
        e_lam = np.exp(lam_k)
        i_sat = p["j_sat_a_m2"] * area
        p_mid = biased_sheath_parameters(
            i_disc_a=(0.75 * e_lam - 1.0) * i_sat, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, sheath_lambda=lam_k,
        )
        p_near = biased_sheath_parameters(
            i_disc_a=(0.99 * e_lam - 1.0) * i_sat, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, sheath_lambda=lam_k,
        )
        self.assertTrue(p_mid["near_electron_sat"])
        self.assertFalse(p_mid["electron_saturated"])
        self.assertGreater(p_mid["g_sh"], p_near["g_sh"])
        self.assertLess(p_near["g_sh"], 0.1)  # smooth roll-off near u -> 1, not ~exp(lam)

        # Electron saturation (1 - j0 >= exp(Lambda)): BOTH lambda_eff and g_sh auto-clamp to 0
        p_esat = biased_sheath_parameters(
            i_disc_a=2000.0, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, sheath_lambda=lam_k,
        )
        self.assertTrue(p_esat["electron_saturated"])
        self.assertTrue(p_esat["near_electron_sat"])
        self.assertEqual(p_esat["lambda_eff"], 0.0)
        self.assertEqual(p_esat["g_sh"], 0.0)

        # 2. High external resistance r_ext_hat -> inf approaches insulating BC
        st = local_braginskii_state(
            ne_cm3=5e12, te_ev=2.0, b0_g=800.0,
            dln_ne_dr_cm=+0.3, dln_te_dr_cm=-0.3, er_v_per_cm=1.2, r_cm=3.0, m=-1,
        )
        kp = float(np.hypot(st.k_theta_m, np.pi / 0.04))
        res_ins = solve_axial_braginskii_bvp(st, kp, bc="insulating", n_nodes=24)
        res_hi_r = solve_axial_braginskii_bvp(
            st, kp, bc="conducting_1", r_ext_hat=1e6, n_nodes=24,
        )
        self.assertIsNone(res_hi_r["cwm"])
        self.assertAlmostEqual(res_hi_r["top"].f_hz, res_ins["top"].f_hz, delta=50.0)
        self.assertAlmostEqual(res_hi_r["top"].gamma_s, res_ins["top"].gamma_s, delta=100.0)

        # 3. Automatic electron-saturation clamp in solve_axial_braginskii_bvp
        # (j0_norm <= 1 - exp(Lambda)) removes the negative-f CWM without manual overrides
        res_flt = solve_axial_braginskii_bvp(st, kp, bc="conducting_1", n_nodes=24)
        self.assertIsNotNone(res_flt["cwm"])
        self.assertLess(res_flt["cwm"].f_hz, -1e3)
        res_esat_auto = solve_axial_braginskii_bvp(
            st, kp, bc="conducting_1", sheath_lambda=3.5,
            j0_norm=1.0 - np.exp(3.5) - 1.0,
            include_current_convective=False, n_nodes=24,
        )
        self.assertTrue(res_esat_auto["cwm"] is None or res_esat_auto["cwm"].f_hz > 0.0)

        # 4. conducting_2 uses opposite outward DC current at z=0 (-j0) vs z=L (+j0):
        # flipping j0 -> -j0 must preserve the eigenvalue spectrum by z <-> L-z symmetry
        res_c2_pos = solve_axial_braginskii_bvp(
            st, kp, bc="conducting_2", j0_norm=-0.4, n_nodes=24,
        )
        res_c2_neg = solve_axial_braginskii_bvp(
            st, kp, bc="conducting_2", j0_norm=+0.4, n_nodes=24,
        )
        self.assertIsNotNone(res_c2_pos["cwm"])
        self.assertIsNotNone(res_c2_neg["cwm"])
        self.assertAlmostEqual(res_c2_pos["cwm"].f_hz, res_c2_neg["cwm"].f_hz, delta=1e-3)
        self.assertAlmostEqual(res_c2_pos["cwm"].gamma_s, res_c2_neg["cwm"].gamma_s, delta=1e-3)

        # 5. Kadomtsev-Nedospasov current-convective drive with j0_norm < 0 excites
        # a negative-f resistive mode on a collisional d(ln T)/dr < 0 column even
        # with insulating ends (absent at j0_norm = 0)
        st_res = local_braginskii_state(
            ne_cm3=1.2e13, te_ev=1.0, b0_g=800.0,
            dln_ne_dr_cm=+0.3, dln_te_dr_cm=-0.3, er_v_per_cm=1.0, r_cm=3.0, m=-1,
        )
        res_ins0 = solve_axial_braginskii_bvp(st_res, kp, bc="insulating", n_nodes=24)
        res_ins_cc = solve_axial_braginskii_bvp(
            st_res, kp, bc="insulating", j0_norm=-0.5,
            include_current_convective=True, n_nodes=24,
        )
        self.assertLess(res_ins_cc["top"].f_hz, 0.0)
        self.assertGreater(res_ins_cc["top"].gamma_s, res_ins0["top"].gamma_s + 1e3)

    def test_jun2026_stability_batch_all_respects_overwrite_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "stability_linear_jun2026.npz")
            np.savez(out_npz, sentinel=np.int64(7))
            res = jst.batch_all(data_dir=tmp, out_path=out_npz, overwrite=False)
            self.assertEqual(res, out_npz)
            with np.load(out_npz) as d:
                self.assertEqual(int(d["sentinel"]), 7)

    def test_analyze_biased_sheath_uses_matched_core_footprint_for_2in_disc(self):
        # Synthetic hollow-core profile: hot, low-ne core (r <= 2.54 cm) vs cooler,
        # denser wall (r = 2-4 cm). The 2" disc diagnostic (p2) must evaluate
        # dv_ohm_2in_v on the matched core (ne_2in, Te_2in), while the 4" annulus
        # (p4) stays matched to the wall state.
        x = np.arange(-8.0, 9.0, 1.0)
        rr = np.abs(x)
        ne = np.where(rr <= 2.54, 3.0e12, 1.2e13)[:, None]
        te = np.where(rr <= 2.54, 4.0, 1.0)[:, None]
        vp = np.where(rr <= 2.54, 24.0, 20.0)[:, None]
        lam = np.where(rr <= 2.54, 4.2, 3.4)[:, None]
        zeros = np.zeros_like(ne)
        dln_ne = np.full_like(ne, +0.3)
        dln_te = np.full_like(ne, -0.3)
        er = np.full_like(ne, +1.0)

        with tempfile.TemporaryDirectory() as tmp:
            prof_p = os.path.join(tmp, "profiles_summary_jun2026.npz")
            np.savez(
                prof_p,
                **{
                    "00__x_cm": x,
                    "00__x_center_cm": np.float64(0.0),
                    "00__ref_sweep_idx": np.int64(0),
                    "00__ne": ne,
                    "00__Te": te,
                    "00__Vp": vp,
                    "00__lambda": lam,
                    "00__dln_ne_dr": dln_ne,
                    "00__dln_te_dr": dln_te,
                    "00__er_r_v_cm": er,
                    "00__ref_vp_core_v": np.float64(24.0),
                    "00__ref_vp_ring_v": np.float64(20.0),
                    "00__ref_ne_core_cm3": np.float64(3.0e12),
                    "00__ref_te_core_ev": np.float64(4.0),
                },
            )
            bs_data = {
                "00": {
                    "gas_puff_ms": 24.0,
                    "f0_hz": 3500.0,
                    "I_2in_delta": 18.0,
                    "I_4in_delta": 22.0,
                    "V_2in_dc": 40.0,
                    "V_4in_dc": 40.0,
                    "R_ext_2in_ohm": 0.15,
                    "R_ext_4in_ohm": 0.18,
                    "g2_VI_2in": 0.9,
                    "g2_VI_4in": 0.9,
                },
            }
            out = jst.analyze_biased_sheath(
                data_dir=tmp, profiles_npz=prof_p, onset_npz=os.path.join(tmp, "none.npz"),
                bdot_2d_npz=os.path.join(tmp, "none_bdot.npz"),
                n_nodes=16, biasscope_data=bs_data,
            )
            r0 = out["00"]
            exp_p2 = biased_sheath_parameters(
                18.0, jst.DISC_AREAS_M2["2in"], 0.15, 3.0e18, 4.0,
                sheath_lambda=4.2, l_ohm_m=jst.L_P29_TO_P35_M,
                mode_coupling=r0["mode_coupling_2in"],
            )
            wall_p2 = biased_sheath_parameters(
                18.0, jst.DISC_AREAS_M2["2in"], 0.15, 1.2e19, 1.0,
                sheath_lambda=3.4, l_ohm_m=jst.L_P29_TO_P35_M,
                mode_coupling=r0["mode_coupling_2in"],
            )
            self.assertAlmostEqual(r0["ne_2in_m3"], 3.0e18, places=6)
            self.assertAlmostEqual(r0["te_2in_ev"], 4.0, places=12)
            self.assertAlmostEqual(r0["lambda_2in"], 4.2, places=12)
            self.assertAlmostEqual(r0["dv_ohm_2in_v"], exp_p2["dv_ohm_v"], places=12)
            # Core Te=4 eV gives ~8x smaller Ohmic drop than cooler wall Te=1 eV
            self.assertLess(r0["dv_ohm_2in_v"], 0.25 * wall_p2["dv_ohm_v"])
            self.assertAlmostEqual(r0["j0_2in_norm"], exp_p2["j0_norm"], places=12)

    def test_evaluate_axial_sheath_closure_sign_convention_and_isat_cancellation(self):
        # Hollow inner wall (dn/dr > 0, dTe/dr < 0 -> eta_e = -1.06 < 0):
        # dTe/Te and dn/n are ~180 deg out of phase, causing partial cancellation in
        # dI_sat/I_sat = dn/n + 0.5*dTe/Te (|I_sat|/|n| < 1) and shifting dphi(j - I_sat)
        # into the positive in-phase quadrant.
        st = local_braginskii_state(
            ne_cm3=1.29e13, te_ev=0.91, b0_g=800.0,
            dln_ne_dr_cm=+0.221, dln_te_dr_cm=-0.235,
            er_v_per_cm=+0.95, r_cm=3.0, m=-1, b_sign_z=+1,
        )
        kp = float(np.hypot(st.k_theta_m, np.pi / 0.04))
        L = 14.0
        z_p28 = L - jst.L_P28_TO_P35_M
        z_p29 = L - jst.L_P29_TO_P35_M

        res_flt = solve_axial_braginskii_bvp(
            st, kp, l_par_m=L, sheath_lambda=3.83, bc="conducting_1", n_nodes=24,
        )
        res_rext = solve_axial_braginskii_bvp(
            st, kp, l_par_m=L, sheath_lambda=3.83, bc="conducting_1",
            r_ext_hat=1.44, n_nodes=24,
        )
        res_bsh = solve_axial_braginskii_bvp(
            st, kp, l_par_m=L, sheath_lambda=3.83, bc="conducting_1",
            j0_norm=-0.44, r_ext_hat=1.44, n_nodes=24,
        )
        res_ins = solve_axial_braginskii_bvp(
            st, kp, l_par_m=L, sheath_lambda=3.83, bc="insulating", n_nodes=24,
        )

        ev_flt = jst.evaluate_axial_sheath_closure(res_flt, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=L)
        ev_rext = jst.evaluate_axial_sheath_closure(res_rext, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=L)
        ev_bsh = jst.evaluate_axial_sheath_closure(res_bsh, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=L)
        ev_ins = jst.evaluate_axial_sheath_closure(res_ins, z_p28_m=z_p28, z_p29_m=z_p29, l_par_m=L)

        self.assertTrue(ev_flt["has_cwm"])
        self.assertGreater(abs(ev_flt["dphi_T_n_sheath_deg"]), 150.0)
        self.assertLess(ev_flt["isat_over_n_sheath"], 0.85)
        self.assertGreater(ev_flt["j_over_isat_sheath"], ev_flt["j_over_n_sheath"])
        self.assertGreater(ev_flt["dphi_j_isat_sheath_deg"], 0.0)
        self.assertLess(ev_flt["dphi_j_isat_sheath_deg"], 70.0)
        self.assertGreater(ev_flt["dphi_j_isat_p28_deg"], 0.0)
        self.assertLess(ev_flt["dphi_j_isat_p28_deg"], 70.0)

        # Series resistance & biased sheath throttle the normalized sheath current ratio
        self.assertLess(ev_rext["j_over_isat_sheath"], ev_flt["j_over_isat_sheath"])
        self.assertLess(ev_bsh["j_over_isat_sheath"], ev_rext["j_over_isat_sheath"])

        # Insulating BVP has no CWM -> has_cwm=False and NaNs
        self.assertFalse(ev_ins["has_cwm"])
        self.assertTrue(np.isnan(ev_ins["j_over_isat_sheath"]))

    def test_analyze_sheath_closure_peak_rms_invariance_polarity_and_eta_flip(self):
        # Synthetic matched Run 03 profile (hollow core r<=2.5 cm, wall 2.5-4.5 cm)
        # and synthetic calibrated Run 26 (800 G) & Run 31 (600 G) 2-D phasor grids.
        x_prof = np.arange(-8.0, 9.0, 1.0)
        rr_p = np.abs(x_prof)
        ne = np.where(rr_p <= 2.5, 6.4e12, 1.4e13)[:, None]
        te = np.where(rr_p <= 2.5, 1.74, 0.85)[:, None]
        lam = np.where(rr_p <= 2.5, 3.6, 3.85)[:, None]
        dln_ne = np.full_like(ne, +0.22)
        dln_te = np.full_like(ne, -0.24)
        er = np.full_like(ne, +0.95)

        ux = np.arange(-8.0, 9.0, 1.0)
        uy = np.arange(-8.0, 9.0, 1.0)
        XX, YY = np.meshgrid(ux, uy)
        RR = np.hypot(XX, YY)
        theta = np.arctan2(YY, XX)

        # Exact synthetic peak phasors: Jz leads LP by +15.0 deg everywhere
        dphi_true_deg = 15.0
        jz_core_pk, jz_wall_pk = 1400.0, 500.0
        lp_core_pk, lp_wall_pk = 0.28, 0.12
        dc_core_26, dc_wall_26 = 1.00, 1.20
        dc_scale_31 = 0.70

        jz_amp_26 = np.where(RR <= 2.5, jz_core_pk, np.where(RR <= 4.5, jz_wall_pk, 50.0))
        lp_amp_26 = np.where(RR <= 2.5, lp_core_pk, np.where(RR <= 4.5, lp_wall_pk, 0.05))
        lp_26 = lp_amp_26 * np.exp(-1j * theta)
        jz_26 = jz_amp_26 * np.exp(-1j * theta + 1j * np.radians(dphi_true_deg))
        g2_hi = np.full_like(RR, 0.95)
        dc_26 = np.where(RR <= 2.5, dc_core_26, dc_wall_26)
        dc_31 = dc_scale_31 * dc_26

        with tempfile.TemporaryDirectory() as tmp:
            prof_p = os.path.join(tmp, "profiles_summary_jun2026.npz")
            bdot_p = os.path.join(tmp, "bdot_2d_jun2026.npz")
            np.savez(
                prof_p,
                **{
                    "03__x_cm": x_prof,
                    "03__x_center_cm": np.float64(0.0),
                    "03__ref_sweep_idx": np.int64(0),
                    "03__ne": ne,
                    "03__Te": te,
                    "03__lambda": lam,
                    "03__dln_ne_dr": dln_ne,
                    "03__dln_te_dr": dln_te,
                    "03__er_r_v_cm": er,
                },
            )
            np.savez(
                bdot_p,
                **{
                    "26__x_cm": ux, "26__y_cm": uy,
                    "26__f0_hz": np.float64(4110.0),
                    "26__bdot_calibrated": np.bool_(True),
                    "26__abs_coil_polarity_verified": np.bool_(True),
                    "26__h1__Jz": jz_26, "26__h1__LP": lp_26, "26__h1__g2_LP": g2_hi,
                    "26__h1__center_x_cm": np.float64(0.0), "26__h1__center_y_cm": np.float64(0.0),
                    "26__h1__dphi_J_minus_LP_deg": np.float64(dphi_true_deg),
                    "26__h1__R_J_LP": np.float64(1.0),
                    "31__x_cm": ux, "31__y_cm": uy,
                    "31__f0_hz": np.float64(3529.0),
                    "31__bdot_calibrated": np.bool_(True),
                    "31__abs_coil_polarity_verified": np.bool_(True),
                    "31__h1__Jz": 0.7 * jz_26, "31__h1__LP": 0.7 * lp_26, "31__h1__g2_LP": g2_hi,
                    "31__h1__center_x_cm": np.float64(0.0), "31__h1__center_y_cm": np.float64(0.0),
                    "31__h1__dphi_J_minus_LP_deg": np.float64(dphi_true_deg),
                    "31__h1__R_J_LP": np.float64(1.0),
                },
            )
            lp_dc_data = {
                "26": {"x_cm": ux, "y_cm": uy, "lp_dc": dc_26, "lp_polarity_sign": +1, "source": "synthetic"},
                "31": {"x_cm": ux, "y_cm": uy, "lp_dc": dc_31, "lp_polarity_sign": +1, "source": "synthetic"},
            }
            res = jst.analyze_sheath_closure(
                data_dir=tmp, profiles_npz=prof_p, bdot_2d_npz=bdot_p,
                n_nodes=20, lp_dc_data=lp_dc_data,
            )

            e26 = res["exp_26"]
            e31 = res["exp_31"]
            rp = res["ref_profile"]

            # Exact peak-vs-RMS normalization and sqrt(2) cancellation in ratio
            self.assertAlmostEqual(
                e26["core_j_par_time_rms_a_m2"], jz_core_pk / np.sqrt(2.0), places=9,
            )
            self.assertAlmostEqual(
                e26["core_disat_over_isat_rms"], (lp_core_pk / dc_core_26) / np.sqrt(2.0), places=9,
            )
            exp_core_ratio = (jz_core_pk / rp["core_j_sat_a_m2"]) / (lp_core_pk / dc_core_26)
            exp_wall_ratio = (jz_wall_pk / rp["wall_j_sat_a_m2"]) / (lp_wall_pk / dc_wall_26)
            self.assertAlmostEqual(e26["core_j_over_isat_ratio"], exp_core_ratio, places=9)
            self.assertAlmostEqual(e26["wall_j_over_isat_ratio"], exp_wall_ratio, places=9)
            self.assertAlmostEqual(e26["core_dphi_j_isat_deg"], dphi_true_deg, places=6)
            self.assertAlmostEqual(e26["wall_dphi_j_isat_deg"], dphi_true_deg, places=6)
            self.assertAlmostEqual(e26["core_R_j_isat"], 1.0, places=9)

            # Run 31 scaled both Jz and LP_dc by 0.7 -> DC-scaled ratio equals Run 26!
            self.assertAlmostEqual(e31["core_lp_dc_scale_vs_800G"], dc_scale_31, places=9)
            self.assertAlmostEqual(e31["core_j_over_isat_ratio_dc_scaled"], exp_core_ratio, places=9)
            self.assertAlmostEqual(e31["wall_j_over_isat_ratio_dc_scaled"], exp_wall_ratio, places=9)

            # Inverted LP polarity flips dphi(J - I_sat) by exactly 180 deg
            lp_dc_inv = {
                "26": {"x_cm": ux, "y_cm": uy, "lp_dc": -dc_26, "lp_polarity_sign": -1, "source": "synthetic_inv"},
                "31": {"x_cm": ux, "y_cm": uy, "lp_dc": -dc_31, "lp_polarity_sign": -1, "source": "synthetic_inv"},
            }
            res_inv = jst.analyze_sheath_closure(
                data_dir=tmp, profiles_npz=prof_p, bdot_2d_npz=bdot_p,
                n_nodes=20, lp_dc_data=lp_dc_inv,
            )
            self.assertAlmostEqual(
                res_inv["exp_26"]["core_dphi_j_isat_deg"], dphi_true_deg - 180.0, places=6,
            )

            # Hollow (eta_e < 0) vs Peaked (eta_e > 0) flips sheath dphi(j - I_sat) by ~180 deg
            ef = res["eta_flip"]
            self.assertLess(ef["eta_e_hollow"], 0.0)
            self.assertGreater(ef["eta_e_peaked"], 0.0)
            for mtag in ("conducting_1", "flt_rext", "bsh_rext"):
                self.assertTrue(ef[f"{mtag}_hollow_has_cwm"])
                self.assertTrue(ef[f"{mtag}_peaked_has_cwm"])
                self.assertGreater(abs(ef[f"{mtag}_hollow_dphi_T_n_sheath_deg"]), 150.0)
                self.assertLess(abs(ef[f"{mtag}_peaked_dphi_T_n_sheath_deg"]), 30.0)
                self.assertLess(ef[f"{mtag}_hollow_isat_over_n_sheath"], 0.85)
                self.assertGreater(ef[f"{mtag}_peaked_isat_over_n_sheath"], 1.2)
                self.assertGreater(ef[f"{mtag}_abs_delta_dphi_sheath_deg"], 150.0)

            # Uncalibrated bdot_2d file must raise ValueError
            bdot_uncal = os.path.join(tmp, "bdot_uncal.npz")
            np.savez(
                bdot_uncal,
                **{
                    "26__x_cm": ux, "26__y_cm": uy,
                    "26__f0_hz": np.float64(4110.0),
                    "26__bdot_calibrated": np.bool_(False),
                    "26__abs_coil_polarity_verified": np.bool_(True),
                    "26__h1__Jz": jz_26, "26__h1__LP": lp_26, "26__h1__g2_LP": g2_hi,
                    "26__h1__center_x_cm": np.float64(0.0), "26__h1__center_y_cm": np.float64(0.0),
                },
            )
            with self.assertRaises(ValueError):
                jst.analyze_sheath_closure(
                    data_dir=tmp, profiles_npz=prof_p, bdot_2d_npz=bdot_uncal,
                    n_nodes=16, lp_dc_data=lp_dc_data,
                )


if __name__ == "__main__":
    unittest.main()
