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
        # 1. Exact biased_sheath_parameters scaling
        area = np.pi * (0.0254 ** 2)
        p = biased_sheath_parameters(
            i_disc_a=20.0, area_m2=area, r_ext_ohm=0.20,
            ne_m3=5e18, te_ev=2.0, sheath_lambda=3.7, l_ohm_m=1.92,
        )
        self.assertLess(p["j0_norm"], 0.0)
        self.assertGreater(p["g_sh"], 1.0)
        self.assertLess(p["lambda_eff"], 3.7)
        self.assertAlmostEqual(p["r_ext_hat"], 0.20 / p["r_sh_ohm"], places=12)
        self.assertGreater(p["dv_ohm_v"], 0.0)

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

        # 3. Pure flat electron-saturation (g_sh=0, lam_eff=0, no CC) removes negative-f CWM
        res_flt = solve_axial_braginskii_bvp(st, kp, bc="conducting_1", n_nodes=24)
        self.assertIsNotNone(res_flt["cwm"])
        self.assertLess(res_flt["cwm"].f_hz, -1e3)
        res_esat = solve_axial_braginskii_bvp(
            st, kp, bc="conducting_1", j0_norm=-0.5,
            sheath_conductance=0.0, sheath_lambda_eff=0.0,
            include_current_convective=False, n_nodes=24,
        )
        self.assertTrue(res_esat["cwm"] is None or res_esat["cwm"].f_hz > 0.0)

        # 4. Kadomtsev-Nedospasov current-convective drive with j0_norm < 0 excites
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


if __name__ == "__main__":
    unittest.main()
