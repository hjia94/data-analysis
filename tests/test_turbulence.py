"""Unit tests for broadband spectrum/intermittency/Reynolds helpers and Jun2026_turbulence."""
import os
import sys
import tempfile
import unittest

import numpy as np

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(WS, "src"))
sys.path.insert(0, os.path.join(WS, "experiments", "ucla-lapd", "Jun-2026"))

from data_analysis.signal import (
    amplitude_normalized_cross_phase,
    fit_broadband_spectrum,
    fluctuation_moments,
    harmonic_steepening,
    reynolds_shear_production,
)
import Jun2026_turbulence as jtb


class TestTurbulenceHelpers(unittest.TestCase):
    def test_fit_broadband_spectrum_distinguishes_exponential_vs_power_law(self):
        freqs = np.linspace(500.0, 80000.0, 320)
        bg = np.full_like(freqs, 1e-12)

        # Pure exponential continuum + strong 4 kHz coherent peak
        f_e = 8000.0
        psd_exp = 1e-4 * np.exp(-freqs / f_e)
        psd_exp[np.abs(freqs - 4000.0) <= 500.0] += 1.0
        fit_e = fit_broadband_spectrum(freqs, psd_exp, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_e.preferred_model, "exponential")
        self.assertGreater(fit_e.r2_exp, 0.999)
        self.assertAlmostEqual(fit_e.f_e_hz, f_e, delta=1e-3 * f_e)
        self.assertAlmostEqual(fit_e.tau_lorentz_s, 1.0 / (4.0 * np.pi * f_e), delta=1e-8)
        self.assertGreater(fit_e.coh_harmonic_frac, 0.95)

        # Pure power-law continuum f^-3.5
        psd_pow = 1e6 * (freqs ** -3.5)
        fit_p = fit_broadband_spectrum(freqs, psd_pow, bg_psd=bg, harmonic_f0_hz=4000.0)
        self.assertEqual(fit_p.preferred_model, "power_law")
        self.assertGreater(fit_p.r2_pow, 0.999)
        self.assertAlmostEqual(fit_p.alpha, 3.5, places=3)

        # Below SNR floor -> insufficient_data
        fit_low = fit_broadband_spectrum(freqs, bg * 2.0, bg_psd=bg, snr_min=5.0)
        self.assertEqual(fit_low.preferred_model, "insufficient_data")
        self.assertTrue(np.isnan(fit_low.r2_exp))

    def test_fluctuation_moments_sine_wave_and_gaussian(self):
        t = np.linspace(0.0, 10.0, 4000, endpoint=False)
        # Pure sinusoid superposed on linear ramp -> exact excess kurtosis = -1.5, skew = 0
        sine_stack = 10.0 + 0.5 * t[None, :] + 1.0 * np.sin(2.0 * np.pi * 5.0 * t)[None, :]
        m_sin = fluctuation_moments(sine_stack, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_sin.skewness_med, 0.0, delta=1e-2)
        self.assertAlmostEqual(m_sin.excess_kurtosis_med, -1.5, delta=0.05)

        # Gaussian noise -> skew ~ 0, excess kurtosis ~ 0
        rng = np.random.default_rng(123)
        gauss = 5.0 + 0.5 * rng.standard_normal((20, 4000))
        m_g = fluctuation_moments(gauss, normalize_by_mean=True, detrend=True)
        self.assertAlmostEqual(m_g.skewness_med, 0.0, delta=0.1)
        self.assertAlmostEqual(m_g.excess_kurtosis_med, 0.0, delta=0.15)
        self.assertAlmostEqual(m_g.rel_rms_med, 0.1, delta=0.01)

    def test_reynolds_shear_production_signs_and_rigid_rotation_zero(self):
        r = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        rs = np.full_like(r, +2e-4)

        # Rigid-body rotation v_theta = Omega * r -> shear = 0 -> P_shear = 0
        prod_rigid = reynolds_shear_production(r, rs, -0.05 * r)
        np.testing.assert_allclose(prod_rigid.shear_per_cm, 0.0, atol=1e-12)
        self.assertAlmostEqual(prod_rigid.p_shear_integrated, 0.0, places=12)

        # Decaying clockwise flow M_theta = -0.25 (constant) -> Omega = -0.25/r -> dOmega/dr = +0.25/r^2 > 0
        # Positive Reynolds stress * positive shear -> P_shear < 0 (wave feeds mean flow)
        prod_feed = reynolds_shear_production(r, rs, np.full_like(r, -0.25))
        self.assertTrue(np.all(prod_feed.shear_per_cm > 0.0))
        self.assertTrue(np.all(prod_feed.p_shear_local < 0.0))
        self.assertLess(prod_feed.p_shear_integrated, 0.0)

        # Opposite-sign Reynolds stress -> P_shear > 0 (KH extraction from mean flow)
        prod_kh = reynolds_shear_production(r, -rs, np.full_like(r, -0.25))
        self.assertGreater(prod_kh.p_shear_integrated, 0.0)

    def test_compute_mach_transport_recovers_quadrature_and_reynolds_stress(self):
        # Synthetic 4 kHz wave at position (dx, dy) = (3.0, 0.0) -> r_hat = x_hat, theta_hat = y_hat
        fs = 100e3
        t = np.arange(0, 300) / fs
        w = 2.0 * np.pi * 4000.0
        # Construct X+, X-, Y+, Y- with known dn/n (10%), dM_r in quadrature (+90 deg to dn),
        # and dM_theta in phase with dM_r (so c_nr ~ 0, c_rt ~ +1)!
        from data_analysis.plasma.mach import K_HUTCHINSON

        dn = 0.10 * np.cos(w * t)
        dmr = 0.05 * np.sin(w * t)
        dmt = 0.05 * np.sin(w * t)
        # mach_single uses FLOW_SIGN = -1 -> I_+- = n * exp(-+ M / (2 * K_H)), I_+/I_- = exp(-M / K_H)
        xp = (1.0 + dn) * np.exp(-dmr / (2.0 * K_HUTCHINSON))
        xm = (1.0 + dn) * np.exp(+dmr / (2.0 * K_HUTCHINSON))
        yp = (1.0 + dn) * np.exp(-dmt / (2.0 * K_HUTCHINSON))
        ym = (1.0 + dn) * np.exp(+dmt / (2.0 * K_HUTCHINSON))
        ch_d = {k: np.tile(v, (4, 1)) for k, v in (("X+", xp), ("X-", xm), ("Y+", yp), ("Y-", ym))}
        rec = jtb._compute_mach_transport_at_position(
            ch_d, {"X": 1.0, "Y": 1.0}, dx_cm=3.0, dy_cm=0.0, fs_hz=fs, ref_ds=np.tile(1.0 + dn, (4, 1))
        )
        self.assertIsNotNone(rec)
        self.assertAlmostEqual(rec["mr_rms"], float(np.std(dmr)), delta=0.005)
        self.assertAlmostEqual(rec["mt_rms"], float(np.std(dmt)), delta=0.005)
        self.assertAlmostEqual(rec["c_nr"], 0.0, delta=0.05)
        self.assertAlmostEqual(rec["c_rt"], +1.0, delta=0.05)
        self.assertGreater(rec["rs_f0"], 0.0)
        self.assertGreater(rec["et_f0"], 0.0)
        self.assertLessEqual(abs(rec["rs_f0"]) / rec["et_f0"], 1.0 + 1e-9)

    def test_eich_manz_shear_transfer_invariance_and_signs(self):
        r_cm = np.array([2.0, 3.0, 4.0, 5.0, 6.25])
        rs = np.full_like(r_cm, +2.0e-4)
        et = np.full_like(r_cm, 4.0e-4)  # rs / et = +0.5
        te = np.full_like(r_cm, 1.0)
        ne = np.full_like(r_cm, 1.0e13)
        # Constant clockwise E x B velocity: v_ExB = -Er/B0 = -1000 m/s -> Er = +0.8 V/cm at 800 G
        er = np.full_like(r_cm, +0.8)
        mth = np.full_like(r_cm, -0.20)
        res = jtb._eich_manz_shear_transfer(
            r_cm, mth, rs, rs, et, et, te, ne, er,
            b0_g=800.0, gamma_exp_s=2000.0, gamma_cwm_s=1000.0,
        )
        # Exact c_s and K_H invariance for LP v_ExB when Te & ne are uniform:
        # rescaling rs & et by any factor k^2 (same as changing K_H) or scaling Te by 4x
        # leaves gamma_sh_exb_f0_s identically unchanged!
        res_scaled = jtb._eich_manz_shear_transfer(
            r_cm, mth, 4.0 * rs, 4.0 * rs, 4.0 * et, 4.0 * et, 4.0 * te, ne, er,
            b0_g=800.0, gamma_exp_s=2000.0, gamma_cwm_s=1000.0,
        )
        self.assertGreater(res["gamma_sh_mach_f0_s"], 0.0)
        self.assertGreater(res["gamma_sh_exb_f0_s"], 0.0)
        self.assertAlmostEqual(res["gamma_sh_exb_f0_s"], res_scaled["gamma_sh_exb_f0_s"], places=8)
        self.assertAlmostEqual(
            res["R_em_mach_f0_vs_exp"], res["gamma_sh_mach_f0_s"] / 2000.0, places=12
        )
        self.assertAlmostEqual(
            res["R_em_exb_f0_vs_cwm"], res["gamma_sh_exb_f0_s"] / 1000.0, places=12
        )
        # Rigid-body rotation (v_theta proportional to r) -> zero shear transfer
        res_rigid = jtb._eich_manz_shear_transfer(
            r_cm, -0.05 * r_cm, rs, rs, et, et, te, ne, +0.2 * r_cm,
            b0_g=800.0, gamma_exp_s=2000.0,
        )
        self.assertAlmostEqual(res_rigid["gamma_sh_mach_f0_s"], 0.0, places=9)
        self.assertAlmostEqual(res_rigid["gamma_sh_exb_f0_s"], 0.0, places=9)

    def test_amplitude_normalized_cross_phase_separates_growth_from_saturation(self):
        # Exact synthetic test: exponential ramp (gamma = 10 ms^-1) with 0 deg cross-phase during
        # growth (t < 1.5 ms) and +60 deg cross-phase in saturation (t >= 1.8 ms).
        fs = 100e3
        t_s = np.arange(0, 450) / fs  # 0..4.5 ms
        t_ms = t_s * 1e3
        w = 2.0 * np.pi * 4000.0
        amp = np.minimum(0.01 * np.exp(10.0 * (t_ms - 1.0)), 1.0)
        # Smooth phase switch from 0 deg on [1.0, 1.45] ms to +60 deg for t >= 1.8 ms
        phi_rad = np.where(t_ms < 1.50, 0.0, np.where(t_ms >= 1.80, np.radians(60.0),
                           np.radians(60.0) * (t_ms - 1.50) / 0.30))
        sig_a = np.tile(amp * np.cos(w * t_s), (4, 1))
        sig_b = np.tile(amp * np.cos(w * t_s + phi_rad), (4, 1))

        m_grow = (t_ms >= 1.05) & (t_ms <= 1.45)
        m_sat = (t_ms >= 2.20) & (t_ms <= 4.00)
        plv_g, phi_g, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=m_grow)
        plv_s, phi_s, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, (2500.0, 5500.0), mask=m_sat)

        self.assertGreater(plv_g, 0.98)
        self.assertGreater(plv_s, 0.98)
        self.assertAlmostEqual(phi_g, 0.0, delta=3.0)
        self.assertAlmostEqual(phi_s, +60.0, delta=3.0)

    def test_per_shot_ramp_masks_early_sat_does_not_overlap_fast_ramp(self):
        # Fast exponential ramp (gamma = 10 ms^-1) reaching saturation around t = 1.6 ms:
        # 0 deg during growth (amp < 0.85 * peak), +60 deg once saturated (amp >= 0.85 * peak).
        fs = 100e3
        t_s = np.arange(0, 490) / fs
        t_ms = t_s * 1e3
        w = 2.0 * np.pi * 4000.0
        # DC baseline = 1.0, relative AC amplitude grows 0.002 -> 0.20 (100x) at gamma = 10 ms^-1
        rel_amp = np.minimum(0.002 * np.exp(10.0 * (t_ms - 1.15)), 0.20)
        phi_rad = np.where(rel_amp < 0.17, 0.0, np.radians(60.0))
        sig_a = np.tile(1.0 + rel_amp * np.cos(w * t_s), (6, 1))
        sig_b = np.tile(1.0 + rel_amp * np.cos(w * t_s + phi_rad), (6, 1))
        f_band = (2500.0, 5500.0)

        m_grow, m_early, m_late, t_ons, ramp_durs, t_90s = jtb._per_shot_ramp_masks(t_s, sig_a, f_band)
        self.assertTrue(np.all(np.isfinite(t_ons)))
        # Hilbert t_90 must land at/after saturation (~1.60 ms) so early_sat sees pure +60 deg!
        self.assertGreater(float(np.median(t_90s)), 1.55)
        _, phi_g, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_grow)
        _, phi_e, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_early)
        _, phi_s, _ = amplitude_normalized_cross_phase(sig_a, sig_b, fs, f_band, mask=m_late)
        self.assertAlmostEqual(phi_g, 0.0, delta=5.0)
        self.assertAlmostEqual(phi_e, +60.0, delta=5.0)
        self.assertAlmostEqual(phi_s, +60.0, delta=5.0)

    def test_harmonic_steepening_separates_curvature_vs_sawtooth_and_folds_waveform(self):
        fs = 100e3
        t = np.arange(0, 2000) / fs
        f0 = 4000.0
        rng = np.random.default_rng(7)
        # Random per-shot phase and slight frequency jitter to test phase-shift invariance
        phases = rng.uniform(-np.pi, np.pi, size=8)
        f_jit = f0 + rng.uniform(-120.0, 120.0, size=8)

        def _make_stack(a2_cos, a2_sin):
            rows = []
            for ph, fj in zip(phases, f_jit):
                th = 2.0 * np.pi * fj * t + ph
                rows.append(10.0 + 0.2 * t + np.cos(th) + a2_cos * np.cos(2.0 * th) + a2_sin * np.sin(2.0 * th))
            return np.vstack(rows)

        # (a) Pure symmetric peakedness: beta = 0 deg, curv = +0.35, steep = 0, slope_ratio = 1, rise_frac = 0.5
        st_sym = harmonic_steepening(_make_stack(+0.35, 0.0), fs, f0)
        self.assertGreater(st_sym.biphase_lock, 0.99)
        self.assertAlmostEqual(st_sym.a2_a1_med, 0.35, delta=0.02)
        self.assertAlmostEqual(st_sym.biphase_deg, 0.0, delta=2.0)
        self.assertAlmostEqual(st_sym.curvature_comp, +0.35, delta=0.02)
        self.assertAlmostEqual(st_sym.steepening_comp, 0.0, delta=0.02)
        self.assertGreater(st_sym.skewness_med, 0.5)
        self.assertAlmostEqual(st_sym.hilbert_asymmetry_med, 0.0, delta=0.05)
        self.assertAlmostEqual(st_sym.slope_ratio_med, 1.0, delta=0.08)
        self.assertAlmostEqual(st_sym.rise_frac_med, 0.5, delta=0.03)
        expected_sym = np.cos(st_sym.phase_rad) + 0.35 * np.cos(2.0 * st_sym.phase_rad)
        np.testing.assert_allclose(st_sym.folded_wave, expected_sym, atol=0.08)

        # (b) Steep rising sawtooth: -0.35*sin(2*th) = +0.35*cos(2*th + 90 deg) -> beta = +90 deg
        st_rise = harmonic_steepening(_make_stack(0.0, -0.35), fs, f0)
        self.assertGreater(st_rise.biphase_lock, 0.99)
        self.assertAlmostEqual(st_rise.biphase_deg, +90.0, delta=2.0)
        self.assertAlmostEqual(st_rise.curvature_comp, 0.0, delta=0.02)
        self.assertAlmostEqual(st_rise.steepening_comp, +0.35, delta=0.02)
        self.assertAlmostEqual(st_rise.skewness_med, 0.0, delta=0.05)
        self.assertLess(st_rise.hilbert_asymmetry_med, -0.5)
        self.assertGreater(st_rise.slope_ratio_med, 1.7)
        self.assertLess(st_rise.rise_frac_med, 0.40)
        expected_rise = np.cos(st_rise.phase_rad) - 0.35 * np.sin(2.0 * st_rise.phase_rad)
        np.testing.assert_allclose(st_rise.folded_wave, expected_rise, atol=0.08)

        # (c) Steep falling sawtooth: +0.35*sin(2*th) -> beta = -90 deg
        st_fall = harmonic_steepening(_make_stack(0.0, +0.35), fs, f0)
        self.assertAlmostEqual(st_fall.biphase_deg, -90.0, delta=2.0)
        self.assertAlmostEqual(st_fall.steepening_comp, -0.35, delta=0.02)
        self.assertGreater(st_fall.hilbert_asymmetry_med, 0.5)
        self.assertLess(st_fall.slope_ratio_med, 0.6)
        self.assertGreater(st_fall.rise_frac_med, 0.60)

        # (d) 1-D pure sinusoid & invalid/Nyquist f0 edge cases
        st_1d = harmonic_steepening(np.cos(2.0 * np.pi * f0 * t), fs, f0)
        self.assertEqual(st_1d.n_shots, 1)
        self.assertFalse(st_1d.biphase_locked)
        self.assertLess(st_1d.a2_a1_med, 0.02)
        self.assertAlmostEqual(st_1d.slope_ratio_med, 1.0, delta=0.05)
        st_bad = harmonic_steepening(np.cos(2.0 * np.pi * f0 * t), fs, 0.3 * fs)
        self.assertEqual(st_bad.n_shots, 0)
        self.assertTrue(np.isnan(st_bad.biphase_deg))

    def test_harmonic_steepening_gates_unlocked_random_biphase_to_nan(self):
        fs = 100e3
        t = np.arange(0, 2000) / fs
        f0 = 4000.0
        rng = np.random.default_rng(42)
        nsh = 25
        # Independent random phase for 2f0 relative to f0 -> unlocked bispectrum
        rows = []
        for _ in range(nsh):
            p1 = rng.uniform(-np.pi, np.pi)
            p2 = rng.uniform(-np.pi, np.pi)
            th = 2.0 * np.pi * f0 * t
            rows.append(10.0 + np.cos(th + p1) + 0.30 * np.cos(2.0 * th + p2))
        st = harmonic_steepening(np.vstack(rows), fs, f0)
        self.assertFalse(st.biphase_locked)
        self.assertLess(st.biphase_lock, st.biphase_lock_crit)
        self.assertTrue(np.isnan(st.biphase_deg))
        self.assertTrue(np.isnan(st.curvature_comp))
        self.assertTrue(np.isnan(st.steepening_comp))
        # Disabling gating still returns finite raw numbers for diagnostics
        st_raw = harmonic_steepening(np.vstack(rows), fs, f0, gate_unlocked=False)
        self.assertFalse(st_raw.biphase_locked)
        self.assertTrue(np.isfinite(st_raw.biphase_deg))

    def test_mach_linear_tip_bracket_exponential_vs_linear_tip_models(self):
        # Verify BOTH sides of the Mach model-dependence bracket:
        # (a) Under Hutchinson's exponential law I_+- = n * exp(-+ M / (2 K_H)), a pure
        #     sinusoidal M + 20% sinusoidal density wave yields 0.000 measured M
        #     steepening (exact density cancellation), while the linear-tip bracket
        #     reads ~0.10 (proving the bracket must NOT be subtracted).
        # (b) Under linear sinusoidal tips (20% / 20%, 20 deg lag), the bracket
        #     reproduces ~0.10 apparent M steepening while geometric-mean Isat_0 ~ 0.
        from data_analysis.plasma.mach import K_HUTCHINSON, mach_single

        fs = 100e3
        t = np.arange(0, 2000) / fs
        w = 2.0 * np.pi * 4000.0
        n_t = 1.0 + 0.20 * np.cos(w * t)
        m_true = 0.30 * np.sin(w * t)
        # mach_single: M = -K_H * ln(I_+/I_-) -> I_+- = n * exp(-+ M / (2 * K_H))
        xp_h = np.tile(n_t * np.exp(-m_true / (2.0 * K_HUTCHINSON)), (6, 1))
        xm_h = np.tile(n_t * np.exp(+m_true / (2.0 * K_HUTCHINSON)), (6, 1))
        yp_h = np.tile(n_t, (6, 1))
        ym_h = np.tile(n_t, (6, 1))
        m_meas = mach_single(xp_h / xm_h, 1.0)
        np.testing.assert_allclose(m_meas[0], m_true, atol=1e-12)
        st_meas = harmonic_steepening(m_meas, fs, 4000.0, gate_unlocked=False)
        self.assertLess(st_meas.a2_a1_med, 0.005)
        self.assertAlmostEqual(abs(st_meas.steepening_comp), 0.0, delta=0.005)

        recs_h = [{"tips_raw": {"X+": xp_h, "X-": xm_h, "Y+": yp_h, "Y-": ym_h}, "ur": 1.0, "ut": 0.0}]
        br_h = jtb._mach_linear_tip_bracket(recs_h, {"X": 1.0, "Y": 1.0}, fs, 4000.0)
        st_br_h = harmonic_steepening(br_h["mr"], fs, 4000.0, gate_unlocked=False)
        self.assertGreater(abs(st_br_h.steepening_comp), 0.06)

        lag = np.radians(20.0)
        xp_l = np.tile(1.0 + 0.20 * np.cos(w * t), (6, 1))
        xm_l = np.tile(1.0 + 0.20 * np.cos(w * t + lag), (6, 1))
        yp_l = np.tile(1.0 + 0.20 * np.cos(w * t + 0.5), (6, 1))
        ym_l = np.tile(1.0 + 0.20 * np.cos(w * t + 0.5 + lag), (6, 1))
        recs_l = [{"tips_raw": {"X+": xp_l, "X-": xm_l, "Y+": yp_l, "Y-": ym_l}, "ur": 1.0, "ut": 0.0}]
        br_l = jtb._mach_linear_tip_bracket(recs_l, {"X": 1.0, "Y": 1.0}, fs, 4000.0)
        self.assertIsNotNone(br_l)
        st_mr = harmonic_steepening(br_l["mr"], fs, 4000.0)
        st_is = harmonic_steepening(br_l["isat0"], fs, 4000.0, gate_unlocked=False)
        self.assertAlmostEqual(st_mr.a2_a1_med, 0.10, delta=0.015)
        self.assertAlmostEqual(abs(st_mr.steepening_comp), 0.10, delta=0.015)
        self.assertAlmostEqual(abs(st_is.steepening_comp), 0.0, delta=0.01)

    def test_plot_turbulence_steepening_smoke(self):
        import matplotlib
        matplotlib.use("Agg")
        import Jun2026_plot as jp

        ph = np.linspace(-np.pi, np.pi, 65)
        ph = 0.5 * (ph[:-1] + ph[1:])
        synth = {
            "r32__steep_phase_rad": ph,
            "radst_11__r_cm": np.arange(8, dtype=float),
            "radst_11__steepening_comp": np.linspace(-0.2, 0.4, 8),
            "radst_11__curvature_comp": np.linspace(0.5, -0.2, 8),
            "radst_12__r_cm": np.arange(8, dtype=float),
            "radst_12__steepening_comp": np.linspace(-0.1, 0.3, 8),
            "radst_12__curvature_comp": np.linspace(0.2, -0.1, 8),
        }
        for rnum in ("00", "01", "02", "03", "04", "28", "27", "30"):
            p = f"spec_{rnum}__"
            synth[p + "folded_phase_rad"] = ph
            synth[p + "folded_wave"] = np.cos(ph) - 0.3 * np.sin(2.0 * ph)
            synth[p + "folded_std"] = np.full_like(ph, 0.05)
            synth[p + "biphase_deg"] = 75.0 if rnum in ("00", "01", "02", "03") else -90.0
            synth[p + "steepening_comp"] = 0.3 if rnum in ("00", "01", "02", "03") else -0.25
            synth[p + "curvature_comp"] = 0.1
            synth[p + "slope_ratio_med"] = 2.5 if rnum in ("00", "01", "02", "03") else 0.4
            synth[p + "rise_frac_med"] = 0.3 if rnum in ("00", "01", "02", "03") else 0.7
        for tag in ("isat0", "mr", "mt"):
            synth[f"r32__steep_{tag}_folded_waves"] = np.tile(np.cos(ph) - 0.25 * np.sin(2.0 * ph), (5, 1))
            synth[f"r32__steep_{tag}_biphase_deg"] = np.full(5, 80.0)
            synth[f"r32__steep_{tag}_steepening_comp"] = np.full(5, 0.3)
            synth[f"r32__steep_{tag}_slope_ratio"] = np.full(5, 2.0)
            synth[f"r32__steep_linear_tip_{tag}_steepening_comp"] = np.full(5, 0.02)

        with tempfile.TemporaryDirectory() as tmp:
            npz_p = os.path.join(tmp, jtb.OUT_NAME)
            out_png = os.path.join(tmp, "steepening_test.png")
            np.savez(npz_p, **synth)
            jp.plot_turbulence_steepening(npz_path=npz_p, save_fig=out_png, show=False)
            self.assertTrue(os.path.exists(out_png))
            self.assertGreater(os.path.getsize(out_png), 1000)

    def test_batch_all_respects_overwrite_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, jtb.OUT_NAME)
            np.savez(p, sentinel=np.array(99))
            loaded = jtb.batch_all(data_dir=tmp, overwrite=False)
            self.assertEqual(int(loaded["sentinel"]), 99)

    def test_resolve_run32_channels_reads_wiring_not_hardcoded_order(self):
        from unittest import mock

        swapped_wiring = {
            ("scope", "C2"): "Isat, Mach@P33-X+",
            ("scope", "C1"): "Isat, Mach@P33-X-",
            ("scope", "C4"): "Isat, Mach@P33-Y+",
            ("scope", "C3"): "Isat, Mach@P33-Y-",
            ("scope", "C5"): "Isat, Mach@P33-Z+",
            ("scope", "C6"): "Isat, Mach@P33-Z-",
            ("scope", "C7"): "Isat, LP@P29-R",
        }
        with mock.patch.object(jtb.jmf, "channel_wiring", return_value=swapped_wiring):
            ch = jtb._resolve_run32_channels("dummy.hdf5")
        self.assertEqual(ch, {"X+": "C2", "X-": "C1", "Y+": "C4", "Y-": "C3", "REF": "C7"})

        no_ref = {k: v for k, v in swapped_wiring.items() if "P29-R" not in v}
        with mock.patch.object(jtb.jmf, "channel_wiring", return_value=no_ref):
            with self.assertRaises(ValueError):
                jtb._resolve_run32_channels("dummy.hdf5")


if __name__ == "__main__":
    unittest.main()
