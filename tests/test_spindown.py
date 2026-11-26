"""Unit tests for exponential relaxation fitting, Poulos Pedersen closure, and Jun2026_spindown."""

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

from data_analysis.plasma.flow import fit_exponential_relaxation
from data_analysis.plasma.stability import pedersen_closure_diagnostics
import Jun2026_spindown as jsd


class TestSpindownAndPedersen(unittest.TestCase):
    def test_fit_exponential_relaxation_recovers_decay_and_offset(self):
        t = np.linspace(5.0, 8.0, 61)
        tau_true = 0.55
        amp_true = -0.22
        off_true = 0.015
        y = amp_true * np.exp(-(t - 5.1) / tau_true) + off_true
        fit = fit_exponential_relaxation(t, y, t_start=5.1, t_end=7.9)
        self.assertAlmostEqual(fit["tau"], tau_true, places=5)
        self.assertAlmostEqual(fit["rate"], 1.0 / tau_true, places=5)
        self.assertAlmostEqual(fit["amp"], amp_true, places=5)
        self.assertAlmostEqual(fit["offset"], off_true, places=5)
        self.assertGreater(fit["r2"], 0.999999)

        # Too few finite points returns NaN gracefully
        fit_empty = fit_exponential_relaxation(t[:2], y[:2], min_points=4)
        self.assertTrue(np.isnan(fit_empty["tau"]))

    def test_pedersen_closure_diagnostics_alfven_and_poulos_scaling(self):
        # 800 G He plasma: L_pen scales as sigma_par^(1/2) / nu_in^(1/2) ~ T_e^(3/4) / nu_in^(1/2)
        hot = pedersen_closure_diagnostics(
            i_tot_a=55.0, er_v_m=145.0, ne_m3=5e18, te_ev=2.8, b0_t=0.08,
            r_wall_m=0.035, l_par_m=14.0, nu_in_s=500.0,
        )
        cold = pedersen_closure_diagnostics(
            i_tot_a=40.0, er_v_m=85.0, ne_m3=1.4e19, te_ev=0.6, b0_t=0.08,
            r_wall_m=0.035, l_par_m=14.0, nu_in_s=2000.0,
        )
        # Alfven transit time scales as sqrt(n_e)
        self.assertAlmostEqual(
            cold["tau_alfven_s"] / hot["tau_alfven_s"],
            np.sqrt(1.4e19 / 5e18),
            places=6,
        )
        # Ion-neutral Pedersen conductivity falls well short of sigma_p_req in both regimes
        self.assertLess(hot["sigma_p_in_over_req"], 0.05)
        self.assertLess(cold["sigma_p_in_over_req"], 0.25)
        # Parallel shorting beats ion-neutral Pedersen in both regimes
        self.assertLess(hot["pedersen_to_parallel"], 0.02)
        self.assertLess(cold["pedersen_to_parallel"], 0.5)
        # Missing i_tot_a stays NaN (no silent default)
        nan_i = pedersen_closure_diagnostics(
            i_tot_a=np.nan, er_v_m=85.0, ne_m3=1.4e19, te_ev=0.6, b0_t=0.08, nu_in_s=2000.0,
        )
        self.assertTrue(np.isnan(nan_i["sigma_p_req_s_m"]))
        self.assertTrue(np.isnan(nan_i["sigma_p_in_over_req"]))

    def test_excursion_peak_finder_handles_both_signs(self):
        t = np.linspace(5.1, 6.0, 19)
        base = 0.02
        y_pos = base + 0.05 * np.exp(-((t - 5.25) / 0.1) ** 2)
        y_neg = base - 0.05 * np.exp(-((t - 5.30) / 0.1) ** 2)
        kp = jsd._find_excursion_peak(t, y_pos, base, 5.15, 5.70)
        kn = jsd._find_excursion_peak(t, y_neg, base, 5.15, 5.70)
        self.assertAlmostEqual(t[kp], 5.25, places=6)
        self.assertAlmostEqual(t[kn], 5.30, places=6)

    def test_jun2026_spindown_batch_all_respects_overwrite_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_npz = os.path.join(tmp, "spindown_pedersen_jun2026.npz")
            np.savez(out_npz, sentinel=np.int64(42))
            res = jsd.batch_all(data_dir=tmp, out_path=out_npz, overwrite=False)
            self.assertEqual(res, out_npz)
            with np.load(out_npz) as d:
                self.assertEqual(int(d["sentinel"]), 42)


if __name__ == "__main__":
    unittest.main()
