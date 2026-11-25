"""Unit tests for bispectral controls and Jun2026_bicoherence campaign adapter."""

import os
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from data_analysis.signal import (
    bicoherence,
    bicoherence_surrogate,
    shot_phase_consistency,
    summed_bicoherence,
)

JUN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if JUN_DIR not in sys.path:
    sys.path.insert(0, JUN_DIR)

import Jun2026_bicoherence as jbi  # noqa: E402


class TestBispectralControls(unittest.TestCase):
    def setUp(self):
        self.dt = 1e-5          # 100 kHz
        self.nseg = 256         # df = 390.625 Hz
        self.df = 1.0 / (self.nseg * self.dt)
        self.rng = np.random.default_rng(42)

    def _tone(self, f_hz, phi, nt=None, amp=1.0):
        nt = self.nseg if nt is None else nt
        t = np.arange(nt, dtype=float) * self.dt
        return amp * np.cos(2.0 * np.pi * f_hz * t[None, :] + np.asarray(phi)[:, None])

    def test_shot_phase_consistency_locked_vs_random(self):
        n = 200
        # Non-integer cycles across segments to verify multi-segment RMS preserves locking
        f_lock = 10.0 * self.df
        nt = self.nseg * 3
        locked = self._tone(f_lock, np.full(n, 0.35), nt=nt)
        res_lock = shot_phase_consistency(locked, self.dt, nperseg=self.nseg, f_band=(f_lock, f_lock))
        self.assertEqual(res_lock.n_shots, n)
        self.assertAlmostEqual(float(res_lock.consistency[0]), 1.0, places=10)

        rand_phi = self.rng.uniform(0, 2 * np.pi, n)
        unlocked = self._tone(f_lock, rand_phi, nt=nt)
        res_rand = shot_phase_consistency(unlocked, self.dt, nperseg=self.nseg, f_band=(f_lock, f_lock))
        self.assertLess(float(res_rand.consistency[0]), res_rand.threshold)
        self.assertAlmostEqual(res_rand.threshold, np.sqrt(-np.log(0.05) / n), places=10)

    def test_shot_phase_consistency_edge_cases(self):
        # All-zero input returns NaN without RuntimeWarning
        zeros = np.zeros((10, self.nseg))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = shot_phase_consistency(zeros, self.dt, nperseg=self.nseg)
        self.assertEqual(len(caught), 0)
        self.assertTrue(np.all(np.isnan(res.consistency)))

        # NaN row is dropped cleanly
        x = self._tone(10.0 * self.df, np.zeros(10))
        x[3, 5] = np.nan
        res2 = shot_phase_consistency(x, self.dt, nperseg=self.nseg)
        self.assertEqual(res2.n_shots, 9)

        with self.assertRaises(ValueError):
            shot_phase_consistency(np.zeros(self.nseg), self.dt, nperseg=self.nseg)
        with self.assertRaises(ValueError):
            shot_phase_consistency(np.ones((5, self.nseg), dtype=complex), self.dt, nperseg=self.nseg)

    def test_surrogate_distinguishes_true_coupling_from_trigger_locking(self):
        n = 200
        f0 = 10.0 * self.df
        f2 = 20.0 * self.df
        band = dict(f1_band=(f0, f0), f2_band=(f0, f0))

        # Case 1: genuine quadratic coupling with random fundamental phase per shot
        phi = self.rng.uniform(0, 2 * np.pi, n)
        theta = 0.6
        coupled = (
            self._tone(f0, phi, amp=1.0)
            + self._tone(f2, 2.0 * phi + theta, amp=0.4)
            + 0.05 * self.rng.standard_normal((n, self.nseg))
        )
        r_true = bicoherence(coupled, self.dt, self.nseg, **band)
        r_shuf = bicoherence_surrogate(coupled, coupled, coupled, self.dt, self.nseg, **band)
        r_shuf_rng = bicoherence_surrogate(
            coupled, coupled, coupled, self.dt, self.nseg, rng=7, **band
        )
        self.assertGreater(float(r_true.b2[0, 0]), 0.95)
        self.assertLess(float(r_shuf.b2[0, 0]), r_shuf.threshold)
        self.assertLess(float(r_shuf_rng.b2[0, 0]), r_shuf_rng.threshold)

        # Case 2: trigger-locked independent tones (constant phase across shots)
        locked = (
            self._tone(f0, np.full(n, 0.2), amp=1.0)
            + self._tone(f2, np.full(n, -1.1), amp=0.4)
            + 0.05 * self.rng.standard_normal((n, self.nseg))
        )
        r_lock = bicoherence(locked, self.dt, self.nseg, **band)
        r_lock_shuf = bicoherence_surrogate(locked, locked, locked, self.dt, self.nseg, **band)
        # Surrogate stays high -> exposes trigger-locking false positive!
        self.assertGreater(float(r_lock.b2[0, 0]), 0.95)
        self.assertGreater(float(r_lock_shuf.b2[0, 0]), 0.95)

        with self.assertRaises(ValueError):
            bicoherence_surrogate(coupled, coupled, coupled, self.dt, self.nseg, shift=n, **band)

        # Case 3: slowly drifting phase across sequential shots (adjacent shots correlated,
        # global phase uniform): shift=1 leaks coupling, whereas default shift=n//2 and
        # random derangement break it cleanly.
        phi_drift = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        drifted = self._tone(f0, phi_drift, amp=1.0) + self._tone(f2, 2.0 * phi_drift, amp=0.4)
        r_s1 = bicoherence_surrogate(drifted, drifted, drifted, self.dt, self.nseg, shift=1, **band)
        r_rng = bicoherence_surrogate(drifted, drifted, drifted, self.dt, self.nseg, rng=0, **band)
        self.assertGreater(float(r_s1.b2[0, 0]), 0.9)
        self.assertLess(float(r_rng.b2[0, 0]), 0.1)

    def test_summed_bicoherence_peaks_at_harmonic_and_respects_symmetry(self):
        n = 150
        f0 = 10.0 * self.df
        phi = self.rng.uniform(0, 2 * np.pi, n)
        x = (
            self._tone(f0, phi, amp=1.0)
            + self._tone(2 * f0, 2 * phi, amp=0.5)
            + 0.05 * self.rng.standard_normal((n, self.nseg))
        )
        res = bicoherence(x, self.dt, self.nseg, f1_band=(4 * self.df, 24 * self.df),
                          f2_band=(-24 * self.df, 24 * self.df))
        f3_sym, _, bmax_sym, cnt_sym = summed_bicoherence(res, symmetric=True)
        _, _, _, cnt_full = summed_bicoherence(res, symmetric=False)
        f3_pos, _, bmax_pos, cnt_pos = summed_bicoherence(res, symmetric=True, positive_only=True)
        # Peak of summed b2_max in principal triangle is at 2*f0
        self.assertAlmostEqual(float(f3_pos[int(np.argmax(bmax_pos))]), 2 * f0, places=4)
        # Symmetry avoids double-counting off-diagonal positive pairs
        self.assertTrue(np.all(cnt_sym <= cnt_full))
        self.assertTrue(np.any(cnt_sym < cnt_full))
        # positive_only drops negative-f2 conjugate reflections (min positive f2 unmasked is 3*df)
        self.assertLess(int(cnt_pos.sum()), int(cnt_sym.sum()))
        self.assertTrue(np.all(f3_pos >= 7 * self.df - 1e-6))


class TestJun2026Bicoherence(unittest.TestCase):
    def setUp(self):
        self.dt = 1e-5
        self.nseg = 256
        self.df = 1.0 / (self.nseg * self.dt)
        self.rng = np.random.default_rng(123)

    def test_harmonic_triad_bins_integer_and_half_integer(self):
        # Integer bin: f0/df = 10.07 -> 2f0 at bin 20 = 10 + 10
        k1, h2, h3 = jbi._harmonic_triad_bins(10.07 * self.df, self.df)
        self.assertEqual(k1, 10)
        self.assertEqual(h2, (10, 10, 20))
        self.assertEqual(h3, (20, 10, 30))

        # Half-integer bin: f0/df = 10.52 -> 2f0 at bin 21 = 11 + 10 (hits exact 2f0 bin!)
        k1, h2, h3 = jbi._harmonic_triad_bins(10.52 * self.df, self.df)
        self.assertEqual(k1, 11)
        self.assertEqual(h2, (11, 10, 21))
        self.assertEqual(h3, (21, 11, 32))

    def test_analyze_bicoherence_stack_recovers_harmonics_and_controls(self):
        n = 160
        t = np.arange(self.nseg + 2, dtype=float) * self.dt
        f0 = 10.0 * self.df
        phi = self.rng.uniform(0, 2 * np.pi, n)
        theta2 = 0.5   # rad -> biphase = -theta2
        theta3 = -0.8
        stack = (
            1.0 * np.cos(2 * np.pi * f0 * t[None, :] + phi[:, None])
            + 0.35 * np.cos(2 * np.pi * (2 * f0) * t[None, :] + 2 * phi[:, None] + theta2)
            + 0.15 * np.cos(2 * np.pi * (3 * f0) * t[None, :] + 3 * phi[:, None] + theta3)
            + 0.02 * self.rng.standard_normal((n, t.size))
        )
        out = jbi.analyze_bicoherence_stack(t, stack, f0_hz=None, nperseg=self.nseg)
        self.assertEqual(out["n_avg"], n)
        self.assertAlmostEqual(out["f0_hz"], f0, places=3)
        self.assertGreater(out["b2_h2"], 0.95)
        self.assertLess(out["b2_h2_shuf"], out["threshold"])
        self.assertAlmostEqual(out["biphase_h2_deg"], -np.degrees(theta2), delta=5.0)
        self.assertGreater(out["b2_h3"], 0.90)
        self.assertLess(out["b2_h3_shuf"], out["threshold"])
        self.assertLess(out["pc_R_h1"], out["pc_threshold"])
        self.assertAlmostEqual(out["amp_h2_over_h1"], 0.35, delta=0.02)
        self.assertAlmostEqual(out["amp_h3_over_h1"], 0.15, delta=0.02)

    def test_analyze_cross_bicoherence_stacks(self):
        n = 160
        t = np.arange(self.nseg, dtype=float) * self.dt
        f0 = 10.0 * self.df
        phi = self.rng.uniform(0, 2 * np.pi, n)
        x = np.cos(2 * np.pi * f0 * t[None, :] + phi[:, None]) + 0.03 * self.rng.standard_normal((n, self.nseg))
        y = 0.4 * np.cos(2 * np.pi * (2 * f0) * t[None, :] + 2 * phi[:, None] + 0.3) + 0.03 * self.rng.standard_normal((n, self.nseg))
        out = jbi.analyze_cross_bicoherence_stacks(t, x, x, y, f0_hz=f0, nperseg=self.nseg)
        self.assertGreater(out["b2_h2"], 0.95)
        self.assertLess(out["b2_h2_shuf"], out["threshold"])

    def test_batch_all_incremental_settings_and_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            onset_path = os.path.join(tmp, jbi.jon.OUT_NAME)
            np.savez(
                onset_path,
                **{
                    "00__f_peak_hz": 3209.0, "00__t_onset_med_ms": 2.88,
                    "02__f_peak_hz": 3934.0, "02__t_onset_med_ms": 1.65,
                    "26__f_peak_hz": 4110.0, "26__t_onset_med_ms": 1.40,
                },
            )

            auto_calls = []
            bias_calls = []

            def fake_auto(spec, **kw):
                auto_calls.append(spec.run_num)
                t_on = float(kw.get("t_onset_med_ms", np.nan))
                res = {
                    "b2_h2": 0.9,
                    "t_onset_med_ms": t_on,
                    "window_post_onset": bool(np.isfinite(t_on) and t_on <= jbi.TMIN_MS),
                }
                if kw.get("return_stack"):
                    return res, (np.zeros(4), np.zeros((2, 4)), np.array([1, 2]))
                return res

            def fake_bias(*, runs=jbi.BIASSCOPE_RUNS, **kw):
                bias_calls.append(tuple(runs))
                return {r: {"i2_b2_h2": 0.8} for r in runs}

            with (
                mock.patch.object(jbi, "analyze_run_auto", side_effect=fake_auto),
                mock.patch.object(jbi, "analyze_biasscope_bicoherence", side_effect=fake_bias),
            ):
                # 1. Initial subset ["00", "02"]: scopes biasscope and sets window_post_onset
                out = jbi.batch_all(data_dir=tmp, run_nums=["00", "02"])
                self.assertEqual(auto_calls, ["00", "02"])
                self.assertEqual(bias_calls, [("00", "02")])
                with np.load(out) as d:
                    self.assertFalse(bool(d["auto_00__window_post_onset"]))
                    self.assertTrue(bool(d["auto_02__window_post_onset"]))
                    self.assertIn("bias_02__i2_b2_h2", d)
                    self.assertNotIn("bias_06__i2_b2_h2", d)

                # 2. Re-running with overwrite=False adds missing "26", skips "02", skips biasscope
                auto_calls.clear()
                bias_calls.clear()
                jbi.batch_all(data_dir=tmp, run_nums=["02", "26"], overwrite=False)
                self.assertEqual(auto_calls, ["26"])
                self.assertEqual(bias_calls, [])
                with np.load(out) as d:
                    self.assertIn("auto_00__b2_h2", d)
                    self.assertIn("auto_02__b2_h2", d)
                    self.assertIn("auto_26__b2_h2", d)

                # 3. Settings mismatch under overwrite=False raises ValueError
                with self.assertRaises(ValueError):
                    jbi.batch_all(data_dir=tmp, run_nums=["26"], max_shots=99, overwrite=False)

                # 4. Partial overwrite=True refreshes ONLY "02" without deleting "00" or "26"
                auto_calls.clear()
                bias_calls.clear()
                jbi.batch_all(data_dir=tmp, run_nums=["02"], overwrite=True)
                self.assertEqual(auto_calls, ["02"])
                self.assertEqual(bias_calls, [("02",)])
                with np.load(out) as d:
                    self.assertIn("auto_00__b2_h2", d)
                    self.assertIn("auto_02__b2_h2", d)
                    self.assertIn("auto_26__b2_h2", d)
                    self.assertIn("bias_00__i2_b2_h2", d)
                    self.assertIn("bias_02__i2_b2_h2", d)


if __name__ == "__main__":
    unittest.main()
