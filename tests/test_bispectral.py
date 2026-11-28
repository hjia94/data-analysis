"""Analytic unit tests for data_analysis.signal.bispectral (bicoherence & controls)."""

import unittest
from unittest import mock
import warnings

import numpy as np

from data_analysis.signal import (
    bicoherence,
    bicoherence_surrogate,
    bicoherence_threshold,
    cross_bicoherence,
    shot_phase_consistency,
    summed_bicoherence,
)

DT = 1e-6
NSEG = 256
DF = 1 / (NSEG * DT)
K1, K2 = 20, 33
F1, F2, F3 = K1 * DF, K2 * DF, (K1 + K2) * DF


def tone(f, phi, nt=NSEG, a=1.0, dt=DT):
    """(nshot, nt) cosines, one phase per shot."""
    t = np.arange(nt) * dt
    return a * np.cos(2 * np.pi * f * t + np.asarray(phi)[:, None])


def at(res, f1, f2):
    return (int(np.argmin(abs(res.f1 - f1))), int(np.argmin(abs(res.f2 - f2))))


class TestBispectral(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_coupled_triad_auto(self):
        n = 200
        p1 = self.rng.uniform(0, 2 * np.pi, n)
        p2 = self.rng.uniform(0, 2 * np.pi, n)
        theta = 0.7
        a1, a2, a3 = 1.0, 0.8, 0.5
        x = tone(F1, p1, a=a1) + tone(F2, p2, a=a2) + tone(F3, p1 + p2 + theta, a=a3)
        r = bicoherence(x, DT, NSEG)
        i = at(r, F1, F2)
        self.assertEqual(r.n_avg, n)
        self.assertAlmostEqual(float(r.b2[i]), 1.0, delta=1e-10)
        np.testing.assert_allclose(abs(r.bispectrum[i]), a1 * a2 * a3, rtol=1e-10)
        self.assertAlmostEqual(float(np.angle(r.bispectrum[i])), -theta, delta=1e-10)
        self.assertAlmostEqual(float(r.b2[at(r, F2, F1)]), 1.0, delta=1e-10)

    def test_random_phase_triad_is_at_noise_level(self):
        n = 400
        p1, p2, p3 = (self.rng.uniform(0, 2 * np.pi, n) for _ in range(3))
        x = tone(F1, p1) + tone(F2, p2) + tone(F3, p3)
        r = bicoherence(x, DT, NSEG, f1_band=(F1, F1), f2_band=(F2, F2), p=1e-3)
        self.assertEqual(r.b2.shape, (1, 1))
        self.assertLess(float(r.b2[0, 0]), r.threshold)
        r2 = bicoherence(
            tone(F1, p1) + tone(F2, p2) + tone(F3, p1 + p2),
            DT, NSEG, f1_band=(F1, F1), f2_band=(F2, F2),
        )
        self.assertAlmostEqual(float(r2.b2[0, 0]), 1.0, delta=1e-10)

    def test_noise_statistics_and_threshold(self):
        x = self.rng.standard_normal((50, NSEG * 4))
        r = bicoherence(x, DT, NSEG, f1_band=(10 * DF, 60 * DF), f2_band=(5 * DF, 50 * DF))
        self.assertEqual(r.n_avg, 200)
        b2 = r.b2[np.isfinite(r.b2)]
        np.testing.assert_allclose(b2.mean(), 1 / r.n_avg, rtol=0.1)
        self.assertAlmostEqual(float(np.mean(b2 > r.threshold)), 0.05, delta=0.015)
        self.assertAlmostEqual(r.threshold, 1 - 0.05 ** (1 / 199), places=10)

    def test_long_record_segments_and_coupling_in_noise(self):
        nt = NSEG * 300
        t = np.arange(nt) * DT
        walk = np.repeat(self.rng.uniform(0, 2 * np.pi, (2, 300)), NSEG, axis=1)
        x = (
            np.cos(2 * np.pi * F1 * t + walk[0])
            + np.cos(2 * np.pi * F2 * t + walk[1])
            + 0.5 * np.cos(2 * np.pi * F3 * t + walk[0] + walk[1])
            + 0.5 * self.rng.standard_normal(nt)
        )
        r = bicoherence(x, DT, NSEG)
        self.assertEqual(r.n_avg, 300)
        self.assertGreater(float(r.b2[at(r, F1, F2)]), 0.5)
        k1, k2 = np.meshgrid(np.rint(r.f1 / DF), np.rint(r.f2 / DF), indexing="ij")
        near = lambda a, b: (abs(k1 - a) <= 2) & (abs(k2 - b) <= 2)
        off = r.b2[~(near(K1, K2) | near(K2, K1)) & np.isfinite(r.b2)]
        np.testing.assert_allclose(off.mean(), 1 / r.n_avg, rtol=0.15)
        self.assertLess(float(off.max()), 0.1)

    def test_cross_bicoherence_three_channels(self):
        n = 200
        p1, p2 = self.rng.uniform(0, 2 * np.pi, n), self.rng.uniform(0, 2 * np.pi, n)
        noise = lambda: 0.1 * self.rng.standard_normal((n, NSEG))
        x, y, z = tone(F1, p1) + noise(), tone(F2, p2) + noise(), tone(F3, p1 + p2) + noise()
        band = dict(f1_band=(F1 - 3 * DF, F1 + 3 * DF), f2_band=(F2 - 3 * DF, F2 + 3 * DF))
        r = cross_bicoherence(x, y, z, DT, NSEG, **band)
        self.assertGreater(float(r.b2[at(r, F1, F2)]), 0.95)
        self.assertLess(float(np.nanmax(bicoherence(x, DT, NSEG, **band).b2)), 0.1)

    def test_difference_triad_with_negative_f2(self):
        n = 200
        p1, p2 = self.rng.uniform(0, 2 * np.pi, n), self.rng.uniform(0, 2 * np.pi, n)
        x, y, z = tone(F1, p1), tone(F2, p2), tone(F2 - F1, p2 - p1)
        r = cross_bicoherence(y, x, z, DT, NSEG, f1_band=(F2, F2), f2_band=(-F1, -F1))
        self.assertAlmostEqual(float(r.f2[0]), -F1)
        self.assertAlmostEqual(float(r.b2[0, 0]), 1.0, delta=1e-10)

    def test_axes_nyquist_and_inputs(self):
        x = self.rng.standard_normal((10, NSEG))
        r = bicoherence(x, DT, NSEG)
        np.testing.assert_allclose(r.f1, np.arange(1, NSEG // 2 + 1) * DF)
        self.assertEqual(r.b2.shape, (NSEG // 2, NSEG // 2))
        self.assertEqual(r.bispectrum.shape, (NSEG // 2, NSEG // 2))
        k1, k2 = np.meshgrid(np.rint(r.f1 / DF), np.rint(r.f2 / DF), indexing="ij")
        over = k1 + k2 > NSEG // 2
        unresolved = (k1 <= 2) | (k2 <= 2)
        self.assertTrue(np.isnan(r.b2[over | unresolved]).all())
        self.assertTrue(np.isfinite(r.b2[~(over | unresolved)]).all())

        x[3, 5] = np.nan
        self.assertEqual(bicoherence(x, DT, NSEG).n_avg, 9)
        with self.assertRaisesRegex(ValueError, "selects no frequency bin"):
            bicoherence(x, DT, NSEG, f1_band=(1.2 * DF, 1.8 * DF))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            bicoherence(x, DT, NSEG, f1_band=(-DF, DF))
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            cross_bicoherence(x, x, x[:, :-1], DT, 128)
        with self.assertRaisesRegex(ValueError, "longer than the record"):
            bicoherence(x, DT, 2 * NSEG)
        with self.assertRaisesRegex(ValueError, "at least 2"):
            bicoherence_threshold(1)

    def test_block_accumulation_matches_single_block(self):
        from data_analysis.signal import bispectral
        x = self.rng.standard_normal((30, NSEG * 2))
        kw = dict(f1_band=(DF, 40 * DF), f2_band=(DF, 40 * DF))
        ref = bicoherence(x, DT, NSEG, **kw)
        with mock.patch.object(bispectral, "_BLOCK_ELEMENTS", 1):
            r = bicoherence(x, DT, NSEG, **kw)
        np.testing.assert_allclose(r.b2, ref.b2, rtol=1e-10)
        np.testing.assert_allclose(r.bispectrum, ref.bispectrum, rtol=1e-10, atol=1e-14)

    def test_block_memory_is_bounded_by_nperseg(self):
        from data_analysis.signal import bispectral
        seen = []
        orig_rfft = np.fft.rfft
        with mock.patch.object(bispectral, "_BLOCK_ELEMENTS", 4096), \
             mock.patch.object(np.fft, "rfft", lambda a, *k, **kw: (seen.append(a.size), orig_rfft(a, *k, **kw))[1]):
            x = self.rng.standard_normal((20, 1024 * 4)).astype(np.float32)
            r = bicoherence(x, DT, 1024, f1_band=(20 * DF / 4, 20 * DF / 4), f2_band=(33 * DF / 4, 33 * DF / 4))
        self.assertEqual(r.b2.shape, (1, 1))
        self.assertEqual(r.n_avg, 80)
        self.assertLessEqual(max(seen), 4096)

    def test_unresolved_triads_are_masked(self):
        for window, lobe in (("hann", 2), ("boxcar", 0), ("blackman", 2)):
            with self.subTest(window=window):
                x = self.rng.standard_normal((200, NSEG)) + 3.0
                r = bicoherence(
                    x, DT, NSEG, f1_band=(0, 20 * DF), f2_band=(-30 * DF, 10 * DF),
                    detrend=None, window=window,
                )
                k1, k2 = np.meshgrid(np.rint(r.f1 / DF), np.rint(r.f2 / DF), indexing="ij")
                a1, a2, a3 = abs(k1), abs(k2), abs(k1 + k2)
                bad = (a1 <= lobe) | (a2 <= lobe) | (a3 <= lobe) | (abs(a3 - a1) <= lobe) | (abs(a3 - a2) <= lobe)
                self.assertTrue(np.isnan(r.b2[bad]).all())
                self.assertTrue(np.isfinite(r.b2[~bad]).all())
                self.assertTrue((k2 == 0).any() and (k1 + k2 == 0).any())
                if window != "boxcar":
                    self.assertLess(float(np.nanmax(r.b2)), 3 * r.threshold)

    def test_nyquist_bin_amplitude(self):
        n = 50
        p1 = self.rng.uniform(0, 2 * np.pi, n)
        fny = NSEG // 2 * DF
        x = (
            tone(F1, p1, a=1.0)
            + tone(fny - F1, -p1, a=0.8)
            + 0.5 * np.cos(np.pi * np.arange(NSEG))[None, :]
        )
        r = bicoherence(x, DT, NSEG, f1_band=(F1, F1), f2_band=(fny - F1, fny - F1))
        np.testing.assert_allclose(abs(r.bispectrum[0, 0]), 1.0 * 0.8 * 0.5, rtol=1e-10)
        self.assertAlmostEqual(float(r.b2[0, 0]), 1.0, delta=1e-10)

    def test_complex_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "real"):
            bicoherence(np.ones((4, NSEG), complex), DT, NSEG)

    def test_shot_phase_consistency_locked_vs_random_and_edges(self):
        dt = 1e-5
        df = 1.0 / (NSEG * dt)
        n = 200
        f_lock = 10.0 * df
        nt = NSEG * 3
        locked = tone(f_lock, np.full(n, 0.35), nt=nt, dt=dt)
        res_lock = shot_phase_consistency(locked, dt, nperseg=NSEG, f_band=(f_lock, f_lock))
        self.assertEqual(res_lock.n_shots, n)
        self.assertAlmostEqual(float(res_lock.consistency[0]), 1.0, places=10)

        unlocked = tone(f_lock, self.rng.uniform(0, 2 * np.pi, n), nt=nt, dt=dt)
        res_rand = shot_phase_consistency(unlocked, dt, nperseg=NSEG, f_band=(f_lock, f_lock))
        self.assertLess(float(res_rand.consistency[0]), res_rand.threshold)
        self.assertAlmostEqual(res_rand.threshold, np.sqrt(-np.log(0.05) / n), places=10)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res_z = shot_phase_consistency(np.zeros((10, NSEG)), dt, nperseg=NSEG)
        self.assertEqual(len(caught), 0)
        self.assertTrue(np.all(np.isnan(res_z.consistency)))

        x = tone(f_lock, np.zeros(10), dt=dt)
        x[3, 5] = np.nan
        self.assertEqual(shot_phase_consistency(x, dt, nperseg=NSEG).n_shots, 9)
        with self.assertRaises(ValueError):
            shot_phase_consistency(np.zeros(NSEG), dt, nperseg=NSEG)
        with self.assertRaises(ValueError):
            shot_phase_consistency(np.ones((5, NSEG), dtype=complex), dt, nperseg=NSEG)

    def test_surrogate_distinguishes_true_coupling_from_trigger_locking(self):
        dt = 1e-5
        df = 1.0 / (NSEG * dt)
        n = 200
        f0, f2 = 10.0 * df, 20.0 * df
        band = dict(f1_band=(f0, f0), f2_band=(f0, f0))

        phi = self.rng.uniform(0, 2 * np.pi, n)
        coupled = tone(f0, phi, dt=dt) + tone(f2, 2.0 * phi + 0.6, a=0.4, dt=dt) + 0.05 * self.rng.standard_normal((n, NSEG))
        r_true = bicoherence(coupled, dt, NSEG, **band)
        r_shuf = bicoherence_surrogate(coupled, coupled, coupled, dt, NSEG, **band)
        r_shuf_rng = bicoherence_surrogate(coupled, coupled, coupled, dt, NSEG, rng=7, **band)
        self.assertGreater(float(r_true.b2[0, 0]), 0.95)
        self.assertLess(float(r_shuf.b2[0, 0]), r_shuf.threshold)
        self.assertLess(float(r_shuf_rng.b2[0, 0]), r_shuf_rng.threshold)

        locked = tone(f0, np.full(n, 0.2), dt=dt) + tone(f2, np.full(n, -1.1), a=0.4, dt=dt) + 0.05 * self.rng.standard_normal((n, NSEG))
        r_lock = bicoherence(locked, dt, NSEG, **band)
        r_lock_shuf = bicoherence_surrogate(locked, locked, locked, dt, NSEG, **band)
        self.assertGreater(float(r_lock.b2[0, 0]), 0.95)
        self.assertGreater(float(r_lock_shuf.b2[0, 0]), 0.95)

        with self.assertRaises(ValueError):
            bicoherence_surrogate(coupled, coupled, coupled, dt, NSEG, shift=n, **band)

        phi_drift = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        drifted = tone(f0, phi_drift, dt=dt) + tone(f2, 2.0 * phi_drift, a=0.4, dt=dt)
        self.assertGreater(float(bicoherence_surrogate(drifted, drifted, drifted, dt, NSEG, shift=1, **band).b2[0, 0]), 0.9)
        self.assertLess(float(bicoherence_surrogate(drifted, drifted, drifted, dt, NSEG, rng=0, **band).b2[0, 0]), 0.1)

    def test_summed_bicoherence_peaks_at_harmonic_and_respects_symmetry(self):
        dt = 1e-5
        df = 1.0 / (NSEG * dt)
        n = 150
        f0 = 10.0 * df
        phi = self.rng.uniform(0, 2 * np.pi, n)
        x = tone(f0, phi, dt=dt) + tone(2 * f0, 2 * phi, a=0.5, dt=dt) + 0.05 * self.rng.standard_normal((n, NSEG))
        res = bicoherence(x, dt, NSEG, f1_band=(4 * df, 24 * df), f2_band=(-24 * df, 24 * df))
        _, _, _, cnt_sym = summed_bicoherence(res, symmetric=True)
        _, _, _, cnt_full = summed_bicoherence(res, symmetric=False)
        f3_pos, _, bmax_pos, cnt_pos = summed_bicoherence(res, symmetric=True, positive_only=True)
        self.assertAlmostEqual(float(f3_pos[int(np.argmax(bmax_pos))]), 2 * f0, places=4)
        self.assertTrue(np.all(cnt_sym <= cnt_full) and np.any(cnt_sym < cnt_full))
        self.assertLess(int(cnt_pos.sum()), int(cnt_sym.sum()))
        self.assertTrue(np.all(f3_pos >= 7 * df - 1e-6))


if __name__ == "__main__":
    unittest.main()
