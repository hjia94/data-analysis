"""Unit tests for Mach probe face-ratio binning and sweep-edge alignment."""

import os
import sys
import unittest

import numpy as np

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

from data_analysis.plasma.mach import binned_face_ratio
import Jun2026_mach_line as jml


class TestBinnedFaceRatio(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.plus = rng.uniform(1.0, 5.0, size=(5, 100))
        self.minus = rng.uniform(1.0, 5.0, size=(5, 100))

    def test_contiguous_bins_match_direct_sums(self):
        edges = np.array([0, 20, 50, 80, 100])
        ratio, counts, amp = binned_face_ratio(self.plus, self.minus, edges)
        for k in range(len(edges) - 1):
            s0, s1 = edges[k], edges[k + 1]
            self.assertEqual(counts[k], 5 * (s1 - s0))
            expected_r = self.plus[:, s0:s1].sum() / self.minus[:, s0:s1].sum()
            expected_a = (self.plus[:, s0:s1].sum() + self.minus[:, s0:s1].sum()) / (5 * (s1 - s0))
            self.assertAlmostEqual(ratio[k], expected_r, places=12)
            self.assertAlmostEqual(amp[k], expected_a, places=12)

    def test_leading_and_trailing_zero_width_bins_are_nan(self):
        # Reproduces Run 08-14 sweep_bin_edges when IV timestamps bracket the Mach record.
        edges = np.array([0, 0, 0, 10, 20, 100, 100, 100])
        ratio, counts, amp = binned_face_ratio(self.plus, self.minus, edges)
        for k in (0, 1, 5, 6):
            self.assertEqual(counts[k], 0)
            self.assertTrue(np.isnan(ratio[k]))
            self.assertTrue(np.isnan(amp[k]))
        self.assertEqual(counts[2], 50)
        self.assertAlmostEqual(
            ratio[2], self.plus[:, 0:10].sum() / self.minus[:, 0:10].sum(), places=12
        )
        self.assertEqual(counts[3], 50)
        self.assertAlmostEqual(
            ratio[3], self.plus[:, 10:20].sum() / self.minus[:, 10:20].sum(), places=12
        )

    def test_final_edge_before_record_end_does_not_bleed_tail(self):
        plus = np.ones((3, 20))
        minus = np.ones((3, 20))
        plus[:, 10:] = 999.0  # spike strictly past edges[-1] = 10
        ratio, counts, amp = binned_face_ratio(plus, minus, np.array([0, 5, 10]))
        np.testing.assert_array_equal(counts, [15, 15])
        np.testing.assert_allclose(ratio, [1.0, 1.0])
        np.testing.assert_allclose(amp, [2.0, 2.0])

    def test_non_positive_samples_are_excluded(self):
        plus_bad = self.plus.copy()
        plus_bad[:, 10:20] = -1.0
        ratio, counts, _ = binned_face_ratio(plus_bad, self.minus, np.array([0, 10, 20, 30]))
        self.assertEqual(counts[0], 50)
        self.assertEqual(counts[1], 0)
        self.assertTrue(np.isnan(ratio[1]))
        self.assertEqual(counts[2], 50)

    def test_sweep_bin_edges_pre_record_bins_are_empty(self):
        tarr = np.linspace(-2e-3, 8e-3, 100001)
        timestamps = np.arange(-11.9, 12.2, 1.0) * 1e-3
        edges, _ = jml.sweep_bin_edges(tarr, timestamps)
        dummy = np.ones((2, tarr.size))
        ratio, counts, _ = binned_face_ratio(dummy, dummy, edges)
        n_pre = int(np.sum(timestamps[1:] <= -2e-3))
        self.assertGreater(n_pre, 0)
        np.testing.assert_array_equal(counts[:n_pre], 0)
        self.assertTrue(np.all(np.isnan(ratio[:n_pre])))
        self.assertTrue(np.all(counts[n_pre:n_pre + 10] > 0))
        np.testing.assert_allclose(ratio[n_pre:n_pre + 10], 1.0)


if __name__ == "__main__":
    unittest.main()
