"""Unit tests for data_analysis.utils.merge_save_npz."""

import os
import tempfile
import unittest

import numpy as np

from data_analysis.utils import merge_save_npz


class TestMergeSaveNpz(unittest.TestCase):
    def test_create_merge_overwrite_and_drop_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "archive.npz")

            # Initial write
            merge_save_npz(path, {"a": np.array([1, 2]), "b": np.array([3, 4])})
            with np.load(path) as d:
                self.assertEqual(set(d.files), {"a", "b"})
                np.testing.assert_array_equal(d["a"], [1, 2])

            # Merge preserves untouched keys and overwrites updated ones
            merge_save_npz(path, {"b": np.array([30, 40]), "c": np.array([5])})
            with np.load(path) as d:
                self.assertEqual(set(d.files), {"a", "b", "c"})
                np.testing.assert_array_equal(d["a"], [1, 2])
                np.testing.assert_array_equal(d["b"], [30, 40])
                np.testing.assert_array_equal(d["c"], [5])

            # drop_keys removes existing keys and ignores absent ones
            merge_save_npz(path, {"d": np.array([6])}, drop_keys=("a", "missing"))
            with np.load(path) as d:
                self.assertEqual(set(d.files), {"b", "c", "d"})
                np.testing.assert_array_equal(d["d"], [6])


if __name__ == "__main__":
    unittest.main()
