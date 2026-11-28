"""Unit tests for data_analysis.utils batch/NPZ/formatting helpers and mach_tip_channels."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from data_analysis.io.probe_map import mach_tip_channels
from data_analysis.utils import (
    find_run_hdf5,
    fmt_num,
    merge_save_npz,
    npz_settings_mismatch,
    pack_nested_npz,
    pack_npz_entries,
    prepare_incremental_npz,
)


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

    def test_find_run_hdf5_single_missing_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                find_run_hdf5(tmp, "02")
            p1 = os.path.join(tmp, "02-He-800G.hdf5")
            open(p1, "wb").close()
            self.assertEqual(find_run_hdf5(tmp, "02"), p1)
            open(os.path.join(tmp, "02-He-dup.hdf5"), "wb").close()
            with self.assertRaises(ValueError):
                find_run_hdf5(tmp, "02")

    def test_pack_entries_nested_and_settings_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "packed.npz")
            payload = pack_npz_entries(
                "02",
                {
                    "flag": True,
                    "count": 10,
                    "val": 2.5,
                    "dt_s": 2.0e-7,
                    "tag": "hann",
                    "mat": np.eye(2),
                    "skipped": None,
                },
            )
            payload.update(pack_nested_npz("grp", {"a": {"x": 1.0}, "b": {"y": 2}}))
            self.assertNotIn("02__skipped", payload)
            self.assertEqual(payload["grp__a__x"].dtype, np.float64)
            self.assertEqual(payload["grp__b__y"].dtype, np.int64)
            merge_save_npz(path, payload)

            want = {"flag": True, "count": 10, "val": 2.5, "dt_s": 2.0e-7, "tag": "hann", "mat": np.eye(2)}
            with np.load(path) as d:
                self.assertEqual(npz_settings_mismatch(d, "02", want), [])
                diffs = npz_settings_mismatch(d, "02", dict(want, count=11, tag="rect"))
                self.assertEqual(len(diffs), 2)
                # Relative tolerance catches small-scale float differences (e.g. target_dt_s)
                dt_diffs = npz_settings_mismatch(d, "02", {"dt_s": 2.0005e-7})
                self.assertEqual(len(dt_diffs), 1)

                # Fallback prefix + alias support
                dc_diffs = npz_settings_mismatch(
                    d, "02__dc", {"flag_used": False},
                    allow_missing_keys=True, fallback_prefix="02",
                    fallback_aliases={"flag_used": "flag"},
                )
                self.assertEqual(len(dc_diffs), 1)

    def test_prepare_incremental_npz_skip_mismatch_and_partial_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "batch.npz")
            merge_save_npz(
                path,
                {
                    **pack_npz_entries("01", {"marker": 1.0, "win": 5}),
                    **pack_npz_entries("02", {"marker": 2.0, "win": 5}),
                },
            )
            # Matching settings keep both runs in existing_runs
            ex, drop = prepare_incremental_npz(path, ["01", "02"], "__marker", want={"win": 5})
            self.assertEqual(ex, {"01", "02"})
            self.assertEqual(drop, [])

            # Mismatched settings fail fast when overwrite=False
            with self.assertRaises(ValueError):
                prepare_incremental_npz(path, ["01"], "__marker", want={"win": 9})

            # Partial overwrite drops only run 01 keys while keeping run 02
            ex2, drop2 = prepare_incremental_npz(
                path, ["01"], "__marker", overwrite=True, full_overwrite=False, want={"win": 9}
            )
            self.assertEqual(ex2, {"02"})
            self.assertEqual(set(drop2), {"01__marker", "01__win"})

            # Full overwrite deletes archive
            ex3, drop3 = prepare_incremental_npz(
                path, ["01", "02"], "__marker", overwrite=True, full_overwrite=True
            )
            self.assertEqual((ex3, drop3), (set(), []))
            self.assertFalse(os.path.exists(path))

    def test_fmt_num_and_mach_tip_channels(self):
        self.assertEqual(fmt_num(3500.0, "+8.2f", scale=1e-3), "   +3.50")
        self.assertEqual(fmt_num(np.nan, "+8.2f"), "     ---")
        self.assertEqual(fmt_num(np.inf, "9.2f"), "      ---")

        wiring = {
            ("scope", "C1"): "Isat, Mach@P33-X+",
            ("scope", "C2"): "Isat, Mach@P33-X-",
            ("scope", "C3"): "Isat, Mach@P33-Y+",
            ("scope", "C4"): "Isat, Mach@P33-Y-",
            ("scope", "C5"): "Unrelated LP",
        }
        with mock.patch("data_analysis.io.probe_map.channel_wiring", return_value=wiring):
            self.assertEqual(
                mach_tip_channels("dummy.hdf5", required_axes=("X", "Y")),
                {"X+": "C1", "X-": "C2", "Y+": "C3", "Y-": "C4"},
            )
            self.assertEqual(
                mach_tip_channels("dummy.hdf5", with_scope=True)["X-"],
                ("scope", "C2"),
            )
            with self.assertRaises(ValueError):
                mach_tip_channels("dummy.hdf5", required_axes=("X", "Y", "Z"))

        dup_wiring = {**wiring, ("other", "C1"): "Isat, Mach@P33-X+"}
        with mock.patch("data_analysis.io.probe_map.channel_wiring", return_value=dup_wiring):
            with self.assertRaises(ValueError):
                mach_tip_channels("dummy.hdf5", with_scope=False)
            with self.assertRaises(ValueError):
                mach_tip_channels("dummy.hdf5", with_scope=True)


if __name__ == "__main__":
    unittest.main()
