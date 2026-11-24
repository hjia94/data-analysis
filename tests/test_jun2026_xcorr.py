"""Unit tests for Jun2026_xcorr coordinate handling, drive resolution, and batch_all."""

import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_JUN2026_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiments", "ucla-lapd", "Jun-2026")
)
if _JUN2026_DIR not in sys.path:
    sys.path.insert(0, _JUN2026_DIR)

from data_analysis.io.probe_map import STATIONARY, UNMAPPED
from data_analysis.viz.plot_utils import grid_frames
import Jun2026_IV as jiv
import Jun2026_plot as jpl
import Jun2026_xcorr as jxc


def _mock_pos(motion_group, shot_nums, npos=3, nshot=2):
    return jiv.ProbePositions(
        pos_array=np.zeros((npos * nshot, 2)),
        xpos=np.arange(npos, dtype=float),
        ypos=np.zeros(1, dtype=float),
        shot_nums=np.asarray(shot_nums, dtype=int),
        npos=npos,
        nshot=nshot,
        motion_group=motion_group,
    )


def _synthetic_iter(px, py, mg="<DriveA>", dt=1e-5, nsamples=2048):
    t = np.arange(nsamples) * dt
    sig = np.sin(2 * np.pi * 4000.0 * t)[None, :]  # (1, nsamples)
    positions = [(i, sig, sig, dt) for i in range(len(px))]
    return np.asarray(px, float), np.asarray(py, float), len(px), positions, mg


class TestXcorrCoordinatesAndMerge(unittest.TestCase):
    def test_opposite_order_pairs_grid_to_same_true_location(self):
        xs = np.array([-4.0, 0.0, 4.0])
        ys = np.array([-2.0, 0.0, 2.0])
        gx, gy = np.meshgrid(xs, ys)
        px_a, py_a = gx.ravel(), gy.ravel()
        px_b, py_b = px_a[::-1], py_a[::-1]

        g2_a = np.where((px_a == 4.0) & (py_a == -2.0), 0.99, 0.1)[:, None]
        g2_b = np.where((px_b == 4.0) & (py_b == -2.0), 0.99, 0.1)[:, None]
        ph_zero = np.zeros_like(g2_a)

        with tempfile.TemporaryDirectory() as tmp:
            npz_p = os.path.join(tmp, "26-xcorr-data.npz")
            ka = jxc._pair_key(("lpscope", "C1"), ("lpscope", "C3"))
            kb = jxc._pair_key(("lpscope", "C3"), ("bdotscope", "C1"))
            np.savez(
                npz_p,
                freq=np.array([4000.0]),
                **{
                    f"{ka}__pos_x": px_a, f"{ka}__pos_y": py_a,
                    f"{ka}__gamma2": g2_a, f"{ka}__phase": ph_zero,
                    f"{kb}__pos_x": px_b, f"{kb}__pos_y": py_b,
                    f"{kb}__gamma2": g2_b, f"{kb}__phase": ph_zero,
                },
            )
            _, ga, _, pxa, pya = jpl._load_xcorr_run(
                "26-dummy.hdf5", ("lpscope", "C1"), ("lpscope", "C3"), npz_path=npz_p
            )
            _, gb, _, pxb, pyb = jpl._load_xcorr_run(
                "26-dummy.hdf5", ("lpscope", "C3"), ("bdotscope", "C1"), npz_path=npz_p
            )
            fa, xsa, ysa = grid_frames(pxa, pya, ga)
            fb, xsb, ysb = grid_frames(pxb, pyb, gb)
            np.testing.assert_array_equal(xsa, xsb)
            np.testing.assert_array_equal(ysa, ysb)
            np.testing.assert_allclose(fa, fb)
            self.assertAlmostEqual(fa[0, 0, 2], 0.99)

    def test_legacy_top_level_coords_fallback(self):
        px = np.array([-2.0, 0.0, 2.0])
        py = np.array([0.0, 0.0, 0.0])
        g2 = np.full((3, 1), 0.8)
        ph = np.zeros((3, 1))
        with tempfile.TemporaryDirectory() as tmp:
            npz_p = os.path.join(tmp, "07-xcorr-data.npz")
            k = jxc._pair_key(("machscope", "C2"), ("machscope", "C3"))
            np.savez(
                npz_p,
                freq=np.array([4000.0]),
                pos_x=px,
                pos_y=py,
                **{f"{k}__gamma2": g2, f"{k}__phase": ph},
            )
            _, _, _, px_ld, py_ld = jpl._load_xcorr_run(
                "07-dummy.hdf5", ("machscope", "C2"), ("machscope", "C3"), npz_path=npz_p
            )
            np.testing.assert_array_equal(px_ld, px)
            np.testing.assert_array_equal(py_ld, py)

    def test_top_level_coords_kept_within_jitter_and_dropped_on_multigrid(self):
        with tempfile.TemporaryDirectory() as tmp:
            ifn = os.path.join(tmp, "99-test.hdf5")
            out_npz = jxc.xcorr_npz_path(ifn)

            # Pair 1: writes top-level pos_x/pos_y
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter([0.0, 1.0, 2.0], [0.0, 0.0, 0.0]),
            ):
                jxc.batch_xcorr(ifn, ("s", "C1"), ("s", "C5"), nperseg=512)
            with np.load(out_npz) as d:
                self.assertIn("pos_x", d.files)
                self.assertIn("pos_y", d.files)

            # Pair 2 with 5 um jitter: stays uniform, keeps top-level pos_x/pos_y
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter(
                    [0.0005, 1.0002, 1.9997], [0.0004, -0.0003, 0.0001]
                ),
            ):
                jxc.batch_xcorr(ifn, ("s", "C2"), ("s", "C5"), nperseg=512)
            with np.load(out_npz) as d:
                self.assertIn("pos_x", d.files)
                self.assertIn("pos_y", d.files)

            # Pair 3 on reversed scan: drops top-level pos_x/pos_y, keeps per-pair
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter([2.0, 1.0, 0.0], [0.0, 0.0, 0.0], "<DriveB>"),
            ):
                jxc.batch_xcorr(
                    ifn, ("s", "C3"), ("s", "C5"), nperseg=512, t_offset_b_ms=12.0
                )
            with np.load(out_npz) as d:
                self.assertNotIn("pos_x", d.files)
                self.assertNotIn("pos_y", d.files)
                np.testing.assert_allclose(d["s-C1__s-C5__pos_x"], [0.0, 1.0, 2.0])
                np.testing.assert_allclose(d["s-C3__s-C5__pos_x"], [2.0, 1.0, 0.0])
                self.assertEqual(float(d["s-C3__s-C5__t_offset_b_ms"]), 12.0)

    def test_legacy_single_pair_backfilled_and_multipair_rejected(self):
        freq = np.fft.rfftfreq(512, 1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            # 1 legacy pair -> backfills old pair's coords before dropping top-level keys
            ifn1 = os.path.join(tmp, "97-legacy1.hdf5")
            npz1 = jxc.xcorr_npz_path(ifn1)
            np.savez(
                npz1,
                freq=freq,
                pos_x=np.array([0.0, 1.0, 2.0]),
                pos_y=np.array([0.0, 0.0, 0.0]),
                **{
                    "s-C1__s-C5__gamma2": np.ones((3, freq.size)),
                    "s-C1__s-C5__phase": np.zeros((3, freq.size)),
                    "s-C1__s-C5__nshots": np.array([5, 5, 5]),
                },
            )
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter([2.0, 1.0, 0.0], [0.0, 0.0, 0.0], "<DriveB>"),
            ):
                jxc.batch_xcorr(ifn1, ("s", "C2"), ("s", "C5"), nperseg=512)
            with np.load(npz1) as d:
                self.assertNotIn("pos_x", d.files)
                self.assertNotIn("pos_y", d.files)
            r1 = jpl._load_xcorr_run(ifn1, ("s", "C1"), ("s", "C5"), npz_path=npz1)
            r2 = jpl._load_xcorr_run(ifn1, ("s", "C2"), ("s", "C5"), npz_path=npz1)
            self.assertIsNotNone(r1)
            self.assertIsNotNone(r2)
            np.testing.assert_allclose(r1[3], [0.0, 1.0, 2.0])
            np.testing.assert_allclose(r2[3], [2.0, 1.0, 0.0])

            # 2+ legacy pairs without per-pair coords -> fails fast advising overwrite=True
            ifn2 = os.path.join(tmp, "98-legacy2.hdf5")
            npz2 = jxc.xcorr_npz_path(ifn2)
            np.savez(
                npz2,
                freq=freq,
                pos_x=np.array([0.0, 1.0, 2.0]),
                pos_y=np.array([0.0, 0.0, 0.0]),
                **{
                    "s-C1__s-C5__gamma2": np.ones((3, freq.size)),
                    "s-C1__s-C5__phase": np.zeros((3, freq.size)),
                    "s-C1__s-C5__nshots": np.array([5, 5, 5]),
                    "s-C3__s-C5__gamma2": np.ones((3, freq.size)),
                    "s-C3__s-C5__phase": np.zeros((3, freq.size)),
                    "s-C3__s-C5__nshots": np.array([5, 5, 5]),
                },
            )
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter([2.0, 1.0, 0.0], [0.0, 0.0, 0.0], "<DriveB>"),
            ):
                with self.assertRaises(ValueError) as ctx:
                    jxc.batch_xcorr(ifn2, ("s", "C2"), ("s", "C5"), nperseg=512)
            self.assertIn("overwrite=True", str(ctx.exception))


class TestXcorrDriveResolutionAndOffset(unittest.TestCase):
    def test_resolve_pair_positions_stationary_unmapped_and_disjoint(self):
        mapping = {
            ("s", "C1"): "<Hermes> p29",
            ("s", "C3"): "<Hades> p25",
            ("s", "C5"): STATIONARY,
            ("b", "C1"): UNMAPPED,
        }
        pos_hermes = _mock_pos("<Hermes> p29", [1, 2, 3, 4, 5, 6])
        pos_hades_disjoint = _mock_pos("<Hades> p25", [7, 8, 9, 10, 11, 12])
        pos_bdot = _mock_pos("<Athena> p28", [1, 2, 3, 4, 5, 6])
        by_group = {
            "<Hermes> p29": pos_hermes,
            "<Hades> p25": pos_hades_disjoint,
            "<Athena> p28": pos_bdot,
        }

        with (
            mock.patch.object(jxc, "probe_channel_map", return_value=mapping),
            mock.patch.object(jiv, "read_lp_positions", side_effect=lambda _f, g: by_group[g]),
        ):
            # Moving vs stationary in both argument orders inherits moving drive
            pa, pb, sp = jxc._resolve_pair_positions("29.hdf5", ("s", "C1"), ("s", "C5"))
            self.assertIs(pa, pos_hermes)
            self.assertIs(pb, pos_hermes)
            self.assertIs(sp, pos_hermes)

            pa, pb, sp = jxc._resolve_pair_positions("29.hdf5", ("s", "C5"), ("s", "C1"))
            self.assertIs(pa, pos_hermes)
            self.assertIs(pb, pos_hermes)
            self.assertIs(sp, pos_hermes)

            # Unmapped channel rejected without explicit motion_group_name
            with self.assertRaises(ValueError):
                jxc._resolve_pair_positions("26.hdf5", ("s", "C5"), ("b", "C1"))

            # Explicit motion_group_name succeeds for unmapped channel
            pa, pb, sp = jxc._resolve_pair_positions(
                "26.hdf5", ("s", "C5"), ("b", "C1"), motion_group_name="<Athena> p28"
            )
            self.assertIs(sp, pos_bdot)

            # Disjoint two-phase moving drives rejected by assert_pairable
            with self.assertRaises(ValueError):
                jxc._resolve_pair_positions("29.hdf5", ("s", "C1"), ("s", "C3"))

    def test_cross_scope_trigger_offset_alignment(self):
        class _MockRun:
            def channel(self, chan, scope_name=None, shots=None):
                if scope_name == "machscope":
                    t = np.linspace(0.0, 5e-3, 5001)
                    sig = np.sin(2 * np.pi * 4000.0 * t)
                    return np.tile(sig, (len(shots), 1)), t
                t = np.linspace(0.0, 25e-3, 125001)
                sig = np.sin(2 * np.pi * 4000.0 * (t - 12e-3))
                return np.tile(sig, (len(shots), 1)), t

        pos = _mock_pos("<Drive>", [1, 2, 3], npos=1, nshot=3)
        sa, sb, _ = jxc._read_pair_at_position(
            _MockRun(), ("machscope", "C2"), ("biasscope", "C1"),
            pos, pos, 0, tmin_ms=1.0, tmax_ms=4.0, t_offset_b_ms=12.0,
        )
        np.testing.assert_allclose(sa, sb, atol=1e-4)


class TestXcorrBatchAll(unittest.TestCase):
    def test_skip_validation_fail_fast_and_offset_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            ifn = os.path.join(tmp, "26-test.hdf5")
            with open(ifn, "wb") as f:
                f.write(b"")

            # Seed one pair with known settings via mocked _iter_run_positions
            with mock.patch.object(
                jxc, "_iter_run_positions",
                return_value=_synthetic_iter([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], "<Hermes>"),
            ):
                jxc.batch_xcorr(
                    ifn, ("lpscope", "C1"), ("lpscope", "C3"),
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=512, t_offset_b_ms=0.0,
                )

            p1 = jxc.PairSpec(("lpscope", "C1"), ("lpscope", "C3"), "<Hermes>", 0.0)

            # Exact match skips without calling batch_xcorr
            with mock.patch.object(jxc, "batch_xcorr") as m_bx:
                jxc.batch_all(
                    data_dir=tmp, runs={"26": [p1]},
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=512, overwrite=False,
                )
                m_bx.assert_not_called()

            # Any parameter mismatch fails fast with overwrite=True hint
            mismatches = [
                dict(runs={"26": [p1]}, tmin_ms=2.0, tmax_ms=4.5, nperseg=512),
                dict(runs={"26": [p1]}, tmin_ms=1.5, tmax_ms=4.0, nperseg=512),
                dict(runs={"26": [p1]}, tmin_ms=1.5, tmax_ms=4.5, nperseg=1024),
                dict(
                    runs={"26": [jxc.PairSpec(("lpscope", "C1"), ("lpscope", "C3"), "<Hermes>", 12.0)]},
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=512,
                ),
                dict(
                    runs={"26": [jxc.PairSpec(("lpscope", "C1"), ("lpscope", "C3"), "<Other>", 0.0)]},
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=512,
                ),
                # New pair with conflicting nperseg fails fast before HDF5 read
                dict(
                    runs={"26": [jxc.PairSpec(("lpscope", "C2"), ("lpscope", "C3"))]},
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=1024,
                ),
            ]
            for kw in mismatches:
                with self.assertRaises(ValueError) as ctx:
                    jxc.batch_all(data_dir=tmp, overwrite=False, **kw)
                self.assertIn("overwrite=True", str(ctx.exception))

            # New pair forwards PairSpec.t_offset_b_ms and motion_group_name
            p_new = jxc.PairSpec(("lpscope", "C2"), ("biasscope", "C1"), "<Athena>", 12.0)
            with mock.patch.object(jxc, "batch_xcorr") as m_bx:
                jxc.batch_all(
                    data_dir=tmp, runs={"26": [p1, p_new]},
                    tmin_ms=1.5, tmax_ms=4.5, nperseg=512, overwrite=False,
                )
                m_bx.assert_called_once()
                self.assertEqual(m_bx.call_args.kwargs["t_offset_b_ms"], 12.0)
                self.assertEqual(m_bx.call_args.kwargs["motion_group_name"], "<Athena>")


if __name__ == "__main__":
    unittest.main()
