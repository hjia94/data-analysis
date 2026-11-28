"""Analytic physics & math checks for data_analysis.plasma.bdot and data_analysis.plasma.fields.

Every case has a closed-form answer: calibrated 3-axis probe round trip,
integration baselines, uniform current rod (curl/J), uniform and r^2 dBz/dt
(Faraday E_phi), uniformly filled current ball (Biot-Savart A and Poisson),
Drude conductivity, cylinder volumes, circular polarization handedness, and
reference-locked phasors.
"""

from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from scipy.constants import epsilon_0, mu_0

from data_analysis.io import (
    load_bdot_vna_dir,
    parse_bdot_gain,
    read_e5100a_binary,
    read_e5100a_csv,
)
from data_analysis.plasma import bdot, fields


def rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


NT, DT, F0 = 20000, 1e-9, 2e6          # 2 MHz RF sampled at 1 ns -> 500 samples/period
T = np.arange(NT) * DT


class TestBdotAndFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 3-axis synthetic probe
        rng = np.random.default_rng(0)
        A = np.array([[27.1, 0.99, 1.44], [-6.37, -25.7, 1.49], [5.69, 3.08, 27.9]]) * 1e-6
        G = 100 / np.sqrt(10)
        B0 = np.array([3e-4, -1e-4, 2e-4])[:, None, None]
        phase = rng.uniform(0, 2 * np.pi, (3, 4, 1))
        B = B0 * np.sin(2 * np.pi * F0 * T + phase)
        Bdot = B0 * 2 * np.pi * F0 * np.cos(2 * np.pi * F0 * T + phase)
        V = G * np.einsum("ij,j...->i...", A, Bdot) + 0.003
        cls.three_axis = (A, G, B, Bdot, V)

        # Infinite uniform current rod (radius a, current I) on a (z, y, x) grid
        h = 2e-3
        x = np.arange(-30, 31) * h
        z = np.arange(-5, 6) * h
        Z, Y, X = np.meshgrid(z, x, x, indexing="ij")
        R = np.hypot(X, Y)
        a, I = 0.03, 1000.0
        Bphi = np.where(R < a, mu_0 * I * R / (2 * np.pi * a**2), mu_0 * I / (2 * np.pi * np.maximum(R, 1e-12)))
        with np.errstate(invalid="ignore", divide="ignore"):
            Bx, By = np.nan_to_num(-Bphi * Y / R), np.nan_to_num(Bphi * X / R)
        cls.rod = (h, R, a, I / (np.pi * a**2), Bx, By)

        # Uniform Jz inside a sphere: Az = mu0 J (3a^2 - r^2)/6 inside, mu0 J a^3/(3r) outside
        hb, n, ab, J0 = 2e-3, 41, 0.03, 1e5
        c = (np.arange(n) - n // 2) * hb
        Zb, Yb, Xb = np.meshgrid(c, c, c, indexing="ij")
        Rb = np.sqrt(Xb**2 + Yb**2 + Zb**2)
        Jz = np.where(Rb < ab, J0, 0.0)
        Az_true = np.where(Rb < ab, mu_0 * J0 * (3 * ab**2 - Rb**2) / 6, mu_0 * J0 * ab**3 / (3 * np.maximum(Rb, 1e-12)))
        Ax, Ay, Az = fields.vector_potential(0 * Jz, 0 * Jz, Jz, hb, hb, hb)
        cls.ball = (hb, Rb, ab, J0, Jz, Ax, Ay, Az, Az_true)

    def test_inverse_area_matrix_forms(self):
        self.assertAlmostEqual(bdot.inverse_area_matrix(5e-5, 10), 1 / 5e-4, delta=1e-6 / 5e-4)
        np.testing.assert_allclose(
            bdot.inverse_area_matrix([1e-5, 2e-5, 4e-5], 2),
            np.diag([1 / 2e-5, 1 / 4e-5, 1 / 8e-5]),
        )
        with self.assertRaises(ValueError):
            bdot.inverse_area_matrix(np.ones((2, 2)))

    def test_voltage_to_bdot_3x3(self):
        A, G, _, Bdot, V = self.three_axis
        bd = bdot.voltage_to_bdot(V, bdot.inverse_area_matrix(A, G))
        self.assertLess(rel(bd - bd.mean(-1, keepdims=True), Bdot), 1e-9)

    def test_integrate_periodic(self):
        A, G, B, _, V = self.three_axis
        bd = bdot.voltage_to_bdot(V, bdot.inverse_area_matrix(A, G))
        self.assertLess(rel(bdot.integrate_bdot(bd, DT, "periodic", period=1 / (F0 * DT)), B), 1e-3)

    def test_integrate_smooth_ramping_drive(self):
        env = np.linspace(0.5, 1.5, NT)
        B = 3e-4 * env * np.sin(2 * np.pi * F0 * T)
        bd = np.gradient(B, DT) + 2.0
        Bs = bdot.integrate_bdot(bd, DT, "smooth", period=1 / (F0 * DT), smooth_periods=4)
        mid = slice(2000, NT - 2000)
        self.assertLess(rel(Bs[mid], B[mid]), 1e-2)

    def test_integrate_pretrigger_pulse(self):
        pulse = np.where(T > 5e-6, 1 - np.exp(-(T - 5e-6) / 2e-6), 0.0) * 1e-3
        Bp = bdot.integrate_bdot(np.gradient(pulse, DT) + 50.0, DT, "pretrigger", pretrigger=4000)
        self.assertLess(rel(Bp, pulse), 1e-3)
        Bs = bdot.integrate_bdot(np.gradient(pulse, DT) + 50.0, DT, "pretrigger", pretrigger=slice(0, 4000))
        np.testing.assert_allclose(Bs, Bp)

    def test_integrate_argument_errors(self):
        x = np.zeros(100)
        for kw in ({"baseline": "pretrigger"}, {"baseline": "periodic"}, {"baseline": "bogus"}):
            with self.assertRaises(ValueError):
                bdot.integrate_bdot(x, 1.0, **kw)

    def test_helmholtz_bench_and_effective_area(self):
        bench = bdot.HelmholtzBench()
        expected_k0 = 0.055 / (32.0 * (4.0 / 5.0) ** 1.5 * mu_0 * 2.0 * np.pi)
        np.testing.assert_allclose(bench.scale_m2_hz, expected_k0, rtol=1e-12)
        np.testing.assert_allclose(bench.scale_m2_hz, 304.220166, rtol=1e-6)

        f = np.array([100e3, 300e3, 500e3])
        A_true = 1.15e-4
        H_pos = 1j * (A_true / bench.scale_m2_hz) * f
        H_neg = -1j * (A_true / bench.scale_m2_hz) * f
        np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_pos, bench).real, A_true, rtol=1e-12)
        np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_neg, bench).real, -A_true, rtol=1e-12)
        np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_pos, bench).imag, 0.0, atol=1e-18)

    def test_fit_effective_area_matrix_signed_and_npz_roundtrip(self):
        bench = bdot.HelmholtzBench()
        k0 = bench.scale_m2_hz
        f = np.linspace(10e3, 2.5e6, 1601)
        A_true = np.array([
            [1.15e-4, 1.0e-7, 1.25e-5],
            [1.5e-6, 9.8e-5, -4.0e-6],
            [2.3e-6, -1.0e-5, 8.7e-5],
        ])
        H = 1j * (A_true[:, :, None] / k0) * f[None, None, :]
        low_f = f < 50e3
        H[:, :, low_f] += 5.0 * (10e3 / f[low_f])[None, None, :] * np.exp(1j * np.deg2rad(160.0))
        high_f = f > 1.2e6
        H[:, :, high_f] *= 0.85

        cal = bdot.fit_effective_area_matrix(
            f, H, fit_band_hz=(150e3, 500e3), offdiag_band_hz=(250e3, 1000e3),
            bench=bench, probe_id="C16",
            provenance={
                "bench_gain": 10.0,
                "bench_atten_db": 0.0,
                "notes": None,
                "tags": ["Red", None, 2026],
                "meta": {"upmark": "5/4/26"},
            },
        )
        np.testing.assert_allclose(cal.area_matrix_m2, A_true, rtol=1e-12, atol=1e-18)
        self.assertTrue(np.all(cal.r2_matrix > 0.9999))
        self.assertTrue(np.all(cal.phase_R_matrix > 0.9999))
        self.assertFalse(cal.is_diagonal_only)
        self.assertAlmostEqual(float(cal.phase_deg_matrix[1, 2]), -90.0, delta=1e-6)
        self.assertAlmostEqual(float(cal.phase_deg_matrix[2, 1]), -90.0, delta=1e-6)

        with tempfile.TemporaryDirectory() as tmp:
            out_npz = Path(tmp) / "C16-bdot-calibration.npz"
            bdot.save_bdot_calibration(out_npz, cal)
            cal2 = bdot.load_bdot_calibration(out_npz)
            self.assertEqual(cal2.probe_id, "C16")
            self.assertAlmostEqual(cal2.provenance["bench_gain"], 10.0)
            self.assertIsNone(cal2.provenance["notes"])
            self.assertEqual(cal2.provenance["tags"], ["Red", None, 2026])
            self.assertEqual(cal2.provenance["meta"], {"upmark": "5/4/26"})
            np.testing.assert_allclose(cal2.area_matrix_m2, A_true, rtol=1e-12)
            np.testing.assert_allclose(cal2.area_at(300e3, mode="interpolated").real, A_true, rtol=1e-12)
            with self.assertRaises(ValueError):
                cal2.area_at(4e3, mode="interpolated")

    def test_load_standalone_nacalc_npz(self):
        area = np.full((3, 3), np.nan)
        np.fill_diagonal(area, [1.15e-4, 9.8e-5, 8.7e-5])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "C19_calibration.npz"
            np.savez(
                p,
                probe_id=np.str_("C19"),
                area_matrix_m2=area,
                r2_matrix=np.eye(3),
                phase_deg_matrix=np.eye(3) * 90.0,
                phase_R_matrix=np.eye(3),
                bench_gain=np.float64(10.0),
                bench_atten_db=np.float64(0.0),
                diag_band_hz=np.array([150e3, 500e3]),
                offdiag_band_hz=np.array([250e3, 1000e3]),
                k0_m2_hz=np.float64(bdot.HelmholtzBench().scale_m2_hz),
            )
            cal = bdot.load_bdot_calibration(p)
            self.assertEqual(cal.probe_id, "C19")
            self.assertTrue(cal.is_diagonal_only)
            self.assertIsNone(cal.freq_hz)
            self.assertAlmostEqual(cal.provenance["bench_gain"], 10.0)
            np.testing.assert_allclose(cal.area_matrix_m2, np.diag([1.15e-4, 9.8e-5, 8.7e-5]))

    def test_per_channel_gain_complex_phasor_and_spectrum(self):
        A = np.array([[1.1e-4, 0.0, 1.2e-5], [1.5e-6, 9.8e-5, 0.0], [2.0e-6, -1.0e-5, 8.7e-5]])
        gains = np.array([10.0, 100.0, 1000.0])
        f0 = 4000.0
        w0 = 2.0 * np.pi * f0
        B_true = np.array([3e-5 + 1e-5j, -2e-5 + 4e-5j, 5e-6 - 2e-6j])[:, None]
        V_coil = -1j * w0 * (A @ B_true)
        V_scope = gains[:, None] * V_coil

        inv_g = bdot.inverse_area_matrix(A, gain=gains)
        np.testing.assert_allclose(bdot.voltage_to_bdot(V_scope, inv_g), -1j * w0 * B_true, rtol=1e-12)
        np.testing.assert_allclose(bdot.phasor_voltage_to_b(V_coil, f0, A, convention="exp_minus_iwt"), B_true, rtol=1e-12)
        np.testing.assert_allclose(
            bdot.phasor_voltage_to_b(+1j * w0 * (A @ B_true), f0, A, convention="exp_plus_iwt"),
            B_true, rtol=1e-12,
        )

        freqs = np.array([0.0, 100e3, 200e3, 400e3])
        B_spec = np.broadcast_to(B_true, (3, freqs.size)).copy()
        V_spec_p = +1j * (2.0 * np.pi * freqs)[None, :] * np.einsum("ij,jf->if", A, B_spec)
        B_out_p = bdot.calibrate_bdot_spectrum(freqs, V_spec_p, A, axis=-1, mode="static")
        self.assertTrue(np.all(np.isnan(B_out_p[:, 0])))
        np.testing.assert_allclose(B_out_p[:, 1:], B_spec[:, 1:], rtol=1e-12)
        V_spec_m = -1j * (2.0 * np.pi * freqs)[None, :] * np.einsum("ij,jf->if", A, B_spec)
        B_out_m = bdot.calibrate_bdot_spectrum(freqs, V_spec_m, A, axis=-1, convention="exp_minus_iwt", mode="static")
        self.assertTrue(np.all(np.isnan(B_out_m[:, 0])))
        np.testing.assert_allclose(B_out_m[:, 1:], B_spec[:, 1:], rtol=1e-12)

    def test_interpolated_phase_convention_and_rfft_spectrum(self):
        bench = bdot.HelmholtzBench()
        k0 = bench.scale_m2_hz
        f_vna = np.linspace(10e3, 2.5e6, 500)
        A0 = np.array([
            [1.15e-4, 2.0e-6, 1.2e-5],
            [-3.0e-6, 9.8e-5, -5.0e-6],
            [4.0e-6, -1.0e-5, 8.7e-5],
        ])
        pole = 1.0 / (1.0 + 1j * (f_vna / 500e3))
        H_vna = 1j * (A0[:, :, None] / k0) * f_vna[None, None, :] * pole[None, None, :]
        cal = bdot.fit_effective_area_matrix(f_vna, H_vna, fit_band_hz=(150e3, 500e3), bench=bench, probe_id="SYN")

        f_test = 400e3
        A_plus = cal.area_at(f_test, mode="interpolated", convention="exp_plus_iwt")
        A_minus = cal.area_at(f_test, mode="interpolated", convention="exp_minus_iwt")
        self.assertGreater(np.max(np.abs(A_plus.imag)), 0.3 * np.max(np.abs(A_plus.real)))
        np.testing.assert_allclose(A_minus, np.conj(A_plus), rtol=1e-14)

        dt, nt = 200e-9, 500
        t = np.arange(nt) * dt
        k0_bin = int(round(f_test * nt * dt))
        self.assertTrue(np.isclose(np.fft.rfftfreq(nt, dt)[k0_bin], f_test))

        B_plus = np.array([3e-4 * np.exp(0.4j), -2e-4 * np.exp(-0.9j), 1.5e-4 * np.exp(1.2j)])
        w0 = 2.0 * np.pi * f_test
        B_t = np.real(B_plus[:, None] * np.exp(1j * w0 * t[None, :]))
        V_plus = (1j * w0) * (A_plus @ B_plus)
        V_t = np.real(V_plus[:, None] * np.exp(1j * w0 * t[None, :]))

        V_ph_plus = fields.phasor(V_t, dt, f_test, window=None)
        B_ph_plus = fields.phasor(B_t, dt, f_test, window=None)
        np.testing.assert_allclose(
            bdot.phasor_voltage_to_b(V_ph_plus, f_test, cal, convention="exp_plus_iwt", mode="interpolated"),
            B_ph_plus, rtol=1e-10,
        )
        np.testing.assert_allclose(
            bdot.phasor_voltage_to_b(np.conj(V_ph_plus), f_test, cal, convention="exp_minus_iwt", mode="interpolated"),
            np.conj(B_ph_plus), rtol=1e-10,
        )

        freqs_rfft = np.fft.rfftfreq(nt, dt)
        V_rfft = np.fft.rfft(V_t, axis=-1)
        B_rfft_true = np.fft.rfft(B_t, axis=-1)
        B_rfft_rec = bdot.calibrate_bdot_spectrum(freqs_rfft, V_rfft, cal)
        self.assertTrue(np.all(np.isnan(B_rfft_rec[:, 0])))
        np.testing.assert_allclose(B_rfft_rec[:, k0_bin], B_rfft_true[:, k0_bin], rtol=1e-10)

        B_rfft_minus = bdot.calibrate_bdot_spectrum(
            freqs_rfft, np.conj(V_rfft), cal, convention="exp_minus_iwt", mode="interpolated",
        )
        np.testing.assert_allclose(B_rfft_minus[:, k0_bin], np.conj(B_rfft_true[:, k0_bin]), rtol=1e-10)

    def test_e5100a_csv_binary_dir_and_parse_bdot_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            f_arr = np.array([100e3, 200e3, 300e3])
            lin_p = tmp_path / "C16BXPX-12-24.CSV"
            lin_p.write_text(
                '"E5100A REV3.14"\n"DATE: Apr  7 2005"\n\n"MEASURE TYPE: AR"\n'
                '"FORMAT TYPE: LINMP"\n"NUMBER of POINTS: 3"\n\n'
                '"Frequency"\t"Main Array"\t"Sub Array"\n'
                '1.0e5\t0.5\t90.0\n2.0e5\t1.0\t90.0\n3.0e5\t1.5\t90.0\n',
                encoding="utf-8",
            )
            rec = read_e5100a_csv(lin_p)
            np.testing.assert_allclose(rec["freq_hz"], f_arr)
            np.testing.assert_allclose(rec["complex_response"], 1j * np.array([0.5, 1.0, 1.5]), atol=1e-12)
            self.assertEqual(rec["metadata"]["NUMBER OF POINTS"], "3")

            log_p = tmp_path / "log.csv"
            log_p.write_text('"FORMAT TYPE: LOGMP"\n1e5,-20.0,-90.0\n2e5,0.0,90.0\n', encoding="utf-8")
            rec_log = read_e5100a_csv(log_p)
            np.testing.assert_allclose(rec_log["complex_response"], [-0.1j, 1.0j], atol=1e-12)

            cal_dir = tmp_path / "c16_dir"
            cal_dir.mkdir()
            for d_i, d in enumerate("XYZ"):
                for p_i, p in enumerate("XYZ"):
                    val = float((p_i + 1) * 10 + (d_i + 1))
                    (cal_dir / f"C16B{d}P{p}.CSV").write_text(
                        f'"MEASURE TYPE: AR"\n"FORMAT TYPE: LINMP"\n1e5\t{val}\t90.0\n2e5\t{2*val}\t90.0\n',
                        encoding="utf-8",
                    )
            vna = load_bdot_vna_dir(cal_dir, probe_id="C16")
            self.assertFalse(vna["is_diagonal_only"])
            self.assertEqual(vna["H_vna"].shape, (3, 3, 2))
            self.assertAlmostEqual(vna["H_vna"][2, 0, 0], 31.0j, delta=1e-12)
            self.assertAlmostEqual(vna["H_vna"][0, 2, 0], 13.0j, delta=1e-12)

            bin_p = tmp_path / "sample.dat"
            hdr = struct.pack("=dd", 10000.0, 180000000.0) + b"\x00" * (0x126 - 16)
            def _pack_rec(vals):
                return struct.pack("=LHL", 0, len(vals), 0) + struct.pack(f"={len(vals)}d", *vals)
            bin_p.write_bytes(hdr + _pack_rec([1.0, 2.0]) + _pack_rec([0.0, 0.0]) + _pack_rec([3.0, 4.0]) + _pack_rec([0.0, 0.0]))
            brec = read_e5100a_binary(bin_p, freq_hz=[1e5, 2e5], format_type="REIM")
            np.testing.assert_allclose(brec["complex_response"], [1.0 + 3.0j, 2.0 + 4.0j])

            desc = (
                "    - Probe\n"
                "      - (NOT USED) P20: Bdot-old\n"
                "        - Connection: X5\n"
                "      - P28: Bdot-10T-C16\n"
                "        - Connection: 2.5BLP => X10 => 2.5BLP =>X100\n"
                "      - P29: LP\n"
            )
            parsed = parse_bdot_gain(desc)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["header"], "P28: Bdot-10T-C16")
            self.assertAlmostEqual(parsed["gain"], 1000.0)
            self.assertAlmostEqual(parsed["atten_db"], 0.0)
            self.assertAlmostEqual(parsed["effective_gain"], 1000.0)
            parsed_pad = parse_bdot_gain("- P28: Bdot\n  - Connection: X100 => 20dB\n")
            self.assertAlmostEqual(parsed_pad["effective_gain"], 10.0)
            self.assertIsNone(parse_bdot_gain("- P29: LP only\n"))

            multi_desc = (
                "- P30: Bdot-5T-B2\n  - Connection: X10\n"
                "- P28: Bdot-10T-C16\n  - Connection: 2.5BLP => X10 => 2.5BLP => X100\n"
            )
            with self.assertRaises(ValueError):
                parse_bdot_gain(multi_desc)
            self.assertAlmostEqual(parse_bdot_gain(multi_desc, probe="P28")["effective_gain"], 1000.0)
            self.assertAlmostEqual(parse_bdot_gain(multi_desc, probe="P30")["effective_gain"], 10.0)
            self.assertIsNone(parse_bdot_gain(multi_desc, probe="P2"))
            self.assertIsNone(parse_bdot_gain(multi_desc, probe="P99"))

    def test_current_density_rod(self):
        h, R, a, J0, Bx, By = self.rod
        _, _, Jz = fields.current_density(Bx, By, 0, h, h, h)
        self.assertLess(np.max(np.abs(Jz[R < a - 2 * h] - J0)) / J0, 1e-9)
        self.assertLess(np.max(np.abs(Jz[R > a + 4 * h])) / J0, 5e-3)

    def test_current_density_plane_and_complex(self):
        h, R, a, J0, Bx, By = self.rod
        ph = np.exp(0.7j)
        _, _, Jz = fields.current_density(Bx[5] * ph, By[5] * ph, 0, h, h, None, axes=(-1, -2, None))
        self.assertTrue(np.iscomplexobj(Jz))
        self.assertLess(np.max(np.abs(Jz[R[5] < a - 2 * h] - J0 * ph)) / J0, 1e-9)

    def test_poynting_and_power_density(self):
        Sx, Sy, Sz = fields.poynting(1.0, 0, 0, 0, 1e-3, 0)
        self.assertEqual((Sx, Sy), (0, 0))
        self.assertAlmostEqual(Sz, 1e-3 / mu_0, delta=1e-9 / mu_0)
        self.assertEqual(fields.power_density(2, 0, 0, 3, 5, 0), 6)

    def test_faraday_ephi_uniform_signed_radii(self):
        r = np.arange(-20, 21) * 1e-2 + 0.005
        E = fields.faraday_ephi(np.full((3, r.size), 7.0), r)
        np.testing.assert_allclose(E, np.broadcast_to(-np.abs(r) * 7.0 / 2, (3, r.size)), rtol=1e-12)
        E0 = fields.faraday_ephi(np.full((r.size, 4), 7.0), r, axis=0)
        np.testing.assert_allclose(E0, np.broadcast_to((-np.abs(r) * 7.0 / 2)[:, None], (r.size, 4)), rtol=1e-12)

    def test_faraday_ephi_r2_profile(self):
        r = np.linspace(0, 0.2, 2001)
        self.assertLess(rel(fields.faraday_ephi(5.0 * r**2, r), -5.0 * r**3 / 4), 1e-5)

    def test_vector_potential_ball(self):
        h, R, a, J0, Jz, Ax, Ay, Az, Az_true = self.ball
        self.assertLess(rel(Az, Az_true), 2e-2)
        self.assertFalse(Ax.any() or Ay.any())

    def test_vector_potential_poisson(self):
        h, R, a, J0, Jz, Ax, Ay, Az, Az_true = self.ball
        lap = (sum(np.roll(Az, s, ax) for ax in range(3) for s in (1, -1)) - 6 * Az) / h**2
        core = R < a - 3 * h
        self.assertLess(np.max(np.abs(lap[core] + mu_0 * J0)) / (mu_0 * J0), 2e-2)

    def test_vector_potential_time_axis_and_complex(self):
        h, R, a, J0, Jz, Ax, Ay, Az, Az_true = self.ball
        Jt = np.stack([Jz, 2 * Jz], axis=-1)
        _, _, Azt = fields.vector_potential(0, 0, Jt, h, h, h, axes=(-2, -3, -4))
        np.testing.assert_allclose(Azt[..., 1], 2 * Az, rtol=1e-10, atol=1e-20)
        _, _, Azc = fields.vector_potential(0, 0, 1j * Jz, h, h, h)
        np.testing.assert_allclose(Azc, 1j * Az, rtol=1e-10, atol=1e-20)

    def test_cycle_means_and_average(self):
        P = 2 + np.sin(2 * np.pi * np.arange(5000) / 333.3)
        self.assertLess(np.max(np.abs(fields.cycle_means(P[None, :], 333.3, start=10) - 2)), 5e-3)
        self.assertAlmostEqual(fields.cycle_average(P, 333.3, 0, 3), 2.0, delta=5e-3)
        with self.assertRaises(ValueError):
            fields.cycle_average(P, 333.3, 0, 100)

    def test_axisymmetric_integral_cylinder(self):
        r, z = np.linspace(0, 0.1, 401), np.linspace(0, 0.3, 301)
        vol = np.pi * 0.1**2 * 0.3
        np.testing.assert_allclose(fields.axisymmetric_integral(np.ones((z.size, r.size)), r, z), vol, rtol=1e-9)
        with self.assertRaises(ValueError):
            fields.axisymmetric_integral(np.ones(5), np.linspace(-1, 1, 5))

    def test_ac_conductivity_drude(self):
        nu, w = 3e7, 2 * np.pi * F0
        wpe2 = 1e17 * 1.602e-19**2 / (9.109e-31 * epsilon_0)
        sigma = epsilon_0 * wpe2 / (nu + 1j * w)
        t = np.arange(3333) * DT
        E = 50 * np.cos(w * t + 0.3)
        J = np.real(sigma * 50 * np.exp(1j * (w * t + 0.3)))
        s = fields.ac_conductivity(J, E, DT, F0)
        self.assertLess(abs(s - sigma) / abs(sigma), 1e-3)
        s2 = fields.ac_conductivity(np.stack([J, J]), np.stack([E, 0 * E]), DT, F0, min_E=1.0)
        self.assertTrue(np.isfinite(s2[0]) and np.isnan(s2[1]))

    def test_circular_polarization_handedness_and_parseval(self):
        bL_m, bR_m, el_m = fields.circular_polarization(1.0 + 0j, -1j, convention="exp_minus_iwt", b_sign_z=+1)
        bL_p, bR_p, el_p = fields.circular_polarization(1.0 + 0j, +1j, convention="exp_plus_iwt", b_sign_z=+1)
        self.assertAlmostEqual(abs(bL_m), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(abs(bR_m), 0.0, places=12)
        self.assertAlmostEqual(float(el_m), -1.0, places=12)
        self.assertAlmostEqual(abs(bL_p), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(abs(bR_p), 0.0, places=12)
        self.assertAlmostEqual(float(el_p), -1.0, places=12)

        bL_rev, bR_rev, el_rev = fields.circular_polarization(1.0 + 0j, -1j, convention="exp_minus_iwt", b_sign_z=-1)
        self.assertAlmostEqual(abs(bL_rev), 0.0, places=12)
        self.assertAlmostEqual(abs(bR_rev), np.sqrt(2.0), places=12)
        self.assertAlmostEqual(float(el_rev), +1.0, places=12)

        rng = np.random.default_rng(7)
        bx = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        by = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
        bL, bR, _ = fields.circular_polarization(bx, by)
        np.testing.assert_allclose(np.abs(bL) ** 2 + np.abs(bR) ** 2, np.abs(bx) ** 2 + np.abs(by) ** 2)

    def test_reference_locked_phasor_recovers_amplitude_and_phase_with_random_shot_phase(self):
        rng = np.random.default_rng(12)
        dt = 1e-5
        t = np.arange(512) * dt
        f0 = 4000.0
        nshots = 40
        phi_rand = rng.uniform(0.0, 2.0 * np.pi, size=(nshots, 1))
        dphi_true = 1.2
        amp_true = 2.5
        ref = 1.3 * np.cos(phi_rand - 2.0 * np.pi * f0 * t[None, :])
        sig = amp_true * np.cos(phi_rand + dphi_true - 2.0 * np.pi * f0 * t[None, :])

        xc, g2 = fields.reference_locked_phasor(sig, ref, dt, f0, convention="exp_minus_iwt")
        self.assertAlmostEqual(abs(xc), amp_true, delta=1e-2)
        self.assertAlmostEqual(float(np.angle(xc)), dphi_true, delta=1e-2)
        self.assertAlmostEqual(float(g2), 1.0, delta=1e-6)

    def test_divergence_2d_analytic_complex_and_solenoidal_polarity(self):
        xs = np.linspace(-0.08, 0.08, 33)
        ys = np.linspace(-0.06, 0.06, 25)
        dx = float(xs[1] - xs[0])
        dy = float(ys[1] - ys[0])
        X, Y = np.meshgrid(xs, ys)

        # Exact linear complex field: dFx/dx + dFy/dy == (2 + 1j) + (5 - 2j) == 7 - 1j
        Fx_lin = (2.0 + 1.0j) * X + 3.0 * Y
        Fy_lin = -1.5 * X + (5.0 - 2.0j) * Y
        div_lin = fields.divergence_2d(Fx_lin, Fy_lin, dx, dy)
        np.testing.assert_allclose(div_lin, 7.0 - 1.0j, atol=1e-11)

        # Solenoidal Gaussian vortex: div B_perp == 0, while flipping By -> -By breaks div/curl << 1
        env = np.exp(-(X ** 2 + Y ** 2) / (0.03 ** 2))
        Bx = -Y * env * np.exp(0.4j)
        By = +X * env * np.exp(0.4j)
        div_ok = fields.divergence_2d(Bx, By, dx, dy)
        _, _, curl_ok = fields.curl(Bx, By, 0.0, dx, dy, None)
        div_bad = fields.divergence_2d(Bx, -By, dx, dy)
        _, _, curl_bad = fields.curl(Bx, -By, 0.0, dx, dy, None)
        inner = (slice(2, -2), slice(2, -2))
        self.assertLess(np.linalg.norm(div_ok[inner]) / np.linalg.norm(curl_ok[inner]), 0.02)
        self.assertGreater(np.linalg.norm(div_bad[inner]) / np.linalg.norm(curl_bad[inner]), 0.5)

    def test_spectral_b_rms_eliminates_harmonic_inflation(self):
        # Two-tone B(t) = A1 cos(w0 t) + A2 cos(2 w0 t):
        # true time-RMS is sqrt((A1^2 + A2^2) / 2), whereas std(dB/dt) / w0
        # inflates to sqrt((A1^2 + 4 A2^2) / 2).
        dt = 1e-6
        n = 5000  # 5.0 ms -> exact integer cycles for 4 kHz & 8 kHz (df = 200 Hz)
        t = np.arange(n) * dt
        f0 = 4000.0
        w0 = 2.0 * np.pi * f0
        a1, a2 = 2.0e-5, 1.5e-5
        bdot_sig = -w0 * a1 * np.sin(w0 * t) - (2.0 * w0) * a2 * np.sin(2.0 * w0 * t)
        b_rms = float(bdot.spectral_b_rms(bdot_sig[None, :], dt, f_min_hz=0.5 * f0)[0])
        b_true = float(np.sqrt(0.5 * (a1 ** 2 + a2 ** 2)))
        b_naive = float(np.std(bdot_sig) / w0)
        self.assertAlmostEqual(b_rms, b_true, delta=1e-11)
        self.assertGreater(b_naive, 1.3 * b_true)


if __name__ == "__main__":
    unittest.main()
