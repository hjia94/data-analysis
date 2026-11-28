"""Analytic checks for data_analysis.plasma.bdot and data_analysis.plasma.fields.

Every case has a closed-form answer, so these test the maths, not a fit:
a calibrated 3-axis probe round trip, the three integration baselines, the
field of a uniform current rod (curl/J), uniform and r^2 dBz/dt (Faraday
E_phi), a uniformly filled current ball (Biot-Savart A and Poisson), a Drude
conductivity, and cylinder volumes.

    pip install -e ".[test]"
    pytest tests/
"""

import struct
import numpy as np
import pytest
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


# ---------------------------------------------------------------------------
# bdot
# ---------------------------------------------------------------------------

NT, DT, F0 = 20000, 1e-9, 2e6          # 2 MHz RF sampled at 1 ns -> 500 samples/period
T = np.arange(NT) * DT


@pytest.fixture(scope="module")
def three_axis():
    rng = np.random.default_rng(0)
    A = np.array([[27.1, 0.99, 1.44], [-6.37, -25.7, 1.49], [5.69, 3.08, 27.9]]) * 1e-6   # m^2
    G = 100 / np.sqrt(10)
    B0 = np.array([3e-4, -1e-4, 2e-4])[:, None, None]       # 3 components x 4 positions
    phase = rng.uniform(0, 2 * np.pi, (3, 4, 1))
    B = B0 * np.sin(2 * np.pi * F0 * T + phase)
    Bdot = B0 * 2 * np.pi * F0 * np.cos(2 * np.pi * F0 * T + phase)
    V = G * np.einsum("ij,j...->i...", A, Bdot) + 0.003      # + DC amplifier offset
    return A, G, B, Bdot, V


def test_inverse_area_matrix_forms():
    assert bdot.inverse_area_matrix(5e-5, 10) == pytest.approx(1 / 5e-4)
    np.testing.assert_allclose(bdot.inverse_area_matrix([1e-5, 2e-5, 4e-5], 2),
                               np.diag([1 / 2e-5, 1 / 4e-5, 1 / 8e-5]))
    with pytest.raises(ValueError):
        bdot.inverse_area_matrix(np.ones((2, 2)))


def test_voltage_to_bdot_3x3(three_axis):
    A, G, _, Bdot, V = three_axis
    bd = bdot.voltage_to_bdot(V, bdot.inverse_area_matrix(A, G))
    assert rel(bd - bd.mean(-1, keepdims=True), Bdot) < 1e-9


def test_integrate_periodic(three_axis):
    A, G, B, _, V = three_axis
    bd = bdot.voltage_to_bdot(V, bdot.inverse_area_matrix(A, G))
    assert rel(bdot.integrate_bdot(bd, DT, "periodic", period=1 / (F0 * DT)), B) < 1e-3


def test_integrate_smooth_ramping_drive():
    env = np.linspace(0.5, 1.5, NT)
    B = 3e-4 * env * np.sin(2 * np.pi * F0 * T)
    bd = np.gradient(B, DT) + 2.0                             # + offset, T/s
    Bs = bdot.integrate_bdot(bd, DT, "smooth", period=1 / (F0 * DT), smooth_periods=4)
    mid = slice(2000, NT - 2000)                              # edges see the reflection
    assert rel(Bs[mid], B[mid]) < 1e-2


def test_integrate_pretrigger_pulse():
    pulse = np.where(T > 5e-6, 1 - np.exp(-(T - 5e-6) / 2e-6), 0.0) * 1e-3
    Bp = bdot.integrate_bdot(np.gradient(pulse, DT) + 50.0, DT, "pretrigger", pretrigger=4000)
    assert rel(Bp, pulse) < 1e-3
    Bs = bdot.integrate_bdot(np.gradient(pulse, DT) + 50.0, DT, "pretrigger", pretrigger=slice(0, 4000))
    np.testing.assert_allclose(Bs, Bp)


def test_integrate_argument_errors():
    x = np.zeros(100)
    for kw in ({"baseline": "pretrigger"}, {"baseline": "periodic"}, {"baseline": "bogus"}):
        with pytest.raises(ValueError):
            bdot.integrate_bdot(x, 1.0, **kw)


def test_helmholtz_bench_and_effective_area():
    bench = bdot.HelmholtzBench()
    expected_k0 = 0.055 / (32.0 * (4.0 / 5.0) ** 1.5 * mu_0 * 2.0 * np.pi)
    assert bench.scale_m2_hz == pytest.approx(expected_k0, rel=1e-12)
    assert bench.scale_m2_hz == pytest.approx(304.220166, rel=1e-6)

    f = np.array([100e3, 300e3, 500e3])
    A_true = 1.15e-4  # m^2
    H_pos = 1j * (A_true / bench.scale_m2_hz) * f       # +90 deg
    H_neg = -1j * (A_true / bench.scale_m2_hz) * f      # -90 deg
    np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_pos, bench).real, A_true, rtol=1e-12)
    np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_neg, bench).real, -A_true, rtol=1e-12)
    np.testing.assert_allclose(bdot.helmholtz_effective_area(f, H_pos, bench).imag, 0.0, atol=1e-18)


def test_fit_effective_area_matrix_signed_and_npz_roundtrip(tmp_path):
    bench = bdot.HelmholtzBench()
    k0 = bench.scale_m2_hz
    f = np.linspace(10e3, 2.5e6, 1601)
    A_true = np.array([
        [1.15e-4, 1.0e-7, 1.25e-5],
        [1.5e-6, 9.8e-5, -4.0e-6],
        [2.3e-6, -1.0e-5, 8.7e-5],
    ])
    # Ideal inductive response inside 100 kHz - 1 MHz, with low-f bench spike (<50 kHz)
    # and gentle high-f droop (>1.2 MHz) outside the fit bands
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
    assert np.all(cal.r2_matrix > 0.9999)
    assert np.all(cal.phase_R_matrix > 0.9999)
    assert not cal.is_diagonal_only
    # Check signs in phase_deg_matrix match negative off-diagonals
    assert cal.phase_deg_matrix[1, 2] == pytest.approx(-90.0, abs=1e-6)
    assert cal.phase_deg_matrix[2, 1] == pytest.approx(-90.0, abs=1e-6)

    # Save & load round-trip (allow_pickle=False even with None/list/dict provenance)
    out_npz = tmp_path / "C16-bdot-calibration.npz"
    bdot.save_bdot_calibration(out_npz, cal)
    cal2 = bdot.load_bdot_calibration(out_npz)
    assert cal2.probe_id == "C16"
    assert cal2.provenance["bench_gain"] == pytest.approx(10.0)
    assert cal2.provenance["notes"] is None
    assert cal2.provenance["tags"] == ["Red", None, 2026]
    assert cal2.provenance["meta"] == {"upmark": "5/4/26"}
    np.testing.assert_allclose(cal2.area_matrix_m2, A_true, rtol=1e-12)
    np.testing.assert_allclose(cal2.area_at(300e3, mode="interpolated").real, A_true, rtol=1e-12)
    with pytest.raises(ValueError):
        cal2.area_at(4e3, mode="interpolated")  # below VNA sweep requires mode='static'


def test_load_standalone_nacalc_npz(tmp_path):
    # Mimics standalone nAcalc.py <probe>_calibration.npz (no freq_hz / area_complex_m2, NaN off-diags)
    area = np.full((3, 3), np.nan)
    np.fill_diagonal(area, [1.15e-4, 9.8e-5, 8.7e-5])
    p = tmp_path / "C19_calibration.npz"
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
    assert cal.probe_id == "C19"
    assert cal.is_diagonal_only
    assert cal.freq_hz is None
    assert cal.provenance["bench_gain"] == pytest.approx(10.0)
    np.testing.assert_allclose(cal.area_matrix_m2, np.diag([1.15e-4, 9.8e-5, 8.7e-5]))


def test_per_channel_gain_complex_phasor_and_spectrum():
    A = np.array([[1.1e-4, 0.0, 1.2e-5], [1.5e-6, 9.8e-5, 0.0], [2.0e-6, -1.0e-5, 8.7e-5]])
    gains = np.array([10.0, 100.0, 1000.0])
    f0 = 4000.0
    w0 = 2.0 * np.pi * f0
    B_true = np.array([3e-5 + 1e-5j, -2e-5 + 4e-5j, 5e-6 - 2e-6j])[:, None]
    # Physics exp(-iwt): V_coil = -i * w * A @ B, digitized V = diag(gains) @ V_coil
    V_coil = -1j * w0 * (A @ B_true)
    V_scope = gains[:, None] * V_coil

    # Both calling conventions match: real-level V_coil (gain=1) and per-channel gain
    inv_g = bdot.inverse_area_matrix(A, gain=gains)
    np.testing.assert_allclose(bdot.voltage_to_bdot(V_scope, inv_g), -1j * w0 * B_true, rtol=1e-12)
    B_rec_m = bdot.phasor_voltage_to_b(V_coil, f0, A, convention="exp_minus_iwt")
    np.testing.assert_allclose(B_rec_m, B_true, rtol=1e-12)
    B_rec_p = bdot.phasor_voltage_to_b(+1j * w0 * (A @ B_true), f0, A, convention="exp_plus_iwt")
    np.testing.assert_allclose(B_rec_p, B_true, rtol=1e-12)

    # Spectrum calibration with DC bin -> NaN (default convention='exp_plus_iwt' matches FFT/STFT)
    freqs = np.array([0.0, 100e3, 200e3, 400e3])
    B_spec = np.broadcast_to(B_true, (3, freqs.size)).copy()
    V_spec_p = +1j * (2.0 * np.pi * freqs)[None, :] * np.einsum("ij,jf->if", A, B_spec)
    B_out_p = bdot.calibrate_bdot_spectrum(freqs, V_spec_p, A, axis=-1, mode="static")
    assert np.all(np.isnan(B_out_p[:, 0]))
    np.testing.assert_allclose(B_out_p[:, 1:], B_spec[:, 1:], rtol=1e-12)
    V_spec_m = -1j * (2.0 * np.pi * freqs)[None, :] * np.einsum("ij,jf->if", A, B_spec)
    B_out_m = bdot.calibrate_bdot_spectrum(freqs, V_spec_m, A, axis=-1, convention="exp_minus_iwt", mode="static")
    assert np.all(np.isnan(B_out_m[:, 0]))
    np.testing.assert_allclose(B_out_m[:, 1:], B_spec[:, 1:], rtol=1e-12)


def test_interpolated_phase_convention_and_rfft_spectrum():
    """End-to-end time-domain check of complex VNA phase in both conventions & default rfft spectrum."""
    bench = bdot.HelmholtzBench()
    k0 = bench.scale_m2_hz
    # Build a synthetic VNA sweep with non-zero coil phase lag (1-pole roll-off at 500 kHz)
    f_vna = np.linspace(10e3, 2.5e6, 500)
    A0 = np.array([
        [1.15e-4, 2.0e-6, 1.2e-5],
        [-3.0e-6, 9.8e-5, -5.0e-6],
        [4.0e-6, -1.0e-5, 8.7e-5],
    ])
    # VNA measures H(f) in exp(+j*w*t): low-pass coil phase lag -> A_vna(f) = A0 / (1 + j*f/f_c)
    pole = 1.0 / (1.0 + 1j * (f_vna / 500e3))
    H_vna = 1j * (A0[:, :, None] / k0) * f_vna[None, None, :] * pole[None, None, :]
    cal = bdot.fit_effective_area_matrix(
        f_vna, H_vna, fit_band_hz=(150e3, 500e3), bench=bench, probe_id="SYN",
    )

    # Verify area_at conjugates under exp_minus_iwt
    f_test = 400e3
    A_plus = cal.area_at(f_test, mode="interpolated", convention="exp_plus_iwt")
    A_minus = cal.area_at(f_test, mode="interpolated", convention="exp_minus_iwt")
    assert np.max(np.abs(A_plus.imag)) > 0.3 * np.max(np.abs(A_plus.real))  # strong phase shift (~39 deg)
    np.testing.assert_allclose(A_minus, np.conj(A_plus), rtol=1e-14)

    # Synthesize real time-domain B(t) and physical coil voltage V(t) at an exact FFT bin (400 kHz)
    dt = 200e-9   # fs = 5 MHz -> rfft spans 0 .. 2.5 MHz (inside VNA band for f > 0? start at 10 kHz)
    nt = 500      # df = 10 kHz -> positive bins 10 kHz .. 2.5 MHz match f_vna exactly
    t = np.arange(nt) * dt
    k0_bin = int(round(f_test * nt * dt))  # bin 40 -> 400 kHz
    assert np.isclose(np.fft.rfftfreq(nt, dt)[k0_bin], f_test)

    B_plus = np.array([3e-4 * np.exp(0.4j), -2e-4 * np.exp(-0.9j), 1.5e-4 * np.exp(1.2j)])
    w0 = 2.0 * np.pi * f_test
    # True physical waveforms: x(t) = Re[X_plus * exp(+j*w*t)]
    B_t = np.real(B_plus[:, None] * np.exp(1j * w0 * t[None, :]))
    V_plus = (1j * w0) * (A_plus @ B_plus)
    V_t = np.real(V_plus[:, None] * np.exp(1j * w0 * t[None, :]))

    # 1. Single-frequency phasors extracted directly from V_t in BOTH conventions
    V_ph_plus = fields.phasor(V_t, dt, f_test, window=None)
    B_ph_plus = fields.phasor(B_t, dt, f_test, window=None)
    np.testing.assert_allclose(
        bdot.phasor_voltage_to_b(V_ph_plus, f_test, cal, convention="exp_plus_iwt", mode="interpolated"),
        B_ph_plus, rtol=1e-10,
    )
    V_ph_minus = np.conj(V_ph_plus)
    B_ph_minus = np.conj(B_ph_plus)
    np.testing.assert_allclose(
        bdot.phasor_voltage_to_b(V_ph_minus, f_test, cal, convention="exp_minus_iwt", mode="interpolated"),
        B_ph_minus, rtol=1e-10,
    )

    # 2. Broadband rfft spectrum with default arguments (exp_plus_iwt, interpolated)
    freqs_rfft = np.fft.rfftfreq(nt, dt)
    V_rfft = np.fft.rfft(V_t, axis=-1)
    B_rfft_true = np.fft.rfft(B_t, axis=-1)
    B_rfft_rec = bdot.calibrate_bdot_spectrum(freqs_rfft, V_rfft, cal)
    assert np.all(np.isnan(B_rfft_rec[:, 0]))
    np.testing.assert_allclose(B_rfft_rec[:, k0_bin], B_rfft_true[:, k0_bin], rtol=1e-10)

    # 3. Conjugated physics-convention spectrum in interpolated mode
    B_rfft_minus = bdot.calibrate_bdot_spectrum(
        freqs_rfft, np.conj(V_rfft), cal, convention="exp_minus_iwt", mode="interpolated",
    )
    np.testing.assert_allclose(B_rfft_minus[:, k0_bin], np.conj(B_rfft_true[:, k0_bin]), rtol=1e-10)


def test_e5100a_csv_binary_dir_and_parse_bdot_gain(tmp_path):
    # Tab-delimited LINMP, LOGMP, REIM parsing
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
    assert rec["metadata"]["NUMBER OF POINTS"] == "3"

    log_p = tmp_path / "log.csv"
    log_p.write_text('"FORMAT TYPE: LOGMP"\n1e5,-20.0,-90.0\n2e5,0.0,90.0\n', encoding="utf-8")
    rec_log = read_e5100a_csv(log_p)
    np.testing.assert_allclose(rec_log["complex_response"], [-0.1j, 1.0j], atol=1e-12)

    # Directory loader: B<drive>P<probe> -> H_vna[probe, drive]
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
    assert not vna["is_diagonal_only"]
    assert vna["H_vna"].shape == (3, 3, 2)
    # BXPZ (drive=0, probe=2) must sit at H_vna[2, 0] (val = 31)
    assert vna["H_vna"][2, 0, 0] == pytest.approx(31.0j, abs=1e-12)
    # BZPX (drive=2, probe=0) must sit at H_vna[0, 2] (val = 13)
    assert vna["H_vna"][0, 2, 0] == pytest.approx(13.0j, abs=1e-12)

    # Binary E5100A round-trip
    bin_p = tmp_path / "sample.dat"
    hdr = struct.pack("=dd", 10000.0, 180000000.0) + b"\x00" * (0x126 - 16)
    def _pack_rec(vals):
        return struct.pack("=LHL", 0, len(vals), 0) + struct.pack(f"={len(vals)}d", *vals)
    bin_p.write_bytes(hdr + _pack_rec([1.0, 2.0]) + _pack_rec([0.0, 0.0]) + _pack_rec([3.0, 4.0]) + _pack_rec([0.0, 0.0]))
    brec = read_e5100a_binary(bin_p, freq_hz=[1e5, 2e5], format_type="REIM")
    np.testing.assert_allclose(brec["complex_response"], [1.0 + 3.0j, 2.0 + 4.0j])

    # parse_bdot_gain
    desc = (
        "    - Probe\n"
        "      - (NOT USED) P20: Bdot-old\n"
        "        - Connection: X5\n"
        "      - P28: Bdot-10T-C16\n"
        "        - Connection: 2.5BLP => X10 => 2.5BLP =>X100\n"
        "      - P29: LP\n"
    )
    parsed = parse_bdot_gain(desc)
    assert parsed is not None
    assert parsed["header"] == "P28: Bdot-10T-C16"
    assert parsed["gain"] == pytest.approx(1000.0)
    assert parsed["atten_db"] == pytest.approx(0.0)
    assert parsed["effective_gain"] == pytest.approx(1000.0)
    parsed_pad = parse_bdot_gain("- P28: Bdot\n  - Connection: X100 => 20dB\n")
    assert parsed_pad["effective_gain"] == pytest.approx(10.0)
    assert parse_bdot_gain("- P29: LP only\n") is None

    # Multiple active B-dot probes: ambiguous without probe=, unambiguous with probe=
    multi_desc = (
        "- P30: Bdot-5T-B2\n"
        "  - Connection: X10\n"
        "- P28: Bdot-10T-C16\n"
        "  - Connection: 2.5BLP => X10 => 2.5BLP => X100\n"
    )
    with pytest.raises(ValueError):
        parse_bdot_gain(multi_desc)
    p28 = parse_bdot_gain(multi_desc, probe="P28")
    p30 = parse_bdot_gain(multi_desc, probe="P30")
    assert p28["effective_gain"] == pytest.approx(1000.0)
    assert p30["effective_gain"] == pytest.approx(10.0)
    assert parse_bdot_gain(multi_desc, probe="P2") is None
    assert parse_bdot_gain(multi_desc, probe="P99") is None


# ---------------------------------------------------------------------------
# fields: curl / J
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rod():
    """B_phi of an infinite uniform current rod (radius a, current I) on a (z, y, x) grid."""
    h = 2e-3
    x = np.arange(-30, 31) * h
    z = np.arange(-5, 6) * h
    Z, Y, X = np.meshgrid(z, x, x, indexing="ij")
    R = np.hypot(X, Y)
    a, I = 0.03, 1000.0
    Bphi = np.where(R < a, mu_0 * I * R / (2 * np.pi * a**2), mu_0 * I / (2 * np.pi * np.maximum(R, 1e-12)))
    with np.errstate(invalid="ignore", divide="ignore"):
        Bx, By = np.nan_to_num(-Bphi * Y / R), np.nan_to_num(Bphi * X / R)
    return h, R, a, I / (np.pi * a**2), Bx, By


def test_current_density_rod(rod):
    h, R, a, J0, Bx, By = rod
    _, _, Jz = fields.current_density(Bx, By, 0, h, h, h)
    assert np.max(np.abs(Jz[R < a - 2 * h] - J0)) / J0 < 1e-9
    # outside: zero up to the finite-difference error of a 1/r field, ~ (h/r)^2
    assert np.max(np.abs(Jz[R > a + 4 * h])) / J0 < 5e-3


def test_current_density_plane_and_complex(rod):
    h, R, a, J0, Bx, By = rod
    ph = np.exp(0.7j)                                        # phasor map stays complex
    _, _, Jz = fields.current_density(Bx[5] * ph, By[5] * ph, 0, h, h, None, axes=(-1, -2, None))
    assert np.iscomplexobj(Jz)
    assert np.max(np.abs(Jz[R[5] < a - 2 * h] - J0 * ph)) / J0 < 1e-9


def test_poynting_and_power_density():
    Sx, Sy, Sz = fields.poynting(1.0, 0, 0, 0, 1e-3, 0)      # x-hat cross y-hat = z-hat
    assert (Sx, Sy) == (0, 0) and Sz == pytest.approx(1e-3 / mu_0)
    assert fields.power_density(2, 0, 0, 3, 5, 0) == 6


# ---------------------------------------------------------------------------
# fields: Faraday E_phi
# ---------------------------------------------------------------------------

def test_faraday_ephi_uniform_signed_radii():
    r = np.arange(-20, 21) * 1e-2 + 0.005                    # crosses the axis, no r = 0 point
    E = fields.faraday_ephi(np.full((3, r.size), 7.0), r)
    np.testing.assert_allclose(E, np.broadcast_to(-np.abs(r) * 7.0 / 2, (3, r.size)), rtol=1e-12)
    E0 = fields.faraday_ephi(np.full((r.size, 4), 7.0), r, axis=0)
    np.testing.assert_allclose(E0, np.broadcast_to((-np.abs(r) * 7.0 / 2)[:, None], (r.size, 4)), rtol=1e-12)


def test_faraday_ephi_r2_profile():
    r = np.linspace(0, 0.2, 2001)                            # dBz/dt = c r^2 -> E = -c r^3 / 4
    assert rel(fields.faraday_ephi(5.0 * r**2, r), -5.0 * r**3 / 4) < 1e-5


# ---------------------------------------------------------------------------
# fields: vector potential
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ball():
    """Jz uniform inside a sphere: Az = mu0 J (3a^2 - r^2)/6 inside, mu0 J a^3/(3r) outside."""
    h, n, a, J0 = 2e-3, 41, 0.03, 1e5
    c = (np.arange(n) - n // 2) * h
    Z, Y, X = np.meshgrid(c, c, c, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Jz = np.where(R < a, J0, 0.0)
    Az_true = np.where(R < a, mu_0 * J0 * (3 * a**2 - R**2) / 6, mu_0 * J0 * a**3 / (3 * np.maximum(R, 1e-12)))
    Ax, Ay, Az = fields.vector_potential(0 * Jz, 0 * Jz, Jz, h, h, h)
    return h, R, a, J0, Jz, Ax, Ay, Az, Az_true


def test_vector_potential_ball(ball):
    h, R, a, J0, Jz, Ax, Ay, Az, Az_true = ball
    assert rel(Az, Az_true) < 2e-2                           # staircase sphere surface
    assert not Ax.any() and not Ay.any()


def test_vector_potential_poisson(ball):
    h, R, a, J0, Jz, Ax, Ay, Az, Az_true = ball
    lap = (sum(np.roll(Az, s, ax) for ax in range(3) for s in (1, -1)) - 6 * Az) / h**2
    core = R < a - 3 * h
    assert np.max(np.abs(lap[core] + mu_0 * J0)) / (mu_0 * J0) < 2e-2


def test_vector_potential_time_axis_and_complex(ball):
    h, R, a, J0, Jz, Ax, Ay, Az, Az_true = ball
    Jt = np.stack([Jz, 2 * Jz], axis=-1)                     # (nz, ny, nx, nt)
    _, _, Azt = fields.vector_potential(0, 0, Jt, h, h, h, axes=(-2, -3, -4))
    np.testing.assert_allclose(Azt[..., 1], 2 * Az, rtol=1e-10, atol=1e-20)
    _, _, Azc = fields.vector_potential(0, 0, 1j * Jz, h, h, h)
    np.testing.assert_allclose(Azc, 1j * Az, rtol=1e-10, atol=1e-20)


# ---------------------------------------------------------------------------
# fields: cycle averages, volume integrals, AC response
# ---------------------------------------------------------------------------

def test_cycle_means_and_average():
    P = 2 + np.sin(2 * np.pi * np.arange(5000) / 333.3)
    assert np.max(np.abs(fields.cycle_means(P[None, :], 333.3, start=10) - 2)) < 5e-3
    assert fields.cycle_average(P, 333.3, 0, 3) == pytest.approx(2, abs=5e-3)
    with pytest.raises(ValueError):
        fields.cycle_average(P, 333.3, 0, 100)


def test_axisymmetric_integral_cylinder():
    r, z = np.linspace(0, 0.1, 401), np.linspace(0, 0.3, 301)
    vol = np.pi * 0.1**2 * 0.3
    assert fields.axisymmetric_integral(np.ones((z.size, r.size)), r, z) == pytest.approx(vol, rel=1e-9)
    with pytest.raises(ValueError):
        fields.axisymmetric_integral(np.ones(5), np.linspace(-1, 1, 5))


def test_ac_conductivity_drude():
    nu, w = 3e7, 2 * np.pi * F0
    wpe2 = 1e17 * 1.602e-19**2 / (9.109e-31 * epsilon_0)
    sigma = epsilon_0 * wpe2 / (nu + 1j * w)                  # J lags E: negative phase
    t = np.arange(3333) * DT                                 # not a whole number of cycles
    E = 50 * np.cos(w * t + 0.3)
    J = np.real(sigma * 50 * np.exp(1j * (w * t + 0.3)))
    s = fields.ac_conductivity(J, E, DT, F0)
    assert abs(s - sigma) / abs(sigma) < 1e-3
    s2 = fields.ac_conductivity(np.stack([J, J]), np.stack([E, 0 * E]), DT, F0, min_E=1.0)
    assert np.isfinite(s2[0]) and np.isnan(s2[1])
