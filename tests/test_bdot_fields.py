"""Analytic checks for data_analysis.plasma.bdot and data_analysis.plasma.fields.

Every case has a closed-form answer, so these test the maths, not a fit:
a calibrated 3-axis probe round trip, the three integration baselines, the
field of a uniform current rod (curl/J), uniform and r^2 dBz/dt (Faraday
E_phi), a uniformly filled current ball (Biot-Savart A and Poisson), a Drude
conductivity, and cylinder volumes.

    pip install -e ".[test]"
    pytest tests/
"""

import numpy as np
import pytest
from scipy.constants import epsilon_0, mu_0

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
