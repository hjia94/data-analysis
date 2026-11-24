"""Bicoherence / bispectrum on signals whose phase coupling is known exactly.

Frequencies sit on FFT bins (dt = 1 us, nperseg = 256 -> df = 3906.25 Hz), so a
Hann-windowed segment spectrum of a sinusoid of amplitude ``a`` is exactly
``a e^{i phi}`` at its bin.
"""

import numpy as np
import pytest

from data_analysis.signal import bicoherence, bicoherence_threshold, cross_bicoherence

DT = 1e-6
NSEG = 256
DF = 1 / (NSEG * DT)
K1, K2 = 20, 33
F1, F2, F3 = K1 * DF, K2 * DF, (K1 + K2) * DF


def tone(f, phi, nt=NSEG, a=1.0):
    """(nshot, nt) cosines, one phase per shot."""
    t = np.arange(nt) * DT
    return a * np.cos(2 * np.pi * f * t + np.asarray(phi)[:, None])


def at(res, f1, f2):
    return (np.argmin(abs(res.f1 - f1)), np.argmin(abs(res.f2 - f2)))


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def test_coupled_triad_auto(rng):
    n = 200
    p1, p2, theta = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n), 0.7
    a1, a2, a3 = 1.0, 0.8, 0.5
    x = tone(F1, p1, a=a1) + tone(F2, p2, a=a2) + tone(F3, p1 + p2 + theta, a=a3)
    r = bicoherence(x, DT, NSEG)
    i = at(r, F1, F2)
    assert r.n_avg == n
    assert r.b2[i] == pytest.approx(1, abs=1e-10)
    assert abs(r.bispectrum[i]) == pytest.approx(a1 * a2 * a3, rel=1e-10)
    assert np.angle(r.bispectrum[i]) == pytest.approx(-theta, abs=1e-10)   # biphase
    # symmetric in f1 <-> f2
    assert r.b2[at(r, F2, F1)] == pytest.approx(1, abs=1e-10)


def test_random_phase_triad_is_at_noise_level(rng):
    n = 400
    p1, p2, p3 = (rng.uniform(0, 2 * np.pi, n) for _ in range(3))
    x = tone(F1, p1) + tone(F2, p2) + tone(F3, p3)
    r = bicoherence(x, DT, NSEG, f1_band=(F1, F1), f2_band=(F2, F2), p=1e-3)
    assert r.b2.shape == (1, 1)
    assert r.b2[0, 0] < r.threshold                               # 1-in-1000 chance level
    # same power spectrum, but coupled -> b2 = 1: only the bispectrum separates them
    r2 = bicoherence(tone(F1, p1) + tone(F2, p2) + tone(F3, p1 + p2), DT, NSEG,
                     f1_band=(F1, F1), f2_band=(F2, F2))
    assert r2.b2[0, 0] == pytest.approx(1, abs=1e-10)


def test_noise_statistics_and_threshold(rng):
    """Gaussian noise: mean b2 = 1/N, P(b2 > threshold) = p."""
    x = rng.standard_normal((50, NSEG * 4))                       # 200 segments
    r = bicoherence(x, DT, NSEG, f1_band=(10 * DF, 60 * DF), f2_band=(5 * DF, 50 * DF))
    assert r.n_avg == 200
    b2 = r.b2[np.isfinite(r.b2)]
    assert b2.mean() == pytest.approx(1 / r.n_avg, rel=0.1)
    assert np.mean(b2 > r.threshold) == pytest.approx(0.05, abs=0.015)
    assert r.threshold == pytest.approx(1 - 0.05 ** (1 / 199))


def test_long_record_segments_and_coupling_in_noise(rng):
    """One 1-D record with fixed phases: segments are the realizations."""
    nt = NSEG * 300
    t = np.arange(nt) * DT
    # A random phase walk per segment keeps the triad coupled but kills the
    # coherence between f1 and f2 themselves.
    walk = np.repeat(rng.uniform(0, 2 * np.pi, (2, 300)), NSEG, axis=1)
    x = (np.cos(2 * np.pi * F1 * t + walk[0]) + np.cos(2 * np.pi * F2 * t + walk[1])
         + 0.5 * np.cos(2 * np.pi * F3 * t + walk[0] + walk[1])
         + 0.5 * rng.standard_normal(nt))
    r = bicoherence(x, DT, NSEG)
    assert r.n_avg == 300
    i = at(r, F1, F2)
    assert r.b2[i] > 0.5
    # Away from the triad and its mirror (Hann leakage couples the +-1 bins too)
    # b2 is at the no-coupling level.
    k1, k2 = np.meshgrid(np.rint(r.f1 / DF), np.rint(r.f2 / DF), indexing="ij")
    near = lambda a, b: (abs(k1 - a) <= 2) & (abs(k2 - b) <= 2)
    off = r.b2[~(near(K1, K2) | near(K2, K1)) & np.isfinite(r.b2)]
    assert off.mean() == pytest.approx(1 / r.n_avg, rel=0.15)
    assert off.max() < 0.1


def test_cross_bicoherence_three_channels(rng):
    n = 200
    p1, p2 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    noise = lambda: 0.1 * rng.standard_normal((n, NSEG))
    x, y, z = tone(F1, p1) + noise(), tone(F2, p2) + noise(), tone(F3, p1 + p2) + noise()
    band = dict(f1_band=(F1 - 3 * DF, F1 + 3 * DF), f2_band=(F2 - 3 * DF, F2 + 3 * DF))
    r = cross_bicoherence(x, y, z, DT, NSEG, **band)
    assert r.b2[at(r, F1, F2)] > 0.95
    # each channel alone has no triad
    assert np.nanmax(bicoherence(x, DT, NSEG, **band).b2) < 0.1


def test_difference_triad_with_negative_f2(rng):
    """Pump at f2 on y, low frequency f1 on x, lower sideband f2 - f1 on z."""
    n = 200
    p1, p2 = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
    x, y, z = tone(F1, p1), tone(F2, p2), tone(F2 - F1, p2 - p1)
    r = cross_bicoherence(y, x, z, DT, NSEG, f1_band=(F2, F2), f2_band=(-F1, -F1))
    assert r.f2[0] == pytest.approx(-F1)
    assert r.b2[0, 0] == pytest.approx(1, abs=1e-10)


def test_axes_nyquist_and_inputs(rng):
    x = rng.standard_normal((10, NSEG))
    r = bicoherence(x, DT, NSEG)
    np.testing.assert_allclose(r.f1, np.arange(1, NSEG // 2 + 1) * DF)
    assert r.b2.shape == r.bispectrum.shape == (NSEG // 2, NSEG // 2)
    over = (r.f1[:, None] + r.f2[None, :]) > NSEG // 2 * DF
    assert np.isnan(r.b2[over]).all() and np.isfinite(r.b2[~over]).all()

    x[3, 5] = np.nan                                              # bad shot is dropped
    assert bicoherence(x, DT, NSEG).n_avg == 9
    with pytest.raises(ValueError, match="selects no frequency bin"):
        bicoherence(x, DT, NSEG, f1_band=(1.2 * DF, 1.8 * DF))
    with pytest.raises(ValueError, match="non-negative"):
        bicoherence(x, DT, NSEG, f1_band=(-DF, DF))
    with pytest.raises(ValueError, match="shapes differ"):
        cross_bicoherence(x, x, x[:, :-1], DT, 128)
    with pytest.raises(ValueError, match="longer than the record"):
        bicoherence(x, DT, 2 * NSEG)
    with pytest.raises(ValueError, match="at least 2"):
        bicoherence_threshold(1)


def test_block_accumulation_matches_single_block(rng, monkeypatch):
    from data_analysis.signal import bispectral
    x = rng.standard_normal((30, NSEG * 2))
    kw = dict(f1_band=(DF, 40 * DF), f2_band=(DF, 40 * DF))
    ref = bicoherence(x, DT, NSEG, **kw)
    monkeypatch.setattr(bispectral, "_BLOCK_ELEMENTS", 1)        # one segment per block
    r = bicoherence(x, DT, NSEG, **kw)
    np.testing.assert_allclose(r.b2, ref.b2, rtol=1e-10)
    np.testing.assert_allclose(r.bispectrum, ref.bispectrum, rtol=1e-10, atol=1e-14)
