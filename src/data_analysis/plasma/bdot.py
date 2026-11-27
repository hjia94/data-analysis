"""B-dot probe maths -- Helmholtz bench calibration, voltage to dB/dt, and B integration.

Everything here operates at **real probe-terminal signal level** and SI units
(areas in m^2, dB/dt in T/s, B in tesla -- multiply by 1e4 for gauss).
Bench and experimental amplifiers / attenuators are experiment-specific and
should be divided out by the caller before calling these helpers (and recorded
explicitly in campaign / calibration provenance); ``gain=1.0`` is retained on
:func:`inverse_area_matrix` and :func:`phasor_voltage_to_b` for convenience
and backward compatibility.

A B-dot coil of effective area ``A`` (turns x loop area) reads
``V_coil = A * dB/dt``. A three-axis probe has cross-talk between its coils,
so the general form is a signed 3x3 calibration matrix::

    [V1]   [A11 A12 A13] [dB1/dt]                  [dB1/dt]         [V1]
    [V2] = [A21 A22 A23] [dB2/dt]      hence       [dB2/dt] = A^-1  [V2]
    [V3]   [A31 A32 A33] [dB3/dt]                  [dB3/dt]         [V3]

Bench calibration (green Helmholtz coil, ``nACalculationRef.pdf``):
    :class:`HelmholtzBench` holds bare bench geometry (radius ``a``, turns
    ``N_H``, reference shunt ``R_shunt`` and divider ``d_ref``, no amplifier).
    :func:`helmholtz_effective_area` converts real probe-level transfer ratio
    ``H_coil(f) = V_coil / V_R`` to complex ``A_eff(f) [m^2]``, and
    :func:`fit_effective_area_matrix` extracts the signed 3x3 matrix
    :class:`BdotCalibration` by zero-intercept linear fit on the clean linear
    inductive band. :func:`save_bdot_calibration` / :func:`load_bdot_calibration`
    persist calibrations to ``.npz`` (and load standalone ``nAcalc.py``
    ``<probe>_calibration.npz`` files directly).

Application to plasma data:
    :func:`inverse_area_matrix` builds ``A^-1`` (or ``(1/G) A^-1``),
    :func:`voltage_to_bdot` converts time-domain or phasor voltages to dB/dt,
    :func:`integrate_bdot` integrates dB/dt in time with drift removal,
    :func:`phasor_voltage_to_b` converts single-frequency voltage phasors to B,
    and :func:`calibrate_bdot_spectrum` calibrates broadband FFT/STFT spectra.

Components stay in the **probe frame**; rotating them into chamber coordinates
depends on how the probe enters the machine and is left to the campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import numpy as np
from scipy import integrate, signal
from scipy.constants import mu_0

_AXES = ("X", "Y", "Z")


# ---------------------------------------------------------------------------
# Helmholtz bench calibration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HelmholtzBench:
    """Bare Helmholtz calibration-bench geometry (no bench amplifier baked in).

    Defaults match the UCLA LAPD green Helmholtz coil setup
    (``nACalculationRef.pdf``): radius ``a = 5.5 cm``, ``1`` turn per coil,
    ``1/16 Ohm`` current shunt tapped through a ``50 / 50 Ohm`` divider
    (``V_R = I / 32``), giving ``scale_m2_hz = 304.220166 m^2 * Hz``.
    """

    coil_radius_m: float = 0.055
    coil_turns: int = 1
    shunt_ohm: float = 1.0 / 16.0
    ref_divider: float = 0.5

    def __post_init__(self):
        if self.coil_radius_m <= 0:
            raise ValueError(f"coil_radius_m must be > 0, got {self.coil_radius_m}")
        if self.coil_turns <= 0:
            raise ValueError(f"coil_turns must be > 0, got {self.coil_turns}")
        if self.shunt_ohm <= 0:
            raise ValueError(f"shunt_ohm must be > 0, got {self.shunt_ohm}")
        if self.ref_divider <= 0:
            raise ValueError(f"ref_divider must be > 0, got {self.ref_divider}")

    @property
    def scale_m2_hz(self) -> float:
        """Bare-bench factor ``K_0 [m^2 * Hz]`` in ``A_eff(f) = K_0 * H_coil(f) / (i*f)``."""
        return float(
            (self.coil_radius_m * self.shunt_ohm * self.ref_divider)
            / ((4.0 / 5.0) ** 1.5 * mu_0 * self.coil_turns * 2.0 * np.pi)
        )


@dataclass
class BdotCalibration:
    """Bench-calibrated bare 3-axis B-dot probe effective area (SI: m^2).

    ``area_matrix_m2`` has shape ``(3, 3)`` with convention
    ``V_coil = area_matrix_m2 @ dB/dt`` (rows = probe axes ``Px, Py, Pz``,
    columns = driven field axes ``Bx, By, Bz``; file ``B<drive>P<probe>``
    populates ``area_matrix_m2[probe_idx, drive_idx]``).
    """

    probe_id: str
    area_matrix_m2: np.ndarray
    r2_matrix: np.ndarray
    phase_deg_matrix: np.ndarray
    phase_R_matrix: np.ndarray
    diag_band_hz: tuple[float, float]
    offdiag_band_hz: tuple[float, float]
    is_diagonal_only: bool = False
    bench: HelmholtzBench = field(default_factory=HelmholtzBench)
    quadrature_ratio_matrix: np.ndarray | None = None
    freq_hz: np.ndarray | None = None
    area_complex_m2: np.ndarray | None = None
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.area_matrix_m2 = np.asarray(self.area_matrix_m2, dtype=float)
        if self.area_matrix_m2.shape != (3, 3):
            raise ValueError(f"area_matrix_m2 must have shape (3, 3), got {self.area_matrix_m2.shape}")
        self.r2_matrix = np.asarray(self.r2_matrix, dtype=float)
        self.phase_deg_matrix = np.asarray(self.phase_deg_matrix, dtype=float)
        self.phase_R_matrix = np.asarray(self.phase_R_matrix, dtype=float)
        self.diag_band_hz = (float(self.diag_band_hz[0]), float(self.diag_band_hz[1]))
        self.offdiag_band_hz = (float(self.offdiag_band_hz[0]), float(self.offdiag_band_hz[1]))
        if self.quadrature_ratio_matrix is not None:
            self.quadrature_ratio_matrix = np.asarray(self.quadrature_ratio_matrix, dtype=float)
        if self.freq_hz is not None:
            self.freq_hz = np.asarray(self.freq_hz, dtype=float)
        if self.area_complex_m2 is not None:
            self.area_complex_m2 = np.asarray(self.area_complex_m2, dtype=complex)

    def area_at(self, freq_hz=None, *, mode="static") -> np.ndarray:
        """Return effective area matrix [m^2] in ``'static'`` or ``'interpolated'`` mode.

        - ``mode='static'``: returns the fitted signed ``(3, 3)`` real matrix
          ``area_matrix_m2`` (extrapolated linearly down to DC).
        - ``mode='interpolated'``: linearly interpolates ``area_complex_m2`` at
          scalar ``freq_hz`` -> ``(3, 3)`` complex, or 1-D ``freq_hz`` ->
          ``(3, 3, nf)`` complex. Raises ``ValueError`` if ``freq_hz`` lies
          outside the measured VNA sweep or if complex sweeps were not stored.
        """
        if mode == "static":
            return self.area_matrix_m2.copy()
        if mode != "interpolated":
            raise ValueError(f"mode must be 'static' or 'interpolated', got {mode!r}")
        if freq_hz is None:
            raise ValueError("mode='interpolated' requires freq_hz")
        if self.freq_hz is None or self.area_complex_m2 is None:
            raise ValueError(
                "This BdotCalibration does not carry full frequency sweeps "
                "(freq_hz / area_complex_m2); use mode='static'."
            )

        f_req = np.asarray(freq_hz, dtype=float)
        scalar = f_req.ndim == 0
        f_1d = np.atleast_1d(f_req)
        if f_1d.ndim != 1:
            raise ValueError(f"freq_hz must be a scalar or 1-D array, got shape {f_req.shape}")
        f_lo, f_hi = float(self.freq_hz[0]), float(self.freq_hz[-1])
        if np.any((f_1d < f_lo) | (f_1d > f_hi)):
            raise ValueError(
                f"Requested freq_hz [{f_1d.min():g}, {f_1d.max():g}] Hz lies outside "
                f"measured VNA band [{f_lo:g}, {f_hi:g}] Hz; use mode='static' to "
                "extrapolate the low-frequency linear fit."
            )

        out = np.zeros((3, 3, f_1d.size), dtype=complex)
        for i in range(3):
            for j in range(3):
                curve = self.area_complex_m2[i, j]
                finite = np.isfinite(curve.real) & np.isfinite(curve.imag)
                if not np.any(finite):
                    continue
                re = np.interp(f_1d, self.freq_hz[finite], curve.real[finite])
                im = np.interp(f_1d, self.freq_hz[finite], curve.imag[finite])
                out[i, j] = re + 1j * im
        return out[..., 0] if scalar else out


def helmholtz_effective_area(freq_hz, H_coil, bench=HelmholtzBench()):
    """Point-by-point complex effective area ``A_eff(f) [m^2]`` from real probe-level ``H_coil``.

    ``H_coil = V_coil / V_R`` must already have any bench amplifier gain and
    attenuators removed. Dividing by ``i * f`` maps an ideal ``+90 deg``
    inductive pickup to the positive real axis and a reversed ``-90 deg``
    winding to the negative real axis.
    """
    f = np.asarray(freq_hz, dtype=float)
    if np.any(f <= 0):
        raise ValueError("freq_hz must be strictly positive")
    H = np.asarray(H_coil, dtype=complex)
    return (bench.scale_m2_hz * H) / (1j * f)


def fit_effective_area_matrix(
    freq_hz,
    H_coil_3x3,
    fit_band_hz,
    *,
    offdiag_band_hz=None,
    bench=HelmholtzBench(),
    min_phase_R=0.9,
    zero_low_coherence=False,
    probe_id="",
    provenance=None,
) -> BdotCalibration:
    """Fit a signed 3x3 :class:`BdotCalibration` from real probe-level ``H_coil_3x3``.

    Parameters
    ----------
    freq_hz : (nf,) array_like
        Frequency axis [Hz], strictly positive and ascending.
    H_coil_3x3 : (3, 3, nf) array_like (complex)
        Real probe-terminal transfer matrix ``H_coil[probe_axis, field_axis, f]``
        (bench amplifier gain / attenuators already divided out). Missing
        off-diagonal entries in a diagonal-only calibration may be ``NaN``.
    fit_band_hz : (f_lo, f_hi)
        Clean linear inductive band [Hz] used for the diagonal zero-intercept
        linear fit ``Im[H_coil] = s * f`` (e.g. ``(150e3, 500e3)``).
    offdiag_band_hz : (f_lo, f_hi), optional
        Separate linear fit band [Hz] for off-diagonal entries (defaults to
        ``fit_band_hz``). Useful when small cross-talk terms emerge above the
        low-frequency bench noise floor at higher frequency (e.g. ``(250e3, 1000e3)``).
    bench : HelmholtzBench, optional
        Bare Helmholtz bench geometry.
    min_phase_R : float, optional
        Phase coherence resultant threshold for ``zero_low_coherence`` (default ``0.9``).
    zero_low_coherence : bool, optional
        If True, set off-diagonal entries with ``phase_R < min_phase_R`` to ``0.0``
        (default False, preserving raw fitted values and flagging via ``phase_R_matrix``).
    probe_id : str, optional
        Probe identifier (e.g. ``'C16'``).
    provenance : dict, optional
        Caller-supplied bench metadata (``bench_gain``, ``bench_atten_db``,
        ``source``, ``upmark``, etc.).
    """
    f = np.asarray(freq_hz, dtype=float)
    if f.ndim != 1 or f.size < 2:
        raise ValueError(f"freq_hz must be 1-D with at least 2 points, got shape {f.shape}")
    if np.any(f <= 0) or np.any(np.diff(f) <= 0):
        raise ValueError("freq_hz must be strictly positive and strictly ascending")

    H = np.asarray(H_coil_3x3, dtype=complex)
    if H.shape != (3, 3, f.size):
        raise ValueError(f"H_coil_3x3 must have shape (3, 3, {f.size}), got {H.shape}")

    diag_band = (float(fit_band_hz[0]), float(fit_band_hz[1]))
    off_band = diag_band if offdiag_band_hz is None else (float(offdiag_band_hz[0]), float(offdiag_band_hz[1]))
    for name, (lo, hi) in (("fit_band_hz", diag_band), ("offdiag_band_hz", off_band)):
        if lo >= hi:
            raise ValueError(f"{name} must satisfy f_lo < f_hi, got ({lo}, {hi})")

    k0 = bench.scale_m2_hz
    area_complex = helmholtz_effective_area(f, H, bench=bench)

    area_m2 = np.zeros((3, 3), dtype=float)
    r2_mat = np.full((3, 3), np.nan, dtype=float)
    phase_deg = np.full((3, 3), np.nan, dtype=float)
    phase_R = np.full((3, 3), np.nan, dtype=float)
    quad_ratio = np.full((3, 3), np.nan, dtype=float)

    any_offdiag = False
    for i in range(3):
        for j in range(3):
            h_ij = H[i, j]
            finite = np.isfinite(h_ij.real) & np.isfinite(h_ij.imag)
            if not np.any(finite):
                if i == j:
                    raise ValueError(f"Diagonal component ({i}, {j}) has no finite VNA samples")
                area_m2[i, j] = 0.0
                continue
            if i != j:
                any_offdiag = True

            lo, hi = diag_band if i == j else off_band
            mask = finite & (f >= lo) & (f <= hi)
            if int(np.sum(mask)) < 2:
                raise ValueError(
                    f"Component ({i}, {j}) has fewer than 2 finite samples in "
                    f"[{lo:g}, {hi:g}] Hz (sweep spans [{f[0]:g}, {f[-1]:g}] Hz)"
                )

            fm = f[mask]
            hm = h_ij[mask]
            ym = hm.imag  # signed inductive component: +|H| at +90 deg, -|H| at -90 deg

            # Zero-intercept linear fit through origin: y = slope * f
            slope = float(np.dot(fm, ym) / np.dot(fm, fm))
            ae_ij = slope * k0

            yhat = slope * fm
            ss_res = float(np.sum((ym - yhat) ** 2))
            ss_tot = float(np.sum((ym - ym.mean()) ** 2))
            r2_ij = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else np.nan

            z_mean = np.mean(np.exp(1j * np.angle(hm)))
            ph_deg_ij = float(np.rad2deg(np.angle(z_mean)))
            ph_R_ij = float(np.abs(z_mean))
            im_med = float(np.median(np.abs(hm.imag)))
            re_med = float(np.median(np.abs(hm.real)))
            quad_ij = (re_med / im_med) if im_med > 0.0 else np.nan

            if zero_low_coherence and i != j and ph_R_ij < min_phase_R:
                ae_ij = 0.0

            area_m2[i, j] = ae_ij
            r2_mat[i, j] = r2_ij
            phase_deg[i, j] = ph_deg_ij
            phase_R[i, j] = ph_R_ij
            quad_ratio[i, j] = quad_ij

    return BdotCalibration(
        probe_id=str(probe_id),
        area_matrix_m2=area_m2,
        r2_matrix=r2_mat,
        phase_deg_matrix=phase_deg,
        phase_R_matrix=phase_R,
        quadrature_ratio_matrix=quad_ratio,
        diag_band_hz=diag_band,
        offdiag_band_hz=off_band,
        is_diagonal_only=not any_offdiag,
        bench=bench,
        freq_hz=f.copy(),
        area_complex_m2=area_complex,
        provenance=dict(provenance or {}),
    )


def save_bdot_calibration(path, cal: BdotCalibration):
    """Persist a :class:`BdotCalibration` to a compact ``.npz`` file."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    payload = {
        "probe_id": np.str_(cal.probe_id),
        "area_matrix_m2": np.asarray(cal.area_matrix_m2, dtype=float),
        "r2_matrix": np.asarray(cal.r2_matrix, dtype=float),
        "phase_deg_matrix": np.asarray(cal.phase_deg_matrix, dtype=float),
        "phase_R_matrix": np.asarray(cal.phase_R_matrix, dtype=float),
        "diag_band_hz": np.asarray(cal.diag_band_hz, dtype=float),
        "offdiag_band_hz": np.asarray(cal.offdiag_band_hz, dtype=float),
        "is_diagonal_only": np.bool_(cal.is_diagonal_only),
        "k0_m2_hz": np.float64(cal.bench.scale_m2_hz),
        "bench_coil_radius_m": np.float64(cal.bench.coil_radius_m),
        "bench_coil_turns": np.int64(cal.bench.coil_turns),
        "bench_shunt_ohm": np.float64(cal.bench.shunt_ohm),
        "bench_ref_divider": np.float64(cal.bench.ref_divider),
    }
    if cal.quadrature_ratio_matrix is not None:
        payload["quadrature_ratio_matrix"] = np.asarray(cal.quadrature_ratio_matrix, dtype=float)
    if cal.freq_hz is not None:
        payload["freq_hz"] = np.asarray(cal.freq_hz, dtype=float)
    if cal.area_complex_m2 is not None:
        payload["area_complex_m2"] = np.asarray(cal.area_complex_m2, dtype=complex)

    for k, v in (cal.provenance or {}).items():
        pk = f"prov__{k}"
        if isinstance(v, str):
            payload[pk] = np.str_(v)
        elif isinstance(v, (bool, np.bool_)):
            payload[pk] = np.bool_(v)
        elif isinstance(v, (int, np.integer)):
            payload[pk] = np.int64(v)
        elif isinstance(v, (float, np.floating)):
            payload[pk] = np.float64(v)
        else:
            payload[pk] = np.asarray(v)

    np.savez(path, **payload)
    return path


def load_bdot_calibration(path) -> BdotCalibration:
    """Load a :class:`BdotCalibration` from ``.npz`` (supports both ``data_analysis`` and standalone ``nAcalc.py`` files)."""
    with np.load(os.fspath(path), allow_pickle=False) as d:
        files = set(d.files)
        if "area_matrix_m2" not in files:
            raise ValueError(f"{path}: missing required 'area_matrix_m2' key")

        area_m2 = np.asarray(d["area_matrix_m2"], dtype=float)
        # Standalone nAcalc.py leaves missing off-diagonals as NaN; normalize off-diag NaN to 0
        # while recording is_diagonal_only.
        diag_nan = np.any(~np.isfinite(np.diag(area_m2)))
        if diag_nan:
            raise ValueError(f"{path}: diagonal entries of area_matrix_m2 must be finite")
        off_mask = ~np.eye(3, dtype=bool)
        if "is_diagonal_only" in files:
            is_diag_only = bool(d["is_diagonal_only"])
        else:
            is_diag_only = bool(np.all(~np.isfinite(area_m2[off_mask]) | (area_m2[off_mask] == 0.0)))
        area_clean = np.where(np.isfinite(area_m2), area_m2, 0.0)

        nan3 = np.full((3, 3), np.nan, dtype=float)
        r2_mat = np.asarray(d["r2_matrix"], dtype=float) if "r2_matrix" in files else nan3.copy()
        ph_mat = np.asarray(d["phase_deg_matrix"], dtype=float) if "phase_deg_matrix" in files else nan3.copy()
        ph_R = np.asarray(d["phase_R_matrix"], dtype=float) if "phase_R_matrix" in files else nan3.copy()
        quad = np.asarray(d["quadrature_ratio_matrix"], dtype=float) if "quadrature_ratio_matrix" in files else None

        diag_band = tuple(float(x) for x in d["diag_band_hz"]) if "diag_band_hz" in files else (np.nan, np.nan)
        off_band = tuple(float(x) for x in d["offdiag_band_hz"]) if "offdiag_band_hz" in files else diag_band

        bench = HelmholtzBench(
            coil_radius_m=float(d["bench_coil_radius_m"]) if "bench_coil_radius_m" in files else 0.055,
            coil_turns=int(d["bench_coil_turns"]) if "bench_coil_turns" in files else 1,
            shunt_ohm=float(d["bench_shunt_ohm"]) if "bench_shunt_ohm" in files else 1.0 / 16.0,
            ref_divider=float(d["bench_ref_divider"]) if "bench_ref_divider" in files else 0.5,
        )

        prov = {}
        for k in sorted(files):
            if k.startswith("prov__"):
                v = d[k]
                prov[k[len("prov__"):]] = v.item() if v.ndim == 0 else v
        # Also pick up standalone nAcalc.py bench metadata keys
        for legacy_k in ("bench_gain", "bench_atten_db"):
            if legacy_k in files and legacy_k not in prov:
                prov[legacy_k] = float(d[legacy_k])

        freq_hz = np.asarray(d["freq_hz"], dtype=float) if "freq_hz" in files else None
        area_complex = np.asarray(d["area_complex_m2"], dtype=complex) if "area_complex_m2" in files else None
        probe_id = str(d["probe_id"]) if "probe_id" in files else ""

    return BdotCalibration(
        probe_id=probe_id,
        area_matrix_m2=area_clean,
        r2_matrix=r2_mat,
        phase_deg_matrix=ph_mat,
        phase_R_matrix=ph_R,
        quadrature_ratio_matrix=quad,
        diag_band_hz=diag_band,
        offdiag_band_hz=off_band,
        is_diagonal_only=is_diag_only,
        bench=bench,
        freq_hz=freq_hz,
        area_complex_m2=area_complex,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# Applying calibration: probe voltage -> dB/dt -> B
# ---------------------------------------------------------------------------

def _inexact(x):
    """Return ``x`` as float, or complex if it already is."""
    arr = np.asarray(x)
    return arr.astype(np.result_type(arr, float), copy=False)


def inverse_area_matrix(area, gain=1.0, *, freq_hz=None, mode="static"):
    """Volts -> dB/dt conversion: ``1/(gain*area)`` or ``inv(area) @ diag(1/gain)``.

    ``area`` is the bare effective coil area in **m^2** (multiply mm^2 by 1e-6), as:

    - a :class:`BdotCalibration` (evaluated via ``area.area_at(freq_hz, mode=mode)``);
    - a scalar, for a single coil -> returns a scalar;
    - a length-3 sequence, for three independent coils -> diagonal 3x3;
    - a 3x3 calibration matrix ``A`` (real or complex) with ``V = G @ A @ dB/dt``,
      whose off-diagonal terms are the coil cross-talk -> ``inv(A) @ diag(1/G)``.

    Recommended convention is to divide digitized voltages by the experiment's
    amplifier/attenuator chain to real probe-terminal level before calling
    :func:`voltage_to_bdot` (leaving ``gain=1.0``). Passing a scalar or
    length-3 ``gain`` is supported for convenience and backward compatibility.
    """
    if isinstance(area, BdotCalibration):
        area = area.area_at(freq_hz=freq_hz, mode=mode)

    a = _inexact(area)
    g = np.asarray(gain, dtype=float)
    if a.ndim == 0:
        if g.ndim != 0:
            raise ValueError(f"scalar area requires scalar gain, got gain shape {g.shape}")
        val = 1.0 / (float(g) * a)
        return complex(val) if np.iscomplexobj(a) else float(val)
    if a.shape == (3,):
        a = np.diag(a)
    if a.shape != (3, 3):
        raise ValueError(f"area must be a BdotCalibration, scalar, length 3, or 3x3; got shape {a.shape}")
    inv_a = np.linalg.inv(a)
    if g.ndim == 0:
        return inv_a / float(g)
    if g.shape == (3,):
        # V = diag(g) @ A @ dB/dt  =>  dB/dt = A^-1 @ diag(1/g) @ V
        return inv_a / g[None, :]
    raise ValueError(f"gain must be a scalar or length 3, got shape {g.shape}")


def voltage_to_bdot(v, inv_area):
    """Apply :func:`inverse_area_matrix` to probe voltages (real or complex) -> dB/dt (T/s).

    For a scalar ``inv_area``, ``v`` has any shape and is simply scaled. For a
    3x3 ``inv_area``, ``v`` carries the three coil channels on **axis 0**, shape
    ``(3, ...)`` (e.g. ``(3, npos, nt)``, or ``np.stack([V1, V2, V3])``), and
    the result has the same shape with the three probe-frame dB/dt components
    on axis 0.

    Any DC amplifier offset is still in the result; :func:`integrate_bdot`
    removes it (``baseline='pretrigger'`` or the periodic fits).
    """
    m = _inexact(inv_area)
    v_arr = _inexact(v)
    if m.ndim == 0:
        return v_arr * m
    if m.shape != (3, 3):
        raise ValueError(f"inv_area must be a scalar or 3x3; got shape {m.shape}")
    if v_arr.ndim == 0 or v_arr.shape[0] != 3:
        raise ValueError(f"3x3 inv_area needs the 3 coil channels on axis 0; got shape {v_arr.shape}")
    return np.einsum("ij,j...->i...", m, v_arr)


def phasor_voltage_to_b(
    v_coil_phasor,
    freq_hz,
    area,
    *,
    convention="exp_minus_iwt",
    gain=1.0,
    mode="static",
):
    """Convert single-frequency B-dot voltage phasor(s) -> magnetic-field phasor ``B`` [T].

    ``v_coil_phasor`` should be at real probe-terminal level (shape ``(3, ...)``
    for 3-axis ``area``, or arbitrary shape for scalar ``area``).

    ``convention`` selects the phasor time-dependence:
    - ``'exp_minus_iwt'`` (physics convention, ``V = -i w A B``): ``B = (+i / w) A^-1 V``
    - ``'exp_plus_iwt'`` (:func:`data_analysis.plasma.fields.phasor` convention,
      ``V = +i w A B``): ``B = (-i / w) A^-1 V``
    """
    if convention not in ("exp_minus_iwt", "exp_plus_iwt"):
        raise ValueError(
            f"convention must be 'exp_minus_iwt' or 'exp_plus_iwt', got {convention!r}"
        )
    f = float(freq_hz)
    if f <= 0.0:
        raise ValueError(f"freq_hz must be positive, got {freq_hz}")
    omega = 2.0 * np.pi * f
    inv_area = inverse_area_matrix(area, gain=gain, freq_hz=f, mode=mode)
    bdot_phasor = voltage_to_bdot(v_coil_phasor, inv_area)
    factor = (1j / omega) if convention == "exp_minus_iwt" else (-1j / omega)
    return factor * bdot_phasor


def calibrate_bdot_spectrum(
    freq_hz,
    v_coil_spec,
    cal,
    *,
    axis=-1,
    convention="exp_minus_iwt",
    gain=1.0,
    mode="interpolated",
):
    """Calibrate a broadband voltage spectrum ``V(f)`` -> magnetic-field spectrum ``B(f)`` [T].

    Evaluates ``B(f) = (±i / (2*pi*f)) * A(f)^-1 @ V(f)`` bin-by-bin along
    ``axis``. Zero-frequency (DC) bins are returned as ``NaN``. For 3-axis
    calibrations, ``v_coil_spec`` must carry the three coil channels on axis 0
    (and ``axis`` must not be axis 0).
    """
    if convention not in ("exp_minus_iwt", "exp_plus_iwt"):
        raise ValueError(
            f"convention must be 'exp_minus_iwt' or 'exp_plus_iwt', got {convention!r}"
        )
    f = np.asarray(freq_hz, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"freq_hz must be 1-D, got shape {f.shape}")
    v_arr = _inexact(v_coil_spec)
    ax = axis % v_arr.ndim
    if v_arr.shape[ax] != f.size:
        raise ValueError(
            f"v_coil_spec axis {axis} length ({v_arr.shape[ax]}) != len(freq_hz) ({f.size})"
        )

    pos = f > 0.0
    sign_i = 1j if convention == "exp_minus_iwt" else -1j
    inv_iw = np.full(f.shape, np.nan + 0j, dtype=complex)
    inv_iw[pos] = sign_i / (2.0 * np.pi * f[pos])

    if isinstance(cal, BdotCalibration) and mode == "interpolated":
        if ax == 0 or v_arr.shape[0] != 3:
            raise ValueError(
                f"3-axis spectrum calibration requires 3 channels on axis 0 and freq on axis != 0; "
                f"got shape {v_arr.shape}, axis={axis}"
            )
        g = np.asarray(gain, dtype=float)
        if g.ndim == 0:
            v_scaled = v_arr / float(g)
        elif g.shape == (3,):
            v_scaled = v_arr / g.reshape((3,) + (1,) * (v_arr.ndim - 1))
        else:
            raise ValueError(f"gain must be a scalar or length 3, got shape {g.shape}")

        # Move frequency axis to last, keeping channel axis at 0: (3, ..., nf)
        v_mv = np.moveaxis(v_scaled, ax, -1)
        out_mv = np.full(v_mv.shape, np.nan + 0j, dtype=complex)
        if np.any(pos):
            A_f = cal.area_at(f[pos], mode="interpolated")      # (3, 3, nf_pos)
            # Invert each (3, 3) slice -> (nf_pos, 3, 3) -> (3, 3, nf_pos)
            inv_A_f = np.linalg.inv(np.moveaxis(A_f, -1, 0))
            inv_A_f = np.moveaxis(inv_A_f, 0, -1)
            bdot_pos = np.einsum("ijf,j...f->i...f", inv_A_f, v_mv[..., pos])
            out_mv[..., pos] = bdot_pos * inv_iw[pos]
        return np.moveaxis(out_mv, -1, ax)

    # Static calibration (scalar, (3,), (3, 3), or BdotCalibration in static mode)
    inv_area = inverse_area_matrix(cal, gain=gain, mode="static")
    bdot_spec = voltage_to_bdot(v_arr, inv_area)
    bdot_mv = np.moveaxis(bdot_spec, ax, -1)
    out_mv = bdot_mv * inv_iw
    return np.moveaxis(out_mv, -1, ax)


def _smooth_last_axis(x, sigma):
    """Gaussian smoothing along the last axis; FFT convolution, reflected edges."""
    half = int(math.ceil(4 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    pad = [(0, 0)] * (x.ndim - 1) + [(half, half)]
    # Reflect padding needs pad < length; fall back to edge padding for short traces.
    mode = "reflect" if half < x.shape[-1] else "edge"
    xp = np.pad(x, pad, mode=mode)
    out = signal.fftconvolve(xp, k.reshape((1,) * (x.ndim - 1) + (-1,)), mode="same", axes=-1)
    return out[..., half:half + x.shape[-1]]


def integrate_bdot(bdot, dt, baseline="pretrigger", *, pretrigger=None,
                   period=None, smooth_periods=4.0):
    """Time-integrate dB/dt (T/s) along the **last axis** -> B (tesla).

    Integration is cumulative trapezoid from ``B = 0`` at the first sample, so
    the result is defined only up to a constant (the field before the record
    started) plus a linear drift from any DC offset in ``bdot``. ``baseline``
    chooses how to remove them:

    ``'pretrigger'`` (default; pulsed shots)
        Subtract the mean of ``bdot`` over the quiet samples ``pretrigger``
        (an ``int`` n -> the first n samples, or a ``slice``) before
        integrating. B is then zero before the event and carries no offset
        drift. Requires ``pretrigger``.
    ``'periodic'`` (steady RF drive)
        Fit and subtract a straight line over an odd number of half cycles,
        then subtract the mean over an integer number of cycles, so a
        stationary sinusoid ends up centred on zero. Requires ``period``
        (samples per cycle, may be fractional, e.g. from
        :func:`data_analysis.signal.first_and_last_zerocrossings`).
    ``'smooth'`` (RF drive whose amplitude changes)
        Subtract a Gaussian-smoothed copy of B. The smoothing width follows
        the original ``gsmooth`` interval of ``period * smooth_periods / 2``
        samples (sigma = interval / sqrt 2); use ``smooth_periods >= 4`` so the
        drive itself is not removed. Requires ``period``.
    ``None``
        Raw cumulative integral.

    **Choosing a baseline for a driven wave.** Integration weights every
    component by 1/f, so slow content in dB/dt -- low-frequency plasma modes,
    and amplifier offsets that shift by a few digitizer counts while a large RF
    signal is present -- dominates B even when it is invisible in dB/dt. In the
    Aug-2026 LAPD Alfven-wave runs (188 kHz drive) slow B was 10-100x the wave
    after ``'pretrigger'`` integration. The wave itself is unaffected: its
    phasor at the drive frequency (:func:`data_analysis.plasma.fields.phasor`)
    agreed between all three baselines to <= 2e-5. So for wave analysis use
    phasors, or ``'smooth'`` (a high-pass at ~f_drive / smooth_periods) when the
    waveform is needed; use ``'pretrigger'`` when the slow field is the signal.

    ``bdot`` may have any leading shape, e.g. ``(npos, nt)`` or the
    ``(3, npos, nt)`` output of :func:`voltage_to_bdot`; every trace is
    treated independently.
    """
    b = np.asarray(bdot, dtype=float)
    nt = b.shape[-1]

    if baseline == "pretrigger":
        if pretrigger is None:
            raise ValueError("baseline='pretrigger' needs pretrigger= (sample count or slice)")
        sl = slice(0, pretrigger) if isinstance(pretrigger, (int, np.integer)) else pretrigger
        quiet = b[..., sl]
        if quiet.shape[-1] == 0:
            raise ValueError(f"pretrigger={pretrigger!r} selects no samples")
        b = b - quiet.mean(axis=-1, keepdims=True)
    elif baseline in ("periodic", "smooth"):
        if period is None or period <= 1:
            raise ValueError(f"baseline={baseline!r} needs period= (samples per cycle, > 1)")
    elif baseline is not None:
        raise ValueError(f"unknown baseline {baseline!r}; "
                         "expected 'pretrigger', 'periodic', 'smooth' or None")

    B = integrate.cumulative_trapezoid(b, dx=dt, axis=-1, initial=0)

    if baseline == "periodic":
        n_cycles = math.floor(nt / period)
        if n_cycles < 1:
            raise ValueError(f"record of {nt} samples is shorter than one period ({period:g})")
        # A line fitted over ^v^v^v (whole cycles) is biased by the last half
        # cycle's slope; an odd number of half cycles (^v^v^) is not.
        n_fit = int(n_cycles * period - period / 2)
        t = np.arange(nt, dtype=float)
        tf = t[:n_fit]
        flat = B.reshape(-1, nt)
        slope, intercept = np.polynomial.polynomial.polyfit(tf, flat[:, :n_fit].T, 1)[::-1]
        flat = flat - (slope[:, None] * t + intercept[:, None])
        # ...but ^v^v^ has the wrong mean, so recentre over whole cycles.
        n_mean = int(n_cycles * period)
        flat = flat - flat[:, :n_mean].mean(axis=1, keepdims=True)
        B = flat.reshape(B.shape)
    elif baseline == "smooth":
        sigma = period * smooth_periods / 2 / math.sqrt(2)
        B = B - _smooth_last_axis(B, sigma)

    return B
