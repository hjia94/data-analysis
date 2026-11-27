"""Jun-2026 2-D electromagnetic B-dot phasors, J_par = (curl B)_z, and B_L/B_R polarization.

Reconstructs coherent, reference-locked 2-D complex magnetic-field phasor maps
on the 17x17 P28 B-dot plane (Runs 26 [800 G] and 31 [600 G], ``x, y in [-8, +8] cm``
at 1 cm spacing) referenced to simultaneous stationary P25-L ``Isat`` (``lpscope/C3``),
alongside simultaneous P29-R ``Isat`` (``lpscope/C2``) on the same grid:

1. **Reference-locked complex phasors** at ``f_0`` (fundamental) and ``2 f_0``
   (2nd harmonic) over the post-onset saturated window ``[2.4, 4.8] ms`` in the
   physics ``exp(i*(m*theta - w*t))`` convention (:func:`reference_locked_phasor`).
2. **Maxwell div/curl coil-polarity check**: compares interior
   ``RMS|div B_perp| / RMS|curl B|_z`` for ``(Bx, +by_sign*By)`` vs.
   ``(Bx, -by_sign*By)`` to verify relative right-handed in-plane coil
   polarity directly from data. By default (``on_polarity_mismatch="raise"``),
   a clearly lower flipped ratio raises ``ValueError`` rather than silently
   using inverted coils (or pass ``on_polarity_mismatch="flip"`` to invert
   ``By`` automatically); ``by_sign_used`` is always recorded. Note that
   negating *both* ``(Bx, By)`` together leaves ``div/curl``, ``m_J``, and
   ``B_L/B_R`` unchanged while shifting ``J_par`` by 180 deg, so
   ``dphi(J_par - LP)`` is determined modulo 180 deg until at least one
   in-plane coil's absolute polarity is bench-verified
   (``abs_coil_polarity_verified=False`` by default).
3. **2-D field-aligned current density** ``J_par = (curl B_perp)_z / mu_0``
   (when calibrated; otherwise uncalibrated ``(G*A)*mu_0*J_par = (curl Phi_perp)_z``
   [V*s/m] with ``bdot_calibrated=False``): azimuthal mode number ``m_J``
   (weighted by transverse B-dot coherence ``min(g2_Bx, g2_By)`` on the
   current-bearing interior ``|J_z| >= 0.2 max|J_z|`` so ``m_J_purity`` is
   comparable with :mod:`Jun2026_modes`), fitted center, peak current radius,
   and phase relative to P29-R ``Isat`` (mod 180 deg until abs coil sign is verified).
4. **Shear-Alfvén vs. compressional ratio** ``|delta B_z| / |delta B_perp|`` and
   **circular polarization decomposition** ``(B_L, B_R, ellipticity)`` across
   core (``r <= 2.5 cm``), inner wall (``2.5 < r <= 4.5 cm``), and outer
   (``4.5 < r <= 8.0 cm``) regions (Rossi 2021 Fig. 3.14 diagnostic).

Saves ``bdot_2d_jun2026.npz`` in ``DATA_DIR`` and prints a concise summary.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_bdot_2d.py
"""

import datetime
import glob
import os
import warnings
from typing import NamedTuple

import numpy as np
from scipy.constants import mu_0

from data_analysis.io import open_lapd, parse_bdot_gain, position_shots
from data_analysis.plasma.bdot import (
    BdotCalibration,
    load_bdot_calibration,
    phasor_voltage_to_b,
)
from data_analysis.plasma.fields import (
    circular_polarization,
    curl,
    divergence_2d,
    reference_locked_phasor,
)
from data_analysis.plasma.modes import fit_azimuthal_mode
from data_analysis.signal import clip_time_window
from data_analysis.utils import merge_save_npz

import Jun2026_IV as jiv
import Jun2026_onset as jon

# --- Campaign defaults ------------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "bdot_2d_jun2026.npz"

BDOT_MOTION_GROUP = "<Athena>    p28_Bdot"
BDOT_PROBE_PORT = "P28"           # header token for parse_bdot_gain(..., probe=...)
LP_MOTION_GROUP = "<Hermes>    p29_LP"
REF_SCOPE_CHAN = ("lpscope", "C3")
LP_SCOPE_CHAN = ("lpscope", "C2")
BDOT_CHANNELS = (("C1", "Bx"), ("C2", "By"), ("C3", "Bz"))

_UNSET = object()

DEFAULT_TMIN_MS = 2.4             # strictly post-onset for Run 26 (1.40 ms) & Run 31 (2.25 ms)
DEFAULT_TMAX_MS = 4.8
DEFAULT_WINDOW = "hann"
DEFAULT_B_SIGN_Z = +1             # B0 along +z (North = West x Up)
DEFAULT_BY_SIGN = +1              # applied sign on By (+1 as-wired, -1 inverted)
DEFAULT_ON_POLARITY_MISMATCH = "raise"  # "raise" | "flip" | "ignore"
DEFAULT_ABS_COIL_POLARITY_VERIFIED = False  # True once one in-plane coil abs sign is bench-verified
DEFAULT_MJ_GAMMA2_MIN = 0.5       # transverse B-dot coherence threshold on current-bearing interior
DEFAULT_GRID_STEP_CM = 1.0
DEFAULT_BDOT_CAL = r"D:\data\LAPD\BDOT CALIBRATION\Bdot-10T-C16\C16-bdot-calibration.npz"
DEFAULT_BDOT_AREA_M2 = None       # bare coil area [m^2]: scalar, (3,), or (3, 3)
DEFAULT_BDOT_GAIN = None          # None: auto-read from HDF5 P28 chain when calibrated (1.0 uncalibrated)
DEFAULT_BDOT_ATTEN_DB = None      # None: auto-read from HDF5 P28 chain when bdot_gain is None (else 0.0)
DEFAULT_CHANNEL_SIGNS = (+1, +1, +1)  # pre-inversion hardware polarity on (C1, C2, C3)
JUN2026_BDOT_SIGNAL_CHAIN = "2.5BLP => X10 => 2.5BLP => X100"
JUN2026_BDOT_GAIN = 1000.0


class Bdot2DRunSpec(NamedTuple):
    run_num: str
    b0_g: float
    f0_fallback_hz: float


# Run 31 filename says 800G, but HDF5 description & f0=3.529 kHz confirm 600 G.
BDOT_2D_RUNS = (
    Bdot2DRunSpec("26", 800.0, 4110.0),
    Bdot2DRunSpec("31", 600.0, 3529.0),
)
BDOT_2D_RUN_MAP = {s.run_num: s for s in BDOT_2D_RUNS}


def _run_ifn(data_dir, run_num):
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    if len(hits) > 1:
        raise ValueError(f"multiple HDF5 files for run {run_num}: {hits}")
    return hits[0]


def _resolve_bdot_calibration(bdot_cal=None, bdot_area_m2=None):
    """Normalize ``bdot_cal`` / ``bdot_area_m2`` -> ``(cal_or_area, area_matrix_3x3, probe_id)``."""
    if bdot_cal is _UNSET:
        bdot_cal = None if bdot_area_m2 is not None else DEFAULT_BDOT_CAL
    if bdot_cal is not None and bdot_area_m2 is not None:
        raise ValueError("Pass either bdot_cal or bdot_area_m2, not both")
    if bdot_cal is not None:
        cal = bdot_cal if isinstance(bdot_cal, BdotCalibration) else load_bdot_calibration(bdot_cal)
        return cal, cal.area_matrix_m2.copy(), str(cal.probe_id)
    if bdot_area_m2 is None:
        return None, np.full((3, 3), np.nan, dtype=float), ""
    a = np.asarray(bdot_area_m2, dtype=float)
    if a.ndim == 0:
        mat = np.eye(3, dtype=float) * float(a)
    elif a.shape == (3,):
        mat = np.diag(a)
    elif a.shape == (3, 3):
        mat = a.copy()
    else:
        raise ValueError(f"bdot_area_m2 must be scalar, (3,), or (3, 3); got shape {a.shape}")
    return bdot_area_m2, mat, ""


def _read_bdot_chain(run, probe=BDOT_PROBE_PORT):
    """Read and parse the active B-dot signal chain from an open LAPD run.

    Returns ``None`` when the run description is unavailable or lists no
    matching B-dot block, while propagating :class:`ValueError` from
    :func:`parse_bdot_gain` on ambiguous multi-probe matches.
    """
    try:
        raw_desc = run.description().raw
    except (OSError, ValueError, NotImplementedError, KeyError):
        return None
    return parse_bdot_gain(raw_desc, probe=probe)


def _resolve_bdot_gain(parsed_chain, *, calibrated, bdot_gain, bdot_atten_db, run_num):
    """Resolve experiment amplifier gain & attenuator [dB] for one run.

    - Uncalibrated (``calibrated=False``): ``None`` defaults to ``(1.0, 0.0)``
      so uncalibrated flux-like outputs remain at raw scope level.
    - Calibrated (``calibrated=True``): ``bdot_gain=None`` reads the run's
      recorded ``P28`` chain from HDF5 (raising ``ValueError`` if absent rather
      than silently using 1.0). When ``bdot_gain`` is explicitly supplied,
      ``bdot_atten_db=None`` defaults to ``0.0`` and a warning is emitted if the
      effective gain disagrees with the recorded HDF5 chain.
    """
    if not calibrated:
        gain_used = 1.0 if bdot_gain is None else float(bdot_gain)
        atten_used = 0.0 if bdot_atten_db is None else float(bdot_atten_db)
        return gain_used, atten_used

    if bdot_gain is None:
        if parsed_chain is None:
            raise ValueError(
                f"Run {run_num}: B-dot calibration requested with bdot_gain=None, "
                f"but no active {BDOT_PROBE_PORT} B-dot signal chain was found in "
                "the HDF5 description; pass bdot_gain= explicitly."
            )
        gain_used = float(parsed_chain["gain"])
        atten_used = float(parsed_chain["atten_db"]) if bdot_atten_db is None else float(bdot_atten_db)
    else:
        gain_used = float(bdot_gain)
        atten_used = 0.0 if bdot_atten_db is None else float(bdot_atten_db)

    if parsed_chain is not None:
        req_eff = gain_used * (10.0 ** (-atten_used / 20.0))
        if not np.isclose(req_eff, parsed_chain["effective_gain"], rtol=1e-3, atol=0.0):
            warnings.warn(
                f"Run {run_num}: requested bdot effective gain {req_eff:g} differs from "
                f"HDF5 description '{parsed_chain['chain']}' ({parsed_chain['effective_gain']:g})",
                UserWarning,
                stacklevel=3,
            )
    return gain_used, atten_used


def _apply_bdot_calibration(
    vx,
    vy,
    vz,
    freq_hz,
    *,
    bdot_cal=None,
    bdot_area_m2=None,
    bdot_gain=1.0,
    bdot_atten_db=0.0,
    channel_signs=DEFAULT_CHANNEL_SIGNS,
    probe_to_lab=None,
    mode="static",
):
    """Convert scope B-dot voltage phasors (``exp(-i*w*t)``) -> ``(Bx, By, Bz, calibrated)``.

    Experiment signal conditioning (``channel_signs``, ``bdot_gain``,
    ``bdot_atten_db``) is applied **first** to produce real probe-terminal
    voltage phasors ``V_coil`` before delegating to
    :func:`data_analysis.plasma.bdot.phasor_voltage_to_b`. Optional
    ``probe_to_lab`` (3x3 rotation/sign matrix) is applied **after** ``A^-1``.
    When neither ``bdot_cal`` nor ``bdot_area_m2`` is given, returns uncalibrated
    flux-like phasors ``(i/w) * V_coil`` [V*s] with ``calibrated=False``.
    """
    signs = np.asarray(channel_signs, dtype=float)
    if signs.shape != (3,) or not np.all(np.isin(signs, (-1.0, +1.0))):
        raise ValueError(f"channel_signs must be 3 values in {{-1, +1}}, got {channel_signs!r}")
    eff_gain = float(bdot_gain) * (10.0 ** (-float(bdot_atten_db) / 20.0))
    if eff_gain <= 0.0:
        raise ValueError(f"effective bdot gain must be > 0, got {eff_gain}")

    # Real probe-terminal voltage phasors (3, ny, nx)
    v_scope = np.stack([vx, vy, vz], axis=0)
    v_coil = (signs[:, None, None] * v_scope) / eff_gain

    cal_obj, _, _ = _resolve_bdot_calibration(bdot_cal, bdot_area_m2)
    if cal_obj is None:
        factor = 1j / (2.0 * np.pi * float(freq_hz))
        out = factor * v_coil
        return out[0], out[1], out[2], False

    B = phasor_voltage_to_b(
        v_coil, freq_hz, cal_obj, convention="exp_minus_iwt", gain=1.0, mode=mode,
    )
    if probe_to_lab is not None:
        R = np.asarray(probe_to_lab, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"probe_to_lab must have shape (3, 3), got {R.shape}")
        B = np.einsum("ij,j...->i...", R, B)
    return B[0], B[1], B[2], True


def analyze_harmonic_fields(x_cm, y_cm, Bx, By, Bz, LP, g2_Bx, g2_By, g2_Bz, g2_LP, *,
                            calibrated=False, b_sign_z=DEFAULT_B_SIGN_Z,
                            by_sign=DEFAULT_BY_SIGN,
                            on_polarity_mismatch=DEFAULT_ON_POLARITY_MISMATCH,
                            abs_coil_polarity_verified=DEFAULT_ABS_COIL_POLARITY_VERIFIED,
                            m_j_gamma2_min=DEFAULT_MJ_GAMMA2_MIN):
    """Compute 2-D ``J_par``, div/curl polarity ratio, ``B_L/B_R`` polarization, and LP phase.

    Checks Maxwell's ``div B_perp = 0`` handedness by comparing interior
    ``RMS|div B_perp| / RMS|curl B|_z`` for ``(Bx, by_sign * By)`` vs.
    ``(Bx, -by_sign * By)``. If the opposite ``By`` sign cuts ``div/curl`` by
    more than 30% (``alt < 0.7 * as_is``):
    - ``"raise"`` (default): raises ``ValueError`` rather than returning a
      corrupted ``m_J`` / polarization map.
    - ``"flip"``: negates ``By`` automatically and records ``by_sign_used = -by_sign``.
    - ``"ignore"``: keeps ``by_sign`` as given and flags ``coil_handedness_ok = False``.

    Note that simultaneously negating *both* ``(Bx, By)`` leaves ``div/curl``,
    ``m_J``, and ``B_L/B_R`` invariant while shifting ``dphi_J_minus_LP_deg`` by
    180 deg; ``abs_coil_polarity_verified`` records whether at least one in-plane
    coil's absolute polarity has been independently verified.
    """
    if by_sign not in (+1, -1):
        raise ValueError(f"by_sign must be +1 or -1, got {by_sign!r}")
    if on_polarity_mismatch not in ("raise", "flip", "ignore"):
        raise ValueError(
            f"on_polarity_mismatch must be 'raise', 'flip', or 'ignore', got {on_polarity_mismatch!r}"
        )

    x_cm = np.asarray(x_cm, dtype=float)
    y_cm = np.asarray(y_cm, dtype=float)
    Bx = np.asarray(Bx, dtype=complex)
    By_used = float(by_sign) * np.asarray(By, dtype=complex)
    By_alt = -By_used
    Bz = np.asarray(Bz, dtype=complex)
    LP = np.asarray(LP, dtype=complex)
    dx_m = float(np.median(np.diff(x_cm))) * 1e-2
    dy_m = float(np.median(np.diff(y_cm))) * 1e-2
    XX, YY = np.meshgrid(x_cm, y_cm)
    interior = slice(1, -1)

    # 1. Div/curl coil-polarity diagnostic (requested by_sign vs opposite By)
    _, _, curl_z = curl(Bx, By_used, 0, dx_m, dy_m, None, axes=(-1, -2, -3))
    div_perp = divergence_2d(Bx, By_used, dx_m, dy_m, axes=(-1, -2))
    rms_div = float(np.sqrt(np.nanmean(np.abs(div_perp[interior, interior]) ** 2)))
    rms_curl = float(np.sqrt(np.nanmean(np.abs(curl_z[interior, interior]) ** 2)))
    div_curl_ratio = rms_div / rms_curl if rms_curl > 0.0 else np.nan

    _, _, curl_z_alt = curl(Bx, By_alt, 0, dx_m, dy_m, None, axes=(-1, -2, -3))
    div_perp_alt = divergence_2d(Bx, By_alt, dx_m, dy_m, axes=(-1, -2))
    rms_div_a = float(np.sqrt(np.nanmean(np.abs(div_perp_alt[interior, interior]) ** 2)))
    rms_curl_a = float(np.sqrt(np.nanmean(np.abs(curl_z_alt[interior, interior]) ** 2)))
    div_curl_alt = rms_div_a / rms_curl_a if rms_curl_a > 0.0 else np.nan

    mismatch = bool(
        np.isfinite(div_curl_ratio) and np.isfinite(div_curl_alt)
        and (div_curl_alt < 0.7 * div_curl_ratio)
    )
    by_sign_used = int(by_sign)
    coil_handedness_ok = True
    polarity_auto_flipped = False
    if mismatch:
        if on_polarity_mismatch == "raise":
            raise ValueError(
                f"In-plane B-dot coil polarity mismatch for by_sign={by_sign:+d}: "
                f"div/curl={div_curl_ratio:.3f} vs flipped By={div_curl_alt:.3f}. "
                f"Verify hardware coil wiring or pass by_sign={-by_sign:+d} / "
                "on_polarity_mismatch='flip'."
            )
        if on_polarity_mismatch == "flip":
            By_used = By_alt
            by_sign_used = -int(by_sign)
            curl_z, div_perp = curl_z_alt, div_perp_alt
            rms_div, rms_curl = rms_div_a, rms_curl_a
            div_curl_ratio, div_curl_alt = div_curl_alt, div_curl_ratio
            polarity_auto_flipped = True
        else:
            coil_handedness_ok = False

    # J_par: A/m^2 if calibrated, else (G*A)*mu_0*J_par = (curl Phi)_z [V*s/m]
    Jz = (curl_z / mu_0) if calibrated else curl_z

    # 2. Azimuthal mode m_J and fitted center on 2-D Jz.
    # Weight by true transverse B-dot coherence min(g2_Bx, g2_By) (so m_J_purity
    # is comparable with Jun2026_modes), gated to the current-bearing interior
    # |Jz| >= 0.2 * max|Jz| where curl_z is above exterior finite-difference noise.
    j_int_abs = np.abs(Jz[interior, interior])
    j_max = float(np.nanmax(j_int_abs)) if np.any(np.isfinite(j_int_abs)) else 0.0
    g2_bperp_int = np.minimum(
        np.asarray(g2_Bx, dtype=float)[interior, interior],
        np.asarray(g2_By, dtype=float)[interior, interior],
    )
    g2_j_fit = (
        np.where(j_int_abs >= 0.2 * j_max, g2_bperp_int, np.nan)
        if j_max > 0.0 else g2_bperp_int
    )
    m_j = fit_azimuthal_mode(
        XX[interior, interior].ravel(), YY[interior, interior].ravel(),
        g2_j_fit.ravel(), np.angle(Jz[interior, interior]).ravel(),
        r_min_cm=1.0, r_max_cm=7.0,
        gamma2_min=float(m_j_gamma2_min),
        moving_is_a=True,
    )
    cx = float(m_j.center_x_cm) if np.isfinite(m_j.center_x_cm) else 2.0
    cy = float(m_j.center_y_cm) if np.isfinite(m_j.center_y_cm) else 0.0
    RR = np.hypot(XX - cx, YY - cy)
    if j_max > 0.0:
        ipk = np.unravel_index(int(np.nanargmax(j_int_abs)), j_int_abs.shape)
        r_peak_j_cm = float(RR[interior, interior][ipk])
    else:
        r_peak_j_cm = np.nan

    # 3. Compressional vs shear-Alfven amplitude ratio |Bz| / |B_perp|
    b_perp_abs = np.sqrt(np.abs(Bx) ** 2 + np.abs(By_used) ** 2)
    rms_bperp = float(np.sqrt(np.nanmean(b_perp_abs[interior, interior] ** 2)))
    rms_bz = float(np.sqrt(np.nanmean(np.abs(Bz[interior, interior]) ** 2)))
    bz_over_bperp = rms_bz / rms_bperp if rms_bperp > 0.0 else np.nan

    # 4. Circular polarization B_L (CW/ion sense) vs B_R (CCW/electron sense)
    B_L, B_R, ellip = circular_polarization(
        Bx, By_used, convention="exp_minus_iwt", b_sign_z=b_sign_z,
    )

    zones = {
        "core": RR <= 2.5,
        "wall": (RR > 2.5) & (RR <= 4.5),
        "outer": (RR > 4.5) & (RR <= 8.0),
    }
    zone_stats = {}
    for zname, zmask in zones.items():
        if np.any(zmask):
            zone_stats[f"{zname}_bL_rms"] = float(np.sqrt(np.nanmean(np.abs(B_L[zmask]) ** 2)))
            zone_stats[f"{zname}_bR_rms"] = float(np.sqrt(np.nanmean(np.abs(B_R[zmask]) ** 2)))
            zone_stats[f"{zname}_ellip_med"] = float(np.nanmedian(ellip[zmask]))
            zone_stats[f"{zname}_jz_rms"] = float(np.sqrt(np.nanmean(np.abs(Jz[zmask]) ** 2)))
            zone_stats[f"{zname}_lp_rms"] = float(np.sqrt(np.nanmean(np.abs(LP[zmask]) ** 2)))
        else:
            for suf in ("bL_rms", "bR_rms", "ellip_med", "jz_rms", "lp_rms"):
                zone_stats[f"{zname}_{suf}"] = np.nan

    # 5. Circular-averaged phase between J_par(P28) and LP Isat(P29-R) on the current ring
    ring = (
        (RR >= 2.0) & (RR <= 5.0)
        & np.isfinite(Jz) & np.isfinite(LP) & np.isfinite(g2_LP)
        & (np.abs(Jz) >= 0.3 * j_max) & (g2_LP >= 0.5)
    )
    if np.any(ring):
        z_jl = np.mean(np.exp(1j * np.angle(Jz[ring] * np.conj(LP[ring]))))
        dphi_j_lp_deg = float(np.degrees(np.angle(z_jl)))
        R_j_lp = float(np.abs(z_jl))
        n_j_lp = int(np.sum(ring))
    else:
        dphi_j_lp_deg, R_j_lp, n_j_lp = np.nan, np.nan, 0

    return {
        "Bx": Bx, "By": By_used, "Bz": Bz, "Jz": Jz,
        "div_B_perp": div_perp, "B_L": B_L, "B_R": B_R, "ellipticity": ellip,
        "LP": LP, "g2_Bx": g2_Bx, "g2_By": g2_By, "g2_Bz": g2_Bz, "g2_LP": g2_LP,
        "rms_div": rms_div, "rms_curl": rms_curl,
        "div_curl_ratio": div_curl_ratio, "div_curl_ratio_flipped": div_curl_alt,
        "by_sign_used": int(by_sign_used),
        "coil_handedness_ok": bool(coil_handedness_ok),
        "polarity_auto_flipped": bool(polarity_auto_flipped),
        "abs_coil_polarity_verified": bool(abs_coil_polarity_verified),
        "m_J": int(m_j.m), "m_J_purity": float(m_j.purity),
        "center_x_cm": cx, "center_y_cm": cy, "r_peak_j_cm": r_peak_j_cm,
        "bz_over_bperp": bz_over_bperp,
        "med_g2_Bx": float(np.nanmedian(g2_Bx)),
        "med_g2_By": float(np.nanmedian(g2_By)),
        "med_g2_Bz": float(np.nanmedian(g2_Bz)),
        "med_g2_LP": float(np.nanmedian(g2_LP)),
        "dphi_J_minus_LP_deg": dphi_j_lp_deg,
        "R_J_LP": R_j_lp,
        "n_J_LP": n_j_lp,
        **zone_stats,
    }


def _grid_reference_locked_channel(stack, ref_seg, pos_obj, shot_to_idx, dt_sig, dt_ref,
                                   freqs, *, t0_offset_s=0.0,
                                   grid_step_cm=DEFAULT_GRID_STEP_CM,
                                   window=DEFAULT_WINDOW):
    """Project shot-by-shot reference-locked phasors onto the snapped 2-D ``(ny, nx)`` grid.

    ``t0_offset_s = t_sig[0] - t_ref[0]`` aligns the signal phasor to the
    reference start time (``exp(+i * 2*pi*f * t0_offset_s)`` in ``exp(-iwt)``
    convention) when the signal and reference come from different scopes.
    """
    from data_analysis.plasma.fields import phasor

    seg = np.asarray(stack, dtype=float)
    seg = seg - seg.mean(axis=1, keepdims=True)
    px = np.round(pos_obj.pos_array["x"][::pos_obj.nshot][:pos_obj.npos] / grid_step_cm) * grid_step_cm
    py = np.round(pos_obj.pos_array["y"][::pos_obj.nshot][:pos_obj.npos] / grid_step_cm) * grid_step_cm
    ux = np.unique(px)
    uy = np.unique(py)

    # Pre-index positions -> shot indices once across harmonics
    pos_indices = []
    for p_i in range(pos_obj.npos):
        sh = position_shots(pos_obj.shot_nums, p_i, pos_obj.nshot)
        idx = [shot_to_idx[int(s)] for s in sh if int(s) in shot_to_idx]
        ix = int(np.searchsorted(ux, px[p_i]))
        iy = int(np.searchsorted(uy, py[p_i]))
        pos_indices.append((iy, ix, idx))

    out = {}
    for f in freqs:
        grid_c = np.full((uy.size, ux.size), np.nan + 0j, dtype=complex)
        grid_g2 = np.full((uy.size, ux.size), np.nan, dtype=float)
        # Vectorize single-frequency DFT across all shots, then group by position.
        # Matches fields.reference_locked_phasor(..., convention="exp_minus_iwt").
        phase_align = np.exp(2j * np.pi * f * t0_offset_s)
        Xp = np.conj(phasor(seg, dt_sig, f, axis=-1, window=window)) * phase_align
        Rp = np.conj(phasor(ref_seg, dt_ref, f, axis=-1, window=window))
        for iy, ix, idx in pos_indices:
            if not idx:
                continue
            xs, rs = Xp[idx], Rp[idx]
            cross = np.mean(xs * np.conj(rs))
            pxx = float(np.mean(np.abs(xs) ** 2))
            prr = float(np.mean(np.abs(rs) ** 2))
            if prr > 0.0:
                grid_c[iy, ix] = cross / np.sqrt(prr)
            if pxx > 0.0 and prr > 0.0:
                grid_g2[iy, ix] = float((np.abs(cross) ** 2) / (pxx * prr))
        out[float(f)] = (grid_c, grid_g2)
    return ux, uy, out


def analyze_run(spec, data_dir=DATA_DIR, onset_table=None, *,
                tmin_ms=DEFAULT_TMIN_MS, tmax_ms=DEFAULT_TMAX_MS,
                window=DEFAULT_WINDOW, b_sign_z=DEFAULT_B_SIGN_Z,
                by_sign=DEFAULT_BY_SIGN,
                on_polarity_mismatch=DEFAULT_ON_POLARITY_MISMATCH,
                abs_coil_polarity_verified=DEFAULT_ABS_COIL_POLARITY_VERIFIED,
                bdot_cal=_UNSET,
                bdot_area_m2=DEFAULT_BDOT_AREA_M2,
                bdot_gain=DEFAULT_BDOT_GAIN,
                bdot_atten_db=DEFAULT_BDOT_ATTEN_DB,
                channel_signs=DEFAULT_CHANNEL_SIGNS,
                probe_to_lab=None):
    """Extract 2-D complex B-dot & LP phasors, ``J_par``, and polarization for one run."""
    ifn = _run_ifn(data_dir, spec.run_num)
    f0 = (float(onset_table[spec.run_num]["f_peak_hz"])
          if (onset_table and spec.run_num in onset_table and np.isfinite(onset_table[spec.run_num]["f_peak_hz"]))
          else float(spec.f0_fallback_hz))
    freqs = (f0, 2.0 * f0)

    cal_obj, area_mat, probe_id = _resolve_bdot_calibration(bdot_cal, bdot_area_m2)

    run = open_lapd(ifn)
    parsed_chain = _read_bdot_chain(run)

    signal_chain = parsed_chain["chain"] if parsed_chain else ""
    gain_used, atten_used = _resolve_bdot_gain(
        parsed_chain,
        calibrated=(cal_obj is not None),
        bdot_gain=bdot_gain,
        bdot_atten_db=bdot_atten_db,
        run_num=spec.run_num,
    )

    pos_b = jiv.read_lp_positions(ifn, motion_group_name=BDOT_MOTION_GROUP)
    pos_lp = jiv.read_lp_positions(ifn, motion_group_name=LP_MOTION_GROUP)

    tb = run.time_array(scope_name="bdotscope")
    tl = run.time_array(scope_name="lpscope")
    ib0, ib1 = clip_time_window(tb, tmin_ms, tmax_ms)
    il0, il1 = clip_time_window(tl, tmin_ms, tmax_ms)
    dtb = float(tb[1] - tb[0])
    dtl = float(tl[1] - tl[0])
    if np.isclose(dtb, dtl, rtol=1e-6, atol=0.0):
        n_samp = min(ib1 - ib0, il1 - il0)
        ib1 = ib0 + n_samp
        il1 = il0 + n_samp
    t0_offset_s = float(tb[ib0] - tl[il0])

    all_shots = np.asarray(run.shots(scope_name="bdotscope"), dtype=int)
    shot_to_idx = {int(s): i for i, s in enumerate(all_shots)}
    shots_list = all_shots.tolist()

    ref_scope, ref_ch = REF_SCOPE_CHAN
    st_ref, _ = run.channel(ref_ch, scope_name=ref_scope, shots=shots_list)
    ref_seg = np.asarray(st_ref[:, il0:il1], dtype=float)
    ref_seg = ref_seg - ref_seg.mean(axis=1, keepdims=True)

    bdot_grids = {}
    ux = uy = None
    for ch, name in BDOT_CHANNELS:
        st, _ = run.channel(ch, scope_name="bdotscope", shots=shots_list)
        ux, uy, res_f = _grid_reference_locked_channel(
            st[:, ib0:ib1], ref_seg, pos_b, shot_to_idx, dtb, dtl, freqs,
            t0_offset_s=t0_offset_s, window=window,
        )
        bdot_grids[name] = res_f

    lp_scope, lp_ch = LP_SCOPE_CHAN
    st_lp, _ = run.channel(lp_ch, scope_name=lp_scope, shots=shots_list)
    ux_lp, uy_lp, lp_grids = _grid_reference_locked_channel(
        st_lp[:, il0:il1], ref_seg, pos_lp, shot_to_idx, dtl, dtl, freqs,
        t0_offset_s=0.0, window=window,
    )
    if not (np.allclose(ux, ux_lp) and np.allclose(uy, uy_lp)):
        raise ValueError(f"Run {spec.run_num}: B-dot and LP snapped grids differ")

    out = {
        "b0_g": float(spec.b0_g),
        "f0_hz": float(f0),
        "x_cm": ux,
        "y_cm": uy,
        "t0_offset_us": float(t0_offset_s * 1e6),
        "bdot_calibrated": bool(cal_obj is not None),
        "bdot_probe_id": str(probe_id),
        "bdot_area_matrix_m2": area_mat,
        "bdot_signal_chain": str(signal_chain),
        "bdot_gain": float(gain_used),
        "bdot_atten_db": float(atten_used),
        "bdot_channel_signs": np.asarray(channel_signs, dtype=np.int64),
        "abs_coil_polarity_verified": bool(abs_coil_polarity_verified),
    }

    for h_lbl, f_hz in (("h1", freqs[0]), ("h2", freqs[1])):
        vx, gx = bdot_grids["Bx"][f_hz]
        vy, gy = bdot_grids["By"][f_hz]
        vz, gz = bdot_grids["Bz"][f_hz]
        lp_c, glp = lp_grids[f_hz]
        Bx, By, Bz, cal_flag = _apply_bdot_calibration(
            vx, vy, vz, f_hz,
            bdot_cal=cal_obj if isinstance(cal_obj, BdotCalibration) else None,
            bdot_area_m2=None if isinstance(cal_obj, BdotCalibration) else cal_obj,
            bdot_gain=gain_used,
            bdot_atten_db=atten_used,
            channel_signs=channel_signs,
            probe_to_lab=probe_to_lab,
        )
        h_res = analyze_harmonic_fields(
            ux, uy, Bx, By, Bz, lp_c, gx, gy, gz, glp,
            calibrated=cal_flag, b_sign_z=b_sign_z,
            by_sign=by_sign, on_polarity_mismatch=on_polarity_mismatch,
            abs_coil_polarity_verified=abs_coil_polarity_verified,
        )
        out[f"{h_lbl}__freq_hz"] = float(f_hz)
        for k, v in h_res.items():
            out[f"{h_lbl}__{k}"] = v

    return out


def _run_settings_dict(tmin_ms, tmax_ms, window, b_sign_z, by_sign,
                       on_polarity_mismatch, abs_coil_polarity_verified,
                       bdot_cal, bdot_area_m2, bdot_gain,
                       bdot_atten_db=DEFAULT_BDOT_ATTEN_DB,
                       channel_signs=DEFAULT_CHANNEL_SIGNS):
    cal_obj, area_mat, probe_id = _resolve_bdot_calibration(bdot_cal, bdot_area_m2)
    if cal_obj is None:
        want_gain = 1.0 if bdot_gain is None else float(bdot_gain)
        want_atten = 0.0 if bdot_atten_db is None else float(bdot_atten_db)
    elif bdot_gain is None:
        want_gain = None
        want_atten = None if bdot_atten_db is None else float(bdot_atten_db)
    else:
        want_gain = float(bdot_gain)
        want_atten = 0.0 if bdot_atten_db is None else float(bdot_atten_db)
    return {
        "tmin_ms": float(tmin_ms),
        "tmax_ms": float(tmax_ms),
        "window": str(window),
        "b_sign_z": int(b_sign_z),
        "by_sign": int(by_sign),
        "on_polarity_mismatch": str(on_polarity_mismatch),
        "abs_coil_polarity_verified": bool(abs_coil_polarity_verified),
        "bdot_calibrated": bool(cal_obj is not None),
        "bdot_probe_id": str(probe_id),
        "bdot_area_matrix_m2": area_mat,
        "bdot_gain": want_gain,
        "bdot_atten_db": want_atten,
        "bdot_channel_signs": np.asarray(channel_signs, dtype=np.int64),
    }


def _run_settings_mismatch(npz_data, run_num, want):
    diffs = []
    for field, val in want.items():
        k = f"{run_num}__{field}"
        if k not in npz_data.files:
            diffs.append(f"{field}: missing")
            continue
        if val is None:
            continue
        if isinstance(val, np.ndarray):
            got = np.asarray(npz_data[k])
            if got.shape != val.shape or not np.allclose(got, val, rtol=1e-6, atol=1e-12, equal_nan=True):
                diffs.append(f"{field}: stored array differs from requested")
        elif isinstance(val, bool):
            if bool(npz_data[k]) != val:
                diffs.append(f"{field}: stored {bool(npz_data[k])} != {val}")
        elif isinstance(val, int):
            if int(npz_data[k]) != val:
                diffs.append(f"{field}: stored {int(npz_data[k])} != {val}")
        elif isinstance(val, str):
            if str(npz_data[k]) != val:
                diffs.append(f"{field}: stored {str(npz_data[k])!r} != {val!r}")
        else:
            if not np.isclose(float(npz_data[k]), val, atol=1e-9, rtol=0.0):
                diffs.append(f"{field}: stored {float(npz_data[k]):g} != {val:g}")
    return diffs


def batch_all(data_dir=DATA_DIR, runs=None, onset_npz=None, out_path=None, *,
              tmin_ms=DEFAULT_TMIN_MS, tmax_ms=DEFAULT_TMAX_MS,
              window=DEFAULT_WINDOW, b_sign_z=DEFAULT_B_SIGN_Z,
              by_sign=DEFAULT_BY_SIGN,
              on_polarity_mismatch=DEFAULT_ON_POLARITY_MISMATCH,
              abs_coil_polarity_verified=DEFAULT_ABS_COIL_POLARITY_VERIFIED,
              bdot_cal=_UNSET,
              bdot_area_m2=DEFAULT_BDOT_AREA_M2,
              bdot_gain=DEFAULT_BDOT_GAIN,
              bdot_atten_db=DEFAULT_BDOT_ATTEN_DB,
              channel_signs=DEFAULT_CHANNEL_SIGNS,
              probe_to_lab=None,
              overwrite=False):
    """Compute 2-D B-dot phasors, ``J_par``, and ``B_L/B_R`` polarization -> npz path."""
    specs = (BDOT_2D_RUNS if runs is None
             else tuple(BDOT_2D_RUN_MAP[r] for r in runs))
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    onset_path = onset_npz or os.path.join(data_dir, jon.OUT_NAME)
    onset_table = jon.load_onset_table(onset_path) if os.path.isfile(onset_path) else {}
    want = _run_settings_dict(
        tmin_ms, tmax_ms, window, b_sign_z, by_sign, on_polarity_mismatch,
        abs_coil_polarity_verified, bdot_cal, bdot_area_m2, bdot_gain,
        bdot_atten_db=bdot_atten_db, channel_signs=channel_signs,
    )

    if overwrite and runs is None and os.path.isfile(out_path):
        os.remove(out_path)

    existing_runs = set()
    drop_keys = []
    if os.path.isfile(out_path):
        with np.load(out_path) as d:
            existing_runs = {
                k.split("__", 1)[0] for k in d.files if k.endswith("__f0_hz")
            }
            if overwrite:
                prefixes = tuple(f"{s.run_num}__" for s in specs)
                drop_keys = [k for k in d.files if k.startswith(prefixes)]
                existing_runs -= {s.run_num for s in specs}
            else:
                for spec in specs:
                    if spec.run_num in existing_runs:
                        run_want = want
                        if want["bdot_gain"] is None or want["bdot_atten_db"] is None:
                            try:
                                ifn = _run_ifn(data_dir, spec.run_num)
                            except FileNotFoundError:
                                ifn = None
                            if ifn is not None:
                                parsed_chain = _read_bdot_chain(open_lapd(ifn))
                                g_exp, a_exp = _resolve_bdot_gain(
                                    parsed_chain,
                                    calibrated=True,
                                    bdot_gain=bdot_gain,
                                    bdot_atten_db=bdot_atten_db,
                                    run_num=spec.run_num,
                                )
                                run_want = dict(want, bdot_gain=g_exp, bdot_atten_db=a_exp)
                        diffs = _run_settings_mismatch(d, spec.run_num, run_want)
                        if diffs:
                            raise ValueError(
                                f"{out_path}: stored run '{spec.run_num}' differs "
                                f"from requested settings ({'; '.join(diffs)}); "
                                "pass overwrite=True to regenerate"
                            )

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
    }
    ran_any = False
    for spec in specs:
        if spec.run_num in existing_runs:
            continue
        res = analyze_run(
            spec, data_dir=data_dir, onset_table=onset_table,
            tmin_ms=tmin_ms, tmax_ms=tmax_ms, window=window, b_sign_z=b_sign_z,
            by_sign=by_sign, on_polarity_mismatch=on_polarity_mismatch,
            abs_coil_polarity_verified=abs_coil_polarity_verified,
            bdot_cal=bdot_cal, bdot_area_m2=bdot_area_m2,
            bdot_gain=bdot_gain, bdot_atten_db=bdot_atten_db,
            channel_signs=channel_signs, probe_to_lab=probe_to_lab,
        )
        p = spec.run_num
        for k, v in res.items():
            if isinstance(v, str):
                payload[f"{p}__{k}"] = np.str_(v)
            elif isinstance(v, (bool, np.bool_)):
                payload[f"{p}__{k}"] = np.bool_(v)
            elif isinstance(v, (int, np.integer)):
                payload[f"{p}__{k}"] = np.int64(v)
            elif isinstance(v, (float, np.floating)):
                payload[f"{p}__{k}"] = np.float64(v)
            else:
                payload[f"{p}__{k}"] = np.asarray(v)
        for sk, sv in want.items():
            if sv is None:
                continue
            if isinstance(sv, np.ndarray):
                payload[f"{p}__{sk}"] = sv
            elif isinstance(sv, bool):
                payload[f"{p}__{sk}"] = np.bool_(sv)
            elif isinstance(sv, int):
                payload[f"{p}__{sk}"] = np.int64(sv)
            elif isinstance(sv, str):
                payload[f"{p}__{sk}"] = np.str_(sv)
            else:
                payload[f"{p}__{sk}"] = np.float64(sv)
        ran_any = True

    if ran_any or not os.path.isfile(out_path):
        stale_keys = [k for k in drop_keys if k not in payload]
        merge_save_npz(out_path, payload, drop_keys=stale_keys)
        print(f"\nWrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print the 2-D electromagnetic J_par & B_L/B_R polarization summary."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path) as d:
        print("\n=== 2-D Electromagnetic B-dot Reconstruction: J_par=(curl B)_z & B_L/B_R Polarization ===")
        for spec in BDOT_2D_RUNS:
            r = spec.run_num
            if f"{r}__f0_hz" not in d.files:
                continue
            cal = bool(d[f"{r}__bdot_calibrated"])
            abs_pol = bool(d[f"{r}__abs_coil_polarity_verified"]) if f"{r}__abs_coil_polarity_verified" in d.files else False
            b_unit = "G" if cal else "uV*s"
            b_scale = 1e4 if cal else 1e6
            j_unit = "A/m^2" if cal else "uV*s/cm"
            j_scale = 1.0 if cal else 1e4
            pol_note = "" if abs_pol else " [mod 180d: abs coil sign unverified]"
            print(
                f"\nRun {r} ({float(d[f'{r}__b0_g']):.0f} G, f0={float(d[f'{r}__f0_hz']):.1f} Hz, "
                f"win=[{float(d[f'{r}__tmin_ms']):.1f}, {float(d[f'{r}__tmax_ms']):.1f}] ms, "
                f"calibrated={cal}, abs_polarity_verified={abs_pol}):"
            )
            for h_lbl, h_name in (("h1", "f0 "), ("h2", "2f0")):
                pfx = f"{r}__{h_lbl}__"
                by_s = int(d[pfx + "by_sign_used"]) if (pfx + "by_sign_used") in d.files else +1
                print(
                    f"  [{h_name}] div/curl (by_sign={by_s:+d}): used={float(d[pfx+'div_curl_ratio']):.3f} vs "
                    f"alt={float(d[pfx+'div_curl_ratio_flipped']):.3f} | "
                    f"J_par: m_J={int(d[pfx+'m_J']):+d} (g2-pur={float(d[pfx+'m_J_purity']):.3f}, "
                    f"ctr=({float(d[pfx+'center_x_cm']):+.1f},{float(d[pfx+'center_y_cm']):+.1f})cm, "
                    f"r_pk={float(d[pfx+'r_peak_j_cm']):.2f}cm) | "
                    f"|Bz|/|Bp|={float(d[pfx+'bz_over_bperp']):.3f} (g2_z={float(d[pfx+'med_g2_Bz']):.2f}) | "
                    f"dphi(J-LP)={float(d[pfx+'dphi_J_minus_LP_deg']):+.1f}d{pol_note} (R={float(d[pfx+'R_J_LP']):.2f})"
                )
                for zn, zl in (("core", "Core  r<=2.5"), ("wall", "Wall 2.5-4.5"), ("outer", "Outer 4.5-8 ")):
                    print(
                        f"        {zl}: |BL|={float(d[pfx+zn+'_bL_rms'])*b_scale:5.2f} "
                        f"|BR|={float(d[pfx+zn+'_bR_rms'])*b_scale:5.2f} {b_unit}  "
                        f"P={float(d[pfx+zn+'_ellip_med']):+.2f}  "
                        f"|Jz|={float(d[pfx+zn+'_jz_rms'])*j_scale:6.1f} {j_unit}  "
                        f"|LP|={float(d[pfx+zn+'_lp_rms'])*1e3:6.2f} mV"
                    )


if __name__ == "__main__":
    out = batch_all()
    print_summary(out)
