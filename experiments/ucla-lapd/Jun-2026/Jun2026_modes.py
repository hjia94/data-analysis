"""Jun-2026 mode structure (m, k_par), B-dot dipole, disc-current coherence & dispersion.

Synthesizes the campaign's cross-correlation, B-dot, Mach-flow, IV-profile,
onset, and biasing-circuit diagnostics into one quantitative summary:

1. **2-D azimuthal mode structure** (Runs 26, 31, 29): physical ``m`` and
   center ``(cx, cy)`` for LP density and P28 B-dot ``Bx/By`` (fundamental and
   2nd harmonic), plus the Biot-Savart ``(Bx, By)`` quadrature check for the
   rotating ``m_J = -1`` field-aligned current filament and nominal
   ``delta_B_perp / B_0`` (explicitly flagged as uncalibrated unless bench-
   calibrated coil area and gain are supplied).
2. **Axial wavenumber ``k_par``** (Runs 27, 28, 29, 30; ``dz = 1.28 m`` between
   P29 and P25): circular-averaged axial phase lag and ``k_par`` split into the
   inner unstable shear layer (``r <= 4 cm``) vs. outer column (``r > 4 cm``).
3. **End-electrode current coherence** (Runs 06, 00-05): cross-scope coherence
   between stationary P33 ``Isat`` (``machscope/C2``) and ``biasscope`` 2", 4",
   6" electrode currents, 2" voltage, and ``Light@P30`` (with ``+12.0 ms``
   trigger alignment), testing axial current sloshing across electrodes.
4. **Plasma-frame dispersion** (24-45 ms puff scan & 600-1000 G B0 scan):
   combines ``onset_summary_jun2026.npz``, ``profiles_summary_jun2026.npz``,
   ``mach_line_rescaled_jun2026.npz``, and ``32-mach-flow-data.npz`` to compare
   ``f_plasma = f_lab - m * f_rot`` against ``f_*n``, ``f_*T``, ``f_*p``,
   ``k_perp * rho_s``, ``beta_e / (m_e / m_i)``, and ``f_Alfven``.

Saves ``modes_dispersion_jun2026.npz`` into ``DATA_DIR`` and prints a concise
synthesis table.

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_modes.py
"""

import datetime
import glob
import os
import warnings

import numpy as np

from data_analysis.io import open_lapd, parse_bdot_gain
from data_analysis.plasma.bdot import (
    BdotCalibration,
    inverse_area_matrix,
    load_bdot_calibration,
    voltage_to_bdot,
)
from data_analysis.plasma.flow import find_flow_centre
from data_analysis.plasma.modes import (
    axial_wavenumber,
    bdot_cartesian_dipole_check,
    fit_azimuthal_mode,
    plasma_frame_dispersion,
)
from data_analysis.signal import avg_cross_spectrum, clip_time_window
from data_analysis.utils import merge_save_npz

import Jun2026_mach_line as jml
import Jun2026_onset as jon
import Jun2026_plot as jpl
import Jun2026_profiles as jpr

# --- Campaign constants -----------------------------------------------------
DATA_DIR = jon.DATA_DIR
OUT_NAME = "modes_dispersion_jun2026.npz"

# Axial port spacing: LAPD ports are spaced by 31.95 cm (4 ports P25->P29 = 1.278 m).
# Sign convention: +z points from P25 toward P29 (toward the Port-35 bias plate),
# so dz = z(P29) - z(P25) = +1.28 m. Reversing +z flips the sign of k_par while
# leaving |k_par| and f_Alfven unchanged. Aliasing limit: |k_par| = pi/1.28 = 2.45 /m.
DZ_P29_P25_M = 1.28

# Bdot-10T-C16 HDF5 run descriptions record `2.5BLP => X10 => 2.5BLP => X100`
# (G = 1000.0), while the campaign-independent bare 3x3 effective area `.npz`
# is generated once per probe by `experiments/ucla-lapd/bdot_cal.py` (or
# standalone `nAcalc.py`). By default, estimate_bdot_amplitude loads
# `DEFAULT_BDOT_CAL` and auto-reads the P28 signal chain from each run's HDF5
# description (raising ValueError if absent rather than silently using 1.0);
# pass `bdot_cal=None` explicitly for uncalibrated raw V_perp / omega output.
_UNSET = object()
BDOT_PROBE_PORT = "P28"
DEFAULT_BDOT_CAL = r"D:\data\LAPD\BDOT CALIBRATION\Bdot-10T-C16\C16-bdot-calibration.npz"
DEFAULT_BDOT_AREA_M2 = None
DEFAULT_BDOT_GAIN = None
DEFAULT_BDOT_ATTEN_DB = None
# Matches Jun2026_bdot_2d.DEFAULT_CHANNEL_SIGNS: uniquely verified from Maxwell
# div/curl, By->Pz cross-talk cancellation, DC turn-on/off Ampère's law, and core dB_z.
DEFAULT_BDOT_CHANNEL_SIGNS = (-1, -1, +1)

# LAPD magnetic field polarity: +1 -> B_0 along +z so outward E_r > 0 gives
# clockwise E x B rotation (v_theta < 0, matching the observed m = -1 mode).
B_SIGN_Z = +1
DISPERSION_R_CM = 3.0            # inner-wall evaluation radius [cm]
DISPERSION_R_HALF_CM = 1.0       # matched radial half-width [r_cm - 1, r_cm + 1] cm
DEFAULT_M = -1                   # verified physical azimuthal mode number (CW)

BIASSCOPE_OFFSET_MS = 12.0       # t_biasscope - t_machscope at same instant
BIASSCOPE_WINDOW_MS = (1.5, 4.5) # in machscope (bias-relative) time [ms]
BIASSCOPE_TARGET_DT_S = 1e-5     # 100 kHz anti-aliased working grid
BIASSCOPE_NPERSEG = 256          # df = 390.6 Hz at 100 kHz
BIASSCOPE_MAX_SHOTS = 60


def _run_ifn(data_dir, run_num):
    hits = sorted(glob.glob(os.path.join(data_dir, f"{run_num}-*.hdf5")))
    if not hits:
        raise FileNotFoundError(f"no HDF5 for run {run_num} in {data_dir}")
    if len(hits) > 1:
        raise ValueError(f"multiple HDF5 files for run {run_num}: {hits}")
    return hits[0]


def _require_xcorr(ifn, ch_a, ch_b, npz_path):
    """Load an xcorr pair or raise ``KeyError`` (instead of returning ``None``)."""
    loaded = jpl._load_xcorr_run(ifn, ch_a, ch_b, npz_path=npz_path)
    if loaded is None:
        raise KeyError(f"pair {ch_a} vs {ch_b} not found in {npz_path}")
    return loaded


def _blockmean_to_grid(t_src, stack, t_grid):
    """Anti-aliased block-mean downsample before sub-sample interpolation onto ``t_grid``.

    Prevents high-frequency RF/digitizer noise (e.g. from 50 MHz ``machscope``
    or 5 MHz ``biasscope``) from folding into the low-frequency band when
    resampling onto a 100 kHz common time grid.
    """
    t_src = np.asarray(t_src, dtype=float)
    arr = np.asarray(stack, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    dt_src = float(t_src[1] - t_src[0])
    dt_tgt = float(t_grid[1] - t_grid[0])
    q = max(1, int(round(dt_tgt / dt_src)))
    if q > 1:
        nblk = t_src.size // q
        if nblk < 2:
            raise ValueError(f"record too short for block-mean factor q={q}")
        t_ds = t_src[:nblk * q].reshape(nblk, q).mean(axis=1)
        s_ds = arr[:, :nblk * q].reshape(arr.shape[0], nblk, q).mean(axis=2)
    else:
        t_ds, s_ds = t_src, arr
    return np.vstack([np.interp(t_grid, t_ds, row) for row in s_ds])


def _nearest_freq_idx(freq_hz, target_hz):
    return int(np.argmin(np.abs(freq_hz - float(target_hz))))


def _snap_to_grid(px, py, vals, step_cm=0.5):
    """Snap scattered 2-D positions (absorbing ~5 um encoder jitter) to a grid."""
    xs = np.round(px / step_cm) * step_cm
    ys = np.round(py / step_cm) * step_cm
    ux = np.unique(xs)
    uy = np.unique(ys)
    grid = np.full((uy.size, ux.size), np.nan, dtype=float)
    ix = np.searchsorted(ux, xs)
    iy = np.searchsorted(uy, ys)
    grid[iy, ix] = vals
    return ux, uy, grid


def analyze_azimuthal_modes(data_dir=DATA_DIR, onset_table=None):
    """Fit physical ``m`` and B-dot dipole quadrature on Runs 26, 31, and 29."""
    results = {}

    # 1. B-dot + LP planes: Run 26 (800 G) and Run 31 (600 G)
    for rnum, b0_g, f_fallback in [("26", 800.0, 4110.0), ("31", 600.0, 3530.0)]:
        npz_p = os.path.join(data_dir, f"{rnum}-xcorr-data.npz")
        if not os.path.isfile(npz_p):
            continue
        f0 = float(onset_table[rnum]["f_peak_hz"]) if (onset_table and rnum in onset_table) else f_fallback
        f2 = 2.0 * f0

        # LP moving P29-R vs stationary P25-L (moving_is_a=True)
        freq, g_lp, p_lp, px_lp, py_lp = _require_xcorr(
            f"{rnum}-dummy.hdf5", ("lpscope", "C2"), ("lpscope", "C3"), npz_p
        )
        i1 = _nearest_freq_idx(freq, f0)
        i2 = _nearest_freq_idx(freq, f2)
        m_lp1 = fit_azimuthal_mode(px_lp, py_lp, g_lp[:, i1], p_lp[:, i1], moving_is_a=True)

        # B-dot stationary P25-L vs moving P28 Bx/By (moving_is_a=False)
        _, g_bx, p_bx, px_b, py_b = _require_xcorr(
            f"{rnum}-dummy.hdf5", ("lpscope", "C3"), ("bdotscope", "C1"), npz_p
        )
        _, g_by, p_by, _, _ = _require_xcorr(
            f"{rnum}-dummy.hdf5", ("lpscope", "C3"), ("bdotscope", "C2"), npz_p
        )
        m_bx1 = fit_azimuthal_mode(px_b, py_b, g_bx[:, i1], p_bx[:, i1], moving_is_a=False)
        m_by1 = fit_azimuthal_mode(px_b, py_b, g_by[:, i1], p_by[:, i1], moving_is_a=False)
        m_bx2 = fit_azimuthal_mode(
            px_b, py_b, g_bx[:, i2], p_bx[:, i2],
            center=(m_bx1.center_x_cm, m_bx1.center_y_cm), moving_is_a=False,
        )
        dip = bdot_cartesian_dipole_check(
            p_bx[:, i1], p_by[:, i1], g_bx[:, i1], g_by[:, i1], moving_is_a=False
        )

        results[rnum] = {
            "b0_g": b0_g,
            "f1_hz": float(freq[i1]),
            "f2_hz": float(freq[i2]),
            "lp_m": m_lp1.m, "lp_purity": m_lp1.purity,
            "lp_cx_cm": m_lp1.center_x_cm, "lp_cy_cm": m_lp1.center_y_cm,
            "bx_m": m_bx1.m, "bx_purity": m_bx1.purity,
            "bx_cx_cm": m_bx1.center_x_cm, "bx_cy_cm": m_bx1.center_y_cm,
            "by_m": m_by1.m, "by_purity": m_by1.purity,
            "bx_h2_m": m_bx2.m, "bx_h2_purity": m_bx2.purity,
            "bdot_dphi_by_bx_deg": dip.dphi_by_minus_bx_deg,
            "bdot_dipole_R": dip.resultant_R,
            "bdot_inferred_m_j": dip.inferred_m_j,
            "bdot_n_pts": dip.n_points,
        }

    # 2. Run 29 two-phase 31x31 LP plane: R-facing tips (P29-R C2 and P25-R C4)
    #    consistently match analyze_axial_kpar; P29-L (C1) is stored beside them.
    npz29 = os.path.join(data_dir, "29-xcorr-data.npz")
    if os.path.isfile(npz29):
        f0 = float(onset_table["29"]["f_peak_hz"]) if (onset_table and "29" in onset_table) else 4260.0
        freq, g_p29r, p_p29r, px29, py29 = _require_xcorr(
            "29-dummy.hdf5", ("scope", "C2"), ("scope", "C5"), npz29
        )
        _, g_p29l, p_p29l, _, _ = _require_xcorr(
            "29-dummy.hdf5", ("scope", "C1"), ("scope", "C5"), npz29
        )
        _, g_p25r, p_p25r, px25, py25 = _require_xcorr(
            "29-dummy.hdf5", ("scope", "C4"), ("scope", "C5"), npz29
        )
        i1 = _nearest_freq_idx(freq, f0)
        m29r = fit_azimuthal_mode(px29, py29, g_p29r[:, i1], p_p29r[:, i1], r_max_cm=6.5, moving_is_a=True)
        m29l = fit_azimuthal_mode(px29, py29, g_p29l[:, i1], p_p29l[:, i1], r_max_cm=6.5, moving_is_a=True)
        m25r = fit_azimuthal_mode(px25, py25, g_p25r[:, i1], p_p25r[:, i1], r_max_cm=6.5, moving_is_a=True)
        results["29"] = {
            "b0_g": 800.0,
            "f1_hz": float(freq[i1]),
            "p29_r_m": m29r.m, "p29_r_purity": m29r.purity,
            "p29_r_cx_cm": m29r.center_x_cm, "p29_r_cy_cm": m29r.center_y_cm,
            "p29_l_m": m29l.m, "p29_l_purity": m29l.purity,
            "p29_l_cx_cm": m29l.center_x_cm, "p29_l_cy_cm": m29l.center_y_cm,
            "p25_r_m": m25r.m, "p25_r_purity": m25r.purity,
            "p25_r_cx_cm": m25r.center_x_cm, "p25_r_cy_cm": m25r.center_y_cm,
        }

    return results


def analyze_axial_kpar(data_dir=DATA_DIR, onset_table=None, *,
                       dz_m=DZ_P29_P25_M, center_2d_cm=(1.5, 0.0),
                       inner_r_max_cm=4.0, gamma2_min=0.7):
    """Estimate axial wavenumber ``k_par`` across Runs 28 (600G), 27/29 (800G), 30 (1kG).

    ``center_2d_cm = (cx, cy)`` sets the radial split center for the 2-D Run 29
    plane (and ``cx`` for the 1-D lines Runs 27, 28, 30).
    """
    cx, cy = float(center_2d_cm[0]), float(center_2d_cm[1])
    results = {}

    # 1-D axial lines: direct P29-R (C2) vs P25-R (C4) phase = phi(P29) - phi(P25)
    for rnum, b0_g, f_fallback in [("28", 600.0, 3529.0), ("27", 800.0, 4079.0), ("30", 1000.0, 4501.0)]:
        npz_p = os.path.join(data_dir, f"{rnum}-xcorr-data.npz")
        if not os.path.isfile(npz_p):
            continue
        f0 = float(onset_table[rnum]["f_peak_hz"]) if (onset_table and rnum in onset_table) else f_fallback
        freq, g2, ph, px, _ = _require_xcorr(
            f"{rnum}-dummy.hdf5", ("scope", "C2"), ("scope", "C4"), npz_p
        )
        i1 = _nearest_freq_idx(freq, f0)
        r = np.abs(px - cx)
        inner = r <= inner_r_max_cm
        outer = (r > inner_r_max_cm) & (r <= 10.0)
        k_in = axial_wavenumber(ph[inner, i1], dz_m, g2[inner, i1], gamma2_min=gamma2_min)
        k_out = axial_wavenumber(ph[outer, i1], dz_m, g2[outer, i1], gamma2_min=gamma2_min)
        results[rnum] = {
            "b0_g": b0_g, "f1_hz": float(freq[i1]),
            "inner_dphi_deg": k_in.mean_dphi_deg, "inner_R": k_in.resultant_R,
            "inner_k_par_m": k_in.k_par_rad_m, "inner_lambda_par_m": k_in.lambda_par_m,
            "inner_n": k_in.n_points,
            "outer_dphi_deg": k_out.mean_dphi_deg, "outer_R": k_out.resultant_R,
            "outer_k_par_m": k_out.k_par_rad_m, "outer_lambda_par_m": k_out.lambda_par_m,
            "outer_n": k_out.n_points,
        }

    # 2-D two-phase plane (Run 29): indirect phi(P29-R vs P33) - phi(P25-R vs P33)
    npz29 = os.path.join(data_dir, "29-xcorr-data.npz")
    if os.path.isfile(npz29):
        f0 = float(onset_table["29"]["f_peak_hz"]) if (onset_table and "29" in onset_table) else 4261.0
        freq, g29, p29, px29, py29 = _require_xcorr(
            "29-dummy.hdf5", ("scope", "C2"), ("scope", "C5"), npz29
        )
        _, g25, p25, px25, py25 = _require_xcorr(
            "29-dummy.hdf5", ("scope", "C4"), ("scope", "C5"), npz29
        )
        i1 = _nearest_freq_idx(freq, f0)
        ux1, uy1, G29 = _snap_to_grid(px29, py29, g29[:, i1])
        _, _, P29 = _snap_to_grid(px29, py29, p29[:, i1])
        ux2, uy2, G25 = _snap_to_grid(px25, py25, g25[:, i1])
        _, _, P25 = _snap_to_grid(px25, py25, p25[:, i1])
        if np.allclose(ux1, ux2) and np.allclose(uy1, uy2):
            XX, YY = np.meshgrid(ux1, uy1)
            RR = np.hypot(XX - cx, YY - cy)
            dphi = np.angle(np.exp(1j * (P29 - P25)))
            inner = RR <= inner_r_max_cm
            outer = RR > inner_r_max_cm
            k_in = axial_wavenumber(dphi[inner], dz_m, G29[inner], G25[inner], gamma2_min=gamma2_min)
            k_out = axial_wavenumber(dphi[outer], dz_m, G29[outer], G25[outer], gamma2_min=gamma2_min)
            results["29"] = {
                "b0_g": 800.0, "f1_hz": float(freq[i1]),
                "center_x_cm": cx, "center_y_cm": cy,
                "inner_dphi_deg": k_in.mean_dphi_deg, "inner_R": k_in.resultant_R,
                "inner_k_par_m": k_in.k_par_rad_m, "inner_lambda_par_m": k_in.lambda_par_m,
                "inner_n": k_in.n_points,
                "outer_dphi_deg": k_out.mean_dphi_deg, "outer_R": k_out.resultant_R,
                "outer_k_par_m": k_out.k_par_rad_m, "outer_lambda_par_m": k_out.lambda_par_m,
                "outer_n": k_out.n_points,
            }

    return results


def estimate_bdot_amplitude(data_dir=DATA_DIR, onset_table=None, *,
                            bdot_cal=_UNSET,
                            bdot_area_m2=DEFAULT_BDOT_AREA_M2,
                            bdot_gain=DEFAULT_BDOT_GAIN,
                            bdot_atten_db=DEFAULT_BDOT_ATTEN_DB,
                            channel_signs=DEFAULT_BDOT_CHANNEL_SIGNS,
                            max_shots=60):
    """Measure P28 B-dot p95 RMS voltage and (when calibrated) ``delta_B_perp / B_0``.

    Always computes raw ``v_perp_p95_v`` [V] and ``v_perp_over_omega_vs``
    [V*s = ``gain * area * delta_B_perp``]. Defaults to ``DEFAULT_BDOT_CAL``
    (bare 3x3 ``.npz``) and auto-reads ``bdot_gain``/``bdot_atten_db`` from the
    run's P28 HDF5 description; converts to real probe-terminal voltage and
    applies 3-axis :func:`data_analysis.plasma.bdot.voltage_to_bdot` **before**
    taking RMS, ``hypot(dBx, dBy)``, and the 95th percentile. Pass
    ``bdot_cal=None`` explicitly for uncalibrated output.
    """
    if bdot_cal is _UNSET:
        bdot_cal = None if bdot_area_m2 is not None else DEFAULT_BDOT_CAL
    if bdot_cal is not None and bdot_area_m2 is not None:
        raise ValueError("Pass either bdot_cal or bdot_area_m2, not both")

    probe_id = ""
    if bdot_cal is not None:
        cal_obj = bdot_cal if isinstance(bdot_cal, BdotCalibration) else load_bdot_calibration(bdot_cal)
        area_mat = cal_obj.area_matrix_m2.copy()
        probe_id = str(cal_obj.probe_id)
        calibrated = True
    elif bdot_area_m2 is not None:
        cal_obj = bdot_area_m2
        a = np.asarray(bdot_area_m2, dtype=float)
        if a.ndim == 0:
            area_mat = np.eye(3, dtype=float) * float(a)
        elif a.shape == (3,):
            area_mat = np.diag(a)
        elif a.shape == (3, 3):
            area_mat = a.copy()
        else:
            raise ValueError(f"bdot_area_m2 must be scalar, (3,), or (3, 3); got shape {a.shape}")
        calibrated = True
    else:
        cal_obj = None
        area_mat = np.full((3, 3), np.nan, dtype=float)
        calibrated = False

    signs = np.asarray(channel_signs, dtype=float)
    if signs.shape != (3,) or not np.all(np.isin(signs, (-1.0, +1.0))):
        raise ValueError(f"channel_signs must be 3 values in {{-1, +1}}, got {channel_signs!r}")

    # Real probe-level inverse area matrix (experiment gain removed before voltage_to_bdot)
    inv_area = inverse_area_matrix(cal_obj, gain=1.0) if calibrated else None

    out = {}
    for rnum, b0_g, f_fallback in [("26", 800.0, 4110.0), ("31", 600.0, 3529.0)]:
        try:
            ifn = _run_ifn(data_dir, rnum)
        except FileNotFoundError:
            continue
        f0 = float(onset_table[rnum]["f_peak_hz"]) if (onset_table and rnum in onset_table) else f_fallback
        run = open_lapd(ifn)
        try:
            raw_desc = run.description().raw
        except (OSError, ValueError, NotImplementedError, KeyError):
            raw_desc = None
        parsed_chain = parse_bdot_gain(raw_desc, probe=BDOT_PROBE_PORT) if raw_desc else None
        signal_chain = parsed_chain["chain"] if parsed_chain else ""

        if not calibrated:
            gain_used = 1.0 if bdot_gain is None else float(bdot_gain)
            atten_used = 0.0 if bdot_atten_db is None else float(bdot_atten_db)
        elif bdot_gain is None:
            if parsed_chain is None:
                raise ValueError(
                    f"Run {rnum}: B-dot calibration requested with bdot_gain=None, "
                    f"but no active {BDOT_PROBE_PORT} B-dot signal chain was found in "
                    "the HDF5 description; pass bdot_gain= explicitly."
                )
            gain_used = float(parsed_chain["gain"])
            atten_used = float(parsed_chain["atten_db"]) if bdot_atten_db is None else float(bdot_atten_db)
        else:
            gain_used = float(bdot_gain)
            atten_used = 0.0 if bdot_atten_db is None else float(bdot_atten_db)

        eff_gain = gain_used * (10.0 ** (-atten_used / 20.0))
        if eff_gain <= 0.0:
            raise ValueError(f"Run {rnum}: effective bdot gain must be > 0, got {eff_gain}")
        if calibrated and parsed_chain is not None:
            if not np.isclose(eff_gain, parsed_chain["effective_gain"], rtol=1e-3, atol=0.0):
                warnings.warn(
                    f"Run {rnum}: requested bdot effective gain {eff_gain:g} differs from "
                    f"HDF5 description '{parsed_chain['chain']}' ({parsed_chain['effective_gain']:g})",
                    UserWarning,
                    stacklevel=2,
                )

        tarr = run.time_array(scope_name="bdotscope")
        i0, i1 = clip_time_window(tarr, *BIASSCOPE_WINDOW_MS)
        all_shots = np.asarray(run.shots(scope_name="bdotscope"), dtype=int)
        idx = np.unique(np.linspace(0, all_shots.size - 1, min(all_shots.size, max_shots), dtype=int))
        shots = all_shots[idx].tolist()

        segs = []
        v_rms = {}
        for ch, name in [("C1", "Bx"), ("C2", "By"), ("C3", "Bz")]:
            st, _ = run.channel(ch, scope_name="bdotscope", shots=shots)
            seg = np.asarray(st[:, i0:i1], dtype=float)
            seg = seg - seg.mean(axis=1, keepdims=True)
            segs.append(seg)
            v_rms[name] = np.std(seg, axis=1)

        # Peak over the sampled scan positions (95th percentile avoids single-shot spikes)
        v_perp_pk = float(np.percentile(np.hypot(v_rms["Bx"], v_rms["By"]), 95))
        v_z_pk = float(np.percentile(v_rms["Bz"], 95))
        omega = 2.0 * np.pi * f0

        if calibrated:
            # Stack (3, nshot, nt) -> real probe-terminal voltage -> 3x3 dB/dt -> RMS/hypot
            v_stack = np.stack(segs, axis=0)
            v_coil = (signs[:, None, None] * v_stack) / eff_gain
            bd_stack = voltage_to_bdot(v_coil, inv_area)
            bd_rms = np.std(bd_stack, axis=-1)  # (3, nshot), T/s
            b_perp_t = float(np.percentile(np.hypot(bd_rms[0], bd_rms[1]), 95)) / omega
            b_z_t = float(np.percentile(bd_rms[2], 95)) / omega
            b0_t = b0_g * 1e-4
            db_perp_g = b_perp_t * 1e4
            db_z_g = b_z_t * 1e4
            db_over_b0 = b_perp_t / b0_t
        else:
            db_perp_g = db_z_g = db_over_b0 = np.nan

        out[rnum] = {
            "b0_g": b0_g,
            "f0_hz": f0,
            "v_perp_p95_v": v_perp_pk,
            "v_z_p95_v": v_z_pk,
            "v_perp_over_omega_vs": v_perp_pk / omega,
            "v_z_over_omega_vs": v_z_pk / omega,
            "delta_b_perp_g": db_perp_g,
            "delta_b_z_g": db_z_g,
            "delta_b_perp_over_b0": db_over_b0,
            "bdot_probe_id": probe_id,
            "bdot_area_matrix_m2": area_mat,
            "bdot_signal_chain": signal_chain,
            "bdot_gain": float(gain_used),
            "bdot_atten_db": float(atten_used),
            "bdot_calibrated": bool(calibrated),
        }
    return out


def analyze_biasscope_coherence(data_dir=DATA_DIR, runs=("06", "00", "01", "02", "03", "04", "05"),
                                onset_table=None, max_shots=BIASSCOPE_MAX_SHOTS):
    """Cross-scope coherence between P33 Isat (``machscope/C2``) and ``biasscope``.

    Anti-aliases each scope onto the 100 kHz common grid via block-mean
    downsampling (:func:`_blockmean_to_grid`) before cross-spectral estimation.
    Evaluates all channel coherences/phases AND direct ``I_2in`` vs. ``I_4in``
    cross-phase at **one shared fundamental frequency bin** per run so phase
    differences are never computed across mismatched frequency bins.
    """
    channels = [("C1", "I_2in"), ("C2", "I_4in"), ("C3", "I_6in"),
                ("C4", "V_2in"), ("C8", "Light_P30")]
    t0_ms, t1_ms = BIASSCOPE_WINDOW_MS
    t_grid = np.arange(t0_ms * 1e-3, t1_ms * 1e-3, BIASSCOPE_TARGET_DT_S)
    results = {}

    for rnum in runs:
        ifn = _run_ifn(data_dir, rnum)
        run = open_lapd(ifn)
        tm = run.time_array(scope_name="machscope")
        tb = run.time_array(scope_name="biasscope")
        im0, im1 = clip_time_window(tm, t0_ms, t1_ms)
        ib0, ib1 = clip_time_window(tb, t0_ms + BIASSCOPE_OFFSET_MS, t1_ms + BIASSCOPE_OFFSET_MS)
        tm_sub = tm[im0:im1]
        tb_sub = tb[ib0:ib1] - BIASSCOPE_OFFSET_MS * 1e-3

        all_shots = np.asarray(run.shots(scope_name="machscope"), dtype=int)
        idx = np.unique(np.linspace(0, all_shots.size - 1, min(all_shots.size, max_shots), dtype=int))
        shots = all_shots[idx].tolist()

        st_m, _ = run.channel("C2", scope_name="machscope", shots=shots)
        s_isat = _blockmean_to_grid(tm_sub, st_m[:, im0:im1], t_grid)

        stacks = {}
        spectra = {}
        freq = None
        for ch, lbl in channels:
            st_b, _ = run.channel(ch, scope_name="biasscope", shots=shots)
            s_b = _blockmean_to_grid(tb_sub, st_b[:, ib0:ib1], t_grid)
            stacks[lbl] = s_b
            freq, g2, ph, _ = avg_cross_spectrum(
                s_isat, s_b, BIASSCOPE_TARGET_DT_S, nperseg=BIASSCOPE_NPERSEG
            )
            spectra[lbl] = (g2, ph)

        # Pick ONE shared fundamental bin per run: nearest to onset f_peak_hz
        # when available, else peak of g2(I_2in) * g2(I_4in) inside 2.5-5.5 kHz.
        band = (freq >= 2500.0) & (freq <= 5500.0)
        bidx = np.flatnonzero(band)
        f_target = (float(onset_table[rnum]["f_peak_hz"])
                    if (onset_table and rnum in onset_table and np.isfinite(onset_table[rnum]["f_peak_hz"]))
                    else np.nan)
        if np.isfinite(f_target) and 2500.0 <= f_target <= 5500.0:
            k_shared = _nearest_freq_idx(freq, f_target)
        else:
            joint = spectra["I_2in"][0][bidx] * spectra["I_4in"][0][bidx]
            k_shared = int(bidx[int(np.argmax(joint))])

        row_res = {"shared_f_hz": float(freq[k_shared])}
        for _, lbl in channels:
            g2, ph = spectra[lbl]
            s_b = stacks[lbl]
            mean_dc = float(np.mean(s_b))
            rel_rms = float(np.std(s_b - s_b.mean(axis=1, keepdims=True)) / max(abs(mean_dc), 1e-6))
            row_res[f"{lbl}_g2"] = float(g2[k_shared])
            row_res[f"{lbl}_phase_deg"] = float(np.degrees(ph[k_shared]))
            row_res[f"{lbl}_rel_rms"] = rel_rms

        # Direct I_2in vs I_4in cross-spectrum at the exact same shared bin
        # (avg_cross_spectrum(I_2in, I_4in) gives phase of I_4in relative to I_2in,
        # or equivalently -angle(P_24); note core.py: phase is y relative to x).
        _, g2_24, ph_24, _ = avg_cross_spectrum(
            stacks["I_2in"], stacks["I_4in"],
            BIASSCOPE_TARGET_DT_S, nperseg=BIASSCOPE_NPERSEG,
        )
        row_res["g2_I2_I4"] = float(g2_24[k_shared])
        row_res["dphi_I4_minus_I2_deg"] = float(np.degrees(ph_24[k_shared]))
        results[rnum] = row_res

    return results


def build_dispersion_table(data_dir=DATA_DIR, *, r_cm=DISPERSION_R_CM,
                           r_half_cm=DISPERSION_R_HALF_CM,
                           m=DEFAULT_M, b_sign_z=B_SIGN_Z, k_par_m=0.65):
    """Combine Modules #1-#3 into plasma-frame dispersion on a **matched** radial band.

    Evaluates ``ne``, ``Te``, ``dln_ne_dr``, ``dln_te_dr``, ``v_ExB``, and
    Mach ``dv_rot`` consistently over the exact same radial band
    ``r in [r_cm - r_half_cm, r_cm + r_half_cm]`` so Doppler shifts from
    Langmuir ``E x B`` and P33 Mach are compared at identical radii.

    Uses the **signed** measured velocities under one shared lab convention
    (``v_theta > 0`` CCW, ``v_theta < 0`` CW):
    - ``v_ExB_r`` stores signed outward ``E_r / B_0``, so azimuthal
      ``v_theta = -b_sign_z * (E_r / B_0)`` (clockwise ``< 0`` when ``E_r > 0``
      and ``B_0`` points along ``+z``).
    - Mach ``dv_rot = 0.5 * (dv_Y(x_c + r) - dv_Y(x_c - r))`` is already
      signed lab-frame ``v_theta`` using the unified :mod:`data_analysis.plasma.mach`
      lab-axis sign convention (``FLOW_SIGN_PLUS_FACING_AXIS = -1``).
    """
    onset = jon.load_onset_table(os.path.join(data_dir, jon.OUT_NAME))
    prof_p = os.path.join(data_dir, jpr.OUT_NAME)
    mach_p = os.path.join(data_dir, jml.RESCALED_OUT_NAME)
    iv_to_mach = {v: k for k, v in jml.MACH_TO_IV.items()}

    rows = {}
    with np.load(prof_p) as pd, np.load(mach_p) as md:
        for spec in jpr.PROFILE_RUNS:
            ir = spec.run_num
            mr = iv_to_mach[ir]
            on = onset[ir]
            f_lab = float(on["f_peak_hz"])
            gamma = float(on["gamma_med_s"])
            unstable = bool(np.isfinite(on["t_onset_med_ms"]))
            f_lab_eval = f_lab if unstable else np.nan

            ref_swp = int(pd[f"{ir}__ref_sweep_idx"])
            ref_bin = int(md[f"{mr}__ref_bin_idx"])
            if ref_swp < 0:
                raise ValueError(f"{prof_p}: IV run {ir} has no valid reference sweep (ref_sweep_idx={ref_swp})")
            if ref_bin < 0:
                raise ValueError(f"{mach_p}: Mach run {mr} has no valid reference bin (ref_bin_idx={ref_bin})")

            # Matched radial band [r_cm - r_half_cm, r_cm + r_half_cm] on BOTH IV and Mach
            x_iv = pd[f"{ir}__x_cm"]
            r_iv = np.abs(x_iv - float(pd[f"{ir}__x_center_cm"]))
            band_iv = (r_iv >= r_cm - r_half_cm) & (r_iv <= r_cm + r_half_cm)

            r_m = md[f"{mr}__r_cm"]
            band_m = (r_m >= r_cm - r_half_cm) & (r_m <= r_cm + r_half_cm)

            ne_w = float(np.nanmedian(pd[f"{ir}__ne"][band_iv, ref_swp]))
            te_w = float(np.nanmedian(pd[f"{ir}__Te"][band_iv, ref_swp]))
            dln_n = float(np.nanmedian(pd[f"{ir}__dln_ne_dr"][band_iv, ref_swp]))
            dln_t = float(np.nanmedian(pd[f"{ir}__dln_te_dr"][band_iv, ref_swp]))

            # Signed azimuthal velocities on the exact same matched radial band [r_cm +- r_half_cm]:
            # v_ExB_r = Er / B0 -> v_theta = -b_sign_z * (Er / B0)
            # dv_rot  = 0.5*(dv_Y(+r) - dv_Y(-r)) is already signed lab-frame v_theta
            v_exb_band = md[f"{mr}__v_ExB_r_km_s"][band_m, ref_bin] * 1e3  # m/s
            v_mach_band = md[f"{mr}__dv_rot_km_s"][band_m, ref_bin] * 1e3  # m/s
            v_rot_exb = -int(np.sign(b_sign_z or 1)) * float(np.nanmedian(v_exb_band))
            v_rot_mach = float(np.nanmedian(v_mach_band))

            d_exb = plasma_frame_dispersion(
                f_lab_eval, m=m, r_cm=r_cm, v_rot_m_s=v_rot_exb,
                b0_g=spec.b0_g, ne_cm3=ne_w, te_ev=te_w,
                dln_ne_dr_cm=dln_n, dln_te_dr_cm=dln_t,
                k_par_m=k_par_m, mu=4.0, b_sign_z=b_sign_z,
            )
            d_mach = plasma_frame_dispersion(
                f_lab_eval, m=m, r_cm=r_cm, v_rot_m_s=v_rot_mach,
                b0_g=spec.b0_g, ne_cm3=ne_w, te_ev=te_w,
                dln_ne_dr_cm=dln_n, dln_te_dr_cm=dln_t,
                k_par_m=k_par_m, mu=4.0, b_sign_z=b_sign_z,
            )
            rows[ir] = {
                "gas_puff_ms": float(spec.gas_puff_ms),
                "unstable": unstable,
                "f_lab_hz": f_lab,
                "gamma_s": gamma,
                "gamma_over_omega": float(gamma / (2.0 * np.pi * f_lab)) if (unstable and np.isfinite(gamma) and f_lab > 0) else np.nan,
                "f_doppler_exb_hz": d_exb["f_doppler_hz"],
                "f_plasma_exb_hz": d_exb["f_plasma_hz"],
                "f_doppler_mach_hz": d_mach["f_doppler_hz"],
                "f_plasma_mach_hz": d_mach["f_plasma_hz"],
                "f_star_n_hz": d_exb["f_star_n_hz"],
                "f_star_T_hz": d_exb["f_star_T_hz"],
                "f_star_p_hz": d_exb["f_star_p_hz"],
                "f_alfven_hz": d_exb["f_alfven_hz"],
                "k_perp_rho_s": d_exb["k_perp_rho_s"],
                "beta_e": d_exb["beta_e"],
                "beta_e_over_me_mi": d_exb["beta_e_over_me_mi"],
                "nu_ei_over_omega": d_exb["nu_ei_over_omega_lab"],
            }
    return rows


def batch_all(data_dir=DATA_DIR, out_path=None, *, include_biasscope=True,
              include_bdot_amp=True, bdot_cal=_UNSET,
              bdot_area_m2=DEFAULT_BDOT_AREA_M2,
              bdot_gain=DEFAULT_BDOT_GAIN,
              bdot_atten_db=DEFAULT_BDOT_ATTEN_DB,
              bdot_channel_signs=DEFAULT_BDOT_CHANNEL_SIGNS,
              overwrite=False):
    """Run all Module #4 analyses and save ``modes_dispersion_jun2026.npz``.

    Skips when ``out_path`` already exists unless ``overwrite=True``.
    """
    out_path = out_path or os.path.join(data_dir, OUT_NAME)
    if os.path.isfile(out_path):
        if not overwrite:
            print(f"  {os.path.basename(out_path)} exists, skipping (pass overwrite=True to regenerate)")
            return out_path
        os.remove(out_path)

    onset = jon.load_onset_table(os.path.join(data_dir, jon.OUT_NAME))
    az = analyze_azimuthal_modes(data_dir, onset)
    # Use the fitted 2-D B-dot mode center (Run 26) for the Run 29 k_par radial split
    # when available, falling back to (1.5, 0.0) cm.
    if "26" in az and np.isfinite(az["26"]["bx_cx_cm"]):
        center_2d = (az["26"]["bx_cx_cm"], az["26"]["bx_cy_cm"])
    else:
        center_2d = (1.5, 0.0)
    ax = analyze_axial_kpar(data_dir, onset, center_2d_cm=center_2d)
    k_par_ref = abs(ax["29"]["inner_k_par_m"]) if "29" in ax else 0.65
    disp = build_dispersion_table(data_dir, k_par_m=k_par_ref)
    bdot_amp = (estimate_bdot_amplitude(
        data_dir, onset, bdot_cal=bdot_cal, bdot_area_m2=bdot_area_m2,
        bdot_gain=bdot_gain, bdot_atten_db=bdot_atten_db,
        channel_signs=bdot_channel_signs,
    ) if include_bdot_amp else {})
    bias_coh = (analyze_biasscope_coherence(data_dir, onset_table=onset)
                if include_biasscope else {})

    # Also record Run 32 3-D Mach plane fitted rotation center for cross-check
    p32 = os.path.join(data_dir, "32-mach-flow-data.npz")
    r32_cx = r32_cy = r32_ratio = np.nan
    if os.path.isfile(p32):
        with np.load(p32) as d32:
            t32 = d32["t_ms"]
            m_win = (t32 >= 0.5) & (t32 <= 1.5)
            m_pre = t32 < -0.5
            vx_sub = d32["v_X"][:, m_win]
            vy_sub = d32["v_Y"][:, m_win]
            vx = np.full(vx_sub.shape[0], np.nan, dtype=float)
            vy = np.full(vy_sub.shape[0], np.nan, dtype=float)
            ok_x = np.any(np.isfinite(vx_sub), axis=1)
            ok_y = np.any(np.isfinite(vy_sub), axis=1)
            vx[ok_x] = np.nanmean(vx_sub[ok_x], axis=1)
            vy[ok_y] = np.nanmean(vy_sub[ok_y], axis=1)
            if np.any(m_pre):
                vx_p = d32["v_X"][:, m_pre]
                vy_p = d32["v_Y"][:, m_pre]
                ok_xp = ok_x & np.any(np.isfinite(vx_p), axis=1)
                ok_yp = ok_y & np.any(np.isfinite(vy_p), axis=1)
                vx[~ok_xp] = np.nan
                vy[~ok_yp] = np.nan
                vx[ok_xp] -= np.nanmean(vx_p[ok_xp], axis=1)
                vy[ok_yp] -= np.nanmean(vy_p[ok_yp], axis=1)
            r32_cx, r32_cy, r32_ratio = find_flow_centre(vx, vy, d32["pos_x"], d32["pos_y"])

    payload = {
        "updated": np.str_(datetime.datetime.now().isoformat(timespec="seconds")),
        "dz_p29_p25_m": np.float64(DZ_P29_P25_M),
        "dispersion_r_cm": np.float64(DISPERSION_R_CM),
        "k_par_ref_m": np.float64(k_par_ref),
        "r32_center_x_cm": np.float64(r32_cx),
        "r32_center_y_cm": np.float64(r32_cy),
        "r32_center_ratio": np.float64(r32_ratio),
    }

    def _pack(prefix, nested):
        for rnum, fields in nested.items():
            for k, v in fields.items():
                key = f"{prefix}__{rnum}__{k}"
                if isinstance(v, np.ndarray):
                    payload[key] = v
                elif isinstance(v, str):
                    payload[key] = np.str_(v)
                elif isinstance(v, (bool, np.bool_)):
                    payload[key] = np.bool_(v)
                elif isinstance(v, (int, np.integer)):
                    payload[key] = np.int64(v)
                else:
                    payload[key] = np.float64(v)

    _pack("az", az)
    _pack("ax", ax)
    _pack("bdot", bdot_amp)
    _pack("bias", bias_coh)
    _pack("disp", disp)

    merge_save_npz(out_path, payload)
    print(f"\nWrote {out_path}")
    return out_path


def print_summary(npz_path=None):
    """Print the complete Module #4 mode-structure & dispersion synthesis."""
    npz_path = npz_path or os.path.join(DATA_DIR, OUT_NAME)
    with np.load(npz_path) as d:
        print(f"\n=== 1. 2-D Azimuthal Mode Structure (physical m, exp(i*(m*theta - w*t))) ===")
        print(f"Run 32 Mach rotation center: ({float(d['r32_center_x_cm']):+.2f}, {float(d['r32_center_y_cm']):+.2f}) cm (ratio={float(d['r32_center_ratio']):.3f})")
        for r in ("26", "31"):
            if f"az__{r}__bx_m" not in d.files:
                continue
            print(
                f"  Run {r} ({float(d[f'az__{r}__b0_g']):.0f}G, f1={float(d[f'az__{r}__f1_hz'])*1e-3:.2f}kHz): "
                f"LP m={int(d[f'az__{r}__lp_m']):+d} (pur={float(d[f'az__{r}__lp_purity']):.2f}) | "
                f"Bdot Bx/By m={int(d[f'az__{r}__bx_m']):+d}/{int(d[f'az__{r}__by_m']):+d} "
                f"(pur={float(d[f'az__{r}__bx_purity']):.2f}/{float(d[f'az__{r}__by_purity']):.2f}, "
                f"ctr=({float(d[f'az__{r}__bx_cx_cm']):+.1f},{float(d[f'az__{r}__bx_cy_cm']):+.1f})), "
                f"h2 m={int(d[f'az__{r}__bx_h2_m']):+d} | "
                f"dphi(By-Bx)={float(d[f'az__{r}__bdot_dphi_by_bx_deg']):+.1f}deg (R={float(d[f'az__{r}__bdot_dipole_R']):.2f} -> m_J={int(d[f'az__{r}__bdot_inferred_m_j']):+d})"
            )
            if f"bdot__{r}__v_perp_p95_v" in d.files:
                vp_mv = float(d[f"bdot__{r}__v_perp_p95_v"]) * 1e3
                vp_w = float(d[f"bdot__{r}__v_perp_over_omega_vs"]) * 1e6
                if bool(d[f"bdot__{r}__bdot_calibrated"]):
                    print(
                        f"    B-dot p95 (calibrated): V_perp={vp_mv:.1f} mV, "
                        f"dB_perp={float(d[f'bdot__{r}__delta_b_perp_g']):.3f} G, "
                        f"dB_z={float(d[f'bdot__{r}__delta_b_z_g']):.3f} G "
                        f"(dB/B0={float(d[f'bdot__{r}__delta_b_perp_over_b0']):.2e})"
                    )
                else:
                    print(
                        f"    B-dot p95 (uncalibrated): V_perp={vp_mv:.1f} mV, "
                        f"V_perp/w=(G*A)*dB_perp={vp_w:.3f} uV*s "
                        "(pass bdot_area_m2 & bdot_gain for Gauss/Tesla)"
                    )

        print(f"\n=== 2. Axial Wavenumber k_par (P29 - P25, dz = {float(d['dz_p29_p25_m']):.2f} m) ===")
        print(f"{'Run':>3} {'B0':>5} {'Inner dphi':>11} {'R_in':>5} {'kpar_in':>9} {'lam_in':>8} | {'Outer dphi':>11} {'R_out':>5} {'kpar_out':>9}")
        for r in ("28", "27", "29", "30"):
            if f"ax__{r}__b0_g" not in d.files:
                continue
            print(
                f"{r:>3} {float(d[f'ax__{r}__b0_g']):5.0f} "
                f"{float(d[f'ax__{r}__inner_dphi_deg']):+10.1f}d {float(d[f'ax__{r}__inner_R']):5.2f} "
                f"{float(d[f'ax__{r}__inner_k_par_m']):+9.2f} {float(d[f'ax__{r}__inner_lambda_par_m']):7.1f}m | "
                f"{float(d[f'ax__{r}__outer_dphi_deg']):+10.1f}d {float(d[f'ax__{r}__outer_R']):5.2f} "
                f"{float(d[f'ax__{r}__outer_k_par_m']):+9.2f}"
            )

        if "bias__02__I_2in_g2" in d.files:
            print("\n=== 3. End-Electrode Current Coherence vs. P33 Isat (biasscope, +12 ms, shared f0) ===")
            print(f"{'Run':>3} {'f0[kHz]':>7} {'g2(2\")':>7} {'dI/I 2\"':>8} {'g2(4\")':>7} {'dI/I 4\"':>8} {'g2(2-4)':>7} {'dphi(4-2)':>10} {'g2(V2\")':>7} {'g2(Light)':>9}")
            for r in ("06", "00", "01", "02", "03", "04", "05"):
                if f"bias__{r}__I_2in_g2" not in d.files:
                    continue
                print(
                    f"{r:>3} {float(d[f'bias__{r}__shared_f_hz'])*1e-3:7.2f} "
                    f"{float(d[f'bias__{r}__I_2in_g2']):7.3f} {float(d[f'bias__{r}__I_2in_rel_rms'])*100:7.2f}% "
                    f"{float(d[f'bias__{r}__I_4in_g2']):7.3f} {float(d[f'bias__{r}__I_4in_rel_rms'])*100:7.2f}% "
                    f"{float(d[f'bias__{r}__g2_I2_I4']):7.3f} "
                    f"{float(d[f'bias__{r}__dphi_I4_minus_I2_deg']):+9.1f}d "
                    f"{float(d[f'bias__{r}__V_2in_g2']):7.3f} {float(d[f'bias__{r}__Light_P30_g2']):9.3f}"
                )

        print(f"\n=== 4. Plasma-Frame Dispersion (m = {DEFAULT_M}, r = {float(d['dispersion_r_cm']):.1f} cm, k_par = {float(d['k_par_ref_m']):.2f} /m) ===")
        print(
            f"{'Run':>3} {'Puff':>5} {'f_lab':>6} {'f_dopE':>7} {'f_plE':>7} {'f_plM':>7} "
            f"{'f*n':>6} {'f*T':>6} {'f*p':>6} {'k_rhos':>6} {'be/emi':>7} {'f_A[kHz]':>9} {'g/w':>6}"
        )
        for spec in jpr.PROFILE_RUNS:
            r = spec.run_num
            if f"disp__{r}__f_lab_hz" not in d.files:
                continue
            gw = float(d[f"disp__{r}__gamma_over_omega"])
            gw_s = f"{gw:6.2f}" if np.isfinite(gw) else "   ---"
            fl = float(d[f"disp__{r}__f_lab_hz"]) * 1e-3 if bool(d[f"disp__{r}__unstable"]) else np.nan
            fl_s = f"{fl:6.2f}" if np.isfinite(fl) else "  stbl"
            fpe = float(d[f"disp__{r}__f_plasma_exb_hz"]) * 1e-3
            fpm = float(d[f"disp__{r}__f_plasma_mach_hz"]) * 1e-3
            fpe_s = f"{fpe:+7.2f}" if np.isfinite(fpe) else "    ---"
            fpm_s = f"{fpm:+7.2f}" if np.isfinite(fpm) else "    ---"
            print(
                f"{r:>3} {float(d[f'disp__{r}__gas_puff_ms']):4.0f}m {fl_s:>6} "
                f"{float(d[f'disp__{r}__f_doppler_exb_hz'])*1e-3:+7.2f} "
                f"{fpe_s:>7} "
                f"{fpm_s:>7} "
                f"{float(d[f'disp__{r}__f_star_n_hz'])*1e-3:+6.2f} "
                f"{float(d[f'disp__{r}__f_star_T_hz'])*1e-3:+6.2f} "
                f"{float(d[f'disp__{r}__f_star_p_hz'])*1e-3:+6.2f} "
                f"{float(d[f'disp__{r}__k_perp_rho_s']):6.2f} "
                f"{float(d[f'disp__{r}__beta_e_over_me_mi']):7.1f} "
                f"{float(d[f'disp__{r}__f_alfven_hz'])*1e-3:9.1f} {gw_s:>6}"
            )


if __name__ == "__main__":
    out = batch_all()
    print_summary(out)
