"""Generate an experiment-independent B-dot probe calibration ``.npz`` from E5100A VNA sweeps.

Reads a folder of Agilent/HP E5100A Helmholtz-bench exports (3 diagonal files
``BXPX/BYPY/BZPZ`` or a full 9-file ``B[XYZ]P[XYZ]`` set), divides raw VNA
``A/R`` by the bench calibration amplifier gain and attenuators to real
probe-terminal signal level, fits the bare signed 3x3 effective area matrix
via :func:`data_analysis.plasma.bdot.fit_effective_area_matrix`, and writes a
reusable ``<probe>-bdot-calibration.npz`` file.

Campaign analysis scripts (e.g. ``Jun2026_bdot_2d.py``, ``Jun2026_modes.py``,
``Aug2026_bdot.py``) load only the resulting ``.npz`` (via
:func:`data_analysis.plasma.bdot.load_bdot_calibration`) and never touch raw
VNA files or bench calibration gains.

Examples::

    .venv/Scripts/python.exe experiments/ucla-lapd/bdot_cal.py \\
        "D:/data/LAPD/BDOT CALIBRATION/Bdot-10T-C16/202603_Raw_Calibration_Data_Red" \\
        --probe-id C16 --gain 10 \\
        --note "202603_Red (matches 2026-05-04 adjusted up-mark)"
"""

from __future__ import annotations

import argparse
import datetime
import os

from data_analysis.io.network_analyzer import load_bdot_vna_dir
from data_analysis.plasma.bdot import (
    BdotCalibration,
    HelmholtzBench,
    fit_effective_area_matrix,
    save_bdot_calibration,
)

AXES = ("X", "Y", "Z")
DEFAULT_DIAG_BAND_KHZ = (150.0, 500.0)
DEFAULT_OFFDIAG_BAND_KHZ = (250.0, 1000.0)


def calibrate_bdot_dir(
    vna_dir,
    *,
    bench_gain: float,
    bench_atten_db: float = 0.0,
    probe_id: str | None = None,
    diag_band_hz: tuple[float, float] = (150e3, 500e3),
    offdiag_band_hz: tuple[float, float] = (250e3, 1000e3),
    bench: HelmholtzBench = HelmholtzBench(),
    out_path: str | None = None,
    note: str = "",
) -> tuple[BdotCalibration, str]:
    """Load E5100A sweeps, divide out bench gain/attenuation, fit 3x3 ``BdotCalibration``, and save ``.npz``."""
    vna_dir = os.path.abspath(os.fspath(vna_dir))
    if bench_gain <= 0.0:
        raise ValueError(f"bench_gain must be > 0, got {bench_gain}")

    vna = load_bdot_vna_dir(vna_dir, probe_id=probe_id)
    pid = probe_id or os.path.basename(vna_dir)

    bench_scale = float(bench_gain) * (10.0 ** (-float(bench_atten_db) / 20.0))
    # Convert raw VNA A/R to real probe-terminal signal level before calling plasma.bdot
    H_coil = vna["H_vna"] / bench_scale

    provenance = {
        "bench_gain": float(bench_gain),
        "bench_atten_db": float(bench_atten_db),
        "vna_dir": vna_dir,
        "updated": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if note:
        provenance["note"] = str(note)

    cal = fit_effective_area_matrix(
        vna["freq_hz"],
        H_coil,
        fit_band_hz=diag_band_hz,
        offdiag_band_hz=offdiag_band_hz,
        bench=bench,
        probe_id=pid,
        provenance=provenance,
    )

    if out_path is None:
        out_path = os.path.join(vna_dir, f"{pid}-bdot-calibration.npz")
    save_bdot_calibration(out_path, cal)
    return cal, out_path


def print_summary(cal: BdotCalibration):
    """Print signed 3x3 effective area matrix and fit diagnostics."""
    print(f"\n=== B-dot {cal.probe_id} Calibration (bare probe, real signal level) ===")
    print(
        f"Bench: gain={cal.provenance.get('bench_gain')}, "
        f"atten={cal.provenance.get('bench_atten_db')} dB | "
        f"K0={cal.bench.scale_m2_hz:.4f} m^2*Hz"
    )
    print(
        f"Bands: diag={cal.diag_band_hz[0]*1e-3:g}-{cal.diag_band_hz[1]*1e-3:g} kHz, "
        f"off-diag={cal.offdiag_band_hz[0]*1e-3:g}-{cal.offdiag_band_hz[1]*1e-3:g} kHz"
    )
    if cal.provenance.get("note"):
        print(f"Note:  {cal.provenance['note']}")
    print("\nSigned 3x3 Effective Area Matrix A[probe, field] (cm^2)  [V_coil = A @ dB/dt]:")
    print("           Bx         By         Bz")
    for i, p_ax in enumerate(AXES):
        row = "  ".join(f"{cal.area_matrix_m2[i, j]*1e4:+9.4f}" for j in range(3))
        print(f"  P{p_ax}: {row}")
    print("\nFit R^2 / phase coherence R:")
    for i, p_ax in enumerate(AXES):
        row = "  ".join(
            f"{cal.r2_matrix[i, j]:.3f}/{cal.phase_R_matrix[i, j]:.3f}" for j in range(3)
        )
        print(f"  P{p_ax}: {row}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vna_dir", help="Folder containing E5100A B[XYZ]P[XYZ] VNA exports")
    ap.add_argument("--probe-id", default=None, help="Probe identifier filter/label (e.g. C16)")
    ap.add_argument(
        "--gain", type=float, required=True,
        help="Bench calibration amplifier linear gain (e.g. 10 for X10 preamp, 1 if unamplified)",
    )
    ap.add_argument("--atten-db", type=float, default=0.0, help="Bench attenuator [dB] (default: 0)")
    ap.add_argument(
        "--diag-band", type=float, nargs=2, default=list(DEFAULT_DIAG_BAND_KHZ),
        metavar=("F_LO_KHZ", "F_HI_KHZ"),
        help="Diagonal linear fit band [kHz] (default: 150 500)",
    )
    ap.add_argument(
        "--offdiag-band", type=float, nargs=2, default=list(DEFAULT_OFFDIAG_BAND_KHZ),
        metavar=("F_LO_KHZ", "F_HI_KHZ"),
        help="Off-diagonal linear fit band [kHz] (default: 250 1000)",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output .npz path (default: <vna_dir>/<probe_id>-bdot-calibration.npz)",
    )
    ap.add_argument("--note", default="", help="Optional provenance note (e.g. up-mark / bench details)")
    args = ap.parse_args()

    cal, npz_path = calibrate_bdot_dir(
        args.vna_dir,
        bench_gain=args.gain,
        bench_atten_db=args.atten_db,
        probe_id=args.probe_id,
        diag_band_hz=(args.diag_band[0] * 1e3, args.diag_band[1] * 1e3),
        offdiag_band_hz=(args.offdiag_band[0] * 1e3, args.offdiag_band[1] * 1e3),
        out_path=args.out,
        note=args.note,
    )
    print_summary(cal)
    print(f"\nWrote {npz_path}")


if __name__ == "__main__":
    main()
