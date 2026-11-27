"""Calibrate the Jun-2026 P28 B-dot probe (Bdot-10T-C16) from E5100A Helmholtz VNA exports.

Reads the 9-file ``202603_Raw_Calibration_Data_Red`` sweep (closest to the
2026-05-04 adjusted up-mark logged in ``C16 Probe Log``), divides raw VNA
``A/R`` by the calibration-bench amplifier gain (``BENCH_GAIN = 10.0``) to
real probe-terminal signal level, fits the signed 3x3 effective area matrix
via :func:`data_analysis.plasma.bdot.fit_effective_area_matrix`, and writes
``C16-bdot-calibration.npz`` into ``DATA_DIR`` (mirroring :mod:`Jun2026_mach_cal`).

    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_bdot_cal.py
    .venv/Scripts/python.exe experiments/ucla-lapd/Jun-2026/Jun2026_bdot_cal.py --vna-dir <path>
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

# --- Campaign / bench constants ---------------------------------------------
DATA_DIR = r"D:\data\LAPD\jun2026-jia"
DEFAULT_VNA_DIR = r"D:\data\LAPD\BDOT CALIBRATION\Bdot-10T-C16\202603_Raw_Calibration_Data_Red"
PROBE_ID = "C16"
STEM = "C16-bdot-calibration"

# Bench signal conditioning (separate from experimental amplifiers in HDF5 runs)
BENCH_GAIN = 10.0
BENCH_ATTEN_DB = 0.0
BENCH_DESCRIPTION = (
    "green Helmholtz coil (a=5.5 cm, 1 turn/coil, 1/16 Ohm shunt + 50/50 divider) "
    "with X10 bench calibration amplifier"
)
UPMARK_NOTE = "202603_Raw_Calibration_Data_Red (closest to 2026-05-04 adjusted upmark)"

# Clean linear inductive fit bands (bypassing <100-250 kHz bench artifact & >1 MHz droop)
DIAG_BAND_HZ = (150e3, 500e3)
OFFDIAG_BAND_HZ = (250e3, 1000e3)
AXES = ("X", "Y", "Z")


def calibrate_probe(
    vna_dir=DEFAULT_VNA_DIR,
    *,
    probe_id=PROBE_ID,
    bench_gain=BENCH_GAIN,
    bench_atten_db=BENCH_ATTEN_DB,
    diag_band_hz=DIAG_BAND_HZ,
    offdiag_band_hz=OFFDIAG_BAND_HZ,
    out_dir=DATA_DIR,
    stem=STEM,
) -> tuple[BdotCalibration, str]:
    """Load E5100A sweeps, divide out bench gain, fit 3x3 calibration, and save ``.npz``."""
    vna = load_bdot_vna_dir(vna_dir, probe_id=probe_id)
    bench_scale = float(bench_gain) * (10.0 ** (-float(bench_atten_db) / 20.0))
    # Convert raw VNA A/R to real probe-terminal signal level before calling plasma.bdot
    H_coil = vna["H_vna"] / bench_scale

    cal = fit_effective_area_matrix(
        vna["freq_hz"],
        H_coil,
        fit_band_hz=diag_band_hz,
        offdiag_band_hz=offdiag_band_hz,
        bench=HelmholtzBench(),
        probe_id=probe_id,
        provenance={
            "bench_gain": float(bench_gain),
            "bench_atten_db": float(bench_atten_db),
            "bench_description": BENCH_DESCRIPTION,
            "upmark_note": UPMARK_NOTE,
            "vna_dir": os.path.abspath(vna_dir),
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
    )

    npz_path = os.path.join(out_dir, f"{stem}.npz")
    save_bdot_calibration(npz_path, cal)
    return cal, npz_path


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
    ap.add_argument("--vna-dir", default=DEFAULT_VNA_DIR, help="Folder with E5100A B[XYZ]P[XYZ] CSVs")
    ap.add_argument("--out-dir", default=DATA_DIR, help="Output folder for calibration .npz")
    ap.add_argument("--probe-id", default=PROBE_ID)
    ap.add_argument("--gain", type=float, default=BENCH_GAIN, help="Bench calibration amplifier gain")
    ap.add_argument("--atten-db", type=float, default=BENCH_ATTEN_DB, help="Bench attenuator [dB]")
    args = ap.parse_args()

    cal, npz_path = calibrate_probe(
        args.vna_dir,
        probe_id=args.probe_id,
        bench_gain=args.gain,
        bench_atten_db=args.atten_db,
        out_dir=args.out_dir,
        stem=f"{args.probe_id}-bdot-calibration",
    )
    print_summary(cal)
    print(f"\nWrote {npz_path}")


if __name__ == "__main__":
    main()
