# analysis
 Data analysis repo for everything. Created on Feb.2024 with the aim to include all data analysis routine in one repository.
 
 This repository will include all old data analysis functions from Bdot_analysis, LP_analysis, and hairpin_analysis. The aim is to not have duplicate code for each analysis routine that does the same thing. Old functions that work for generic cases will be included, and updated with improvement.
 
 For each experiment performed on a different machine, a new branch will be created accordingly with specific analysis routine.

## Repository Structure

This repository is organized to handle various plasma diagnostics analysis routines:

- **Bdot Analysis**: Processing and visualization of magnetic field measurements
- **Langmuir Probe (LP) Analysis**: Plasma parameter extraction from I-V characteristics
- **Hairpin Analysis**: Electron density measurements using resonant probes

## Package layout

The reusable analysis code lives in an installable package under `src/`,
imported as `data_analysis`. Install it in editable mode from the repo root:

```bash
pip install -e .
```

Modules are grouped by **what they do**, not by which experiment produced them:

| Package | Contents |
|---------|----------|
| `data_analysis.io` | File readers, grouped by data shape — the unified LAPD HDF5 reader (`open_lapd`), oscilloscope traces, `.cine` movies, network-analyzer CSV, and the output-path resolver. The *only* place a file format is parsed. See [src/data_analysis/io/README.md](src/data_analysis/io/README.md). |
| `data_analysis.signal` | Generic digital signal processing — filters, STFT, envelopes. |
| `data_analysis.plasma` | Plasma-physics analysis — Langmuir probes, photon/X-ray pulse detection (`Photons`), CTS / Sheffield Thomson scattering, formulas. |
| `data_analysis.tracking` | High-speed-camera object tracking and trajectory fitting. |
| `data_analysis.viz` | Plotting helpers. |
| `data_analysis.utils` | Cross-cutting utilities — file discovery and `.npy` I/O. |

Per-campaign analysis scripts live under [experiments/](experiments/) and import
only from `data_analysis` — they never import another experiment. `compute_B/`
(LAPD coil-field calculator) is a standalone tool outside the package.

## Recent Updates

### November 2024
- Added X-ray and Bdot data processing routines (`process_xray_bdot.py`)
- Implemented STFT spectrograms for frequency analysis of magnetic fluctuations
- Added photon counting analysis for X-ray diagnostics
- Improved data averaging capabilities for multi-shot experiments

## Usage

Example usage for X-ray and Bdot processing:
