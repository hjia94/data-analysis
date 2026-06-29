# data-analysis

Shared data-analysis repository for plasma-physics experiments. Created Feb 2024
to consolidate the analysis routines that used to live in separate
`Bdot_analysis`, `LP_analysis`, and `hairpin_analysis` repos, so that one generic
implementation is shared across experiments instead of being copy-pasted per
campaign.

The reusable code is an installable package (`data_analysis`) under `src/`,
organized by **what the code does** — generic DSP, plasma formulas, I/O — not by
which experiment produced it. Each measurement campaign keeps its own thin
scripts/notebooks under `experiments/`, which import only from `data_analysis`.

## Diagnostics covered

- **Langmuir probe (LP)** — plasma parameters from I–V sweeps (`data_analysis.plasma.langmuir`).
- **Bdot** — magnetic-fluctuation processing and STFT spectrograms (`data_analysis.signal`).
- **X-ray / photon counting** — pulse detection for bremsstrahlung diagnostics (`data_analysis.plasma.photons`).
- **Thomson scattering / CTS** — Sheffield spectral analysis (`data_analysis.plasma.cts`, `sheffield_thomson`).
- **High-speed-camera tracking** — falling-ball trajectory fitting (`data_analysis.tracking`).

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
| `data_analysis.signal` | Generic digital signal processing — filters, STFT, envelopes, zero-crossing detection, and downsampling helpers. |
| `data_analysis.plasma` | Plasma-physics analysis — Langmuir probes, photon/X-ray pulse detection (`Photons`), CTS / Sheffield Thomson scattering, formulas. |
| `data_analysis.tracking` | High-speed-camera object tracking and trajectory fitting. |
| `data_analysis.viz` | Plotting helpers. |
| `data_analysis.utils` | Cross-cutting utilities — file discovery and `.npy` I/O. |

### Optional extras

Some subsystems pull in heavier dependencies, declared as extras in
`pyproject.toml`:

- `pip install -e .[tracking]` — `opencv-python-headless` for `data_analysis.tracking`.
- `pip install -e .[scope]` — the sibling [`LAPD_DAQ`](https://github.com/hjia94/LAPD_DAQ)
  package (`scope_io`, …) used by the oscilloscope / pydaq readers. A local
  editable clone at `../LAPD_DAQ` is discovered automatically (see
  [src/data_analysis/io/README.md](src/data_analysis/io/README.md)).

## Experiments

Per-campaign analysis scripts live under [experiments/](experiments/) and import
only from `data_analysis` — they never import another experiment. `compute_B/`
(LAPD coil-field calculator) is a standalone tool outside the package; see
[compute_B/README.md](compute_B/README.md).

| Folder | Campaign / purpose |
|--------|--------------------|
| [experiments/ucla-lapd/](experiments/ucla-lapd/) | LAPD campaigns, one folder per run (`Jan-2024`, `Nov-2024`, `Aug-2025`, `Mar-2026`, `Jun-2026`). See the per-campaign READMEs, e.g. [Aug-2025](experiments/ucla-lapd/Aug-2025/README.md). |
| [experiments/object_tracking/](experiments/object_tracking/) | High-speed-camera ball-tracking notebook (code now lives in `data_analysis.tracking`). |
| [experiments/cts/](experiments/cts/) | Collective Thomson scattering analysis. |
| [experiments/epfl/](experiments/epfl/) | EPFL CRDS / power-calibration scripts. |

## Branches

Analysis for a specific machine/campaign is developed on its own branch; the
shared `data_analysis` package on `main` carries the generic, reusable code that
every branch builds on.
