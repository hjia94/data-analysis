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
- **Mach probe** — area-ratio calibration and flow velocity (`data_analysis.plasma.mach`), with
  diagnostic-agnostic in-plane flow geometry in `data_analysis.plasma.flow`.
- **Bdot** — magnetic-fluctuation processing and STFT spectrograms (`data_analysis.signal`).
- **Interferometer** — line-averaged density traces merged into pydaq dataruns
  (`data_analysis.io.interferometer`); used to absolutely calibrate probe `n_e`.
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
| `data_analysis.io` | File readers, grouped by data shape — the unified LAPD HDF5 reader (`open_lapd`), interferometer traces, probe↔channel mapping, oscilloscope traces, `.cine` movies, network-analyzer CSV, and the output-path resolver. The *only* place a file format is parsed. See [src/data_analysis/io/README.md](src/data_analysis/io/README.md). |
| `data_analysis.signal` | Generic digital signal processing — filters, STFT, envelopes, cross-spectra, zero-crossing detection, and downsampling helpers. |
| `data_analysis.plasma` | Plasma-physics analysis — Langmuir probes (`langmuir`), Mach probes (`mach`), in-plane flow geometry (`flow`), photon/X-ray pulse detection (`photons`), CTS / Sheffield Thomson scattering, and `formulas`. |
| `data_analysis.tracking` | High-speed-camera object tracking and trajectory fitting. |
| `data_analysis.viz` | Plotting helpers — shared plot utilities, flow-field maps, and the standalone HTML slider renderer. |
| `data_analysis.utils` | Cross-cutting utilities (a single module, not a subpackage) — file discovery and `.npy` I/O. |

### Optional extras

Some subsystems pull in heavier dependencies, declared as extras in
`pyproject.toml`:

- `pip install -e .[tracking]` — `opencv-python-headless` for `data_analysis.tracking`
  and `data_analysis.io.cine` (both import `cv2` at module load).
- `pip install -e .[scope]` — the sibling [`LAPD_DAQ`](https://github.com/hjia94/LAPD_DAQ)
  package (`scope_io`, …) used by the oscilloscope / pydaq readers. A local
  editable clone at `../LAPD_DAQ` is discovered automatically (see
  [src/data_analysis/io/README.md](src/data_analysis/io/README.md)).
- `pip install -e .[gui]` — `screeninfo`, only for the multi-monitor figure-placement
  helpers in `data_analysis.viz.plot_utils`. Plain plotting works without it.

## Experiments

Per-campaign analysis scripts live under [experiments/](experiments/) and import
only from `data_analysis` — they never import another experiment. `compute_B/`
(LAPD coil-field calculator) is a standalone tool outside the package; see
[compute_B/README.md](compute_B/README.md).

| Folder | Campaign / purpose |
|--------|--------------------|
| [experiments/ucla-lapd/](experiments/ucla-lapd/) | LAPD campaigns, one folder per campaign (`Jan-2024`, `Nov-2024`, `Aug-2025`, `Mar-2026`, `Jun-2026`), plus campaign-independent tooling: `dump_campaign.py` (run-log phase-1 extractor, driven by `RUN_LOG_PROMPT.md`) and `interf_save.py`. The folder [README](experiments/ucla-lapd/README.md) indexes the DSP/analysis techniques used across campaigns; see also [Aug-2025](experiments/ucla-lapd/Aug-2025/README.md). |
| [experiments/object_tracking/](experiments/object_tracking/) | High-speed-camera ball-tracking notebook (code now lives in `data_analysis.tracking`). |
| [experiments/cts/](experiments/cts/) | Collective Thomson scattering analysis. |
| [experiments/epfl/](experiments/epfl/) | EPFL CRDS / power-calibration scripts. |
| [experiments/two_chamber_vacuum.py](experiments/two_chamber_vacuum.py) | Standalone two-chamber pinhole vacuum model; imports nothing from the package. |

## Branches

Analysis for a specific machine/campaign is developed on its own branch; the
shared `data_analysis` package on `main` carries the generic, reusable code that
every branch builds on.
