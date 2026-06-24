# Reorganization Plan: `data-analysis`

> Working document. Tracks the planned migration from the current scattered
> layout to a single installable package. Check off steps as they are completed.
> Each step is verified against **real data files** before proceeding to the next.

## Decisions (locked)

- **Import root / package name:** `data_analysis` (underscore). Distinct from the
  repo folder `data-analysis` (hyphen) — `pyproject.toml` maps one to the other.
- **Branch model:** everything stays on `main`. Experiments live under
  `experiments/`. (The README's old "branch per experiment" idea is dropped to
  keep the shared library in sync.)
- **Rollout:** sequenced and controlled step by step by Jia. Every step is tested
  against real data files before the next begins.
- **Generated artifacts live OUTSIDE the repo (configurable path).** Figures and
  processed data must not land next to source. There is one resolver for the output
  root, controlled by Jia via an env var / explicit argument, defaulting to a
  location **outside** the repo tree. See "Output location convention" below.
- **Cross-repo scope dependency:** HDF5 scope reading is **not** owned by this repo.
  It is delegated to `scope_io` in the sibling **LAPD_DAQ** repo (single source of
  truth). `data_analysis` consumes `scope_io`; it does not vendor or fork it.
  See "Cross-repo coupling" below for how the import is wired.

## Goal

A single installable package where data is organized by **shape, not provenance**,
so a new experiment imports one library, calls one reader that auto-detects the
file, and reuses shared analysis without copy-paste or `sys.path` hacks.

This retires the original problem: duplicated functions scattered across folders
that drifted into multiple incompatible versions — most visibly **four separate
HDF5 readers** for the same family of LAPD files.

## Target layout

```
data-analysis/
├── pyproject.toml                  # makes everything importable; kills sys.path
├── README.md
├── REORG_PLAN.md                   # this file
├── src/
│   └── data_analysis/              # the one import root
│       ├── __init__.py
│       ├── io/                     # ── all readers, grouped by data SHAPE ──
│       │   ├── __init__.py
│       │   ├── lapd_hdf5.py        # UNIFIED dispatcher (the 3-way merge)
│       │   ├── _backends/          # provenance hidden here, behind dispatcher
│       │   │   ├── bapsflib_daq.py     # was ucla-lapd/read_hdf5_bapsflib.py
│       │   │   ├── pydaq.py            # was ucla-lapd/read_hdf5_pydaq.py
│       │   │   └── legacy_2018.py      # was ucla-processing/read_hdf5.py
│       │   ├── scope.py            # was read/read_scope_data.py (+ LeCroy header)
│       │   ├── network_analyzer.py # was read/read_network_analyzer_data.py
│       │   ├── cine.py             # was object_tracking/read_cine.py
│       │   ├── spectrometer.py     # CTS + epfl + temp/spectrum readers
│       │   └── npz.py              # save/read_npy, npz bundles
│       ├── signal/                 # generic DSP (no plasma knowledge)
│       │   └── core.py             # filters, STFT, zero-crossings, envelopes
│       ├── plasma/                 # physics
│       │   ├── formulas.py         # was plasma_utils.py
│       │   ├── langmuir.py         # lp_analysis + lp_iv_analysis merged
│       │   └── photons.py          # Photons/PhotonPulse classes
│       ├── compute_b/              # was compute_B/
│       ├── tracking/               # was object_tracking/ (analysis half)
│       └── viz/                    # was plot_utils.py
├── experiments/                    # leaf routines, import data_analysis.*
│   ├── ucla-lapd/
│   │   ├── Jan-2024/ … Mar-2026/
│   │   └── shared/                 # lapd_io.py pattern, promoted & reused
│   └── ucla-processing/
└── scratch/                        # was temp/  (clearly quarantined)
```

**Two firm rules this layout encodes:**
1. `src/data_analysis/io/` is the **only** place a file format is parsed.
2. `experiments/` only ever imports `data_analysis`, **never** a sibling experiment.

## Keystone: unified LAPD HDF5 reader

Today three readers parse the same file family with divergent names/signatures:

| Concept        | `bapsflib`                  | `pydaq`            | `legacy_2018`            |
|----------------|-----------------------------|--------------------|--------------------------|
| describe file  | `show_info(f)`              | `print_info(ifn)`  | `print_data_objects(p)`  |
| read positions | `read_probe_motion_6k(f)`   | `read_positions()` | `read_position_data(p)`  |
| read channel   | `read_data(f, board, chan)` | —                  | `read_channel_data(p,n)` |
| header/timing  | `print_timing(f)`           | header attrs       | `get_header(p)`          |

`src/data_analysis/io/lapd_hdf5.py` sniffs the file (group signatures: bapsflib
mapper present → bapsflib backend; LAPD_DAQ scope groups → pydaq backend; else
legacy), then returns a common object:

```python
def open_lapd(path) -> LapdRun: ...          # detects format, picks backend

class LapdRun:
    def info(self) -> str: ...               # unifies show_info/print_info/...
    def positions(self) -> np.ndarray: ...   # unifies the 3 position readers
    def channel(self, name, shot=None): ...  # unifies read_data/read_channel_data
    def time_array(self): ...
    @property
    def backend(self) -> str: ...            # 'bapsflib' | 'pydaq' | 'legacy'
```

Existing readers move under `_backends/` **largely unchanged** — wrap, don't
rewrite — so behavior is preserved. A change to LAPD handling then lands in one
dispatcher + one backend instead of four files.

## Cross-repo coupling: `data_analysis` ↔ LAPD_DAQ

Scope reading is already being consolidated on branch `consolidate-scope-io`:
recent commits make `read/read_scope_data.py` a **thin wrapper** over the
`scope_io` package in the sibling **LAPD_DAQ** repo (`LAPD_DAQ/scope_io/`,
modules `hdf5.py` + `wavedesc.py`). LAPD_DAQ owns that code; this repo must not
re-implement or vendor it (vendoring would re-create the exact duplication this
reorg exists to kill).

**Current wiring (status quo, keep working):** `read_scope_data.py::_import_scope_io()`
locates LAPD_DAQ at runtime via the `LAPD_DAQ_PATH` env var, else `../../LAPD_DAQ`
(sibling clone), and `sys.path.insert`s it. The `.trc`/`.txt` LeCroy readers in
the same module do **not** need LAPD_DAQ and must keep working without it.

**Complication:** LAPD_DAQ's `pyproject.toml` (`name = "lapd-daq"`) does **not**
ship `scope_io` — its `packages.find` includes `acquisition`, `drivers`,
`lapd_daq`, `motion`, `pi_gpio` only. So `pip install lapd-daq` today does *not*
make `import scope_io` resolve. That is why the runtime path hack exists.

**Decision (recommended):** make the dependency explicit instead of path-discovered.

1. In **LAPD_DAQ**, add `scope_io*` to `[tool.setuptools.packages.find].include`
   (one line) so `scope_io` ships with the package. (Coordinated change in the
   sibling repo — note it; do not assume it's done.)
2. In **data_analysis** `pyproject.toml`, declare the dependency for local
   co-development:
   ```toml
   [project.optional-dependencies]
   scope = ["lapd-daq @ git+https://github.com/hjia94/LAPD_DAQ.git"]
   # dev: pip install -e ../LAPD_DAQ   (editable, points at local checkout)
   ```
3. `io/scope.py` then does a normal `import scope_io`, with the existing
   `_import_scope_io()` sibling-discovery kept as a **fallback** during transition
   so nothing breaks before the LAPD_DAQ packaging change lands.

Alternatives considered: **vendor `scope_io`** → rejected (forks the single source
of truth). **Keep only the runtime path hack** → works but leaves an undeclared,
machine-shaped dependency and a `sys.path` insert, against this reorg's goals.

This coupling gets its own migration step (Step 3a) and is verified before the
unified HDF5 reader work depends on it.

## Output location convention

**Today:** scripts write artifacts via ad-hoc `save_path` / `output_filename`
arguments with no shared base — each script picks its own directory (e.g.
`process_xray_bdot.py` `fig.savefig(output_filename)`, `Mar2026_*.py`
`np.savez(save_path)`). There is no single place that decides *where* outputs go,
so generated files tend to land next to source. The root `.gitignore` already
globs `*.npy/*.npz/*.csv/*.dat/*.txt/*.cine/*.h5` and `temp/`, but **not figures**
(`*.png`/`*.pdf`/`*.svg`), and ignoring is not the same as relocating.

**Decision:** generated artifacts live **outside the repo**, under a single root
that Jia controls. One resolver owns the decision; every writer routes through it.

```python
# src/data_analysis/io/paths.py
import os
from pathlib import Path

def output_root(explicit: str | os.PathLike | None = None) -> Path:
    """Base directory for ALL generated artifacts. Never inside the repo by default.

    Resolution order (first that is set wins):
      1. explicit argument (caller override, e.g. per-run)
      2. $DATA_ANALYSIS_OUTPUT env var      ← Jia's main knob
      3. fallback: ~/data-analysis-output   (outside the repo tree)
    """
    base = explicit or os.environ.get("DATA_ANALYSIS_OUTPUT") or (Path.home() / "data-analysis-output")
    p = Path(base).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p

def output_path(*parts, explicit=None) -> Path:
    """Join under the output root and ensure the parent dir exists."""
    p = output_root(explicit).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
```

Usage in any script: `fig.savefig(output_path("xray", f"{run}.png"))`,
`np.savez(output_path("mach", f"{run}-mach.npz"), ...)`.

**Sub-layout under the root** (created on demand): `figures/` and `processed/`
(or per-diagnostic subfolders like `xray/`, `mach/`). The structure lives under
the *external* root, not the repo.

**Knobs for Jia:**
- `DATA_ANALYSIS_OUTPUT=D:\runs\output` (env var) — primary, set once per machine.
- `explicit=...` argument to `output_root`/`output_path` — per-call override.
- Fallback `~/data-analysis-output` only if neither is set.

**`.gitignore` belt-and-suspenders:** because the default root is outside the repo,
nothing generated should ever be tracked. Still, add `output/` and `*.png`/`*.pdf`/
`*.svg` to `.gitignore` so that *if* someone points the root inside the repo for a
quick local run, artifacts stay untracked. (Repo-internal `output/` is an opt-in
escape hatch, never the default.)

## Data formats the repo can read (inventory)

| Format                              | Source                  | Current reader                          |
|-------------------------------------|-------------------------|-----------------------------------------|
| HDF5 — LabVIEW DAQ + C translator   | LAPD legacy             | `ucla-lapd/read_hdf5_bapsflib.py`       |
| HDF5 — Python `LAPD_DAQ`            | new LAPD pipeline       | `ucla-lapd/read_hdf5_pydaq.py`          |
| HDF5 — 2018-2020 process-plasma     | older LAPD              | `ucla-processing/read_hdf5.py`          |
| HDF5 — scope groups inside DAQ files| LAPD_DAQ scopes         | `read/read_scope_data.py` → **delegates to `LAPD_DAQ/scope_io`** (sibling repo) |
| LeCroy `.trc` / `.txt`             | LeCroy oscilloscope     | `read/read_scope_data.py` (self-contained; no LAPD_DAQ needed) |
| Network analyzer `.csv` (S-params)  | VNA                     | `read/read_network_analyzer_data.py` *(noted broken w/ current pandas)* |
| `.cine`                            | Phantom high-speed cam  | `object_tracking/read_cine.py`          |
| `.npy` / `.npz`                    | internal intermediates  | `data_analysis_utils.py`, `export_xray_npz.py` |
| Spectrometer / CRDS / Thomson       | EPFL, CTS               | `epfl/crds_tools.py`, `CTS/`, `temp/spectrum_analysis.py` |

## Migration steps (test-gated)

Each step ends with verification against a real file Jia points to, then a stop
for go-ahead before the next.

- [ ] **Step 0 — Packaging skeleton, no moves.**
  Add `pyproject.toml` + empty `src/data_analysis/__init__.py`. `pip install -e .`
  - **Verify:** `python -c "import data_analysis"` succeeds; existing
    scripts/notebooks still run unchanged (sys.path hacks still present, harmless).

- [ ] **Step 0b — Output-path resolver (infra only, no script changes yet).**
  Add `src/data_analysis/io/paths.py` with `output_root()` / `output_path()` from
  "Output location convention". Add `output/`, `*.png`, `*.pdf`, `*.svg` to
  `.gitignore` (belt-and-suspenders; default root stays external).
  - **Verify:** with `DATA_ANALYSIS_OUTPUT` set, `output_path("figures","t.png")`
    resolves under that external root and creates parent dirs; with it unset, falls
    back to `~/data-analysis-output` (outside the repo). No existing script changes.

- [ ] **Step 1 — Kill sys.path hacks, one folder at a time.**
  Replace `sys.path.append(...)` with real `data_analysis` imports (re-export
  current root modules so paths resolve during transition).
  - **Verify:** that folder's routine runs against a real data file and matches
    prior output. Repeat per folder.
  - Files with sys.path hacks (14): `compute_B/example.py`, `data_analysis_utils.py`,
    `epfl/crds_tools.py`, `plot_utils.py`, `read/read_scope_data.py`,
    `ucla-lapd/Aug-2025/lapd_io.py`, `ucla-lapd/Aug-2025/movie_maker.py`,
    `ucla-lapd/Jan-2024/Jan2024_Isat.py`, `ucla-lapd/Mar-2026/Mar2026_IV.py`,
    `ucla-lapd/Mar-2026/Mar2026_emissive.py`, `ucla-lapd/Mar-2026/Mar2026_mach.py`,
    `ucla-lapd/Nov-2024/plot_xray_shots.py`, `ucla-lapd/Nov-2024/process_xray_bdot.py`,
    `ucla-lapd/interf_save.py`

- [ ] **Step 2 — Move shared library into the package.**
  Relocate `signal/plasma/viz/compute_b/tracking`; merge `lp_analysis` +
  `lp_iv_analysis` → `plasma/langmuir.py`; extract `Photons`/`PhotonPulse` →
  `plasma/photons.py`. Update experiment imports.
  **Do not move `read_scope_data.py` here** — it carries the LAPD_DAQ coupling and
  is handled separately in Step 3a.
  - **Verify:** re-run an LP routine and a photon/STFT routine on real data;
    compare numeric output.

- [ ] **Step 3a — Wire the LAPD_DAQ `scope_io` dependency (do before Step 3).**
  Move `read/read_scope_data.py` → `io/scope.py`, **preserving** (a) the delegation
  to `scope_io` for HDF5 scope groups and (b) the self-contained LeCroy `.trc`/`.txt`
  path. Implement the recommended dependency wiring from "Cross-repo coupling":
  add `scope_io*` to LAPD_DAQ's packaging (sibling-repo change), declare the
  `scope` optional-dep in this repo's `pyproject.toml`, switch `io/scope.py` to a
  normal `import scope_io` with `_import_scope_io()` retained as fallback.
  - **Verify:** on a real LAPD_DAQ HDF5 file, `io/scope.py` reads a scope channel
    via the installed `scope_io` (not the path hack) **and** still reads a real
    LeCroy `.trc`/`.txt` with LAPD_DAQ absent (uninstalled / `LAPD_DAQ_PATH` unset).

- [ ] **Step 3 — Unified LAPD HDF5 reader (the payoff).**
  Move the three readers to `io/_backends/` unchanged; build `open_lapd()`
  dispatcher returning the common `LapdRun` interface. Promote
  `Aug-2025/lapd_io.py` façade into `experiments/ucla-lapd/shared/`.
  - **Verify (critical):** run `open_lapd` on one real file of *each* provenance
    (bapsflib DAQ, LAPD_DAQ pydaq, legacy 2018) and confirm
    `info()/positions()/channel()` match each old reader's output where applicable.

- [ ] **Step 4 — Migrate experiments to `open_lapd`; route outputs through `output_path`; quarantine `temp/`→`scratch/`; write `io/README.md`.**
  Replace ad-hoc `save_path`/`output_filename` literals in experiment scripts with
  `output_path(...)` (Step 0b resolver) so every artifact lands under the external
  root. (Scripts that take an explicit `save_path` arg keep it — it maps to the
  resolver's `explicit=` override.)
  - **Verify:** each migrated experiment reproduces prior results and writes its
    figures/`.npz` under `DATA_ANALYSIS_OUTPUT` (nothing new appears in the repo
    tree); dead/broken readers (e.g. network-analyzer) confirmed before removal.
  - **Retire back-compat re-exports.** Once experiment scripts import from the
    canonical modules (`data_analysis.signal.core`, `data_analysis.plasma.photons`),
    drop the transitional `# noqa: F401` re-export block in `data_analysis/utils.py`
    (added in Step 2) so the duplicated import surface this reorg exists to kill
    doesn't become permanent.

## Evaluation of current organization (baseline being fixed)

- **Readability — moderate.** Modules well-documented individually, but folder
  names don't reveal that `read/`, `ucla-processing/`, and `ucla-lapd/read_hdf5_*`
  read overlapping data. `temp/` mixes scratch with real readers.
- **Easy to use — weak.** `sys.path.append(r"C:\Users\hjia9\...")` hard-codes the
  machine path into every experiment script; nothing runs on another machine or
  checkout without edits. No install step.
- **Easy to change — mixed.** Generic utils are genuinely reused (good), but the
  four-way HDF5 split means one LAPD change must be reconciled across four files —
  the original "more than one version" failure mode.
- **Future expandability — weak as structured.** No convention for *where* a new
  reader goes, so entropy grows; date-named experiment folders accumulate with no
  shared scaffold.
