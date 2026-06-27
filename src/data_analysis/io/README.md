# `data_analysis.io`

Readers grouped by **data shape**, plus the central output-path resolver. This is
the *only* place in the package where a file format is parsed — experiment code
imports readers from here and never re-implements parsing.

## Modules

| Module | What it reads / does |
|--------|----------------------|
| `lapd_hdf5.py` | Unified LAPD HDF5 reader — `open_lapd()`, `LapdRun`, `LapdSession`. |
| `scope.py` | Oscilloscope traces: LeCroy `.trc`, `.txt`, and scope groups inside HDF5. Optional `scope_io` (from the sibling `LAPD_DAQ` repo) is discovered lazily. |
| `cine.py` | Phantom high-speed-camera `.cine` movies (+ AVI conversion, motion overlays). |
| `network_analyzer.py` | Vector network-analyzer CSV exports (`read_NA_data`). |
| `LeCroy_Scope_Header.py` | LeCroy binary trace header parsing (used by `scope.py`). |
| `paths.py` | Central output-location resolver (`output_path`) so writers don't hard-code paths. |
| `_backends/` | **Private.** Per-provenance LAPD HDF5 parsers behind `lapd_hdf5`. Not a public import. |

## The unified LAPD reader

Three provenances of LAPD HDF5 file exist, each historically read by its own
module:

- **`bapsflib`** — LabVIEW DAQ + C translator.
- **`pydaq`** — the current LAPD_DAQ layout.
- **`legacy`** — the 2018–2020 `process-plasma` layout.

`open_lapd()` sniffs the file's group signatures, picks the matching backend, and
returns a `LapdRun` that delegates to that backend. Behavior is preserved — the
backends wrap the original readers rather than rewriting them. The exact
group-signature rules used for detection live in `detect_backend()` (in
`lapd_hdf5.py`), the single source of truth for provenance sniffing.

```python
from data_analysis.io import open_lapd

run = open_lapd(path)          # detects provenance; does NOT hold the file open
stack, tarr = run.channel("C2")  # (nshot, nsamples) float array + 1-D time axis
```

### Two surfaces: `LapdRun` vs `LapdSession`

**`LapdRun` — per-call, no handle held.** `open_lapd()` records only the path +
backend. Every method (`channel`, `positions`, `info`, …) opens the HDF5 handle,
reads the requested slice, and closes it. Use this for one-off reads; it never
leaks a handle.

`LapdRun.channel()` and `LapdRun.time_array()` return a **normalized schema** for
every backend, so cross-provenance analysis code never branches on `.backend`:

- `channel(name, shots=...)` → `(stack, tarr)`, `stack` shaped `(nshot, nsamples)`.
- `time_array(...)` → `tarr` (1-D).
- `positions()` and `info()` stay **backend-native** — the three position layouts
  share no clean common schema, so forcing one would be lossy.

For large runs, read a subset rather than the whole file:
`channel(name, shots=...)` reads a chosen subset; `iter_shots(name, shots=...)`
yields one shot at a time (read-and-discard); `shots()` lists available shot
numbers.

**`LapdSession` — one open handle for many reads (bapsflib only).** Some
experiment routines read several things from the *same* open file — digitizer
config, probe motion, then a handful of channels. Opening per call would reopen
the (expensive) `lapd.File` each time. `run.session()` is a context manager that
opens the handle once for the `with` body and closes it on exit — no leaked
handle:

```python
with open_lapd(path).session() as sess:
    adc, digi = sess.digitizer_config()
    pos, *_   = sess.positions()
    data, tarr = sess.read_data(4, 5, index_arr=slice(0, n), adc=adc)
    # escape hatch for raw bapsflib calls:
    msi = sess.file.read_msi("Discharge")
```

`LapdSession` methods return each backend's **native** output (e.g. `read_data` →
`(data, tarr)` with `data["signal"]`), so migrating older bapsflib code onto
`session()` is a zero-behavior-change swap.

Only the `bapsflib` backend holds a handle. The `pydaq`/`legacy` backends read
per-path and cheaply, so calling `session()` on them raises `NotImplementedError`
— use the `LapdRun` methods directly. Likewise, where a backend lacks a concept
(e.g. per-shot reads on the 2018 layout), the method raises `NotImplementedError`
rather than guessing.

## Run-description parser & diff (pydaq)

LAPD pydaq files carry a hand-written `description` root attribute (purpose prose
→ `Operator:` → a `Setup:` block of indented bullets: plasma condition, magnetic
field, bias, probe). Across a run series the operator usually changes **one**
setting and re-runs, but the text drifts cosmetically (`800G` vs `800 G`, tabs vs
spaces, `(NOT USED)` markers, typos). `_backends/run_description.py` parses that
text into a structured, drift-tolerant form and diffs two of them.

```python
from data_analysis.io import open_lapd, compare_runs

desc = open_lapd(path).description()      # -> RunDescription (sections/items)
diff = compare_runs(path_a, path_b)       # -> RunDiff
print(diff.summary())                     # "B-field 800G->600G"
diff.summary(arrow="→")                   # real arrow for plot titles
diff.changed                              # [(path, raw_a, raw_b), ...]
```

The diff compares a **normalized** form (so formatting drift is *not* a
difference) but reports the **raw** text. `summary()` lists changed settings
compactly, ranked most-significant first (B-field, gas, bias, density, …), and
collapses reworded prose into a trailing `(+N/-M other)` count. `description()` /
`compare_runs` are pydaq-only (bapsflib/legacy store metadata differently and
raise `NotImplementedError`).

## Conventions

- **Import readers from `data_analysis.io.*`** — never from a flat top-level
  module or a `sys.path` hack. (The reorg-era back-compat shims are gone.)
- **`scope_io` is optional.** `scope.py` first tries a normal import, then a
  `LAPD_DAQ_PATH` env var, then a sibling `LAPD_DAQ` clone. If none resolve it
  raises a message pointing at the `scope` extra (`pip install -e ../LAPD_DAQ`).
- **Heavy backends import lazily.** `lapd_hdf5` imports `bapsflib`/`matplotlib`
  only inside the methods that need them, so opening a `pydaq`/`legacy` file does
  not pull in bapsflib.
