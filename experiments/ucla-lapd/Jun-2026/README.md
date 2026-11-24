# Jun-2026 UCLA LAPD Multi-Electrode Biasing Campaign

Analysis scripts for the June 10–13, 2026 UCLA LAPD campaign (`D:\data\LAPD\jun2026-jia`, Runs `00`–`32`), investigating biasing-plate-driven plasma rotation, hollow profiles, and the onset mechanism of the low-frequency (~3–4.3 kHz) instability across gas-puff, magnetic-field, and axial/2D diagnostic scans.

## Design Convention

Processing, static plotting, and interactive HTML sliders are split across modules:

- **Processing (`Jun2026_IV.py`, `Jun2026_Isat.py`, `Jun2026_mach_*.py`, `Jun2026_xcorr.py`)**: read raw HDF5 files and write compact `.npz` products co-located in `DATA_DIR` (`D:\data\LAPD\jun2026-jia`). Heavy reads run once.
- **Static figures (`Jun2026_plot.py`)**: read saved `.npz` files and emit figures under `<OUTPUT_ROOT>/figures/Jun2026`.
- **Interactive sliders (`Jun2026_slider.py`)**: read saved `.npz` files and emit standalone scrubbable HTML pages.
- **Shared physics / DSP / I/O helpers** live in `src/data_analysis/`; files in this folder are thin campaign adapters.

Always invoke scripts with the repo virtual environment (`.venv\Scripts\python.exe`).

---

## Script & Data-Product Map

| Script | Runs covered | Output product(s) | Purpose |
|---|---|---|---|
| `group_by_gas_puff.py` | `00`–`32` | `gas_puff_groups.npy` | Parse gas-puff settings (`V`, `ms`) from HDF5 run descriptions and group runs. |
| `Jun2026_IV.py` | `00`–`06` (puff scan 23–45 ms), `07` (plane) | `<run>-sweep-data.npz`, `<run>-plasma-data.npz` | Swept Langmuir probe `Vp`, `Te`, `ne` (interferometer-calibrated when available). |
| `Jun2026_Isat.py` | `00`–`06` (stationary ref), per-position scans | `isat_fft_00-06.npz`, `<run>-isat-fft-data.npz` | Fixed-bias `Isat` fluctuation traces and ensemble-averaged amplitude spectra. |
| `Jun2026_mach_cal.py` | `17`–`20` (4-orientation rotation) | `17-20-mach-calibration.npz`, `.png` | P33 6-tip Mach probe area ratios `kappa = A_+/A_-` (stored alongside raw data). |
| `Jun2026_mach_line.py` | `08`–`14` (puff scan 23–45 ms, P33 line) | `08`–`14-mach-flow-line-data.npz` | Calibrated `M_X`, `M_Y` and assumed-`Te` velocities binned to paired IV sweeps (`MACH_TO_IV`). |
| `Jun2026_mach_flow.py` | `32` (41×41 P33 3D Mach plane) | `32-mach-flow-data.npz`, slider HTML | Calibrated 3D flow map (`v_X`, `v_Y`, `v_Z`, polar `v_r`/`v_theta`). |
| `Jun2026_xcorr.py` | `26`–`31` (B-dot/LP planes & axial lines) | `26`–`31-xcorr-data.npz` | Ensemble Welch coherence `gamma2(f)` and cross-phase `phase(f)` with per-pair `(pos_x, pos_y)`. |
| `Jun2026_onset.py` | `00`–`06`, `08`–`14`, `26`–`31` | `onset_summary_jun2026.npz` | Per-shot & onset-aligned instability `t_onset`, exponential growth rate `gamma`, and post-onset full-window `f0` (Rayleigh `1/T_win ~ 300–770 Hz`). |
| `Jun2026_plot.py` | Saved `.npz` files | Static PNGs (`Jun2026/`) | Publication/diagnostic figures for IV lines, IV+Isat combined, and xcorr planes. |
| `Jun2026_slider.py` | Saved `.npz` files | Standalone `.html` sliders | Scrubbable pages for IV lines, Isat FFTs, and multi-pair xcorr scans. |

### Gas-Puff Pairing (`MACH_TO_IV`)

The Day-1 IV scan (`00`–`06`) and Day-2 Mach line scan (`08`–`14`) share the same seven 75 V gas-puff settings, but Run `06` is 23 ms while Run `00` is 24 ms:

| Gas puff (75 V) | IV line (P29) | Mach line (P33) |
|---|---|---|
| 23 ms | `06` | `08` |
| 24 ms | `00` | `09` |
| 25 ms | `01` | `10` |
| 30 ms | `02` | `11` |
| 35 ms | `03` | `12` |
| 40 ms | `04` | `13` |
| 45 ms | `05` | `14` |

---

## Reproducing the Batched `.npz` Products

### Mach Probe Calibration (`17`–`20`) and 3D Plane (`32`)

```powershell
.venv\Scripts\python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_cal.py
.venv\Scripts\python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_flow.py
```

### Mach Lines (`08`–`14`)

Skip existing files (default):

```powershell
.venv\Scripts\python.exe experiments/ucla-lapd/Jun-2026/Jun2026_mach_line.py
```

Force full regeneration (e.g., after a binning or calibration update):

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'experiments/ucla-lapd/Jun-2026'); import Jun2026_mach_line as m; m.batch_all(overwrite=True)"
```

### Cross-Correlation / Coherence (`26`–`31`)

`Jun2026_xcorr.XCORR_RUNS` defines the full `PairSpec` manifest for Runs `26`–`31` (`tmin_ms=1.5`, `tmax_ms=4.5`, `nperseg=4096`, optional `t_offset_b_ms`), including the explicit `<Athena>    p28_Bdot` motion-group override for unmapped B-dot channels in Runs `26` and `31` (`lpscope` and `bdotscope` share a trigger within `42 ns`, so `t_offset_b_ms=0.0`), and the stationary-reference-only pairs for the two-phase Run `29`.

Generate missing pairs (resumable; validates stored `tmin_ms`/`tmax_ms`/`nperseg`/`t_offset_b_ms` and raises `ValueError` on mismatch instead of silently keeping stale pairs):

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'experiments/ucla-lapd/Jun-2026'); import Jun2026_xcorr as x; x.batch_all()"
```

Force full regeneration of all six runs (or pass `runs=['26', '31']` for a subset):

```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, 'experiments/ucla-lapd/Jun-2026'); import Jun2026_xcorr as x; x.batch_all(overwrite=True)"
```

---

## Verified Data & Hardware Caveats

Documented directly from the raw HDF5 files and run logs so downstream scripts do not silently misinterpret channels or coordinates:

1. **Run `31` magnetic field label:** The filename says `800G` (`31-He-800G-bias40V-Bdot-LP-plane_2026-06-13.hdf5`), but the HDF5 run description logs `uniform 600G` — consistent with its ~3.3–3.6 kHz fundamental matching Run `28` (600 G) rather than Run `26` (800 G).
2. **`P29-L` vs. `P29-R` (cathode-facing primary electrons):** In the hollow core, the cathode-facing Langmuir tip (`P29-L`) collects ~50 eV primary electrons (`Vf` down to `-37 V`, `(Vp - Vf)/Te ~ 7–15`). Use the anode-facing tip (`P29-R`, where `(Vp - Vf)/Te ~ 3.1–4.6`) for thermal `Te`, `pe`, and `Vp` profiles.
3. **Hardware sweeper cutoff in Runs `00`–`05`:** The Langmuir voltage sweeper physically stops sweeping early in the higher-puff runs (`3.0, 2.0, 1.5, 1.0, 1.0, 1.0 ms` for Runs `00`–`05`). Later timestamps have no sweeps in the raw data.
4. **`biasscope` scaling and trigger offset:**
   - LeCroy `biasscope` headers already fold the `×200` HV divider attenuation into `vertical_gain` — do **not** multiply by `200` again.
   - `biasscope` triggers `12.0 ms` offset relative to `machscope`; pass `t_offset_b_ms=+12.0` (or `-12.0` depending on channel order) when cross-correlating across those two scopes in `Jun2026_xcorr`.
5. **Multi-drive coordinate quirks:**
   - **Runs `26` & `31`:** `<Hermes> p29_LP` and `<Athena> p28_Bdot` scan in opposite directions (`lpscope` and `bdotscope` share a trigger: `[-2.0, +8.0] ms` within `42 ns`, so `t_offset_b_ms=0.0`). `Jun2026_xcorr` stores coordinates **per pair** (`<pair>__pos_x`, `<pair>__pos_y`) and drops top-level `pos_x`/`pos_y` whenever pairs in one file sit on different grids (Runs `26`, `30`, `31`), so readers must use per-pair coordinates.
   - **Run `29`:** Two-phase 31×31 plane (`<Hermes> p29_LP` on shots 1–4805, `<Hades> p25_LP` on shots 4806–9610). Do not cross-correlate P29 directly against P25; correlate each against stationary P33 (`scope/C5`). Also, `<Hades>` live encoder positions jitter by ~5 µm (`31×34` unique values at 3 decimals vs. `31×31` at 2 decimals), so 2D gridding should snap to the planned 0.5 cm grid.
   - **Run `30` (1 kG axial line):** P25 (`Hades`) and P29 (`Hermes`) sit `1.0 cm` apart in `y` (`+0.5 cm` vs. `-0.5 cm`).
6. **`xcorr` frequency resolution:** At the 5 MHz scope sampling rate, the default `NPERSEG = 4096` gives `df = 1220.7 Hz` (adequate for 2D coherence/phase maps, where the ~4.2 kHz mode spans the `3.66` and `4.88 kHz` bins). Use longer segments / full-window FFTs (`df <= 330 Hz`) for fine dispersion or `f(B0)` tracking.

---

## Branch `analysis-jun-2026` Status

### Completed
- Fixed zero-width and tail-bleed edge bugs in `binned_face_ratio` (`src/data_analysis/plasma/mach.py`, `Jun2026_mach_line.py`).
- Fixed per-pair coordinates, moving-vs-stationary drive resolution, and optional `t_offset_b_ms` cross-scope alignment in `Jun2026_xcorr.py` / `Jun2026_plot.py`.
- Generated and verified `08`–`14-mach-flow-line-data.npz`, `26`–`31-xcorr-data.npz`, and `onset_summary_jun2026.npz`.
- Added generic onset/growth-rate/dominant-frequency DSP helpers (`src/data_analysis/signal/core.py`) and `Jun2026_onset.py`, plus `tests/` unit tests (`python -m unittest discover -s tests`).

### Planned Next
1. Pre-onset profile, electron pressure `pe`, signed logarithmic gradients (`d ln pe / dr = d ln ne / dr + d ln Te / dr`), and `Vf` / sheath-factor validation module (consuming `t_onset` from `onset_summary_jun2026.npz`).
2. Measured-`Te` rescaling of Mach line flows (`Jun2026_mach_line.py` joined on `iv_sweep_index`) and `E x B` shear comparison.
3. Azimuthal/axial mode decomposition (`m`, `k_par`), B-dot calibration, and plasma-frame dispersion comparison (`omega' = omega - k_theta v_ExB` vs. `omega_*n`, `omega_*T`).
