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
| `Jun2026_mach_line.py` | `08`–`14` (puff scan 23–45 ms, P33 line) | `08`–`14-mach-flow-line-data.npz`, `mach_line_rescaled_jun2026.npz` | Calibrated `M_X`, `M_Y` binned to paired IV sweeps (`MACH_TO_IV`), plus measured-`Te` rescaling, antisymmetric rotation `v_rot`, flow shear `gamma_E`, and P29 `E x B` comparison (`batch_rescaled_summary`). |
| `Jun2026_mach_flow.py` | `32` (41×41 P33 3D Mach plane) | `32-mach-flow-data.npz`, slider HTML | Calibrated 3D flow map (`v_X`, `v_Y`, `v_Z`, polar `v_r`/`v_theta`). |
| `Jun2026_xcorr.py` | `26`–`31` (B-dot/LP planes & axial lines) | `26`–`31-xcorr-data.npz` | Ensemble Welch coherence `gamma2(f)` and cross-phase `phase(f)` with per-pair `(pos_x, pos_y)`. |
| `Jun2026_onset.py` | `00`–`06`, `08`–`14`, `26`–`31` | `onset_summary_jun2026.npz` | Per-shot & onset-aligned instability `t_onset`, exponential growth rate `gamma`, and post-onset full-window `f0` (Rayleigh `1/T_win ~ 300–770 Hz`). |
| `Jun2026_profiles.py` | `00`–`06` (calibrated `P29-R`) | `profiles_summary_jun2026.npz` | Pre- vs. post-onset sweep classification, `pe`, signed radial gradients (`dln ne/dr`, `dln Te/dr`, `dln pe/dr`, `eta_e`), `Vf`/`Lambda` validation, and local scales (`rho_s`, `nu_ei`, `beta_e`, `v_ExB`, `omega_*`). |
| `Jun2026_modes.py` | `00`–`06`, `26`–`32` | `modes_dispersion_jun2026.npz` | 2-D azimuthal mode `m`, B-dot rotating-filament dipole & uncalibrated `V_perp`/flux amplitude, axial `k_par`, `biasscope` disc-current coherence, and plasma-frame dispersion (`f_plasma` vs. `f_*n`, `f_*T`, `f_A`). |
| `Jun2026_bicoherence.py` | `00`–`06`, `08`–`14`, `26`–`31` | `bicoherence_jun2026.npz` | Late-bias-window (`2.24–4.80 ms`, per-run `window_post_onset` flag) auto- & cross-bicoherence (`Isat`, `I_2"`, `I_4"`), shot-to-shot phase consistency `R(f)`, and Sattolo shot-shuffled surrogate controls. |
| `Jun2026_stability.py` | `06`, `00`–`05` (+ `600–1000 G`) | `stability_linear_jun2026.npz` | Linear stability benchmarks: pure cylindrical KH, 1-D axial 5-field Braginskii BVP (insulating vs. conducting/biased Port-35 sheath + external impedance + current-convective), and 1-D radial `E x B`-sheared CWM. |
| `Jun2026_bdot_cal.py` | Bench E5100A `C16` (`202603_Red`) | `C16-bdot-calibration.npz` | P28 `Bdot-10T-C16` signed 3×3 bare effective area matrix (real signal level; bench X10 amp removed and recorded in provenance). |
| `Jun2026_bdot_2d.py` | `26` (`800 G`), `31` (`600 G`) | `bdot_2d_jun2026.npz` | 2-D reference-locked B-dot phasors (`f_0`, `2f_0`), Maxwell div/curl polarity check, `J_par = (curl B_perp)_z`, `|Bz|/|B_perp|`, and `B_L/B_R` circular polarization maps (supports `bdot_cal` + `2.5BLP => X10 => 2.5BLP => X100` gain). |
| `Jun2026_spindown.py` | `08`–`14`, `32` (+ `06`, `00`–`05`) | `spindown_pedersen_jun2026.npz` | Post-bias Mach spin-down (fast Alfvén torsional unwinding vs. slow relaxation upper bound `nu_in <= nu_slow`), rigorous ion-neutral Pedersen shortfall (`<= 3–20%` of bias current), parallel-shorting bound, and max-damping stability test. |
| `Jun2026_relaxation.py` | `00`–`06`, `08`–`14`, `26`–`28`, `30`–`32` | `profile_relaxation_jun2026.npz` | Pre- vs. post-onset hollow-core profile relaxation & collapse test: swept IV `H_ne` vs. `H_Isat`, continuous flow-cancelled P33 `Isat_0` & `10 us` onset-aligned collapse check, `B0`/axial lines (`P29` vs. `P25`), and shift-immune 2-D planes (`26`, `31`, `32`). |
| `Jun2026_turbulence.py` | `00`–`06`, `08`–`14`, `26`–`28`, `30`–`32` | `turbulence_transport_jun2026.npz` | Broadband PSD (exponential vs. power-law above pre-bias ambient plasma PSD), coherent harmonic fraction, radial hollow-wall `(skewness, kurtosis)`, per-shot amplitude-normalized growth->sat cross-phase (`Isat` vs. Port-35 discs), Run 32 Reynolds stress, spiral sign, shear-flow energy production & finite-tip flux-artifact benchmark, plus bispectral biphase nonlinear steepening vs. wall-curvature decomposition & phase-folded waveforms (`Isat`, radial `11–14`, Run `32` `M_r`/`M_theta`). |
| `Jun2026_plot.py` | Saved `.npz` files | Static PNGs (`Jun2026/`) | Publication/diagnostic figures for IV lines, IV+Isat combined, xcorr planes, and nonlinear steepening / phase-folded sawtooth waveforms (`plot_turbulence_steepening`). |
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
7. **Cold-ion isothermal sound speed in Mach velocities:** `data_analysis.plasma.mach.flow_velocity` / `rescale_mach_velocity` use isothermal `ion_sound_speed(T_e, mu=4, gamma=1.0)` with `T_i = 0` (matching Hutchinson 1987, `profiles.py`, and `stability.py`). At high puff (`40–45 ms`) where `T_e ~ 0.5 eV` approaches `T_i`, omitting `T_i` underestimates `c_s` (and therefore lower-bounds `v_rot(P33) / v_ExB(P29)`).
8. **P33 Mach probe face-orientation provenance:** `Z+ → South` (opposite `B_0`) is directly data-verified by bias-off South→North background flow (`j_Z+ > j_Z-`, `M_raw,Z > 0` on Cal `17`/`19` and Run `32` pre-bias). `X+ → +x` (West) and `Y+ → +y` (Up) are operator-reported mounting orientation and remain inferred pending a CAD drawing or lab photo for an `E x B`-independent check of `X`/`Y`.
9. **P28 B-dot absolute coil-polarity caveat:** Maxwell's `div/curl` check in `Jun2026_bdot_2d.py` proves relative right-handedness `(Bx, +By)` vs. `(Bx, -By)` (and fails fast if inverted), which uniquely determines `m_J`, `B_L/B_R`, and `ellipticity`. Negating *both* `Bx` and `By` together leaves those quantities unchanged while shifting `J_par` by `180°`, so `dphi(J_par - LP)` is known modulo `180°` (still separating resistive `0°/180°` locking from `±90°` quadrature) until at least one in-plane coil's absolute polarity is bench-verified (`abs_coil_polarity_verified=False`).
10. **P33 Mach `X+` undigitized on Runs `08`–`14` and `Isat ~ ne * sqrt(Te)` hollowness caveat:** On Day-2 Mach line Runs `08`–`14`, `machscope/C1` (`X+`) carries no digitized waveforms; continuous flow-cancelled `Isat_0` in `Jun2026_relaxation.py` therefore uses the opposite azimuthal pair `sqrt(I_Y+ * I_Y-)`. Furthermore, because swept `P29-R` IV stops early on Runs `01`–`05`, deep post-onset profiles track `Isat ~ ne * sqrt(Te)` rather than separated `ne` and `Te`; swept IV shows core electron heating (`Te_core / Te_ring = 1.3–4.5x`) makes true density hollowness `H_ne` `1.14–2.29x` deeper than `H_Isat`.
11. **Column-center shift after Day-3 bias-plate reinsertion:** Between Day-2 Mach lines (`08`–`14`, June 11) and Day-3/4 runs (`26`–`32`, June 12–13), the Port-35 biasing plate was pulled out for obstacle/calibration runs (`16`–`20`) and reinserted, shifting the column center by `~1 cm` in `+x` (Days 1–2 sit at `x ~ 0–1 cm`; Days 3–4 sit at `x ~ 1.4–2.0 cm`). On Run `32`, `find_flow_centre` subtracts the pre-bias baseline (`t < -0.5 ms`) before fitting so uniform pre-bias/$\kappa$ offsets do not pull the stagnation center (`(+1.50, -0.75) cm`, ratio `0.19`, matching P29's `(+1.40, -0.80) cm` hollow core and Run 32's `dM_Z` core).

---

## Branch `analysis-jun-2026` Status

### Completed
- Fixed zero-width and tail-bleed edge bugs in `binned_face_ratio` (`src/data_analysis/plasma/mach.py`, `Jun2026_mach_line.py`).
- Fixed per-pair coordinates, moving-vs-stationary drive resolution, and optional `t_offset_b_ms` cross-scope alignment in `Jun2026_xcorr.py` / `Jun2026_plot.py`.
- Generated and verified `08`–`14-mach-flow-line-data.npz`, `26`–`31-xcorr-data.npz`, `onset_summary_jun2026.npz`, `profiles_summary_jun2026.npz`, `mach_line_rescaled_jun2026.npz`, `modes_dispersion_jun2026.npz`, and `bicoherence_jun2026.npz`.
- Added generic onset/growth-rate/dominant-frequency DSP helpers (`src/data_analysis/signal/core.py`) and `Jun2026_onset.py`.
- Added generic radial profile, electron pressure, signed gradient, `Vf`/sheath-factor, and local scale helpers (`src/data_analysis/plasma/profiles.py`) and `Jun2026_profiles.py`.
- Added measured-`Te` Mach velocity rescaling (`data_analysis.plasma.mach.rescale_mach_velocity`), antisymmetric rotation & cylindrical flow shear (`data_analysis.plasma.flow`), and paired IV/`E x B` shear comparison in `Jun2026_mach_line.py`.
- Added azimuthal/axial mode decomposition, B-dot dipole check, and plasma-frame dispersion helpers (`src/data_analysis/plasma/modes.py`) and `Jun2026_modes.py`.
- Added bispectral trigger-locking & surrogate controls (`shot_phase_consistency`, `bicoherence_surrogate`, `summed_bicoherence` in `src/data_analysis/signal/bispectral.py`) and `Jun2026_bicoherence.py`.
- Added linear stability solvers (`src/data_analysis/plasma/stability.py`: local CWM, local & 1-D axial Chebyshev 5-field Braginskii BVP, and 1-D cylindrical radial KH/CWM) and `Jun2026_stability.py` (`stability_linear_jun2026.npz`).
- Added 2-D electromagnetic field helpers (`divergence_2d`, `circular_polarization`, `reference_locked_phasor` in `src/data_analysis/plasma/fields.py`) and `Jun2026_bdot_2d.py` (`bdot_2d_jun2026.npz`).
- Added post-bias Mach spin-down, Pedersen closure bounds, and max-damping stability tests (`fit_exponential_relaxation` in `flow.py`, `pedersen_closure_diagnostics` in `stability.py`, `Jun2026_spindown.py` -> `spindown_pedersen_jun2026.npz`).
- Added hollow-core profile relaxation & collapse checks (`hollow_profile_metrics` and `orbit_smeared_profile` in `profiles.py`, `Jun2026_relaxation.py` -> `profile_relaxation_jun2026.npz`).
- Added broadband turbulence, intermittency, cross-phase, Reynolds-stress transport, and nonlinear harmonic steepening (`harmonic_steepening` + `plot_turbulence_steepening`) analysis (`src/data_analysis/signal/turbulence.py`, `Jun2026_turbulence.py` -> `turbulence_transport_jun2026.npz`, `Jun2026_plot.py`), plus full `tests/` unit test coverage.

### Planned Next
1. Publication/diagnostic figures in `Jun2026_plot.py` summarizing the onset mechanism, dispersion, and nonlinear harmonic coupling across gas puffs and `B0`.
