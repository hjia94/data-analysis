# Data analysis techniques used in ucla-lapd experiments

## 1. Smoothing

| Technique | Implementation | Where used |
|---|---|---|
| Gaussian kernel smoothing (1-D) | `gaussian_filter1d`| IV curve smoothing before differentiation; sweep prep (Jun-2026, Mar-2026, Jan-2024) |
| Gaussian smoothing via FFT convolution | [`fast_gaussian_filter`](../../src/data_analysis/signal/core.py#L81) | long multi-million-sample traces |
| Savitzky–Golay (polynomial) smoothing | [`Photons`](../../src/data_analysis/plasma/photons.py#L97) (`savgol_window=31`, `savgol_order=3`) | x-ray pulse traces before peak detection (Aug-2025, Nov-2024); also Jan-2024 Isat sweeps |
| Median filtering | [`ndimage.median_filter`](Mar-2026/Mar2026_mach.py#L232) | Mach probe baseline estimate, on the decimated trace |
| Rolling-quantile (percentile) smoothing | [`rolling_baseline`](../../src/data_analysis/signal/core.py#L127) | slow-baseline estimation |

## 2. Filtering

| Technique | Implementation | Where used |
|---|---|---|
| Butterworth low-pass, zero-phase (`filtfilt`) | [`low_pass_filter`](../../src/data_analysis/signal/core.py#L101) | generic trace conditioning |
| Butterworth band-pass, SOS + `sosfiltfilt` | [`butter_bandpass`](../../src/data_analysis/signal/core.py#L114) | Mach probe fluctuation band 0.5–12 kHz ([Mar2026_mach.py](Mar-2026/Mar2026_mach.py#L202)) |
| Detrending / mean subtraction | `cross_correlation` | every spectral estimate |
| Baseline subtraction (linear fit to the ion-sat branch) | [langmuir](../../src/data_analysis/plasma/langmuir.py#L619) | ion-saturation baseline removed from IV |
| Baseline subtraction (envelope interpolation) | [photons](../../src/data_analysis/plasma/photons.py#L150) | x-ray photon traces |
| Baseline subtraction (decimate → median → interpolate) | [Mar2026_mach](Mar-2026/Mar2026_mach.py#L232-L233) | Mach probe |
| Envelope detection (upper/lower extrema) | [`hl_envelopes_idx`](../../src/data_analysis/signal/core.py#L143) | photon baseline, Nov-2024 x-ray |
| Moving-RMS envelope | [Mar2026_mach](Mar-2026/Mar2026_mach.py#L243) | Mach oscillation amplitude |

## 3. Resampling / decimation

| Technique | Implementation | Notes |
|---|---|---|
| Stride decimation (no anti-alias) | [`downsample_stride`](../../src/data_analysis/signal/core.py#L174) | quick look only; aliases |
| Block-mean (boxcar LPF + decimate) | [`downsample_blockmean`](../../src/data_analysis/signal/core.py#L184) | used in Jun-2026 plotting |
| Polyphase FIR anti-aliased decimation | [`downsample_decimate`](../../src/data_analysis/signal/core.py#L197) | correct choice before FFT |
| Automatic decimation-factor selection | [`analyze_downsample_options`](../../src/data_analysis/signal/core.py#L234) | driven by min timescale |
| Linear interpolation onto a common grid | `np.interp` / `interp1d` | baseline re-expansion |

## 4. Frequency-domain analysis

### 4a. Single-signal spectra
| Technique | Implementation | Where used |
|---|---|---|
| FFT single-sided amplitude spectrum | [`amplitude_spectrum`](../../src/data_analysis/signal/core.py#L464) | Jun-2026 Isat fluctuation spectra |
| Incoherent ensemble-averaged FFT (amplitude average over shots) | [`avg_amplitude_spectrum`](../../src/data_analysis/signal/core.py#L484) | Jun-2026 `batch_fft`; preserves broadband power that coherent averaging would cancel |
| STFT / spectrogram (windowed, overlapping, Hanning/Blackman) | [`calculate_stft`](../../src/data_analysis/signal/core.py#L291) | Aug-2025 & Nov-2024 B-dot microwave-band spectrograms (50 MHz–2 GHz) |
| Group-averaged STFT (running sum over shots, divided at the end — one STFT is ~0.5 GB, so slabs are never all held) | [`compute_group_avg_stft`](Aug-2025/compare_bdot_groups.py#L236) | pass/fail shot-group comparison |
| Band power vs time (STFT collapsed into a frequency band, binned in time) | [`_band_power_vs_time`](Aug-2025/compare_bdot_groups.py#L216) | per-channel pass/fail comparison |

### 4b. Two-signal (cross) analysis
| Technique | Implementation | Where used |
|---|---|---|
| Time-lag cross-correlation (FFT-based, Pearson-normalized) | [`cross_correlation`](../../src/data_analysis/signal/core.py#L530) | Jun-2026 `xcorr`; peak lag = propagation delay between probes |
| Magnitude-squared coherence (Welch) | [`coherence_spectrum`](../../src/data_analysis/signal/core.py#L571) | per-shot coherence |
| Cross-phase spectrum | [`cross_phase_spectrum`](../../src/data_analysis/signal/core.py#L590) | per-shot phase |
| Welch cross-spectral density `Pxy`/`Pxx`/`Pyy` | [`_avg_welch_spectra`](../../src/data_analysis/signal/core.py#L608) | shared core |
| **Ensemble-averaged coherence + cross-phase (Smith 1974)** — complex spectra averaged *before* the ratio | [`avg_cross_spectrum`](../../src/data_analysis/signal/core.py#L646) | Jun-2026 `xcorr_averaged`; the single retained estimator on this branch |
| Narrow-band collapse to scalar coherence/phase | [`band_cross_spectrum`](../../src/data_analysis/signal/core.py#L676) | single-frequency plane maps |
| Peak-tracking band collapse (follows a drifting mode) | [`peak_cross_spectrum`](../../src/data_analysis/signal/core.py#L731) | mode frequency varying across a probe plane |
| Ensemble-mean-trace cross-correlation | [`_ensemble_xcorr`](Jun-2026/Jun2026_xcorr.py#L144) | lag from shot-averaged traces |

## 5. Differentiation and integration

| Technique | Implementation | Where used |
|---|---|---|
| Numerical first derivative `dI/dV` | `np.gradient` — [langmuir.py:121](../../src/data_analysis/plasma/langmuir.py#L121), [:289](../../src/data_analysis/plasma/langmuir.py#L289) | Langmuir probe characteristic |
| Second derivative `d²I/dV²` | [langmuir.py:301](../../src/data_analysis/plasma/langmuir.py#L301) | plasma potential / EEDF |
| Smoothed differentiation (Gaussian-then-gradient) | [`derivative`](../../src/data_analysis/plasma/langmuir.py#L116) | noise-suppressed `dI/dV` |
| Trapezoidal integration | `np.trapezoid` — [langmuir.py:317](../../src/data_analysis/plasma/langmuir.py#L317), [photons.py:230](../../src/data_analysis/plasma/photons.py#L230) | density as area under EEPF; photon pulse area |
| Chord integral / average of a spatial profile, NaN-tolerant | [`line_integral`](../../src/data_analysis/signal/core.py#L450), [`line_average`](../../src/data_analysis/signal/core.py#L468) | interferometer cross-calibration (integral: an average carries the length it was divided by) |

## 6. Curve fitting / regression

| Technique | Implementation | Where used |
|---|---|---|
| Linear least squares (`polyfit`, order 1) | [`_apply_linear_fit`](../../src/data_analysis/plasma/langmuir.py#L505) | ion-sat and e-sat branch fits |
| Quadratic polynomial fit | [langmuir.py:74](../../src/data_analysis/plasma/langmuir.py#L74), [track_object.py:169](../../src/data_analysis/tracking/track_object.py#L169) | `V_p` location; parabolic free-fall trajectory |
| Nonlinear exponential fit (`curve_fit`) | [langmuir.py:654](../../src/data_analysis/plasma/langmuir.py#L654) | electron-retarding region → `T_e` |
| Semi-log linear fit | [`temperature`](../../src/data_analysis/plasma/langmuir.py#L226) | `T_e` from log-I slope |
| Piecewise fit + intersection | [`analyze_IV`](../../src/data_analysis/plasma/langmuir.py#L588) | `V_p` from transition/e-sat line crossing. The crossing is `(d₁-c₁)/(c₀-d₀)`; reversing the operand order negates every `V_p` ([:726](../../src/data_analysis/plasma/langmuir.py#L726)). |
| Weighted least squares on `ln R` | [`fit_calibration`](../../src/data_analysis/plasma/mach.py#L227) | Mach-probe area-ratio (κ) calibration |

## 7. Peak / event / threshold detection

| Technique | Implementation | Where used |
|---|---|---|
| Threshold peak finding (`find_peaks`) with noise-σ thresholds | [`Photons._detect_pulses`](../../src/data_analysis/plasma/photons.py#L47) | x-ray photon counting |
| Noise statistics (mean, σ) → detection threshold | [`_compute_thresholds`](../../src/data_analysis/plasma/photons.py#L162) | `lower/upper_threshold = mean + k·σ`. The "quiet window" is assumed to be the **first 0.1%** of the trace — a pulse landing there inflates σ and silently raises both thresholds. |
| Zero-crossing detection | [`first_and_last_zerocrossings`](../../src/data_analysis/signal/core.py#L19), [`find_all_zerocrossing`](../../src/data_analysis/signal/core.py#L60) | sweep boundaries |
| Sweep-segment detection from ramp voltage | [`find_sweep_indices`](../../src/data_analysis/plasma/langmuir.py#L324) | splitting a swept trace into individual IV sweeps |
| Level-crossing index search | [`_find_crossing_index`](../../src/data_analysis/plasma/langmuir.py#L476) | IV branch boundaries |

## 8. Ensemble statistics and binning

| Technique | Implementation | Where used |
|---|---|---|
| Shot-ensemble mean and SEM | [`mean_sem`](../../src/data_analysis/plasma/langmuir.py#L832) | per-position `V_p`, `T_e`, `n_e` with error bars |
| Log-space (geometric) averaging of ratios | [`combine_log`](../../src/data_analysis/plasma/mach.py#L204) | Mach face ratios — the arithmetic mean of ratios is biased high, and `log_std` is the population spread, not the SEM |
| Time-binned ensemble of face ratios | [`binned_face_ratio`](../../src/data_analysis/plasma/mach.py#L166) | Mach `R(t)` with a per-bin minimum sample count |
| NaN-aware masking of bad shots/fits | [`finite_row_mask`](../../src/data_analysis/signal/core.py#L438), `np.nanmean` | failed-fit rejection before averaging |
| Time-window clipping before analysis | [`clip_time_window`](../../src/data_analysis/signal/core.py#L424) | restricting to the plasma-active window |
| Histogram / time-binned counting | [`counts_per_bin`](../../src/data_analysis/plasma/photons.py#L254) | x-ray counts per 0.2 ms bin |
| Amplitude-gated counting | `amplitude_min/max` in `counts_per_bin` | photon energy-band selection |
| Group comparison of averaged spectra | [`compare_bdot_groups`](Aug-2025/compare_bdot_groups.py#L349) | pass vs. fail shot populations |

## 9. Spatial / profile analysis

| Technique | Implementation | Where used |
|---|---|---|
| Position-indexed shot grouping | [`position_shots`](../../src/data_analysis/io/lapd_hdf5.py#L279) | mapping shots → probe positions |
| Probe ↔ channel join by port number | [`probe_channel_map`](../../src/data_analysis/io/probe_map.py#L217) | which motion group recorded which channel; raises rather than guessing when two groups claim a port |
| 2-D regridding of scattered probe positions | [`grid_by_position`](../../src/data_analysis/viz/plot_utils.py#L642) | plane maps of coherence/phase, `n_e` |
| Radial/line profile extraction | [Jun-2026 `plot_iv_line`](Jun-2026/Jun2026_plot.py#L206), [Mar-2026 `plot_result_line`](Mar-2026/Mar2026_IV.py#L136) | 1-D cuts through a plane scan |
| Polar resolution about a fitted rotation centre | [`polar_components`](../../src/data_analysis/plasma/flow.py#L18), [`find_flow_centre`](../../src/data_analysis/plasma/flow.py#L41) | Mach flow fields → radial / azimuthal components |

## 10. Cross-diagnostic calibration

| Technique | Implementation | Where used |
|---|---|---|
| Interferometer-to-Langmuir absolute density calibration (chord-integral matching over a time window) | [`interferometer_calibration`](../../src/data_analysis/plasma/langmuir.py#L1180), [`calibrate_plasma_npz`](../../src/data_analysis/plasma/langmuir.py#L1264) | Jun-2026 — scales probe `n_e` to the line-integrated measurement |
| Mach-probe area-ratio (κ) calibration from rotation runs | [`fit_calibration`](../../src/data_analysis/plasma/mach.py#L227) | Jun-2026 P33 6-tip; κ binds to the **probe**, so it carries across ports and campaigns |
| Trigger-time alignment between instruments | [`compare_trigger_times`](../../src/data_analysis/io/scope_reader.py#L167) | Nov-2024 scope/camera sync |
| Scope-time ↔ chamber-time mapping | [`scope_ms_to_chamber_s`](Aug-2025/tracking_utils.py#L88) | Aug-2025 x-ray/camera correlation |

## 11. Mach-probe flow estimation

| Technique | Implementation | Notes |
|---|---|---|
| Upstream/downstream face ratio | [`face_ratio`](../../src/data_analysis/plasma/mach.py#L59) | `j+/j-` per tip pair, masked by [`valid_current_mask`](../../src/data_analysis/plasma/mach.py#L46) |
| Area ratio from two orientations | [`area_ratio`](../../src/data_analysis/plasma/mach.py#L85) | geometry-only; cannot be fooled by a design-matrix sign error, so it cross-checks `fit_calibration` |
| Mach number, two opposing orientations | [`mach_number`](../../src/data_analysis/plasma/mach.py#L95) | `M = (K/4)·ln(R_a/R_b)` — κ cancels in the ratio; swapping the arguments flips the sign |
| Mach number, one orientation + known κ | [`mach_single`](../../src/data_analysis/plasma/mach.py#L105) | `M = (K/2)·ln(R/κ)`; limited by κ's systematic error, not shot noise |
| Mach → velocity via sound speed | [`flow_velocity`](../../src/data_analysis/plasma/mach.py#L114) | `v = M·c_s ∝ sqrt(T_e)`; **`T_e` is an input assumption**, so a wrong `T_e` rescales every velocity without changing any diagnostic |
