# Data analysis techniques used in ucla-lapd experiments

## 1. Smoothing

| Technique | Implementation | Where used |
|---|---|---|
| Gaussian kernel smoothing (1-D) | `gaussian_filter1d`| IV curve smoothing before differentiation; sweep prep (Jun-2026, Mar-2026, Jan-2024) |
| Gaussian smoothing via FFT convolution | [`fast_gaussian_filter`](../../src/data_analysis/signal/core.py#L81) | long multi-million-sample traces |
| Savitzky–Golay (polynomial) smoothing | defining baseline | x-ray pulse traces before peak detection (Aug-2025, Nov-2024) |
| Median filtering | defining baseline | Mach probe baseline estimate |
| Rolling-quantile (percentile) smoothing | [`rolling_baseline`](../../src/data_analysis/signal/core.py#L127) | slow-baseline estimation |

## 2. Filtering

| Technique | Implementation | Where used |
|---|---|---|
| Butterworth low-pass, zero-phase (`filtfilt`) | [`low_pass_filter`](../../src/data_analysis/signal/core.py#L101) | generic trace conditioning |
| Butterworth band-pass, SOS + `sosfiltfilt` | [`butter_bandpass`](../../src/data_analysis/signal/core.py#L114) | Mach probe fluctuation band 0.5–12 kHz ([Mar2026_mach.py](Mar-2026/Mar2026_mach.py#L237)) |
| Detrending / mean subtraction | `cross_correlation` | every spectral estimate |
| Baseline subtraction (polynomial fit) | [langmuir](../../src/data_analysis/plasma/langmuir.py#L499) | ion-saturation baseline removed from IV |
| Baseline subtraction (envelope interpolation) | [photons](../../src/data_analysis/plasma/photons.py#L150) | x-ray photon traces |
| Baseline subtraction (decimate → median → interpolate) | [Mar2026_mach](Mar-2026/Mar2026_mach.py#L230-L234) | Mach probe |
| Envelope detection (upper/lower extrema) | [`hl_envelopes_idx`](../../src/data_analysis/signal/core.py#L143) | photon baseline, Nov-2024 x-ray |
| Moving-RMS envelope | [Mar2026_mach](Mar-2026/Mar2026_mach.py#L240) | Mach oscillation amplitude |

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
| Group-averaged STFT | [compare_bdot_groups](Aug-2025/compare_bdot_groups.py#L195) | pass/fail shot-group comparison |

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
| Ensemble-mean-trace cross-correlation | [`_ensemble_xcorr`](Jun-2026/Jun2026_xcorr.py#L138) | lag from shot-averaged traces |

## 5. Differentiation and integration

| Technique | Implementation | Where used |
|---|---|---|
| Numerical first derivative `dI/dV` | `np.gradient` — [langmuir.py:112](../../src/data_analysis/plasma/langmuir.py#L112), [:280](../../src/data_analysis/plasma/langmuir.py#L280) | Langmuir probe characteristic |
| Second derivative `d²I/dV²` | [langmuir.py:292](../../src/data_analysis/plasma/langmuir.py#L292) | plasma potential / EEDF |
| Smoothed differentiation (Gaussian-then-gradient) | [`derivative`](../../src/data_analysis/plasma/langmuir.py#L107) | noise-suppressed `dI/dV` |
| Trapezoidal integration | `np.trapezoid` — [langmuir.py:308](../../src/data_analysis/plasma/langmuir.py#L308), [photons.py:230](../../src/data_analysis/plasma/photons.py#L230) | density as area under EEPF; photon pulse area |
| Chord (line) average of a spatial profile, NaN-tolerant | [`line_average`](../../src/data_analysis/signal/core.py#L450) | interferometer cross-calibration |

## 6. Curve fitting / regression

| Technique | Implementation | Where used |
|---|---|---|
| Linear least squares (`polyfit`, order 1) | [`_apply_linear_fit`](../../src/data_analysis/plasma/langmuir.py#L456) | ion-sat and e-sat branch fits |
| Quadratic polynomial fit | [langmuir.py:74](../../src/data_analysis/plasma/langmuir.py#L74), [track_object.py:169](../../src/data_analysis/tracking/track_object.py#L169) | `V_p` location; parabolic free-fall trajectory |
| Nonlinear exponential fit (`curve_fit`) | [langmuir.py:545](../../src/data_analysis/plasma/langmuir.py#L545) | electron-retarding region → `T_e` |
| Semi-log linear fit | [`temperature`](../../src/data_analysis/plasma/langmuir.py#L217) | `T_e` from log-I slope |
| Piecewise fit + intersection | [`analyze_IV`](../../src/data_analysis/plasma/langmuir.py#L480) | `V_p` from transition/e-sat line crossing |

## 7. Peak / event / threshold detection

| Technique | Implementation | Where used |
|---|---|---|
| Threshold peak finding (`find_peaks`) with noise-σ thresholds | [`Photons._detect_pulses`](../../src/data_analysis/plasma/photons.py#L47) | x-ray photon counting |
| Noise statistics (mean, σ) from a quiet window → detection threshold | [photons.py:165-171](../../src/data_analysis/plasma/photons.py#L165-L171) | `lower/upper_threshold = mean + k·σ` |
| Zero-crossing detection | [`first_and_last_zerocrossings`](../../src/data_analysis/signal/core.py#L19), [`find_all_zerocrossing`](../../src/data_analysis/signal/core.py#L60) | sweep boundaries |
| Sweep-segment detection from ramp voltage | [`find_sweep_indices`](../../src/data_analysis/plasma/langmuir.py#L315) | splitting a swept trace into individual IV sweeps |
| Level-crossing index search | [`_find_crossing_index`](../../src/data_analysis/plasma/langmuir.py#L427) | IV branch boundaries |

## 8. Ensemble statistics and binning

| Technique | Implementation | Where used |
|---|---|---|
| Shot-ensemble mean and SEM | [`mean_sem`](../../src/data_analysis/plasma/langmuir.py#L730) | per-position `V_p`, `T_e`, `n_e` with error bars |
| NaN-aware masking of bad shots/fits | [`finite_row_mask`](../../src/data_analysis/signal/core.py#L438), `np.nanmean` | failed-fit rejection before averaging |
| Time-window clipping before analysis | [`clip_time_window`](../../src/data_analysis/signal/core.py#L424) | restricting to the plasma-active window |
| Histogram / time-binned counting | [`counts_per_bin`](../../src/data_analysis/plasma/photons.py#L254) | x-ray counts per 0.2 ms bin |
| Amplitude-gated counting | `amplitude_min/max` in `counts_per_bin` | photon energy-band selection |
| Group comparison of averaged spectra | [`compare_bdot_groups`](Aug-2025/compare_bdot_groups.py#L313) | pass vs. fail shot populations |

## 9. Spatial / profile analysis

| Technique | Implementation | Where used |
|---|---|---|
| Position-indexed shot grouping | [`position_shots`](../../src/data_analysis/io/lapd_hdf5.py#L223) | mapping shots → probe positions |
| 2-D regridding of scattered probe positions | [`grid_by_position`](../../src/data_analysis/viz/plot_utils.py#L638) | plane maps of coherence/phase, `n_e` |
| Radial/line profile extraction | Jun-2026 `plot_iv_line`, Mar-2026 `plot_result_line` | 1-D cuts through a plane scan |

## 10. Cross-diagnostic calibration

| Technique | Implementation | Where used |
|---|---|---|
| Interferometer-to-Langmuir absolute density calibration (chord-average matching over a time window) | [`interferometer_calibration`](../../src/data_analysis/plasma/langmuir.py#L911), [`calibrate_plasma_npz`](../../src/data_analysis/plasma/langmuir.py#L980) | Jun-2026 — scales probe `n_e` to the line-integrated measurement |
| Trigger-time alignment between instruments | [`compare_trigger_times`](../../src/data_analysis/io/scope_reader.py#L167) | Nov-2024 scope/camera sync |
| Scope-time ↔ chamber-time mapping | [`scope_ms_to_chamber_s`](Aug-2025/tracking_utils.py#L88) | Aug-2025 x-ray/camera correlation |
