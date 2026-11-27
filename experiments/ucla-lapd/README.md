# Data analysis techniques used in ucla-lapd experiments

## 1. Smoothing

| Technique | Implementation | Where used |
|---|---|---|
| Gaussian kernel smoothing (1-D) | `gaussian_filter1d`| IV curve smoothing before differentiation; sweep prep (Jun-2026, Mar-2026, Jan-2024) |
| Gaussian smoothing via FFT convolution | [`fast_gaussian_filter`](../../src/data_analysis/signal/core.py#L81) | long multi-million-sample traces |
| Savitzky–Golay (polynomial) smoothing | [`Photons`](../../src/data_analysis/plasma/photons.py#L47) (`savgol_window=31`, `savgol_order=3`) | x-ray pulse traces before peak detection (Aug-2025, Nov-2024); also Jan-2024 Isat sweeps |
| Median filtering | [`ndimage.median_filter`](Mar-2026/Mar2026_mach.py#L232) | Mach probe baseline estimate, on the decimated trace |
| Rolling-quantile (percentile) smoothing | [`rolling_baseline`](../../src/data_analysis/signal/core.py#L127) | slow-baseline estimation |

## 2. Filtering

| Technique | Implementation | Where used |
|---|---|---|
| Butterworth low-pass, zero-phase (`filtfilt`) | [`low_pass_filter`](../../src/data_analysis/signal/core.py#L101) | generic trace conditioning |
| Butterworth band-pass, SOS + `sosfiltfilt` | [`butter_bandpass`](../../src/data_analysis/signal/core.py#L114) | Mach probe fluctuation band 0.5–12 kHz ([Mar2026_mach.py](Mar-2026/Mar2026_mach.py#L202)) |
| Detrending / mean subtraction | `cross_correlation` | every spectral estimate |
| Baseline subtraction (linear fit to the ion-sat branch) | [langmuir](../../src/data_analysis/plasma/langmuir.py#L614) | ion-saturation baseline removed from IV |
| Baseline subtraction (envelope interpolation) | [photons](../../src/data_analysis/plasma/photons.py#L160) | x-ray photon traces |
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
| FFT single-sided amplitude spectrum | [`amplitude_spectrum`](../../src/data_analysis/signal/core.py#L483) | Jun-2026 Isat fluctuation spectra |
| Incoherent ensemble-averaged FFT (amplitude average over shots) | [`avg_amplitude_spectrum`](../../src/data_analysis/signal/core.py#L503) | Jun-2026 `batch_fft`; preserves broadband power that coherent averaging would cancel |
| STFT / spectrogram (windowed, overlapping, Hanning/Blackman) | [`calculate_stft`](../../src/data_analysis/signal/core.py#L291) | Aug-2025 & Nov-2024 B-dot microwave-band spectrograms (50 MHz–2 GHz) |
| Group-averaged STFT (running sum over shots, divided at the end — one STFT is ~0.5 GB, so slabs are never all held) | [`compute_group_avg_stft`](Aug-2025/compare_bdot_groups.py#L236) | pass/fail shot-group comparison |
| Band power vs time (STFT collapsed into a frequency band, binned in time) | [`_band_power_vs_time`](Aug-2025/compare_bdot_groups.py#L216) | per-channel pass/fail comparison |

### 4b. Two-signal (cross) analysis
| Technique | Implementation | Where used |
|---|---|---|
| Time-lag cross-correlation (FFT-based, Pearson-normalized) | [`cross_correlation`](../../src/data_analysis/signal/core.py#L549) | Jun-2026 `xcorr`; peak lag = propagation delay between probes |
| Magnitude-squared coherence (Welch) | [`coherence_spectrum`](../../src/data_analysis/signal/core.py#L590) | per-shot coherence |
| Cross-phase spectrum | [`cross_phase_spectrum`](../../src/data_analysis/signal/core.py#L609) | per-shot phase |
| Welch cross-spectral density `Pxy`/`Pxx`/`Pyy` | [`_avg_welch_spectra`](../../src/data_analysis/signal/core.py#L627) | shared core |
| **Ensemble-averaged coherence + cross-phase (Smith 1974)** — complex spectra averaged *before* the ratio | [`avg_cross_spectrum`](../../src/data_analysis/signal/core.py#L665) | Jun-2026 `xcorr_averaged`; the single retained estimator on this branch |
| Narrow-band collapse to scalar coherence/phase | [`band_cross_spectrum`](../../src/data_analysis/signal/core.py#L695) | single-frequency plane maps |
| Peak-tracking band collapse (follows a drifting mode) | [`peak_cross_spectrum`](../../src/data_analysis/signal/core.py#L750) | mode frequency varying across a probe plane |
| Ensemble-mean-trace cross-correlation | [`_ensemble_xcorr`](Jun-2026/Jun2026_xcorr.py#L144) | lag from shot-averaged traces |

### 4c. Three-wave (higher-order) analysis
| Technique | Implementation | Where used |
|---|---|---|
| Bicoherence and bispectrum `<X(f1) Y(f2) Z*(f1+f2)>`, Kim–Powers normalization (0–1), averaged over shots and segments, evaluated in chosen `f1`/`f2` bands (negative `f2` for difference triads) | [`bicoherence`](../../src/data_analysis/signal/bispectral.py#L192), [`cross_bicoherence`](../../src/data_analysis/signal/bispectral.py#L215) | quadratic phase coupling, e.g. pump / sideband / zonal-flow triads; rewritten from the HOSA port `synergetics/spectrum` |
| Chance level of b² for N realizations, `1 - p^(1/(N-1))` | [`bicoherence_threshold`](../../src/data_analysis/signal/bispectral.py#L87) | significance of a bicoherence peak |

## 5. Differentiation and integration

| Technique | Implementation | Where used |
|---|---|---|
| Numerical first derivative `dI/dV` | `np.gradient` — [langmuir.py:119](../../src/data_analysis/plasma/langmuir.py#L119), [:287](../../src/data_analysis/plasma/langmuir.py#L287) | Langmuir probe characteristic |
| Second derivative `d²I/dV²` | [langmuir.py:299](../../src/data_analysis/plasma/langmuir.py#L299) | plasma potential / EEDF |
| Smoothed differentiation (Gaussian-then-gradient) | [`derivative`](../../src/data_analysis/plasma/langmuir.py#L114) | noise-suppressed `dI/dV` |
| Trapezoidal integration | `np.trapezoid` — [langmuir.py:315](../../src/data_analysis/plasma/langmuir.py#L315), [photons.py:240](../../src/data_analysis/plasma/photons.py#L240) | density as area under EEPF; photon pulse area |
| Chord integral / average of a spatial profile, NaN-tolerant | [`line_integral`](../../src/data_analysis/signal/core.py#L457), [`line_average`](../../src/data_analysis/signal/core.py#L471) | interferometer cross-calibration (integral: an average carries the length it was divided by) |

## 6. Curve fitting / regression

| Technique | Implementation | Where used |
|---|---|---|
| Linear least squares (`polyfit`, order 1) | [`_apply_linear_fit`](../../src/data_analysis/plasma/langmuir.py#L500) | ion-sat and e-sat branch fits |
| Quadratic polynomial fit | [langmuir.py:81](../../src/data_analysis/plasma/langmuir.py#L81), [track_object.py:170](../../src/data_analysis/tracking/track_object.py#L170) | `V_p` location; parabolic free-fall trajectory |
| Nonlinear exponential fit (`curve_fit`) | [langmuir.py:659](../../src/data_analysis/plasma/langmuir.py#L659) | electron-retarding region → `T_e` |
| Semi-log linear fit | [`temperature`](../../src/data_analysis/plasma/langmuir.py#L224) | `T_e` from log-I slope |
| Piecewise fit + intersection | [`analyze_IV`](../../src/data_analysis/plasma/langmuir.py#L583) | `V_p` from transition/e-sat line crossing. The crossing is `(d₁-c₁)/(c₀-d₀)`; reversing the operand order negates every `V_p` ([:724](../../src/data_analysis/plasma/langmuir.py#L724)). |
| Weighted least squares on `ln R` | [`fit_calibration`](../../src/data_analysis/plasma/mach.py#L227) | Mach-probe area-ratio (κ) calibration |

## 7. Peak / event / threshold detection

| Technique | Implementation | Where used |
|---|---|---|
| Threshold peak finding (`find_peaks`) with noise-σ thresholds | [`Photons._detect_pulses`](../../src/data_analysis/plasma/photons.py#L182) | x-ray photon counting |
| Noise statistics (mean, σ) → detection threshold | [`_compute_thresholds`](../../src/data_analysis/plasma/photons.py#L172) | `lower/upper_threshold = mean + k·σ`. The "quiet window" is assumed to be the **first 0.1%** of the trace — a pulse landing there inflates σ and silently raises both thresholds. |
| Zero-crossing detection | [`first_and_last_zerocrossings`](../../src/data_analysis/signal/core.py#L19), [`find_all_zerocrossing`](../../src/data_analysis/signal/core.py#L60) | sweep boundaries |
| Sweep-segment detection from ramp voltage | [`find_sweep_indices`](../../src/data_analysis/plasma/langmuir.py#L322) | splitting a swept trace into individual IV sweeps |
| Level-crossing index search | [`_find_crossing_index`](../../src/data_analysis/plasma/langmuir.py#L471) | IV branch boundaries |

## 8. Ensemble statistics and binning

| Technique | Implementation | Where used |
|---|---|---|
| Shot-ensemble mean and SEM | [`mean_sem`](../../src/data_analysis/plasma/langmuir.py#L824) | per-position `V_p`, `T_e`, `n_e` with error bars |
| Log-space (geometric) averaging of ratios | [`combine_log`](../../src/data_analysis/plasma/mach.py#L204) | Mach face ratios — the arithmetic mean of ratios is biased high, and `log_std` is the population spread, not the SEM |
| Time-binned ensemble of face ratios | [`binned_face_ratio`](../../src/data_analysis/plasma/mach.py#L166) | Mach `R(t)` with a per-bin minimum sample count |
| NaN-aware masking of bad shots/fits | [`finite_row_mask`](../../src/data_analysis/signal/core.py#L437), `np.nanmean` | failed-fit rejection before averaging |
| Time-window clipping before analysis | [`clip_time_window`](../../src/data_analysis/signal/core.py#L423) | restricting to the plasma-active window |
| Histogram / time-binned counting | [`counts_per_bin`](../../src/data_analysis/plasma/photons.py#L264) | x-ray counts per 0.2 ms bin |
| Amplitude-gated counting | `amplitude_min/max` in `counts_per_bin` | photon energy-band selection |
| Group comparison of averaged spectra | [`compare_bdot_groups`](Aug-2025/compare_bdot_groups.py#L348) | pass vs. fail shot populations |

## 9. Spatial / profile analysis

| Technique | Implementation | Where used |
|---|---|---|
| Position-indexed shot grouping | [`position_shots`](../../src/data_analysis/io/lapd_hdf5.py#L316) | mapping shots → probe positions |
| Probe ↔ channel join by port number | [`probe_channel_map`](../../src/data_analysis/io/probe_map.py#L217) | which motion group recorded which channel; raises rather than guessing when two groups claim a port |
| 2-D regridding of scattered probe positions | [`grid_by_position`](../../src/data_analysis/viz/plot_utils.py#L643) | plane maps of coherence/phase, `n_e` |
| Radial/line profile extraction | [Jun-2026 `plot_iv_line`](Jun-2026/Jun2026_plot.py#L181), [Mar-2026 `plot_result_line`](Mar-2026/Mar2026_IV.py#L149) | 1-D cuts through a plane scan |
| Polar resolution about a fitted rotation centre | [`polar_components`](../../src/data_analysis/plasma/flow.py#L18), [`find_flow_centre`](../../src/data_analysis/plasma/flow.py#L41) | Mach flow fields → radial / azimuthal components |

## 10. Cross-diagnostic calibration

| Technique | Implementation | Where used |
|---|---|---|
| Interferometer-to-Langmuir absolute density calibration (chord-integral matching over a time window) | [`interferometer_calibration`](../../src/data_analysis/plasma/langmuir.py#L1197), [`calibrate_plasma_npz`](../../src/data_analysis/plasma/langmuir.py#L1260) | Jun-2026 — scales probe `n_e` to the line-integrated measurement |
| Mach-probe area-ratio (κ) calibration from rotation runs | [`fit_calibration`](../../src/data_analysis/plasma/mach.py#L227) | Jun-2026 P33 6-tip; κ binds to the **probe**, so it carries across ports and campaigns |
| Trigger-time alignment between instruments | [`compare_trigger_times`](../../src/data_analysis/io/scope_reader.py#L156) | Nov-2024 scope/camera sync |
| Scope-time ↔ chamber-time mapping | [`scope_ms_to_chamber_s`](Aug-2025/tracking_utils.py#L88) | Aug-2025 x-ray/camera correlation |

## 11. Mach-probe flow estimation

| Technique | Implementation | Notes |
|---|---|---|
| Upstream/downstream face ratio | [`face_ratio`](../../src/data_analysis/plasma/mach.py#L59) | `j+/j-` per tip pair, masked by [`valid_current_mask`](../../src/data_analysis/plasma/mach.py#L46) |
| Area ratio from two orientations | [`area_ratio`](../../src/data_analysis/plasma/mach.py#L85) | geometry-only; cannot be fooled by a design-matrix sign error, so it cross-checks `fit_calibration` |
| Mach number, two opposing orientations | [`mach_number`](../../src/data_analysis/plasma/mach.py#L106) | `M = flow_sign·(K/2)·ln(R_a/R_b)` (`K = M_c = 0.45`, `flow_sign=-1` when `+` tip faces `+axis`); κ cancels in the ratio |
| Mach number, one orientation + known κ | [`mach_single`](../../src/data_analysis/plasma/mach.py#L118) | `M = flow_sign·K·ln(R/κ)` (`K = M_c = 0.45`, `flow_sign=-1` for lab-axis flow when `+` tip faces `+axis`); limited by κ's systematic error |
| Mach → velocity via sound speed | [`flow_velocity`](../../src/data_analysis/plasma/mach.py#L114) | `v = M·c_s ∝ sqrt(T_e)`; **`T_e` is an input assumption**, so a wrong `T_e` rescales every velocity without changing any diagnostic |

## 12. B-dot probes and gridded field operators

| Technique | Implementation | Where used |
|---|---|---|
| Coil calibration → volts-to-dB/dt (scalar, 3 areas, or 3×3 cross-talk matrix with gain) | [`inverse_area_matrix`](../../src/data_analysis/plasma/bdot.py#L33), [`voltage_to_bdot`](../../src/data_analysis/plasma/bdot.py#L58) | Aug-2026 `Aug2026_bdot` (unit area until C11/C13 are calibrated) |
| Time integration dB/dt → B with drift baselines (`pretrigger` / `periodic` / `smooth`) | [`integrate_bdot`](../../src/data_analysis/plasma/bdot.py#L92) | Aug-2026: integration weights slow content by 1/f, so analyze the driven wave as a phasor or with `smooth` |
| Single-frequency & reference-locked phasors (exact-f DFT, windowed, shot-phase locked) | [`phasor`](../../src/data_analysis/plasma/fields.py#L374), [`reference_locked_phasor`](../../src/data_analysis/plasma/fields.py#L151) | Aug-2026 188 kHz wave maps (trigger-referenced); Jun-2026 2-D B-dot self-excited mode maps (stationary LP reference-locked) |
| Curl / J = curl B/μ₀, 2-D divergence & `B_L`/`B_R` circular polarization on a grid | [`curl`](../../src/data_analysis/plasma/fields.py#L66), [`current_density`](../../src/data_analysis/plasma/fields.py#L88), [`divergence_2d`](../../src/data_analysis/plasma/fields.py#L93), [`circular_polarization`](../../src/data_analysis/plasma/fields.py#L109) | Aug-2026 run 30 field-aligned current map; Jun-2026 `Jun2026_bdot_2d` (Runs 26 & 31 Maxwell div/curl polarity proof, `J_par`, `|Bz|/|B_perp|`, core LHCP → outer RHCP flip) |
| Per-cycle averages over a drive burst | [`cycle_means`](../../src/data_analysis/plasma/fields.py#L214) | Aug-2026 wave envelope |
| Poynting, J·E, Faraday E_φ (symmetric about the axis), Biot–Savart A, volume integrals, AC conductivity | [`fields`](../../src/data_analysis/plasma/fields.py) | ported from the 2019 process-plasma B-dot analysis; not yet used by a campaign (needs E data or symmetry about the axis) |
