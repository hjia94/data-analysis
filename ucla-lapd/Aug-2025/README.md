# Aug-2025

Analysis routines for the August 2025 LAPD campaign. The campaign measured
X-ray bremsstrahlung from a falling tungsten ball, magnetron drive power,
Bdot magnetic fluctuations, and Langmuir probe (LP) profiles across ports
P21 / P23 / P24.

These scripts process the per-shot scope HDF5 files (one file per port-day)
and combine the X-ray pulse train with the high-speed-camera trajectory
fits produced by [object_tracking](../../object_tracking/) into time- and
position-resolved count maps.

## Files

| File | Purpose |
|------|---------|
| [lapd_io.py](lapd_io.py)             | Shared HDF5 readers, `log()` helper, and `sys.path` bootstrap so `read/` and `data_analysis_utils.py` resolve from the repo root. |
| [process_xray.py](process_xray.py)   | Per-shot X-ray pulse detection across every shot in an HDF5 file (or every file in a directory); persists to `analysis_results.npy`. |
| [process_bdot.py](process_bdot.py)   | Per-shot STFT of every Bdot channel and shot-average across the file. |
| [plot_bdot.py](plot_bdot.py)         | Plots the shot-averaged STFT spectrograms (LogNorm, `jet`, one panel per channel). |
| [movie_maker.py](movie_maker.py)     | Interactive / MP4 animation of normalized X-ray counts vs. y-position over time, plus a trajectory-coverage diagnostic movie. |
| [lp.ipynb](lp.ipynb)                 | Working notebook for the XY-plane Langmuir probe sweep (`lpscope` C1 / C2 → averaged Isat profiles). |
| [test.ipynb](test.ipynb)             | Single-shot sandbox for tuning the X-ray `Photons` detector parameters. |

## Typical workflow

```python
# 1. Detect X-ray pulses for every shot in every file under base_dir.
#    Writes/updates analysis_results.npy with keys "{file_prefix}_{shot:03d}".
from process_xray import batch_process_xray
batch_process_xray(r"E:\AUG2025\P23")

# 2. Run the camera tracker (in object_tracking/) to produce
#    tracking_result.npy alongside analysis_results.npy. The tracker stores
#    the per-shot sparse line fit (y_intercept, y_slope) and cm_per_px.
#    See ../../object_tracking/README.md.

# 3. Build the count-vs-y movie by combining both .npy files.
from movie_maker import plot_result
plot_result(r"E:\AUG2025\P23", uw_start=30, frame_step_ms=1)
```

For the Bdot side, run [process_bdot.py](process_bdot.py) directly on a
single HDF5 file — it is independent of the X-ray / camera pipeline.

## `lapd_io.py`

| Function | Description |
|----------|-------------|
| `log(tag, msg)` | Prefixed `print` used throughout the Aug-2025 scripts. |
| `get_magnetron_power_data(f, result, scope_name='magscope')` | Read current/voltage/Pref channels by description string, scale current using the `<n> a/v` token in the description, and return `(tarr, P)` with `P = gaussian_filter1d(I*(-V)*0.6, sigma=100)`. |
| `get_xray_data(result, scope_name='xrayscope')` | Return `(tarr, C2)` for the X-ray scope. |
| `get_bdot_data(f, result, scope_name='bdotscope')` | Return `(tarr, channels_dict, descriptions_dict)`. |

Importing the module also rewrites `sys.path` so `read.read_scope_data`
and `data_analysis_utils` import without a manual `sys.path.append` in the
caller.

## `process_xray.py`

| Function | Description |
|----------|-------------|
| `process_shot_xray(tarr_x, xray_data, min_ts, d, threshold, debug=False)` | Run `data_analysis_utils.Photons` on a single shot; return `(pulse_times, pulse_amplitudes)`. |
| `xray_wt_cam(base_dir, fn)` | Process every shot in one HDF5 file. Skips shots already present in `analysis_results.npy` so the call is incremental. Default detector params: `threshold=[10, 80]`, `min_ts=0.8e-6 s`, `distance_mult=0.1`. |
| `batch_process_xray(base_dir)` | Run `xray_wt_cam` on every `.hdf5` in `base_dir`. |

`analysis_results.npy` is a dict keyed by `"{file_prefix}_{shot:03d}"`
where `file_prefix` is the leading two characters of the HDF5 filename
(e.g. `"02"` for `02_He1kG430G_..._2025-08-12.hdf5`).

> **Per-port tuning.** The threshold list at the top of `xray_wt_cam` was
> `[5, 70]` for P23/P24 data and `[10, 80]` for P21. Adjust before
> re-running on a new port.

## `process_bdot.py` & `plot_bdot.py`

| Function | Description |
|----------|-------------|
| `calculate_bdot_stft(tarr, bdot_data, freq_bins=1000, overlap_fraction=0.05, freq_min=200e6, freq_max=2000e6)` | Hanning-windowed STFT (via `data_analysis_utils.calculate_stft`) for each channel; returns `(stft_time, freq, {channel: matrix})`. |
| `process_bdot(ifn, freq_bins=1000, overlap_fraction=0.05, freq_min=50e6, freq_max=1000e6, plot=True)` | Iterate every shot in the file, compute per-shot STFT, then average across shots per channel. Plots when `plot=True`. |
| `plot_averaged_bdot_stft(stft_matrices, description, stft_tarr, freq_arr)` | One LogNorm `jet` panel per channel, x in ms, y in MHz. |
| `_floor_for_lognorm(matrix)` | Replace non-positive entries with the smallest positive value so `LogNorm` does not blow up. |

The plotter uses `colors.LogNorm`, which would error on zero/negative
entries — `_floor_for_lognorm` is the reason there is no special-case in
the plot loop itself.

## `movie_maker.py`

Builds the count-vs-y animation that combines the X-ray pulse trains with
the camera trajectory fits.

| Function | Description |
|----------|-------------|
| `plot_result(base_dir, uw_start=30, frame_step_ms=1.0, save_mp4=False, output_filename='animation.mp4', fps=10)` | Read `analysis_results.npy` and `tracking_result.npy` from `base_dir`, normalize each shot's counts by the number of valid trajectories crossing the same `(y, t)` bin (`object_tracking.generate_tracking.count_y_passes`), and animate the bar chart. Interactive (slider + Play/Pause) by default; saves an MP4 when `save_mp4=True`. |
| `plot_trajectory_coverage(base_dir, frame_step_ms=1.0, y_min=-50, y_max=50, y_bin_width=0.5, t_min=0, t_max=45, save_mp4=False, ...)` | Coverage diagnostic — animates valid-trajectory crossings per `(y, t)` bin without using any X-ray data. Useful for spotting under-sampled regions before interpreting `plot_result`. |
| `draw_frame`, `draw_coverage_frame`, `Player` | Internal helpers for the two animations. |

### Inputs

| File | Producer |
|------|----------|
| `analysis_results.npy` | [process_xray.py](process_xray.py) — per-shot pulse train. |
| `tracking_result.npy`  | [object_tracking/generate_tracking.py](../../object_tracking/generate_tracking.py) — sparse line fit `y_px(t)` per shot, plus `cm_per_px`. |

The two files must live in the same `base_dir`. The tracking entries
carry their own `cm_per_px`, so no calibration file is loaded here.

### How the y-axis is computed

For each tracked shot, position vs. time is reconstructed from the line
fit:

```
t_s     = (bin_centers + uw_start) * 1e-3              # seconds
r_arr   = -(y_intercept + y_slope * t_s)               # cm below center
```

`uw_start` is the offset (ms) between the X-ray scope time origin and the
microwave start used by `generate_tracking.py`. The default `uw_start=30`
matches the August 2025 timing.

### Legacy cache guard

If a `tracking_result.npy` predates the sparse-fit schema (`y_slope` /
`y_intercept` / `n_points`), the loader raises `TypeError` rather than
silently falling back. Re-run `object_tracking/generate_tracking.py` to
rebuild it.

## `lp.ipynb`

Notebook-only Langmuir probe pipeline for the XY-plane sweep file (e.g.
`31-LP-p24-XYplane+line-He1kG380G-6300A_2025-08-21.hdf5`):

1. `read_positions(ifn)` → `(m_list, pos_array, npos, nshot)`
   (motion list, positions, number of unique positions, shots per
   position).
2. Loop over `npos × nshot`, average `lpscope/C1` and `C2` per position
   into `lp_data_a`, `lp_data_b` (shape `(npos, nt)`).
3. Time-evolution panel + density profile panel for several `tndx`
   ranges.
4. Per-time-range XY scatter colored by averaged Isat.

Set the path in cell 2 (`ifn = ...`) before running.

## Conventions

- All entry-point scripts hardcode `base_dir`/`ifn` at the bottom under
  `if __name__ == "__main__":` for the run that produced the figure in
  use. Edit there rather than wrapping in a CLI.
- Two-digit file prefixes (`02_…`, `27_…`, `86_…`) are how shots are
  keyed across the `analysis_results.npy`/`tracking_result.npy` pair.
  Do not rename HDF5 files after processing.
- `%matplotlib qt` is required for the interactive widgets in
  `movie_maker.plot_result` and the LP notebook; they will not work in
  inline backends.

## Dependencies

Beyond the standard scientific stack (`numpy`, `scipy`, `matplotlib`,
`h5py`), this folder pulls in three sibling modules from the repo root:

- [read/read_scope_data.py](../../read/read_scope_data.py) — HDF5 scope
  readers.
- [data_analysis_utils.py](../../data_analysis_utils.py) — `Photons`,
  `calculate_stft`, `counts_per_bin`.
- [object_tracking/generate_tracking.py](../../object_tracking/generate_tracking.py)
  — `count_y_passes` and the producer of `tracking_result.npy`.

Saving an MP4 from `movie_maker.py` additionally requires `ffmpeg` on
`PATH`.
