# object_tracking

High-speed-camera analysis for the falling-tungsten-ball experiment: read a
`.cine` / `.avi` recording and track the ball's trajectory through the chamber.

The code now lives in the installed `data_analysis` package — this folder keeps
only the working notebook:

| File | Purpose |
|------|---------|
| `fastcam_test.ipynb` | End-to-end examples: load frames, track the ball, overlay motion, calibrate. |

## Where the code moved

| Old flat module | Canonical import |
|-----------------|------------------|
| `read_cine.py` | `data_analysis.io.cine` |
| `track_object.py` | `data_analysis.tracking.track_object` |
| `generate_tracking.py` | `data_analysis.tracking.generate_tracking` |
| `evaluate_freefall_accuracy.py` | `data_analysis.tracking.evaluate_freefall_accuracy` |

The old top-level import names (`from read_cine import ...`, etc.) were retired
in the package reorg — import from the canonical paths above.

## Typical workflow

```python
from data_analysis.io.cine import read_cine, overlay_motion_frames
from data_analysis.tracking.track_object import (
    track_object_per_frame, track_object_sparse, chamber_cm_per_px)

cine_path = "path/to/movie.cine"
avi_path  = cine_path.replace(".cine", ".avi")

# 1. Load raw frames (also produces a usable .avi via convert_cine_to_avi)
tarr, frarr, dt = read_cine(cine_path)

# 2a. Every-frame tracking (chamber is hardcoded; see below)
result = track_object_per_frame(avi_path)
positions, frame_numbers, min_ydiff_frame = result   # TrackingResult is tuple-unpackable

# 2b. Or the sparse tracker: ~5 well-separated frames, fit a line. Needs the
#     .cine for timing (fps/t_start) even though detection runs on the .avi.
fit = track_object_sparse(avi_path, cine_path, cm_per_px=chamber_cm_per_px())

# 3. Visualize the trail around any frame
overlay_motion_frames(frarr, center_frame=min_ydiff_frame, n_frames=30, mode="min")
```

**Which tracker.** `track_object_sparse` is what produces the
`tracking_result.npy` line fits the Aug-2025 X-ray pipeline consumes; it samples
only enough frames to constrain a line, so it is far cheaper over a campaign.
`track_object_per_frame` returns every detection and is for inspecting a single
shot.

## `data_analysis.io.cine`

| Function | Description |
|----------|-------------|
| `read_cine(ifn)` | Parse a Phantom `.cine` file. Returns `(time_arr, frame_arr, dt)` where `frame_arr` is `(N, H, W)` uint8/uint16. |
| `convert_cine_to_avi(frame_arr, avi_path, scale_factor=8)` | Write an AVI (MJPG, vertically flipped, upscaled) for downstream OpenCV use. |
| `batch_convert_cine_to_avi(base_path)` | Convert every `.cine` in a directory; skips files that already have an `.avi`. |
| `overlay_motion_frames(frame_arr, center_frame, n_frames, mode="min", step=1, ax=None, ...)` | Stack frames in `[center-n, center+n]` into one image. `mode="min"` for dark objects on bright background, `"max"` for the inverse. `step>1` samples every Nth frame anchored on `center_frame`. Returns `(ax, overlay)`. |

## `data_analysis.tracking.track_object`

### Chamber

The camera mount is fixed across all runs in this analysis, so the chamber
geometry is a single hardcoded constant rather than per-shot detection.
Override only if the camera is physically remounted.

| Function | Description |
|----------|-------------|
| `get_chamber()` | Returns `(CHAMBER_CX, CHAMBER_CY, CHAMBER_RADIUS) = (1121, 1113, 609)`. |
| `chamber_cm_per_px(chamber_radius_px=CHAMBER_RADIUS)` | `(CHAMBER_DIAMETER_CM / 2) / chamber_radius_px`. Uses only the known 36 cm chamber disk, independent of the parabola fit — but it lies on the chamber **back-wall** plane, farther from the camera than any port plane, so it is NOT expected to equal the per-port gravity-fit cm/px (see `data_analysis.tracking.evaluate_freefall_accuracy --port-ratio`). Using one where the other belongs rescales every position. |

### Tracking

| Function | Description |
|----------|-------------|
| `track_object_per_frame(avi_path, cx=None, cy=None, chamber_radius=None, n_workers=1)` | Track the ball through every frame of the video. Chamber comes from `get_chamber()` unless `cx/cy/chamber_radius` are passed explicitly. Uses a fast cropped-ROI search around the last detection and falls back to full-chamber Hough after `BALL_ROI_LOSS_LIMIT` misses. Set `n_workers>1` to split frames into contiguous ranges across a `multiprocessing.Pool`. Returns a `TrackingResult`. |
| `track_object_sparse(avi_path, cine_path, cm_per_px, cx=None, cy=None, chamber_radius=None)` | Sample ~`SPARSE_TARGET_POINTS` frames at least `SPARSE_MIN_SEPARATION_CM` apart and fit a line. Returns a **dict** (the `tracking_result.npy` entry schema), not a `TrackingResult`. Detection runs on the `.avi`; fps and `t_start` come from `cine_path`. |
| `TrackingResult` | Fields: `positions` `(N, 2)`, `frame_numbers` `(N,)`, `min_ydiff_frame`. Iterable, so `pos, fn, mf = track_object_per_frame(...)` still works. |
| `position_from_fit(t_ms, fit)`, `get_ball_position_at_time(...)` | Evaluate a saved sparse line fit at a time. |

### Calibration

| Function | Description |
|----------|-------------|
| `extract_calibration(cine_path)` | Track one cine and fit `y_px(τ) = a + bτ + cτ²` to deduce `cm/px = -0.5 g·100 / c`. Returns `(cm_per_px_gravity, cm_per_px_chamber, x_cm)`, or `(None, None, None)` on failure — check before using, since the three are not distinguishable from a valid result by shape. |
| `average_calibration(dir_path, n=5, pattern="*.cine", out_dir=r"E:\\")` | Run `extract_calibration` over up to `n` files in `dir_path` and save a per-port summary under `out_dir`. The port tag (`P21`, `P30`, …) is parsed from the filenames and must be consistent across the sampled files. Drift is summarized per shot as `np.ptp(x_cm)`. |

### Tracking results dictionary

A separate `np.save`'d dict keyed by full cine path, holding the
user-confirmed center-crossing frame and time. Useful for downstream
analysis that needs `t = 0` aligned to the chamber crossing.

| Function | Description |
|----------|-------------|
| `update_tracking_result(tr_ifn, filepath, cf_new, ct_new)` | Insert/update an entry. |
| `show_tracking_results(tr_ifn)` | Print all entries. |
| `delete_tracking_entry(tr_ifn, filepath)` | Remove an entry. |

### Free-fall reference model

Used to compare tracked trajectories against ideal kinematics.

| Function | Description |
|----------|-------------|
| `get_vel_freefall(h)` | Speed `sqrt(2 g h)` after falling distance `h` from rest. |
| `get_pos_freefall(t, t0, height=0.5, ...)` | Center-relative position vs. time, given the ball passes the chamber center at `t0` after pre-falling `height`. |
| `get_vel_freefall_time(t, t0, height=0.5)` | Velocity vs. time for the same model. |

## Tunable parameters

Detection thresholds live as module-level constants at the top of
`data_analysis/tracking/track_object.py` — edit there rather than in function
bodies:

- `CHAMBER_CX`, `CHAMBER_CY`, `CHAMBER_RADIUS`, `CHAMBER_DIAMETER_CM`
- `BALL_RADIUS_PX_RANGE`, `BALL_HOUGH_PARAMS`,
  `BALL_ROI_RADIUS_PX`, `BALL_ROI_LOSS_LIMIT`
- Sparse tracker: `SPARSE_FIRST_DETECT_STRIDE`, `SPARSE_TARGET_POINTS`,
  `SPARSE_MIN_SEPARATION_CM`, `SPARSE_SWEEP_MAX_FRAMES` (the last is a safety cap
  on the forward sweep, so a shot the tracker never resolves terminates instead
  of scanning the whole video)

## Logging

Both modules use the standard `logging` library. To see info/warning output
in a notebook or script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Common pitfalls

- **Chamber circle is in the wrong place.** The hardcoded `(1121, 1113, 609)`
  is for the current 2048×2048 camera mount. If the camera was repositioned,
  edit the constants at the top of `data_analysis/tracking/track_object.py`.
- **`overlay_motion_frames` chamber circle appears flipped.** The function
  uses `origin="lower"`; pass `(cx, H - cy)` for circle/scatter overlays, or
  call `ax.invert_yaxis()` and use `cy` directly.
- **`n_workers>1` yields slightly different frame indices.** Worker processes
  seek with `cv2.CAP_PROP_POS_FRAMES`, which on non-all-intra AVIs may snap to
  the nearest preceding keyframe. Compare against `n_workers=1` once per new
  codec to confirm results agree.
