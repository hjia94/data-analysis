# object_tracking

Tools for reading high-speed camera (.cine / .avi) recordings and tracking
the trajectory of a tungsten ball falling through the chamber.

## Files

| File | Purpose |
|------|---------|
| `read_cine.py`        | Parse `.cine` binary files, convert to `.avi`, and visualize motion. |
| `track_object.py`     | Track the ball through a video using a hardcoded chamber location. |
| `evaluate_freefall_accuracy.py` | Per-shot gravity-fit calibration and the cross-port chamber/gravity ratio plot. |
| `fastcam_test.ipynb`  | Working notebook with end-to-end examples. |

## Typical workflow

```python
from read_cine import read_cine, overlay_motion_frames
from track_object import track_object

cine_path = "path/to/movie.cine"
avi_path  = cine_path.replace(".cine", ".avi")

# 1. Load raw frames (also produces a usable .avi via convert_cine_to_avi)
tarr, frarr, dt = read_cine(cine_path)

# 2. Track the ball across the whole video (chamber is hardcoded; see below)
result = track_object(avi_path)
positions, frame_numbers, min_ydiff_frame = result   # tuple-unpackable

# 3. Visualize the trail around any frame
overlay_motion_frames(frarr, center_frame=min_ydiff_frame, n_frames=30, mode="min")
```

## `read_cine.py`

| Function | Description |
|----------|-------------|
| `read_cine(ifn)` | Parse a Phantom `.cine` file. Returns `(time_arr, frame_arr, dt)` where `frame_arr` is `(N, H, W)` uint8/uint16. |
| `convert_cine_to_avi(frame_arr, avi_path, scale_factor=8)` | Write an AVI (MJPG, vertically flipped, upscaled) for downstream OpenCV use. |
| `batch_convert_cine_to_avi(base_path)` | Convert every `.cine` in a directory; skips files that already have an `.avi`. |
| `overlay_motion_frames(frame_arr, center_frame, n_frames, mode="min", step=1, ax=None, ...)` | Stack frames in `[center-n, center+n]` into one image. `mode="min"` for dark objects on bright background, `"max"` for the inverse. `step>1` samples every Nth frame anchored on `center_frame`. Returns `(ax, overlay)`. |

## `track_object.py`

### Chamber

The camera mount is fixed across all runs in this analysis, so the chamber
geometry is a single hardcoded constant rather than per-shot detection.
Override only if the camera is physically remounted.

| Function | Description |
|----------|-------------|
| `get_chamber()` | Returns `(CHAMBER_CX, CHAMBER_CY, CHAMBER_RADIUS) = (1121, 1113, 609)`. |
| `chamber_cm_per_px(radius_px=CHAMBER_RADIUS)` | `18 cm / radius_px`. Lies on the chamber back-wall plane, so it does NOT equal the per-port gravity-fit cm/px (see `evaluate_freefall_accuracy.py --port-ratio`). |

### Tracking

| Function | Description |
|----------|-------------|
| `track_object(avi_path, cx=None, cy=None, chamber_radius=None, n_workers=1)` | Track the ball through the whole video. Chamber comes from `get_chamber()` unless `cx/cy/chamber_radius` are passed explicitly. Uses a fast cropped-ROI search around the last detection and falls back to full-chamber Hough after `BALL_ROI_LOSS_LIMIT` misses. Set `n_workers>1` to split frames into contiguous ranges across a `multiprocessing.Pool`. Returns a `TrackingResult` dataclass. |
| `TrackingResult` | Fields: `positions` `(N, 2)`, `frame_numbers` `(N,)`, `min_ydiff_frame`. Iterable, so `pos, fn, mf = track_object(...)` still works. |

### Calibration

| Function | Description |
|----------|-------------|
| `extract_calibration(cine_path)` | Track one cine and fit `y_px(τ) = a + bτ + cτ²` to deduce `cm/px = -0.5 g·100 / c`. Returns `(cm_per_px_gravity, cm_per_px_chamber, x_cm)`. |
| `average_calibration(dir_path, n=5)` | Run `extract_calibration` over up to `n` files matching a port tag in `dir_path`, save the per-port summary to `E:/calibration_factor_P{N}.npy`. |

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
`track_object.py` — edit there rather than in function bodies:

- `CHAMBER_CX`, `CHAMBER_CY`, `CHAMBER_RADIUS`, `CHAMBER_DIAMETER_CM`
- `BALL_RADIUS_PX_RANGE`, `BALL_HOUGH_PARAMS`,
  `BALL_ROI_RADIUS_PX`, `BALL_ROI_LOSS_LIMIT`

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
  edit the constants at the top of `track_object.py`.
- **`overlay_motion_frames` chamber circle appears flipped.** The function
  uses `origin="lower"`; pass `(cx, H - cy)` for circle/scatter overlays, or
  call `ax.invert_yaxis()` and use `cy` directly.
- **`n_workers>1` yields slightly different frame indices.** Worker processes
  seek with `cv2.CAP_PROP_POS_FRAMES`, which on non-all-intra AVIs may snap to
  the nearest preceding keyframe. Compare against `n_workers=1` once per new
  codec to confirm results agree.
