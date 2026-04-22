# object_tracking

Tools for reading high-speed camera (.cine / .avi) recordings and tracking
the trajectory of a tungsten ball falling through the chamber.

## Files

| File | Purpose |
|------|---------|
| `read_cine.py`        | Parse `.cine` binary files, convert to `.avi`, and visualize motion. |
| `track_object.py`     | Detect chamber geometry and track the ball through a video. |
| `chamber_cache.npy`   | Persistent cache of detected chamber `(cx, cy, radius)` keyed by frame shape `(H, W)`. Auto-managed. |
| `fastcam_test.ipynb`  | Working notebook with end-to-end examples. |

## Typical workflow

```python
from read_cine import read_cine, overlay_motion_frames
from track_object import detect_and_cache_chamber, track_object

cine_path = "path/to/movie.cine"
avi_path  = cine_path.replace(".cine", ".avi")

# 1. Load raw frames (also produces a usable .avi via convert_cine_to_avi)
tarr, frarr, dt = read_cine(cine_path)

# 2. Detect + cache chamber geometry once per camera setup (seconds, not minutes)
cx, cy, radius = detect_and_cache_chamber(avi_path)

# 3. Track the ball across the whole video (uses the cached chamber)
result = track_object(avi_path)
positions, frame_numbers, min_ydiff_frame = result   # tuple-unpackable

# 4. Visualize the trail around any frame
overlay_motion_frames(frarr, center_frame=min_ydiff_frame, n_frames=30, mode="min")
```

## `read_cine.py`

| Function | Description |
|----------|-------------|
| `read_cine(ifn)` | Parse a Phantom `.cine` file. Returns `(time_arr, frame_arr, dt)` where `frame_arr` is `(N, H, W)` uint8/uint16. |
| `convert_cine_to_avi(frame_arr, avi_path, scale_factor=8)` | Write an AVI (MJPG, vertically flipped, upscaled) for downstream OpenCV use. |
| `batch_convert_cine_to_avi(base_path)` | Convert every `.cine` in a directory; skips files that already have an `.avi`. |
| `overlay_motion_frames(frame_arr, center_frame, n_frames, mode="min", ax=None, ...)` | Stack `2*n_frames+1` frames into one image. `mode="min"` for dark objects on bright background, `"max"` for the inverse. Returns `(ax, overlay)`. |

## `track_object.py`

### Chamber cache

The chamber is essentially fixed for a given camera mount, so it is detected
once and cached on disk. All cache helpers use `CHAMBER_CACHE_PATH`
(`object_tracking/chamber_cache.npy`) by default.

| Function | Description |
|----------|-------------|
| `detect_and_cache_chamber(video_path, redetect=False)` | Read one frame, run `detect_chamber`, write `(cx, cy, radius)` to cache. Cheap. |
| `load_chamber_cache(path=...)` / `save_chamber_cache(cache, path=...)` | Low-level dict access. |
| `set_chamber(shape, cx, cy, radius)` | Manually seed/override an entry (useful when auto-detection fails). |
| `clear_chamber_cache()` / `show_chamber_cache()` | Cache maintenance. |

### Detection / tracking

| Function | Description |
|----------|-------------|
| `detect_chamber(frame, debug=False)` | Locate the bright chamber circle in a single BGR frame using OTSU + Hough; falls back to largest-contour. Returns `((cx, cy), radius)`. |
| `track_object(avi_path, cx=None, cy=None, chamber_radius=None, redetect=False)` | Track the ball through the whole video. Chamber is resolved by: explicit args → cache hit → fresh detection (then cached). Returns a `TrackingResult` dataclass. |
| `TrackingResult` | Fields: `positions` `(N, 2)`, `frame_numbers` `(N,)`, `min_ydiff_frame`. Iterable, so `pos, fn, mf = track_object(...)` still works. |

### Tracking results dictionary (separate from chamber cache)

A second `np.save`'d dict keyed by full cine path, holding the user-confirmed
center-crossing frame and time. Useful for downstream analysis that needs
`t = 0` aligned to the chamber crossing.

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
| `extract_calibration(cine_filename)` | Look up cm/pixel calibration from filename tags (`P30`, `P24`). |

## Tunable parameters

Detection thresholds live as module-level constants at the top of
`track_object.py` — edit there rather than in function bodies:

- `CHAMBER_RADIUS_PX_RANGE`, `CHAMBER_HOUGH_PARAMS`,
  `CHAMBER_BRIGHTNESS_MIN`, `CHAMBER_BRIGHT_PIXEL_RATIO_MIN`
- `BALL_RADIUS_PX_RANGE`, `BALL_HOUGH_PARAMS`

## Logging

Both modules use the standard `logging` library. To see info/warning output
in a notebook or script:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Common pitfalls

- **`detect_chamber` returned a bad chamber.** Override manually with
  `set_chamber(shape, cx, cy, radius)`.
- **`KeyError` from cache lookup.** No entry yet for this frame shape — run
  `detect_and_cache_chamber(avi_path)` once.
- **`overlay_motion_frames` chamber circle appears flipped.** The function
  uses `origin="lower"`; pass `(cx, H - cy)` for circle/scatter overlays, or
  call `ax.invert_yaxis()` and use `cy` directly.
