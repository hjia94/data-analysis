import cv2
import logging
import numpy as np
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, Tuple
from scipy.constants import g

logger = logging.getLogger(__name__)

# Chamber detection parameters
CHAMBER_RADIUS_PX_RANGE = (300, 600)
CHAMBER_HOUGH_PARAMS = dict(dp=1.2, param1=150, param2=25)
CHAMBER_BRIGHTNESS_MIN = 200
CHAMBER_BRIGHT_PIXEL_RATIO_MIN = 0.85

# Ball detection parameters
BALL_RADIUS_PX_RANGE = (1, 5)
BALL_HOUGH_PARAMS = dict(dp=1, param1=50, param2=12)

# Persistent chamber cache (camera mount is static across runs).
# Keyed by frame shape (height, width); value: {"cx", "cy", "radius"}.
CHAMBER_CACHE_PATH = Path(__file__).parent / "chamber_cache.npy"


def load_chamber_cache(path: Path = CHAMBER_CACHE_PATH) -> dict:
    """Load the chamber cache dict, or return empty dict if missing."""
    if not os.path.exists(path):
        return {}
    return np.load(path, allow_pickle=True).item()


def save_chamber_cache(cache: dict, path: Path = CHAMBER_CACHE_PATH) -> None:
    np.save(path, cache)


def set_chamber(
    shape: Tuple[int, int],
    cx: int,
    cy: int,
    radius: int,
    path: Path = CHAMBER_CACHE_PATH,
) -> None:
    """Manually set/override the cached chamber for a given frame shape."""
    cache = load_chamber_cache(path)
    cache[tuple(shape)] = {"cx": int(cx), "cy": int(cy), "radius": int(radius)}
    save_chamber_cache(cache, path)
    logger.info("Chamber cache updated for shape %s: cx=%d cy=%d r=%d",
                tuple(shape), cx, cy, radius)


def clear_chamber_cache(path: Path = CHAMBER_CACHE_PATH) -> None:
    if os.path.exists(path):
        os.remove(path)
        logger.info("Chamber cache cleared at %s", path)


def show_chamber_cache(path: Path = CHAMBER_CACHE_PATH) -> None:
    cache = load_chamber_cache(path)
    if not cache:
        print(f"No chamber cache at {path}")
        return
    print(f"Chamber cache ({len(cache)} entries) at {path}:")
    for shape, v in cache.items():
        print(f"  shape={shape}: cx={v['cx']} cy={v['cy']} radius={v['radius']}")


def detect_and_cache_chamber(
    video_path: str,
    frame_idx: int = 0,
    chamber_cache_path: Optional[Path] = CHAMBER_CACHE_PATH,
    redetect: bool = False,
    debug: bool = False,
) -> Tuple[int, int, int]:
    """
    Open `video_path`, read one frame, run detect_chamber, and write the
    result to the chamber cache. Cheap (~one frame read + one detection)
    compared to track_object which processes the entire video.

    Args:
        video_path: Path to AVI/video file.
        frame_idx: Which frame to use for detection (default 0).
        chamber_cache_path: Cache file path. Pass None to skip writing.
        redetect: If True, ignore any existing cache entry and re-detect.
        debug: Forwarded to detect_chamber.

    Returns:
        (cx, cy, radius)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    with _video_capture(video_path) as cap:
        if frame_idx != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    shape_key = (frame.shape[0], frame.shape[1])
    cache = (
        load_chamber_cache(chamber_cache_path)
        if chamber_cache_path is not None else {}
    )

    if not redetect and shape_key in cache:
        v = cache[shape_key]
        logger.info("Chamber already cached for shape %s; use redetect=True to override",
                    shape_key)
        return v["cx"], v["cy"], v["radius"]

    (cx, cy), radius = detect_chamber(frame, debug=debug)
    if chamber_cache_path is not None:
        cache[shape_key] = {"cx": int(cx), "cy": int(cy), "radius": int(radius)}
        save_chamber_cache(cache, chamber_cache_path)
        logger.info("Cached chamber for shape %s: cx=%d cy=%d r=%d",
                    shape_key, cx, cy, radius)
    return int(cx), int(cy), int(radius)


@dataclass
class TrackingResult:
    """Result of track_object. Iterable for backward-compatible tuple unpacking."""
    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=int))
    frame_numbers: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=int))
    min_ydiff_frame: Optional[int] = None

    def __iter__(self):
        yield self.positions
        yield self.frame_numbers
        yield self.min_ydiff_frame


@contextmanager
def _video_capture(path: str) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        yield cap
    finally:
        cap.release()


def _iter_frames(cap: cv2.VideoCapture) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_index, frame) until the capture is exhausted."""
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            return
        yield idx, frame
        idx += 1


#===============================================================================================================================================
def extract_calibration(cine_filename):
    """Extract calibration factor from filename"""
    if "P30" in cine_filename:
        calibration = 1.5e-2
    elif "P24" in cine_filename:
        calibration = 0.031707
    else:
        raise ValueError(f"Unknown calibration for {cine_filename}")
    return calibration

#===============================================================================================================================================
def detect_chamber(
    frame: np.ndarray,
    debug: bool = False,
    min_radius_px: int = CHAMBER_RADIUS_PX_RANGE[0],
    max_radius_px: int = CHAMBER_RADIUS_PX_RANGE[1],
) -> Tuple[Tuple[int, int], int]:
    """
    Detects the bright chamber circle using optimized thresholding and validation.

    Args:
        frame: Input frame (BGR format).
        debug: If True, log detection details.
        min_radius_px: Minimum candidate chamber radius in pixels.
        max_radius_px: Maximum candidate chamber radius in pixels.

    Returns:
        (origin, radius) where origin is (x, y) pixel coordinates and radius is in pixels.
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy array")
    if frame.ndim != 3:
        raise ValueError("frame must be a 3D array (height, width, channels)")

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optimized preprocessing for bright circles.
    # Note: with THRESH_OTSU the manual threshold value is ignored by OpenCV,
    # so we pass 0 to make that explicit.
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to enhance circular shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect circles with optimized parameters
    circles = cv2.HoughCircles(
        closed,
        cv2.HOUGH_GRADIENT,
        minDist=frame.shape[1] // 2,  # Assume only one main chamber
        minRadius=min_radius_px,
        maxRadius=max_radius_px,
        **CHAMBER_HOUGH_PARAMS,
    )

    # Validate and select best candidate
    best_circle = None
    if circles is not None:
        circles = np.int32(np.around(circles))[0]

        for x, y, r in circles:
            if x - r < 0 or y - r < 0 or x + r > frame.shape[1] or y + r > frame.shape[0]:
                continue  # Skip edge-touching circles

            # Brightness verification within the candidate disk
            mask = np.zeros_like(gray)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]

            # Real validity check: fraction of bright (thresholded) pixels inside
            # the candidate disk. The disk is synthetic so cv2.findContours-based
            # circularity would always be ~1.0 — useless as a check.
            disk_pixels = int(np.count_nonzero(mask))
            if disk_pixels == 0:
                continue
            bright_inside = int(np.count_nonzero(cv2.bitwise_and(closed, mask)))
            bright_ratio = bright_inside / disk_pixels

            if (
                bright_ratio > CHAMBER_BRIGHT_PIXEL_RATIO_MIN
                and mean_brightness > CHAMBER_BRIGHTNESS_MIN
                and (best_circle is None or r > best_circle[2])
            ):
                best_circle = (int(x), int(y), int(r))

    # Fallback to contour detection if Hough fails
    if best_circle is None:
        logger.warning("Hough failed, using contour fallback")
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest_contour)
            best_circle = (int(x), int(y), int(r))

    if best_circle:
        x, y, r = best_circle
        origin, radius = (x, y), r
    else:
        logger.warning("No valid circle found, using frame center")
        origin = (frame.shape[1] // 2, frame.shape[0] // 2)
        radius = (min_radius_px + max_radius_px) // 2

    if debug:
        logger.info("Chamber detected at %s with radius %dpx", origin, radius)
    return origin, radius


#===============================================================================================================================================
def _detect_ball_in_frame(
    frame: np.ndarray,
    cx: int,
    cy: int,
    chamber_radius: int,
) -> Optional[Tuple[int, int]]:
    """Return (px, py) of the brightest valid ball candidate, or None."""
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), chamber_radius, 255, -1)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = 255 - blurred

    circles = cv2.HoughCircles(
        inverted,
        cv2.HOUGH_GRADIENT,
        minDist=chamber_radius // 4,
        minRadius=BALL_RADIUS_PX_RANGE[0],
        maxRadius=BALL_RADIUS_PX_RANGE[1],
        **BALL_HOUGH_PARAMS,
    )
    if circles is None:
        return None

    circles = np.int32(np.around(circles[0]))

    # Vectorized chamber-containment + image-bounds filter
    inside_chamber = np.hypot(circles[:, 0] - cx, circles[:, 1] - cy) < chamber_radius
    in_bounds = (
        (circles[:, 0] >= 0) & (circles[:, 0] < w)
        & (circles[:, 1] >= 0) & (circles[:, 1] < h)
    )
    valid = circles[inside_chamber & in_bounds]
    if valid.size == 0:
        return None

    # Select brightest candidate (now safe — bounds checked)
    brightness = gray[valid[:, 1], valid[:, 0]]
    px, py = valid[int(np.argmax(brightness)), :2]
    return int(px), int(py)


def track_object(
    avi_path: str,
    cx: Optional[int] = None,
    cy: Optional[int] = None,
    chamber_radius: Optional[int] = None,
    chamber_cache_path: Optional[Path] = CHAMBER_CACHE_PATH,
    redetect: bool = False,
) -> TrackingResult:
    """
    Track tungsten ball through entire video sequence.

    Chamber resolution priority:
        1. Explicit cx/cy/chamber_radius args (highest).
        2. Persistent cache at `chamber_cache_path`, keyed by frame shape.
        3. detect_chamber() on the first readable frame (slow); result is
           written to the cache for reuse.

    Args:
        avi_path: Path to input AVI file.
        cx, cy, chamber_radius: Optional pre-computed chamber geometry.
        chamber_cache_path: Path to persistent cache file. Pass None to disable.
        redetect: If True, ignore the cache and force re-detection (and update
            the cache with the new value).

    Returns:
        TrackingResult — iterable, so existing `pos, fn, mf = track_object(...)`
        callers still work.
    """
    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"Video file not found: {avi_path}")

    chamber_known = cx is not None and cy is not None and chamber_radius is not None

    positions: list[Tuple[int, int]] = []
    frame_numbers: list[int] = []
    min_ydiff: float = float("inf")
    min_ydiff_frame: Optional[int] = None

    try:
        with _video_capture(avi_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info("Processing %d frames", total_frames)

            for frame_idx, frame in _iter_frames(cap):
                if not chamber_known:
                    shape_key = (frame.shape[0], frame.shape[1])
                    cache = (
                        load_chamber_cache(chamber_cache_path)
                        if chamber_cache_path is not None else {}
                    )
                    if not redetect and shape_key in cache:
                        v = cache[shape_key]
                        cx, cy, chamber_radius = v["cx"], v["cy"], v["radius"]
                        logger.info("Using cached chamber for shape %s: "
                                    "cx=%d cy=%d r=%d", shape_key, cx, cy, chamber_radius)
                    else:
                        (cx, cy), chamber_radius = detect_chamber(frame)
                        if chamber_cache_path is not None:
                            cache[shape_key] = {
                                "cx": int(cx), "cy": int(cy),
                                "radius": int(chamber_radius),
                            }
                            save_chamber_cache(cache, chamber_cache_path)
                            logger.info("Chamber detected and cached for shape %s",
                                        shape_key)
                    chamber_known = True

                detection = _detect_ball_in_frame(frame, cx, cy, chamber_radius)
                if detection is None:
                    continue

                px, py = detection
                rel_x = px - cx
                rel_y = cy - py
                positions.append((rel_x, rel_y))
                frame_numbers.append(frame_idx)

                ay = abs(rel_y)
                if ay < min_ydiff:
                    min_ydiff = ay
                    min_ydiff_frame = frame_idx

    except (FileNotFoundError, ValueError):
        raise
    except Exception as e:
        raise RuntimeError(f"Error during tracking: {e}") from e

    logger.info("Frame closest to chamber center: %s", min_ydiff_frame)
    return TrackingResult(
        positions=np.array(positions, dtype=int) if positions else np.empty((0, 2), dtype=int),
        frame_numbers=np.array(frame_numbers, dtype=int),
        min_ydiff_frame=min_ydiff_frame,
    )

#===============================================================================================================================================
def get_vel_freefall(h=1):
    '''Return magnitude of velocity (m/s) after free falling from rest through height h (m).

    Uses v = sqrt(2 * g * h). Direction (sign) is handled by caller.
    '''
    return np.sqrt(2 * g * h)

def get_pos_freefall(t, t0, height=0.5, chamber_radius=None, enforce_bounds=False):
    '''Return vertical position relative to chamber center with physically correct kinematics.

    Coordinate system:
        - Upward is positive.
        - y = 0 when the ball passes the chamber center at time t0.
        - y > 0 above center; y < 0 below center.

    Parameter height:
        Distance (m) the ball has ALREADY fallen before reaching the center.
        Therefore the release-from-rest position was at y = +height at time
        t_release = t0 - fall_time, where fall_time = sqrt(2*height/g).

    Derivation:
        Released from rest -> y_rel(τ) = height - 0.5*g*τ^2, τ in [0, fall_time].
        At τ = fall_time: y = 0 and v = -g*fall_time = -sqrt(2*g*height).
        Let dt = t - t0 (center-crossing frame). For motion phase (t >= t_release):
            y(dt) = -0.5 * g * dt * (dt + 2*fall_time).
        This gives y(t0)=0 and dy/dt(t0) = -g*fall_time.

    Piecewise model implemented:
        if t < t_release: y = height (still at rest prior to drop)
        else:            y = -0.5 * g * dt * (dt + 2*fall_time)

    Optional bounds:
        If chamber_radius is provided and enforce_bounds is True, y is clipped to
        [-chamber_radius, chamber_radius]. No bounce dynamics are modeled.

    Args:
        t (float | array-like): Evaluation time(s) [s].
        t0 (float): Time of center crossing [s].
        height (float): Distance already fallen before center [m] (>=0).
        chamber_radius (float | None): Physical half-height of chamber [m] for validation.
        enforce_bounds (bool): Clip output to physical bounds if chamber_radius given.

    Returns:
        float or np.ndarray: Position(s) y relative to center.
    '''
    # Basic validation (minimal per project guideline)
    if height < 0:
        raise ValueError("height must be non-negative")
    if chamber_radius is not None and height > chamber_radius:
        raise ValueError("height exceeds chamber_radius; inconsistent initial condition")

    t_arr = np.asarray(t, dtype=float)
    if height == 0:
        # Degenerate case: ball at center at t0 with zero prior fall.
        y = np.zeros_like(t_arr)
        if np.isscalar(t):
            return float(y)
        return y

    v0 = get_vel_freefall(height)  # sqrt(2*g*height)
    fall_time = v0 / g             # sqrt(2*height/g)
    dt = t_arr - t0
    # Motion-phase formula
    y_motion = -0.5 * g * dt * (dt + 2 * fall_time)
    # Prior to release: constant height
    y = np.where(dt < -fall_time, height, y_motion)

    if chamber_radius is not None and enforce_bounds:
        y = np.clip(y, -chamber_radius, chamber_radius)

    if np.isscalar(t):
        return float(y)
    return y

def get_vel_freefall_time(t, t0, height=0.5):
    '''Velocity (m/s, upward positive) as a function of time for the same model.

    For t < t_release: v = 0
    For t >= t_release: v = -g * (dt + fall_time)
    where dt = t - t0 and fall_time = sqrt(2*height/g).
    '''
    t_arr = np.asarray(t, dtype=float)
    v0 = get_vel_freefall(height)
    fall_time = v0 / g
    dt = t_arr - t0
    v = -g * (dt + fall_time)
    v = np.where(dt < -fall_time, 0.0, v)
    if np.isscalar(t):
        return float(v)
    return v

#===============================================================================================================================================
def update_tracking_result(tr_ifn, filepath, cf_new, ct_new):
    """
    Update tracking results for a specific file
    Args:
        tr_ifn: tracking results file path
        filepath: full path to the cine file
        cf_new: new frame number where ball is at chamber center
        ct_new: new time value corresponding to cf_new
    """
    if os.path.exists(tr_ifn):
        # Load existing dictionary
        tracking_dict = np.load(tr_ifn, allow_pickle=True).item()
        print(f"Loaded existing tracking results with {len(tracking_dict)} entries")

        # Show current value if it exists
        if filepath in tracking_dict:
            cf_old, ct_old = tracking_dict[filepath]
            print(f"\nCurrent values for {os.path.basename(filepath)}:")
            print(f"  Frame: {cf_old}")
            print(f"  Time: {ct_old:.6f}s")
        else:
            print(f"\nNo existing entry for {os.path.basename(filepath)}")

        # Update with new values
        tracking_dict[filepath] = (cf_new, ct_new)
        np.save(tr_ifn, tracking_dict)
        print(f"\nUpdated values:")
        print(f"  Frame: {cf_new}")
        print(f"  Time: {ct_new:.6f}s")
    else:
        print(f"No tracking results file found at {tr_ifn}")

# Example usage:
# filepath = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\Y20241102_P30_z13_x200_y0@-40_011.cine"
# update_tracking_result(filepath, cf_new=100, ct_new=0.001)

# Function to display current tracking results
def show_tracking_results(tr_ifn):
    if os.path.exists(tr_ifn):
        tracking_dict = np.load(tr_ifn, allow_pickle=True).item()
        print(f"Found {len(tracking_dict)} entries in tracking results\n")

        for filepath, (cf, ct) in tracking_dict.items():
            print(filepath)
            if cf is None or ct is None:
                print("  Frame: None")
                print("  Time: None")
            else:
                print(f"  Frame: {cf}")
                print(f"  Time: {ct:.6f}s")
            print()
    else:
        print(f"No tracking results file found at {tr_ifn}")

def delete_tracking_entry(tr_ifn, filepath):
    if os.path.exists(tr_ifn):
        tracking_dict = np.load(tr_ifn, allow_pickle=True).item()
        if filepath in tracking_dict:
            del tracking_dict[filepath]
            np.save(tr_ifn, tracking_dict)
            print(f"Deleted entry for {os.path.basename(filepath)}")
        else:
            print(f"No entry found for {os.path.basename(filepath)}")
    else:
        print(f"No tracking results file found at {tr_ifn}")
