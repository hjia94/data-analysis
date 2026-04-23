import cv2
import logging
import multiprocessing as mp
import numpy as np
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator, List, Tuple
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

# ROI tracking parameters
BALL_ROI_RADIUS_PX = 60          # half-width of crop around last detection
BALL_ROI_LOSS_LIMIT = 3          # consecutive ROI misses before giving up tracking

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




#===============================================================================================================================================
def extract_calibration(cine_filename):
    """Return cm/px calibration for a cine file based on its port number tag.

    Perspective model:  cal = K * (P_CAM - port)
    Fitted to two measured points:
        P30 → 0.015 cm/px,  P24 → 0.031707 cm/px
    Solving the 2x2 system gives:
        K     = (0.031707 - 0.015) / (30 - 24) = 0.0027845 cm/px per port
        P_CAM = 30 + 0.015 / K                 ≈ 35.39
    The effective camera port (~35.4) is the optical nodal point, not the
    physical mounting port (~P60); the discrepancy reflects that "1 ft per
    port" is approximate and the lens focal offset matters.
    Valid for port < P_CAM (i.e. P1–P35, the far side of the machine).
    """
    import re
    m = re.search(r'[Pp](\d+)', os.path.basename(cine_filename))
    if m is None:
        raise ValueError(
            f"No port number (e.g. 'P23') found in filename: {cine_filename}"
        )
    port = int(m.group(1))

    K = (0.031707 - 0.015) / (30 - 24)      # cm/px per port
    P_CAM = 30 + 0.015 / K                   # effective camera port ≈ 35.39
    calibration = K * (P_CAM - port)

    if calibration <= 0:
        raise ValueError(
            f"Port {port} is at or beyond the effective camera position "
            f"(P_CAM ≈ {P_CAM:.1f}); calibration formula not valid here."
        )
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
def _hough_ball_brightest(
    gray: np.ndarray,
    min_dist: int,
    chamber_check: Optional[Tuple[int, int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """Run GaussianBlur + invert + HoughCircles on a grayscale region;
    return (x, y) of the brightest valid candidate (local to `gray`), or None.

    If `chamber_check` = (cx, cy, radius) is given, candidates outside
    that disk are also rejected before picking the brightest.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    inverted = 255 - blurred
    circles = cv2.HoughCircles(
        inverted,
        cv2.HOUGH_GRADIENT,
        minDist=min_dist,
        minRadius=BALL_RADIUS_PX_RANGE[0],
        maxRadius=BALL_RADIUS_PX_RANGE[1],
        **BALL_HOUGH_PARAMS,
    )
    if circles is None:
        return None

    circles = np.int32(np.around(circles[0]))
    h, w = gray.shape[:2]
    valid_mask = (
        (circles[:, 0] >= 0) & (circles[:, 0] < w)
        & (circles[:, 1] >= 0) & (circles[:, 1] < h)
    )
    if chamber_check is not None:
        ccx, ccy, cradius = chamber_check
        valid_mask &= np.hypot(circles[:, 0] - ccx, circles[:, 1] - ccy) < cradius

    valid = circles[valid_mask]
    if valid.size == 0:
        return None

    brightness = gray[valid[:, 1], valid[:, 0]]
    px, py = valid[int(np.argmax(brightness)), :2]
    return int(px), int(py)


def _detect_ball_in_frame(
    frame: np.ndarray,
    cx: int,
    cy: int,
    chamber_radius: int,
    last_pos: Optional[Tuple[int, int]] = None,
    roi_radius: int = BALL_ROI_RADIUS_PX,
) -> Optional[Tuple[int, int]]:
    """Return (px, py) of the brightest valid ball candidate, or None.

    If `last_pos` is provided, a fast cropped-ROI search runs first;
    on a hit we return immediately. On miss (or no last_pos), the
    full-chamber Hough pipeline runs.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    if last_pos is not None:
        px0, py0 = last_pos
        x0 = max(0, px0 - roi_radius)
        x1 = min(w, px0 + roi_radius)
        y0 = max(0, py0 - roi_radius)
        y1 = min(h, py0 + roi_radius)
        crop = gray[y0:y1, x0:x1]
        if crop.size > 0:
            local = _hough_ball_brightest(crop, min_dist=max(roi_radius // 2, 5))
            if local is not None:
                return local[0] + x0, local[1] + y0

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), chamber_radius, 255, -1)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    return _hough_ball_brightest(
        masked_gray,
        min_dist=chamber_radius // 4,
        chamber_check=(cx, cy, chamber_radius),
    )


def _track_frame_range(
    args: Tuple[str, int, int, int, int, int],
) -> Tuple[List[Tuple[int, int]], List[int], float, Optional[int]]:
    """Worker: process frames [start_frame, end_frame) of avi_path.

    Module-level so multiprocessing (spawn on Windows) can pickle it.
    Each worker opens its own cv2.VideoCapture (capture objects are not picklable).
    """
    cv2.setNumThreads(1)  # avoid oversubscription when N processes each spawn N threads
    avi_path, start_frame, end_frame, cx, cy, chamber_radius = args

    local_positions: List[Tuple[int, int]] = []
    local_frame_numbers: List[int] = []
    local_min_ydiff: float = float("inf")
    local_min_ydiff_frame: Optional[int] = None

    last_pos: Optional[Tuple[int, int]] = None
    miss_count = 0

    with _video_capture(avi_path) as cap:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for offset in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_idx = start_frame + offset

            detection = _detect_ball_in_frame(
                frame, cx, cy, chamber_radius, last_pos=last_pos,
            )

            if detection is None:
                if last_pos is not None:
                    miss_count += 1
                    if miss_count >= BALL_ROI_LOSS_LIMIT:
                        last_pos = None
                        miss_count = 0
                continue

            px, py = detection
            last_pos = (px, py)
            miss_count = 0

            rel_x = px - cx
            rel_y = cy - py
            local_positions.append((rel_x, rel_y))
            local_frame_numbers.append(frame_idx)

            ay = abs(rel_y)
            if ay < local_min_ydiff:
                local_min_ydiff = ay
                local_min_ydiff_frame = frame_idx

    return local_positions, local_frame_numbers, local_min_ydiff, local_min_ydiff_frame


def track_object(
    avi_path: str,
    cx: Optional[int] = None,
    cy: Optional[int] = None,
    chamber_radius: Optional[int] = None,
    chamber_cache_path: Optional[Path] = CHAMBER_CACHE_PATH,
    redetect: bool = False,
    n_workers: int = 1,
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
        n_workers: Process pool size for parallel frame-range decode + detect.
            Default 1 = single-process (bit-exact prior behavior).
            Values >1 split frames into contiguous ranges and dispatch via
            multiprocessing.Pool. Each worker opens its own cv2.VideoCapture
            and seeks to its start frame; cv2.CAP_PROP_POS_FRAMES on AVIs that
            are not all-intra may snap to the nearest preceding keyframe and
            yield slightly different frame indices than n_workers=1. Compare
            results with n_workers=1 once for any new codec to confirm.

    Returns:
        TrackingResult — iterable, so existing `pos, fn, mf = track_object(...)`
        callers still work.
    """
    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"Video file not found: {avi_path}")

    if n_workers > 1:
        # Resolve chamber once so workers don't each re-detect and so cache
        # writes only happen in one place.
        if cx is None or cy is None or chamber_radius is None:
            cx, cy, chamber_radius = detect_and_cache_chamber(
                avi_path,
                frame_idx=0,
                chamber_cache_path=chamber_cache_path,
                redetect=redetect,
            )

        with _video_capture(avi_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing %d frames across %d workers", total_frames, n_workers)

        chunk = total_frames // n_workers
        ranges = [
            (i * chunk, total_frames if i == n_workers - 1 else (i + 1) * chunk)
            for i in range(n_workers)
        ]
        worker_args = [
            (avi_path, s, e, int(cx), int(cy), int(chamber_radius))
            for (s, e) in ranges if s < e
        ]

        positions: List[Tuple[int, int]] = []
        frame_numbers: List[int] = []
        min_ydiff: float = float("inf")
        min_ydiff_frame: Optional[int] = None

        with mp.Pool(processes=n_workers) as pool:
            # imap (not imap_unordered) preserves submission order; since worker
            # args are sorted by start_frame, result lists stay globally ordered.
            for lp, lfn, lmin, lmin_frame in pool.imap(_track_frame_range, worker_args):
                positions.extend(lp)
                frame_numbers.extend(lfn)
                if lmin_frame is not None and (lmin, lmin_frame) < (
                    min_ydiff, min_ydiff_frame if min_ydiff_frame is not None else float("inf")
                ):
                    min_ydiff = lmin
                    min_ydiff_frame = lmin_frame

        logger.info("Frame closest to chamber center: %s", min_ydiff_frame)
        return TrackingResult(
            positions=np.asarray(positions, dtype=int) if positions else np.empty((0, 2), dtype=int),
            frame_numbers=np.asarray(frame_numbers, dtype=int),
            min_ydiff_frame=min_ydiff_frame,
        )

    chamber_known = cx is not None and cy is not None and chamber_radius is not None

    positions: List[Tuple[int, int]] = []
    frame_numbers: List[int] = []
    min_ydiff: float = float("inf")
    min_ydiff_frame: Optional[int] = None

    last_pos: Optional[Tuple[int, int]] = None
    miss_count = 0
    frame_idx = 0

    with _video_capture(avi_path) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing %d frames", total_frames)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

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

            detection = _detect_ball_in_frame(
                frame, cx, cy, chamber_radius, last_pos=last_pos,
            )

            if detection is None:
                if last_pos is not None:
                    miss_count += 1
                    if miss_count >= BALL_ROI_LOSS_LIMIT:
                        last_pos = None
                        miss_count = 0
                frame_idx += 1
                continue

            px, py = detection
            last_pos = (px, py)
            miss_count = 0

            rel_x = px - cx
            rel_y = cy - py
            positions.append((rel_x, rel_y))
            frame_numbers.append(frame_idx)

            ay = abs(rel_y)
            if ay < min_ydiff:
                min_ydiff = ay
                min_ydiff_frame = frame_idx
            frame_idx += 1

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


# ===============================================================================================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from read_cine import read_cine, convert_cine_to_avi, overlay_motion_frames

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cine_path = r"E:\fast_cam\He3kA_B380G800G_pl0t20_uw15t35\Y20241115_kapton_P30_-16deg_x36_y0@1780_085.cine"
    avi_path = cine_path.replace(".cine", ".avi")
    n_workers = 4
    n_frames = 300  # half-window for overlay
    step = 60 # how many frames to skip between overlayed motion frames (for visibility)

    tarr, frarr, dt = read_cine(cine_path)
    if not os.path.exists(avi_path):
        logger.info("AVI not found; converting %s -> %s", cine_path, avi_path)
        convert_cine_to_avi(frarr, avi_path)

    cx, cy, chamber_radius = detect_and_cache_chamber(avi_path)
    parr, frarr_idx, cf = track_object(avi_path, n_workers=n_workers)
    logger.info("track_object: %d positions, center-cross frame=%s", len(parr), cf)

    if cf is None:
        raise SystemExit("No center-crossing frame found; cannot build overlay.")

    fig, ax = plt.subplots(figsize=(15, 5))
    ax, _ = overlay_motion_frames(
        frarr, center_frame=cf, n_frames=n_frames, step=step, ax=ax,
    )
    # overlay_motion_frames uses origin="lower"; flip the chamber circle's y to match
    ax.add_patch(plt.Circle((cx, frarr.shape[1] - cy), chamber_radius,
                            fill=False, color="green", linewidth=2))
    ax.set_title(f"t={tarr[cf] * 1e3:.3f} ms  ±{n_frames} frames")
    ax.axis("off")
    plt.show()
