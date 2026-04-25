"""
Track a tungsten ball falling through a cylindrical plasma chamber.

Coordinate system
-----------------
All positions are stored as (rel_x, rel_y) relative to the chamber center
(CHAMBER_CX, CHAMBER_CY):
    rel_x = px - cx        (positive = rightward)
    rel_y = cy - py        (positive = upward; inverts the image y-axis)

Core tracking logic (track_object)
-----------------------------------
For each video frame the detector (Hough circles on an inverted, blurred
grayscale image) finds the brightest ball candidate inside the chamber disk.
A narrow ROI crop around the last known position is tried first for speed;
the full-chamber search is the fallback.

On a successful detection the relative position is appended and the tracker
updates min_ydiff_frame: the index of the frame whose |rel_y| is smallest,
i.e. where the ball is vertically closest to the chamber center.  Because the
ball falls straight through the chamber, this is the center-crossing frame
(rel_y ≈ 0, t0 in all downstream kinematics).

Calibration (extract_calibration)
-----------------------------------
The parabolic y(t) trajectory in pixel space is fit with np.polyfit.  The
leading coefficient gives g in px/s², from which cm/px is derived.  t0 is
set to tarr[min_ydiff_frame] so that the fit is expressed in physical time
relative to the center crossing.
"""
import cv2
import logging
import multiprocessing as mp
import numpy as np
from scipy import constants
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Iterator, List, Tuple
from scipy.constants import g

logger = logging.getLogger(__name__)


# Hardcoded chamber geometry. Verified visually against the bright chamber
# disk in the 2048x2048 fast-cam frames (see fastcam_test.ipynb): the
# Hough/contour detector's (1121, 1113, 609) result was overlaid on a real
# frame and confirmed to lie on the bright background. The camera mount is
# fixed across all runs in this analysis, so this is the single source of
# truth — no per-shot detection is needed.
CHAMBER_CX = 1121
CHAMBER_CY = 1113
CHAMBER_RADIUS = 609

# Physical diameter of the bright chamber disk visible in the fast-cam frames.
# 36 cm across (radius = 18 cm). Source of truth for the chamber-based cm/px
# cross-check used to validate the gravity-derived calibration.
CHAMBER_DIAMETER_CM = 36.0

# Ball detection parameters
BALL_RADIUS_PX_RANGE = (1, 5)
BALL_HOUGH_PARAMS = dict(dp=1, param1=50, param2=12)

# ROI tracking parameters
BALL_ROI_RADIUS_PX = 60          # half-width of crop around last detection
BALL_ROI_LOSS_LIMIT = 3          # consecutive ROI misses before giving up tracking


def get_chamber() -> Tuple[int, int, int]:
    """Return the hardcoded chamber geometry as (cx, cy, radius_px)."""
    return CHAMBER_CX, CHAMBER_CY, CHAMBER_RADIUS


def chamber_cm_per_px(chamber_radius_px: int = CHAMBER_RADIUS) -> float:
    """cm/px from the chamber disk: (CHAMBER_DIAMETER_CM / 2) / radius_px.

    Independent of the parabola fit — uses only the known physical size of
    the bright chamber circle (36 cm diameter = 18 cm radius). Lies on the
    chamber back-wall plane, which is farther from the camera than any port
    plane, so it is NOT expected to equal the gravity-derived per-port
    cm/px (see evaluate_freefall_accuracy.py --port-ratio).
    """
    return (CHAMBER_DIAMETER_CM / 2.0) / float(chamber_radius_px)


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
def ensure_avi(cine_path):
    """Convert cine to AVI next to it if the AVI does not yet exist."""
    avi_path = os.path.splitext(cine_path)[0] + ".avi"
    if os.path.exists(avi_path):
        return avi_path
    _tarr, frarr, _dt = read_cine(cine_path)
    convert_cine_to_avi(frarr, avi_path)
    return avi_path

def extract_calibration(cine_path):
    """Track a cine file and deduce cm/px from the parabolic y(t) trajectory.

    Returns:
        (cm_per_px_gravity, cm_per_px_chamber, x_cm) on success, or
        (None, None, None) on failure.
    """
    cine_name = os.path.basename(cine_path)

    try:
        tarr, _frarr, _dt = read_cine(cine_path)
    except Exception as e:
        print(f"[SKIP] {cine_name}: read_cine failed: {e}")
        return None, None, None

    try:
        avi_path = ensure_avi(cine_path)
    except Exception as e:
        print(f"[SKIP] {cine_name}: avi conversion failed: {e}")
        return None, None, None

    result = track_object(avi_path)
    positions = np.asarray(result.positions)
    frame_numbers = np.asarray(result.frame_numbers)
    min_ydiff_frame = result.min_ydiff_frame

    if positions.size == 0 or min_ydiff_frame is None:
        print(f"[SKIP] {cine_name}: tracker returned no usable frames")
        return None, None, None

    x_px = positions[:, 0].astype(float)
    y_px = positions[:, 1].astype(float)
    t = tarr[frame_numbers]
    t0 = float(tarr[min_ydiff_frame])
    tau = t - t0

    if tau.size < 5:
        print(f"[SKIP] {cine_name}: too few tracked frames ({tau.size})")
        return None, None, None

    # Fit parabola in pixel space: y_px = a + b*tau + c*tau^2
    c_px, _b_px, _a_px = np.polyfit(tau, y_px, 2)

    # Deduce cm/px from gravity. Ball falls => c_px < 0 (y upward-positive).
    #   y_m(tau)  = -0.5 * g * tau^2 + v0 * tau         (with y(t0)=0)
    #   y_px(tau) = y_m / (cm/px * 0.01)
    #   => coefficient of tau^2 in px: c_px = -0.5 * g * 100 / (cm/px)
    #   => cm/px = -0.5 * (g*100) / c_px
    G_CM_PER_S2 = constants.g * 100.0                       # 981.0 cm/s^2
    if c_px >= 0:
        print(f"[WARN] {cine_name}: non-falling quadratic (c_px={c_px:.2f}); cm/px unreliable")
        return float("nan"), float("nan"), float("nan")

    cm_per_px_gravity = float(-0.5 * G_CM_PER_S2 / c_px)
    cm_per_px_chamber = chamber_cm_per_px()

    x_cm = x_px * cm_per_px_gravity
    return cm_per_px_gravity, cm_per_px_chamber, x_cm


def average_calibration(dir_path, n=5, pattern="*.cine", out_dir=r"E:\\"):
    """Run extract_calibration on up to `n` cine files in `dir_path`,
    print averaged calibration / horizontal drift statistics to the terminal,
    and save the results as an .npy file to `out_dir` keyed by port number.

    The port tag (e.g. "P21", "P30") is parsed from the filenames and must
    be consistent across the sampled files. Drift is summarized per shot
    as the peak-to-peak horizontal excursion (np.ptp(x_cm)).
    """
    import glob
    import re

    cine_files = sorted(glob.glob(os.path.join(dir_path, pattern)))[:n]
    if not cine_files:
        print(f"[average_calibration] no files matching {pattern} in {dir_path}")
        return

    cm_vals = []
    cm_chamber_vals = []
    drift_vals = []
    used = []
    ports = set()
    for path in cine_files:
        cm_per_px, cm_per_px_chamber, x_cm = extract_calibration(path)
        if cm_per_px is None or np.isnan(cm_per_px):
            continue
        cm_vals.append(cm_per_px)
        if cm_per_px_chamber is not None and not np.isnan(cm_per_px_chamber):
            cm_chamber_vals.append(cm_per_px_chamber)
        drift_vals.append(float(np.ptp(np.asarray(x_cm))))
        name = os.path.basename(path)
        used.append(name)
        m = re.search(r"P\d+", name)
        if m:
            ports.add(m.group(0))

    if not cm_vals:
        print("[average_calibration] no usable calibrations")
        return

    cm_arr = np.asarray(cm_vals)
    drift_arr = np.asarray(drift_vals)
    summary = {
        "cm_per_px_mean": float(cm_arr.mean()),
        "cm_per_px_std": float(cm_arr.std(ddof=0)),
        "drift_cm_mean": float(drift_arr.mean()),
        "drift_cm_std": float(drift_arr.std(ddof=0)),
        "n_used": len(cm_vals),
        "n_requested": len(cine_files),
        "files": used,
    }
    if cm_chamber_vals:
        ch_arr = np.asarray(cm_chamber_vals)
        summary["cm_per_px_chamber_mean"] = float(ch_arr.mean())
        summary["cm_per_px_chamber_std"] = float(ch_arr.std(ddof=0))

    if len(ports) == 1:
        port = ports.pop()
    elif len(ports) == 0:
        port = "Punknown"
        print(f"[average_calibration] WARN: no port tag (P\\d+) in filenames; using {port}")
    else:
        port = "_".join(sorted(ports))
        print(f"[average_calibration] WARN: mixed port tags {sorted(ports)}; using {port}")

    print()
    print(f"=== calibration summary ({port}) ===")
    print(f"  source dir     : {dir_path}")
    print(f"  files used     : {summary['n_used']}/{summary['n_requested']}")
    print(f"  cm/px (gravity): {summary['cm_per_px_mean']:.5f} ± {summary['cm_per_px_std']:.5f}")
    if "cm_per_px_chamber_mean" in summary:
        print(f"  cm/px (chamber): {summary['cm_per_px_chamber_mean']:.5f} ± {summary['cm_per_px_chamber_std']:.5f}")
    print(f"  drift (cm, pp) : {summary['drift_cm_mean']:.3f} ± {summary['drift_cm_std']:.3f}")
    for f in used:
        print(f"    - {f}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"calibration_factor_{port}.npy")
    np.save(out_path, summary)
    print(f"  saved to       : {out_path}")


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


def track_object(avi_path: str, cx: Optional[int] = None, cy: Optional[int] = None, chamber_radius: Optional[int] = None, n_workers: int = 1,) -> TrackingResult:
    """
    Track tungsten ball through entire video sequence.

    Chamber is the hardcoded value from get_chamber() unless cx/cy/chamber_radius
    are passed explicitly.

    Args:
        avi_path: Path to input AVI file.
        cx, cy, chamber_radius: Optional override of the hardcoded chamber.
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

    if cx is None or cy is None or chamber_radius is None:
        cx, cy, chamber_radius = get_chamber()

    if n_workers > 1:
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
def get_ball_position_at_time(
    t_ms: float,
    result: TrackingResult,
    tarr: np.ndarray,
    cm_per_px: float,
) -> Optional[Tuple[float, float]]:
    """Return (x_cm, y_cm) at time t_ms by extending linear fits of the tracked trajectory.

    Fits straight lines to the x(t) and y(t) data observed in the cine frames
    (converted to cm via cm_per_px), then evaluates those lines at any arbitrary
    time — including before or after the recording window.

    Args:
        t_ms: Query time in milliseconds (same reference as tarr * 1000).
        result: TrackingResult from track_object.
        tarr: Time array in seconds from read_cine.
        cm_per_px: Calibration factor in cm/px — use cm_per_px_mean from the
            .npy file saved by average_calibration.

    Returns:
        (x_cm, y_cm) relative to chamber center, or None if result has no data.
    """
    if result.frame_numbers.size == 0:
        return None

    t_tracked = tarr[result.frame_numbers]
    t_s = t_ms / 1000.0

    positions = np.asarray(result.positions, dtype=float)
    x_cm_tracked = positions[:, 0] * cm_per_px
    y_cm_tracked = positions[:, 1] * cm_per_px

    x_slope, x_intercept = np.polyfit(t_tracked, x_cm_tracked, 1)
    y_slope, y_intercept = np.polyfit(t_tracked, y_cm_tracked, 1)

    return float(x_intercept + x_slope * t_s), float(y_intercept + y_slope * t_s)


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

    # logging.basicConfig(level=logging.INFO,
    #                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cine_path = r"E:\fast_cam\He3kA_B380G800G_pl0t20_uw15t35\Y20241115_kapton_P30_-16deg_x36_y0@1780_085.cine"
    avi_path = cine_path.replace(".cine", ".avi")
    n_workers = 4
    n_frames = 300  # half-window for overlay
    step = 60       # how many frames to skip between overlayed motion frames (for visibility)

    tarr, frarr, dt = read_cine(cine_path)
    if not os.path.exists(avi_path):
        logger.info("AVI not found; converting %s -> %s", cine_path, avi_path)
        convert_cine_to_avi(frarr, avi_path)

    cx, cy, chamber_radius = get_chamber()
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
