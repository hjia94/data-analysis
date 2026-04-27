import cv2
import matplotlib.pyplot as plt
import numpy as np
import struct
import time
import os
from pathlib import Path

##############################################################################
#                          CINE File Reading Functions                       #
##############################################################################

def read_L(f):
    try:
        return int(struct.unpack('<l', f.read(4))[0])
    except struct.error:
        raise ValueError("Failed to read 4-byte integer from file")

def read_Q(f):
    try:
        return struct.unpack('Q', f.read(8))[0]
    except struct.error:
        raise ValueError("Failed to read 8-byte integer from file")

def read_Q_array(f, n):
    try:
        a = np.zeros(n, dtype='Q')
        for i in range(n):
            a[i] = read_Q(f)
        return a
    except (struct.error, ValueError) as e:
        raise ValueError(f"Failed to read Q array: {str(e)}")

def read_B_2Darray(f, ypix, xpix):
    try:
        n = xpix * ypix
        a = np.array(struct.unpack(f'{n}B', f.read(n * 1)), dtype='B')
        return a.reshape(ypix, xpix)
    except struct.error:
        raise ValueError("Failed to read byte array from file")

def read_H_2Darray(f, ypix, xpix):
    try:
        n = xpix * ypix
        a = np.array(struct.unpack(f'{n}H', f.read(n * 2)), dtype='H')
        return a.reshape(ypix, xpix)
    except struct.error:
        raise ValueError("Failed to read short array from file")

def read_cine(ifn):
    if not os.path.exists(ifn):
        raise FileNotFoundError(f"CINE file not found: {ifn}")
        
    with open(ifn, 'rb') as cf:
        t_read = time.time()
        print("Reading .cine file...")

        try:
            cf.read(16)  # Skip header
            baseline_image = read_L(cf)
            image_count = read_L(cf)

            if image_count <= 0:
                raise ValueError(f"Invalid image count: {image_count}")

            pointers = np.zeros(3, dtype='L')
            pointers[0] = read_L(cf)
            pointers[1] = read_L(cf)
            pointers[2] = read_L(cf)

            cf.seek(58)
            nbit = read_L(cf)

            if nbit not in [8, 16]:
                raise ValueError(f"Unsupported bit depth: {nbit}")

            cf.seek(int(pointers[0]) + 4)
            xpix = read_L(cf)
            ypix = read_L(cf)

            if xpix <= 0 or ypix <= 0:
                raise ValueError(f"Invalid image dimensions: {xpix}x{ypix}")

            cf.seek(int(pointers[1]) + 768)
            pps = read_L(cf)
            exposure = read_L(cf)

            cf.seek(int(pointers[2]))
            pimage = read_Q_array(cf, image_count)

            dtype = 'B' if nbit == 8 else 'H'
            frame_arr = np.zeros((image_count, ypix, xpix), dtype=dtype)

            for i in range(image_count):
                p = struct.unpack('<l', struct.pack('<L', pimage[i] & 0xffffffffffffffff))[0]
                cf.seek(p)
                ofs = read_L(cf)
                cf.seek(p + ofs)
                frame_arr[i] = read_B_2Darray(cf, ypix, xpix) if nbit == 8 else read_H_2Darray(cf, ypix, xpix)

            time_arr = np.linspace(
                baseline_image / pps, 
                (baseline_image + image_count) / pps, 
                image_count, 
                endpoint=False
            )
            
            print("starting frame number:", baseline_image)
            print("total frames:", image_count)
            print("frame rate:", pps)
            print(f"Done reading .cine file ({time.time() - t_read:.1f} s)")
            dt = 1/pps
            return time_arr, frame_arr, dt

        except Exception as e:
            raise RuntimeError(f"Error reading CINE file: {str(e)}")

def read_cine_header(cine_path):
    """Read fps and t_start from the CINE header without loading frames.

    Returns (fps, t_start) where t_start is the time in seconds of the first
    frame, matching tarr[0] returned by read_cine.
    """
    with open(cine_path, "rb") as cf:
        cf.read(16)
        baseline_image = read_L(cf)
        read_L(cf)  # image_count
        pointers = [read_L(cf), read_L(cf), read_L(cf)]
        cf.seek(int(pointers[1]) + 768)
        fps = read_L(cf)
    if fps <= 0:
        raise ValueError(f"Invalid CINE frame rate: {fps}")
    return float(fps), float(baseline_image) / float(fps)


def overlay_motion_frames(
    frame_arr,
    center_frame,
    n_frames,
    mode="min",
    step=1,
    ax=None,
    cmap="gray",
    show_window=True,
):
    """
    Overlay a set of frames centered on `center_frame` into a single image
    showing the moving object's trail.

    Args:
        frame_arr: (N, H, W) array as returned by read_cine().
        center_frame: index into frame_arr to center the window on.
        n_frames: half-window size; window is [center-n, center+n] inclusive.
        mode: "min" stacks via per-pixel min (dark object on bright bg);
              "max" stacks via per-pixel max (bright object on dark bg).
        step: sample every `step`-th frame within the window (anchored on
              center_frame). step=1 is the default continuous overlay; step=5
              keeps the center frame and every 5th frame on either side,
              yielding discrete ball snapshots instead of a continuous trail.
        ax: matplotlib Axes to draw on. If None, a new figure is created.
        cmap: matplotlib colormap.
        show_window: if True, include frame range and count in the title.

    Returns:
        (ax, overlay) — the Axes and the (H, W) overlay array.
    """
    if frame_arr.ndim != 3:
        raise ValueError("frame_arr must be 3-D (N, H, W)")
    n_total = frame_arr.shape[0]
    if not 0 <= center_frame < n_total:
        raise ValueError(
            f"center_frame {center_frame} out of range [0, {n_total - 1}]"
        )
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if step < 1:
        raise ValueError("step must be >= 1")

    lo = max(0, center_frame - n_frames)
    hi = min(n_total - 1, center_frame + n_frames)
    if (lo, hi) != (center_frame - n_frames, center_frame + n_frames):
        print(f"Window clipped to frames [{lo}, {hi}]")

    # Build indices anchored on center_frame so the center is always included
    # regardless of step, with symmetric sampling outward.
    offsets = np.arange(-((center_frame - lo) // step), ((hi - center_frame) // step) + 1)
    indices = center_frame + offsets * step

    window = frame_arr[indices]
    if mode == "min":
        overlay = window.min(axis=0)
    elif mode == "max":
        overlay = window.max(axis=0)
    else:
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(overlay, cmap=cmap, origin="lower")

    if show_window:
        step_note = "" if step == 1 else f", every {step}th"
        ax.set_title(
            f"Frames {indices[0]}-{indices[-1]} "
            f"({len(indices)} frames{step_note}, centered on {center_frame})"
        )

    return ax, overlay


def convert_cine_to_avi(frame_arr, avi_path, scale_factor=8):
    """Convert CINE frame array to AVI video"""
    if not isinstance(frame_arr, np.ndarray):
        raise ValueError("frame_arr must be a numpy array")
    
    if len(frame_arr.shape) != 3:
        raise ValueError("frame_arr must be a 3D array (frames, height, width)")
    
    orig_height, orig_width = frame_arr.shape[1], frame_arr.shape[2]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(avi_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(avi_path, fourcc, 30, 
                            (orig_width*scale_factor, orig_height*scale_factor), False)
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {avi_path}")

        print(f"Converting to {avi_path}...")
        total_frames = len(frame_arr)
        
        for i, frame in enumerate(frame_arr):
            # Normalize frame to 0-255 range
            norm_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to uint8
            if norm_frame.dtype != np.uint8:
                norm_frame = norm_frame.astype(np.uint8)
            
            # Resize frame
            resized = cv2.resize(norm_frame, 
                               (orig_width*scale_factor, orig_height*scale_factor),
                               interpolation=cv2.INTER_LINEAR)
            
            # Flip vertically
            flipped = cv2.flip(resized, 0)
            
            # Write frame
            out.write(flipped)
            
            # Print progress
            if (i + 1) % 100 == 0 or (i + 1) == total_frames:
                print(f"Progress: {i + 1}/{total_frames} frames")
        
        out.release()
        print(f"Conversion complete. Saved to {avi_path}")
        
    except Exception as e:
        if 'out' in locals():
            out.release()
        raise RuntimeError(f"Error during video conversion: {str(e)}")
    

def batch_convert_cine_to_avi(base_path):
    """
    Convert all .cine files in the specified directory to .avi videos.
    Each .avi file will be saved in the same directory with the same base name.
    Skip conversion if .avi file already exists.
    """
    dir_path = Path(base_path)
    cine_files = list(dir_path.glob('*.cine'))
    if not cine_files:
        print(f"No .cine files found in {base_path}")
        return

    for cine_file in cine_files:
        avi_path = cine_file.with_suffix('.avi')
        if avi_path.exists():
            print(f"{avi_path.name} already exists.")
            continue
        try:
            print(f"Processing {cine_file}...")
            time_arr, frame_arr, dt = read_cine(str(cine_file))
            convert_cine_to_avi(frame_arr, str(avi_path))
        except Exception as e:
            print(f"Failed to convert {cine_file}: {str(e)}")

if __name__ == "__main__":
    # --- Overlay a motion trail from a single .cine file -------------------
    # Edit these three values and run this file directly.
    cine_path = r"F:\AUG2025\P23\your_file.cine"
    center_frame = 1549
    n_frames = 30   # half-window
    step = 10       # plot every Nth frame; 1 = continuous

    tarr, frarr, dt = read_cine(cine_path)
    fig, ax = plt.subplots(figsize=(8, 8))
    overlay_motion_frames(
        frarr,
        center_frame=center_frame,
        n_frames=n_frames,
        step=step,
        mode="min",
        ax=ax,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()