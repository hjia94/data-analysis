import cv2
import numpy as np
import os
from pathlib import Path
from scipy.constants import g

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
def detect_chamber(frame, debug=False):
    """
    Detects the bright chamber circle using optimized thresholding and validation.
    
    Args:
        frame (np.ndarray): Input frame (BGR format)
        calibration (float): Calibration factor in cm/pixel
        
    Returns:
        tuple: (origin, radius) where origin is (x,y) coordinates and radius is in pixels
    """
    if not isinstance(frame, np.ndarray):
        raise ValueError("frame must be a numpy array")
    
    if frame.ndim != 3:
        raise ValueError("frame must be a 3D array (height, width, channels)")

    # radius constraints
    min_radius_px = 300
    max_radius_px = 600

    # Convert to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optimized preprocessing for bright circles
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to enhance circular shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Detect circles with optimized parameters
    circles = cv2.HoughCircles(
        closed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=frame.shape[1]//2,  # Assume only one main chamber
        param1=150,  # Lower Canny threshold
        param2=25,   # Accumulator threshold (lower for better detection)
        minRadius=min_radius_px,
        maxRadius=max_radius_px
    )

    # Validate and select best candidate
    best_circle = None
    if circles is not None:
        circles = np.int32(np.around(circles))[0]
        
        # Score circles by brightness and circularity
        for circle in circles:
            x, y, r = circle
            if x-r < 0 or y-r < 0 or x+r > frame.shape[1] or y+r > frame.shape[0]:
                continue  # Skip edge-touching circles
            
            # Create mask for brightness verification
            mask = np.zeros_like(gray)
            cv2.circle(mask, (x, y), r, 255, -1)
            mean_brightness = cv2.mean(gray, mask=mask)[0]
            
            # Circularity check
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * (cv2.contourArea(contour)) / (perimeter ** 2)
            
            if circularity > 0.85 and mean_brightness > 200:
                if best_circle is None or r > best_circle[2]:
                    best_circle = (x, y, r)

    # Fallback to contour detection if Hough fails
    if best_circle is None:
        print("Hough failed, using contour fallback")
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(largest_contour)
            best_circle = (int(x), int(y), int(r))

    # Final validation
    if best_circle:
        x, y, r = best_circle
        origin = (x, y)
        radius = r
    else:
        print("Warning: No valid circle found, using frame center")
        origin = (frame.shape[1]//2, frame.shape[0]//2)
        radius = int((min_radius_px + max_radius_px)/2)

    if debug:
        print(f"Chamber detected at {origin} with radius {radius}px")
    return origin, radius

#===============================================================================================================================================
def track_object(avi_path):
    """
    Track tungsten ball through entire video sequence
    
    Args:
        avi_path (str): Path to input AVI file
        cx (int): X-coordinate of chamber center
        cy (int): Y-coordinate of chamber center
        chamber_radius (int): Radius of chamber in pixels
        
    Returns:
    tuple: A tuple containing:
        - positions (list): List of (x,y) coordinates of the tracked object
        - frame_numbers (list): List of frame numbers where object was detected
        - min_ydiff (float): Minimum y-difference between consecutive positions
        - min_ydiff_frame (int): Frame number where minimum y-difference occurred
    """
    # Input validation
    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"Video file not found: {avi_path}")

    # Initialize video capture
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {avi_path}")
    
    # Prepare tracking data structures
    positions = []
    frame_numbers = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        min_ydiff = 9999
        min_ydiff_frame = None

        print(f"Processing {total_frames} frames")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if frame_idx == 0:
                (cx, cy), chamber_radius = detect_chamber(frame)
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue

            # Detect ball position
            mask = np.zeros_like(frame[:,:,0])
            cv2.circle(mask, (cx, cy), chamber_radius, 255, -1)
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            
            gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            inverted = 255 - blurred

            # Constants
            min_radius = 1   # Tungsten ball size
            max_radius = 5            
            circles = cv2.HoughCircles(
                inverted,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=chamber_radius//4,
                param1=50,
                param2=12,
                minRadius=min_radius,
                maxRadius=max_radius
            )

            if circles is not None:
                circles = np.int32(np.around(circles[0]))
                valid = []
                for c in circles:
                    # Validate position within chamber
                    if np.hypot(c[0] - cx, c[1] - cy) < chamber_radius:
                        valid.append(c)
                
                if valid:
                    # Select brightest candidate
                    brightest = max(valid, key=lambda c: gray[c[1], c[0]])
                    px, py, radius = brightest
                    
                    # Convert to chamber-relative coordinates
                    rel_x = px - cx
                    rel_y = cy - py
                    positions.append((rel_x, rel_y))
                    frame_numbers.append(frame_idx)

                    if np.abs(rel_y) < min_ydiff:
                        min_ydiff = np.abs(rel_y)
                        min_ydiff_frame = frame_idx

    except Exception as e:
        raise RuntimeError(f"Error during tracking: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
    print(f"Frame closest to chamber center: {min_ydiff_frame}")
    return np.array(positions), np.array(frame_numbers), min_ydiff_frame

#===============================================================================================================================================
def get_vel_freefall(h=1):
    '''Get velocity (m) of object freefalling from height h in meter'''
    return np.sqrt(2*9.8*h)

def get_pos_freefall(t, t0):
    '''
    Get position of object in freefall relative to chamber center
    At time t0, the ball reaches chamber center (y = 0) having fallen 1 meter
    t is in seconds, t0 is time when ball reaches chamber center
    return is in meters relative to chamber center (positive = above, negative = below)
    '''
    # Time to fall 1 meter from rest
    fall_time = np.sqrt(2.0 / g)
    t_start = t0 - fall_time
    
    # Time since falling started (negative means hasn't started yet)
    dt = np.asarray(t) - t_start
    
    # Position: -(0.5*g*dt² - 1), but only for dt >= 0 (after falling started)
    # Negative sign corrects coordinate system: positive = above center, negative = below
    return -np.where(dt >= 0, 0.5 * g * dt**2 - 1.0, -1.0)

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