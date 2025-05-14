import cv2
import numpy as np
import os
from pathlib import Path

def track_object(avi_path, time_arr, cx, cy, chamber_radius):
    """
    Track tungsten ball through entire video sequence
    
    Args:
        avi_path (str): Path to input AVI file
        time_arr (np.ndarray): Array of timestamps for each frame
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
        
    if not isinstance(time_arr, np.ndarray):
        raise ValueError("time_arr must be a numpy array")
        
    if chamber_radius <= 0:
        raise ValueError("chamber_radius must be positive")

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
    return np.array(positions), time_arr[frame_numbers], np.array(frame_numbers), min_ydiff_frame
