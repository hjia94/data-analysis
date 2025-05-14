'''
Original code from: https://github.com/lyraYi/video_tracking.git (last modified 2025-03-09)
Modified using claude AI on 2025-05-07

This script is used to track the object in the cine video file taken by Phantom fast camera.
TODO: tracking is not working with initial test May.7.2025
'''

import cv2
import numpy as np
import struct
import time
import matplotlib.pyplot as plt
import os
import re
import sys
from pathlib import Path

from read_cine import read_cine, convert_cine_to_avi
from track_object import track_object

#===============================================================================================================================================
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

def detect_chamber(frame, calibration):
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
    
    if calibration <= 0:
        raise ValueError("calibration must be positive")

    # Convert physical radius constraints (6-8cm in pixels)
    min_radius_px = int(6 / calibration)
    max_radius_px = int(8 / calibration)

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

    print(f"Chamber detected at {origin} with radius {radius}px")
    return origin, radius


def process_video(cine_path):
    """Process a single video file"""
    if not os.path.exists(cine_path):
        raise FileNotFoundError(f"CINE file not found: {cine_path}")
        
    # Use a single temporary AVI file
    avi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_video.avi")


    try:
        # Read CINE file
        time_arr, frame_arr = read_cine(cine_path)
        # Convert to AVI
        convert_cine_to_avi(frame_arr, avi_path)

        # Extract calibration from filename
        calibration = extract_calibration(os.path.basename(cine_path))
        if calibration is None:
            raise ValueError("Could not extract calibration from filename")

        # Open AVI file for chamber detection
        cap = cv2.VideoCapture(avi_path)
        ret, initial_frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read first frame for chamber detection")
        
        # Detect chamber
        (cx, cy), chamber_radius = detect_chamber(initial_frame, calibration)
        cap.release()

        # Show chamber visualization using matplotlib
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(initial_frame, cv2.COLOR_BGR2RGB))
        circle = plt.Circle((cx, cy), chamber_radius, fill=False, color='green', linewidth=2)
        plt.gca().add_patch(circle)
        plt.title('Chamber Detection')
        plt.axis('off')
        plt.show()

        # Track object using extracted calibration & chamber parameters
        positions, times = track_object(avi_path, time_arr, calibration, cx, cy, chamber_radius)
        
        # Create and display analysis plots
        if len(positions) > 1:
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            
            plt.figure(figsize=(15, 5))
            
            # Trajectory plot
            plt.subplot(131)
            plt.scatter(x, y)
            plt.gca().invert_yaxis()
            plt.title('Trajectory in Chamber')
            plt.xlabel('X Position (cm)')
            plt.ylabel('Y Position (cm)')
            plt.grid(True)
            
            # Vertical motion plot
            plt.subplot(132)
            plt.scatter(times, y)
            plt.gca().invert_yaxis()
            plt.title('Vertical Motion')
            plt.xlabel('Time (s)')
            plt.ylabel('Y Position (cm)')
            plt.grid(True)
            
            # Horizontal motion plot
            plt.subplot(133)
            plt.scatter(times, x)
            plt.title('Horizontal Motion')
            plt.xlabel('Time (s)')
            plt.ylabel('X Position (cm)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        return positions, times

    except Exception as e:
        print(f"Error processing {cine_path}: {str(e)}")
        return None, None
    finally:
        # Clean up temporary AVI file
        if os.path.exists(avi_path):
            os.remove(avi_path)

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':
    ifn = r"E:\good_data\He3kA_B250G500G_pl0t20_uw15t35_P30\Y20241102_P30_z13_x200_y0@-40_022.cine"
    positions, times = process_video(ifn)
    
    if positions is not None:
        print("\nTracking Results:")
        print(f"Total frames tracked: {len(positions)}")
        print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")
        print(f"X position range: {min(p[0] for p in positions):.2f} to {max(p[0] for p in positions):.2f} cm")
        print(f"Y position range: {min(p[1] for p in positions):.2f} to {max(p[1] for p in positions):.2f} cm")
