import cv2
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

            print(f"Done reading .cine file ({time.time() - t_read:.1f} s)")
            return time_arr, frame_arr

        except Exception as e:
            raise RuntimeError(f"Error reading CINE file: {str(e)}")

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