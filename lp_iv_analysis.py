# -*- coding: utf-8 -*-
"""
Langmuir Probe IV Curve Analysis
--------------------------------
This script processes Langmuir probe current-voltage (IV) sweeps to extract 
fundamental plasma parameters: Electron Temperature (Te), Plasma Potential (Vp), 
and Electron Density (ne). 

It utilizes a linear baseline fit for ion saturation (Isat) removal, exponential 
curve fitting for the electron transition region (with explicit boundary padding), 
and intersecting linear fits to determine the plasma potential.

Originally taken from lp_analysis.py, but moved to a separate file due to its length and complexity.
Improved by Google Gemini after iterating with known data for different cases.

Authors: J. Han, Google Gemini
Original function created: 2018-09
File created: 2026-03

Main Analysis Logic
-------------------
The IV curve analysis follows a multi-stage approach:

1. **Ion Saturation Baseline Removal**
   - Smooth the raw current signal with Gaussian filter
   - Detect the "knee" where current begins to rise significantly
   - Fit a linear baseline to ion saturation region
   - Subtract baseline from entire IV curve

2. **Transition Region Identification**
   - Apply medium smoothing to baseline-corrected signal
   - Find upper/lower amplitude thresholds (30% and 5% respectively)
   - Extract the exponential transition region between these bounds
   - Minimum 3 points required; raises exception if insufficient data

3. **Electron Temperature Extraction (Te)**
   - Fit exponential function I = a·exp(b·V) to transition region
   - Te = 1/b (inverse of exponential slope)
   - Flag Te as NaN if > 10 eV or <= 0 eV
   - Uses dummy Te = 0.1 eV for downstream ne calculation if Te is flagged

4. **Plasma Potential Extraction (Vp)**
   - Define two linear fit regions: transition and esat
   - Fit separate lines to each region
   - Compute Vp from intersection of these two fit lines
   - Flag Vp as NaN if > 100 V or denominator < 1e-10

5. **Electron Density Calculation (ne)**
   - Compute thermal velocity: v_th = sqrt(e·Te/m_e)
   - Use saturation current at Vp: ne = I_Vp / (v_th·e)
   - Returns NaN if Te was flagged as invalid
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from scipy.ndimage import gaussian_filter1d

# ============================================
# Configuration Constants
# ============================================
# Gaussian smoothing parameters
SIGMA_ISAT = 50  # Smoothing width for initial Isat baseline detection (larger = more aggressive smoothing)
SIGMA_GUIDE = 25  # Smoothing width for finding transition region boundaries (medium smoothing)

# Amplitude thresholds (% of max amplitude)
KNEE_THRESHOLD_PCT = 0.05  # Knee crossing at 5% of max amplitude identifies Isat region start
LOWER_THRESHOLD_PCT = 0.05  # Lower bound at 5% of max amplitude for transition region
UPPER_THRESHOLD_PCT = 0.30  # Upper bound at 30% of max amplitude for transition region
TRANS_LOWER_PCT = 0.40  # Transition fit region lower threshold at 40% of max amplitude
TRANS_UPPER_PCT = 0.60  # Transition fit region upper threshold at 60% of max amplitude
ESAT_THRESHOLD_PCT = 0.80  # Electron saturation region at 80% of max amplitude

# Physical parameter limits
TE_MAX_EV = 10  # Flag Te as unreasonable if > 10 eV
TE_DUMMY_EV = 0.1  # Placeholder constant (deprecated - no longer used)
VP_MAX_V = 100  # Flag Vp as unreasonable if > 100 V (extreme voltage limit)

# Boundary detection parameters
MIN_ISAT_IDX = 5  # Minimum index to avoid edge effects when fitting Isat region
ISAT_KNEE_FRACTION = 0.80  # Scale knee crossing by 80% to set Isat fit window start
ISAT_FALLBACK_FRACTION = 0.20  # Fallback: use 20% of total points if knee not found
BOUNDARY_PAD_POINTS = 50  # Padding points on each side of transition region for extended fit plot
MIN_FIT_POINTS = 3  # Minimum points required in transition region for exponential fit
MIN_ESAT_POINTS = 2  # Minimum points required in electron saturation region for linear fit
MIN_TRANS_POINTS = 2  # Minimum points required in transition region for Vp fit
MIN_STOP_IDX_GAP = 5  # Minimum gap between start_idx and stop_idx for transition region

# Fit parameters
EXP_FIT_MAXFEV = 5000  # Maximum number of iterations for exponential curve_fit optimization
LIN_FIT_ORDER = 1  # Polynomial order for linear fits (1 = linear, 2 = quadratic, etc.)
DENOM_THRESHOLD = 1e-10  # Tolerance threshold to avoid division by zero when computing Vp intersection

def exponential_func(V, a, b):
    return a * np.exp(b * V)

def _find_crossing_index(signal, threshold, start=0, direction='up'):
    """
    Find the first index where signal crosses threshold.
    
    Parameters
    ----------
    signal : np.ndarray
        Signal to search
    threshold : float
        Threshold value
    start : int
        Start searching from this index
    direction : str
        'up' for crossing from below, 'down' for crossing from above
    
    Returns
    -------
    int or None
        Index of crossing, or None if not found
    """
    if direction == 'up':
        crossings = np.argwhere(signal >= threshold)
    else:
        crossings = np.argwhere(signal <= threshold)
    
    if len(crossings) == 0:
        return None
    return crossings[0][0]

def _apply_linear_fit(V, I, order=1):
    """
    Apply polynomial fit and return coefficients.
    
    Parameters
    ----------
    V : np.ndarray
        X data (voltage)
    I : np.ndarray
        Y data (current)
    order : int
        Polynomial order
    
    Returns
    -------
    np.ndarray
        Polynomial coefficients
    """
    return np.polyfit(V, I, order)

def _eval_polyfit(coeffs, V):
    """Evaluate polynomial fit at given voltages."""
    return np.polyval(coeffs, V)

def analyze_IV(voltage, current, plot=False):
    """Main function to analyze IV curve and extract Vp, Te, ne."""
    sort_idx = np.argsort(voltage)
    V = voltage[sort_idx]
    I_raw = current[sort_idx]
    
    # ==========================================
    # 1. Linear Isat Fit and Subtraction
    # ==========================================
    I_raw_guide = gaussian_filter1d(I_raw, sigma=SIGMA_ISAT)
    
    amp_raw = np.max(I_raw_guide) - np.min(I_raw_guide)
    knee_threshold = np.min(I_raw_guide) + KNEE_THRESHOLD_PCT * amp_raw
    
    knee_crossings = np.argwhere(I_raw_guide >= knee_threshold)
    isat_idx = max(MIN_ISAT_IDX, int(knee_crossings[0][0] * ISAT_KNEE_FRACTION)) \
        if len(knee_crossings) > 0 else int(len(V) * ISAT_FALLBACK_FRACTION)
        
    p_isat = _apply_linear_fit(V[:isat_idx], I_raw[:isat_idx])
    I_baseline = _eval_polyfit(p_isat, V)
    I_sub = I_raw - I_baseline
    
    # ==========================================
    # 2. Find Transition Bounds (Original Logic)
    # ==========================================
    I_guide = gaussian_filter1d(I_sub, sigma=SIGMA_GUIDE)
    
    amplitude = np.max(I_guide)
    upper_limit = UPPER_THRESHOLD_PCT * amplitude
    
    upper_idx = _find_crossing_index(I_guide, upper_limit, direction='up')
    if upper_idx is None:
        raise Exception('Signal never reaches the 30% amplitude limit.')
    
    lower_threshold = LOWER_THRESHOLD_PCT * amplitude
    lower_idx = upper_idx
    for idx in range(upper_idx, -1, -1):
        if I_guide[idx] <= lower_threshold:
            lower_idx = idx
            break
    
    V_fit = V[lower_idx:upper_idx]
    I_fit = I_sub[lower_idx:upper_idx]
    
    if len(I_fit) < MIN_FIT_POINTS:
        raise Exception('Not enough points in the transition region.')

    # ==========================================
    # 3. Initial Guesses for Exponential
    # ==========================================
    valid_log = I_fit > 0 
    
    if np.sum(valid_log) > 3:
        with np.errstate(divide='ignore', invalid='ignore'):
            p = _apply_linear_fit(V_fit[valid_log], np.log(I_fit[valid_log]))
            b_guess = p[0]
            a_guess = np.exp(p[1])
    else:
        b_guess = 0.5
        a_guess = 0.01

    # ==========================================
    # 4. Fit the Exponential
    # ==========================================
    try:
        popt, _ = curve_fit(exponential_func, V_fit, I_fit, 
                            p0=[a_guess, b_guess], 
                            maxfev=EXP_FIT_MAXFEV)
    except Exception as e:
        raise Exception(f'Exponential fit failed: {e}')
        
    Te = 1.0 / popt[1]

    # === DIAGNOSTIC PLOT 1: EXPONENTIAL FIT ===
    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(V, I_raw, label='Input (User Smoothed)', color='tab:blue', linewidth=2)
        plt.plot(V, I_baseline, '--', label='Isat Baseline (Linear)', color='tab:orange')
        plt.plot(V, I_sub, label='Isat Subtracted', color='tab:green', linewidth=2)
        plt.plot(V_fit, I_fit, color='red', label='Region Sent to Fitter', linewidth=4.5, zorder=5)
        
        V_ext = V[max(0, lower_idx - BOUNDARY_PAD_POINTS) : min(len(V), upper_idx + BOUNDARY_PAD_POINTS)]
        plt.plot(V_ext, exponential_func(V_ext, *popt), '--', color='purple', 
                 label=f'Exp Fit (Te={Te:.2f} eV)', linewidth=2.5, zorder=6)
        
        plt.ylim(np.min(I_raw)*1.1, np.max(I_raw)*1.1)
        plt.title('Langmuir Probe IV Analysis: Exponential Fit')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density (A/cm²)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ==========================================
    # 4b. Check for Unreasonable Te
    # ==========================================
    if Te > TE_MAX_EV or Te <= 0:  # Hard threshold
        Te = np.nan  # Flag returned Te as NaN
        Te_calc = TE_DUMMY_EV  # Use dummy value for ne calculation
    else:
        Te_calc = Te

    # ==========================================
    # 5. Find Vp by two linear fit and cross point
    # ==========================================
    I_sub_max = np.max(I_sub)
    trans_upper_thresh = I_sub_max * TRANS_UPPER_PCT
    trans_lower_thresh = I_sub_max * TRANS_LOWER_PCT 
    esat_thresh = I_sub_max * ESAT_THRESHOLD_PCT

    # Find transition region
    lower_bound = np.argwhere(I_sub > trans_lower_thresh)
    if len(lower_bound) == 0:
        raise Exception('Signal too weak for Vp extraction.')
    start_idx = lower_bound[0][0]
    
    upper_bound = np.argwhere(I_sub < trans_upper_thresh)
    stop_idx = upper_bound[-1][0] if len(upper_bound) > 0 else len(I_sub)-1
    
    if start_idx >= stop_idx:
        stop_idx = min(len(V)-1, start_idx + MIN_STOP_IDX_GAP)

    trans_voltage = V[start_idx:stop_idx]
    trans_current = I_sub[start_idx:stop_idx]
    
    if len(trans_voltage) < MIN_TRANS_POINTS:
        raise Exception('Not enough points for Vp transition fit.')
    
    # Fit transition region
    c_trans = _apply_linear_fit(trans_voltage, trans_current)
    
    # Find Esat region
    esat_pos = np.argwhere(I_sub > esat_thresh)
    if len(esat_pos) < MIN_ESAT_POINTS:
        raise Exception('Not enough points for Esat fit.')
        
    esat_volt = V[esat_pos[:, 0]]
    esat_curr = I_sub[esat_pos[:, 0]]
    
    d_esat = _apply_linear_fit(esat_volt, esat_curr)

    # Compute Vp from intersection
    denom = d_esat[0] - c_trans[0]
    if abs(denom) < DENOM_THRESHOLD:
        Vp = np.nan
        I_Vp = np.nan
    else:
        Vp = abs((d_esat[1] - c_trans[1]) / denom) 
        I_Vp = _eval_polyfit(d_esat, np.array([Vp]))[0]

    if Vp >= VP_MAX_V:  # Unreasonable Vp; hard threshold
        Vp = np.nan

    # === DIAGNOSTIC PLOT 2: VP INTERSECTION ===
    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(V, I_sub, label='Isat Subtracted Signal', color='tab:green', linewidth=2)
        
        y_trans_full = _eval_polyfit(c_trans, V)
        plt.plot(V, y_trans_full, '--', color='tab:red', label='Transition Linear Fit', linewidth=2)
        
        z_esat_full = _eval_polyfit(d_esat, V)
        plt.plot(V, z_esat_full, '--', color='tab:purple', label='Esat Linear Fit', linewidth=2)
        
        plt.plot(trans_voltage, trans_current, 'o', color='tab:red', label='Transition Data Points', markersize=5)
        plt.plot(esat_volt, esat_curr, 'o', color='tab:purple', label='Esat Data Points', markersize=5)
        
        if not np.isnan(Vp):
            plt.axvline(Vp, color='k', linestyle=':', linewidth=2, label=f'Vp = {Vp:.2f} V')
            plt.plot(Vp, I_Vp, 'X', color='black', markersize=10)
        
        plt.ylim(np.min(I_sub) - 0.1 * np.max(I_sub), np.max(I_sub) * 1.1)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density (A/cm²)')
        plt.title('Plasma Potential (Vp) Determination')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    # ==========================================
    # 6. Calculate Electron Density
    # ==========================================
    if Te_calc > 0:
        vth_SI = math.sqrt(constants.e * Te_calc / constants.m_e) 
        vth_cm = vth_SI * 100                     
        ne = I_Vp / (vth_cm * constants.e)           
    else:
        ne = np.nan

    return (Vp, Te, ne)

def analyze_IV_safe(voltage, current, file_name="", verbose=False):
    """
    Wrapper function to safely execute analyze_IV. 
    Catches any fitting errors or data quality exceptions, logs them, 
    and returns NaNs to prevent the batch loop from crashing.
    """
    try:
        # Try to run the main analysis function
        Vp, Te, ne = analyze_IV(voltage, current)
        return Vp, Te, ne
        
    except Exception as e:
        # If ANY exception is raised (including our new Te >= 100 check),
        # it gets caught here. We print the ID and the specific error message.
        if verbose:
            print(f"[{file_name}] Analysis failed: {e}")
        
        # Return NaNs so the main loop can store them and safely move on
        return np.nan, np.nan, np.nan