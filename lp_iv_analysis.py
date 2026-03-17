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
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
from scipy.ndimage import gaussian_filter1d

me = constants.m_e

def exponential_func(V, a, b):
    return a * np.exp(b * V)

def analyze_IV(voltage, current, plot=False):
    """
    Analyze the IV curve.
    Input:
        voltage: probe sweep voltage (V)
        current: pre-smoothed current density (A/cm²)
    Returns:
        Tuple containing (Vp, Te, ne)
    """
    # Sort data to ensure strict left-to-right processing
    sort_idx = np.argsort(voltage)
    V = voltage[sort_idx]
    I_raw = current[sort_idx]
    
    # ==========================================
    # 1. Linear Isat Fit and Subtraction
    # ==========================================
    sigma_initial = max(10, int(len(V) * 0.02))
    I_raw_guide = gaussian_filter1d(I_raw, sigma=sigma_initial)
    
    amp_raw = np.max(I_raw_guide) - np.min(I_raw_guide)
    knee_threshold = np.min(I_raw_guide) + 0.05 * amp_raw 
    
    knee_crossings = np.argwhere(I_raw_guide >= knee_threshold)
    if len(knee_crossings) > 0:
        isat_idx = max(5, int(knee_crossings[0][0] * 0.80)) 
    else:
        isat_idx = int(len(V) * 0.20)
        
    # Linear fit for Isat baseline
    p_isat = np.polyfit(V[:isat_idx], I_raw[:isat_idx], 1)
    I_baseline = p_isat[0] * V + p_isat[1]
    
    I_sub = I_raw - I_baseline
    
    # ==========================================
    # 2. Find Transition Bounds (Threshold + Padding)
    # ==========================================
    sigma_guide = max(15, int(len(V) * 0.03)) 
    I_guide = gaussian_filter1d(I_sub, sigma=sigma_guide)
    
    amplitude = np.max(I_guide)
    upper_limit = 0.20 * amplitude
    
    upper_crossings = np.argwhere(I_guide >= upper_limit)
    if len(upper_crossings) == 0:
        raise Exception('Signal never reaches the 20% limit.')
    upper_idx = upper_crossings[0][0]
    
    # Walk backward until hitting the 0.5% threshold
    lower_threshold = 0.005 * amplitude 
    lower_idx = upper_idx
    for idx in range(upper_idx, -1, -1):
        if I_guide[idx] <= lower_threshold:
            lower_idx = idx
            break
            
    # Apply explicit padding to capture the start of the exponential tail
    padding = max(2, int(len(V) * 0.01)) 
    lower_idx = max(0, lower_idx - padding)
    
    V_fit = V[lower_idx:upper_idx]
    I_fit = I_sub[lower_idx:upper_idx]
    
    if len(I_fit) < 3:
        raise Exception('Not enough points in the transition region.')

    # ==========================================
    # 3. Smart Initial Guesses via Log-Linear Fit
    # ==========================================
    valid_log = I_fit > (0.01 * amplitude) 
    
    if np.sum(valid_log) > 3:
        p = np.polyfit(V_fit[valid_log], np.log(I_fit[valid_log]), 1)
        b_guess = p[0]
        a_guess = np.exp(p[1])
    else:
        b_guess = 0.5
        a_guess = I_fit[-1] * np.exp(-0.5 * V_fit[-1]) if len(I_fit) > 0 else 0.01
        
    if b_guess <= 0: 
        b_guess = 0.1 
        
    # ==========================================
    # 4. Fit the Exponential
    # ==========================================
    try:
        popt, _ = curve_fit(exponential_func, V_fit, I_fit, 
                            p0=[a_guess, b_guess], 
                            bounds=([0, 0], [np.inf, np.inf]), 
                            maxfev=5000)
    except Exception as e:
        raise Exception(f'Exponential fit failed: {e}')
        
    Te = 1.0 / popt[1]

    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(V, I_raw, label='Input (User Smoothed)', color='tab:blue', linewidth=2)
        plt.plot(V, I_baseline, '--', label='Isat Baseline (Linear)', color='tab:orange')
        plt.plot(V, I_sub, label='Isat Subtracted', color='tab:green', linewidth=2)
        plt.plot(V_fit, I_fit, color='red', label='Region Sent to Fitter', linewidth=4.5, zorder=5)
        
        V_ext = V[max(0, lower_idx - 50) : min(len(V), upper_idx + 50)]
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
    # 5. Plasma Potential (Vp) and Density (ne)
    # ==========================================
    dif3 = np.max(I_sub) * 0.6 
    dif4 = np.max(I_sub) * 0.4 

    lower_bound = np.argwhere(I_sub > dif4)
    start_idx = lower_bound[0][0]
    upper_bound = np.argwhere(I_sub < dif3)
    stop_idx = upper_bound[len(upper_bound)-1][0]

    trans_voltage = V[start_idx:stop_idx]
    trans_current = I_sub[start_idx:stop_idx]
    
    c_trans = np.polyfit(trans_voltage, trans_current, 1)
    
    dif5 = np.max(I_sub) * 0.8
    esat_pos = np.argwhere(I_sub > dif5)
    esat_volt = V[esat_pos[:, 0]]
    esat_curr = I_sub[esat_pos[:, 0]]
    
    d_esat = np.polyfit(esat_volt, esat_curr, 1)

    Vp = abs((d_esat[1] - c_trans[1]) / (d_esat[0] - c_trans[0])) 
    I_Vp = d_esat[0] * Vp + d_esat[1]                  

    if plot:
        plt.figure(figsize=(10, 7))
        plt.plot(V, I_sub, label='Isat Subtracted Signal', color='tab:green', linewidth=2)
        
        y_trans_full = c_trans[0] * V + c_trans[1]
        plt.plot(V, y_trans_full, '--', color='tab:red', label='Transition Linear Fit', linewidth=2)
        
        z_esat_full = d_esat[0] * V + d_esat[1]
        plt.plot(V, z_esat_full, '--', color='tab:purple', label='Esat Linear Fit', linewidth=2)
        
        plt.plot(trans_voltage, trans_current, 'o', color='tab:red', label='Transition Data Points', markersize=5)
        plt.plot(esat_volt, esat_curr, 'o', color='tab:purple', label='Esat Data Points', markersize=5)
        
        plt.axvline(Vp, color='k', linestyle=':', linewidth=2, label=f'Vp = {Vp:.2f} V')
        plt.plot(Vp, I_Vp, 'X', color='black', markersize=10)
        
        plt.ylim(np.min(I_sub) - 0.1 * np.max(I_sub), np.max(I_sub) * 1.1)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current density (A/cm²)')
        plt.title('Plasma Potential (Vp) Determination')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    if Te > 0:
        vth_SI = math.sqrt(constants.e * Te / me) 
        vth_cm = vth_SI * 100                     
        ne = I_Vp / (vth_cm * constants.e)           
    else:
        ne = 0
        raise Exception('Te is negative')

    return (Vp, Te, ne)

def analyze_IV_safe(voltage, current, file_name="Unknown", plot=False):
    """
    Wrapper to safely run IV analysis and catch noisy/failed files.
    """
    try:
        # Try to run your existing good function
        Vp, Te, ne = analyze_IV(voltage, current, plot=plot)
        return Vp, Te, ne
        
    except Exception as e:
        # If it crashes (e.g., "Not enough points", "Exponential fit failed")
        print(f"[{file_name}] Analysis failed: {e}")
        
        # Return np.nan so your 2D plane plotting ignores this pixel
        return (np.nan, np.nan, np.nan)