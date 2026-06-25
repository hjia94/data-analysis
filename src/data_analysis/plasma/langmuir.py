# -*- coding: utf-8 -*-
"""
Langmuir probe analysis (canonical, single source).

Merged from the former root-level ``lp_analysis.py`` (sweep helpers + legacy
analysis methods) and ``lp_iv_analysis.py`` (the current IV-curve analyzer).
``analyze_IV`` is the canonical analyzer extracting (Vp, Te, ne) from one IV
sweep; ``analyze_IV_safe`` is the batch-loop wrapper that swallows fitting
failures and returns NaNs. Single-curve / interactive callers use ``analyze_IV``;
batch callers use ``analyze_IV_safe``. ``derivative``, ``find_sweep_indices`` and
``reshape_IV`` are the live sweep-preparation helpers. The remaining functions
are alternative/legacy analysis methods kept available pending robustness work
(the routine is subject to change).

Authors: Jia Han (orig. 2018), Google Gemini (IV analyzer, 2026)
"""

import math
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy import integrate, constants
from scipy.ndimage import gaussian_filter1d

# np.trapz was renamed np.trapezoid in numpy 2.0 (np.trapz deprecated).
# Prefer the new name, fall back for numpy < 2.0.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

# physical constants (from former lp_analysis.py)
qe = constants.e          # electron charge (C)
me = constants.electron_mass  # electron mass (kg)
mi = constants.proton_mass    # proton mass (kg)
epsilon = constants.epsilon_0 # permittivity (F/m)

#=== from lp_analysis.py: sweep helpers + legacy analysis methods ============
def analyze_Isat(Iisat, cs): #Iisat in A/cm^2, cs in cm/s

	n = Iisat / (cs * qe) #Iisat = density*ion sound speed*q*area

	return n

def analyze_Esat(Iesat, area, Te): # Esat in mA, Te in eV 
	'''
	Iesat = A*ne*sqrt(Te/(2*pi*me))
	'''
	return 1.49e9 * Iesat /(area * np.sqrt(Te))

#----------------------------------------------------------------------------------------------------
def find_Vp(ss_V, ss_dIdV, magic_num=10):
	'''
	Find max dI/dV by fitting a polynomial over the peak
	numbers are chosen with respect to standard deviation of the trace (and some magic)
	'''
	num = int(np.std(ss_dIdV)/np.average(ss_dIdV) *magic_num)
	ind = np.argpartition(ss_dIdV, -num)[-num:]

	if len(ind) <= 3:
		max_ind = int(np.median(ind))
	else:
		z = np.polyfit(ss_V[ind], ss_dIdV[ind], 2)
		if z[0] < 0:
			p = np.poly1d(z)
			max_ind = np.argmax(p(ss_V))
			if max_ind==0:
				max_ind = ind[-1]
			elif max_ind==len(ss_V)-1:
				max_ind = ind[0]
		else:
			max_ind = np.argmax(ss_dIdV)

	Vp = ss_V[max_ind]
	Vnew = Vp - ss_V
	
	return max_ind, Vp, Vnew

#----------------------------------------------------------------------------------------------------
def EEDF(dIdV, phi, area):
	# Electron energy distribution function for 1D geometry => proportional to first derivative of IV curve
	# See Scott Robertson's paper for detail calculation

	# Compute distribution function
	f = me/(qe**2*area) * dIdV

	# Integrate to find density
	ne = math.sqrt(2/me) * integrate.simps(f/np.sqrt(qe*phi), qe*phi)

	g = f*phi

	Te = integrate.trapz(g, phi) / integrate.trapz(f, phi)

	return f, ne, Te
#----------------------------------------------------------------------------------------------------
def derivative(I, V, sigma=30, smth=True):
	'''
	sigma: smoothing factor for gaussian filter
	threshold: threshold for finding max dIdV
	'''
	dIdV = np.gradient(I, V, axis=-1)
	ss_dIdV = gaussian_filter1d(dIdV, sigma, axis=-1)
	max_inds = np.argmax(ss_dIdV, axis=-1)

	if smth:
		return ss_dIdV, max_inds
	else:
		return dIdV, max_inds
#----------------------------------------------------------------------------------------------------
def distribution(V, dIdV,max_ind, length=100, verbose=False):

	warnings.simplefilter("error")

	
	while True:

		try:
			Vp = V[max_ind]
			Vnew = Vp - V

			dIdV_sub = dIdV[:max_ind+1]
			Vnew_sub = Vnew[:max_ind+1]

			def func(x, a, b, c):
				return a * np.exp(b * x) + c

			popt, _ = curve_fit(func, Vnew_sub, dIdV_sub)

			Vfake = np.linspace(0,np.max(V),length)
			f = me/(qe**2) * func(Vfake, *popt)

			return f

		except RuntimeWarning:
			max_ind -= 1
			if max_ind < 0:
				raise Exception('Cannot find valid distribution function')
	# 		# ===================
		
			dIdV_sub = dIdV[:max_ind+1]
			Vnew_sub = Vnew[:max_ind+1]

			# ===================
			# Fits first derivative to a function
			def func(x, a, b, c):
				return a * np.exp(-b*x**c)

			popt, pcov = np.curve_fit(func, Vnew_sub, dIdV_sub, p0=[np.max(dIdV)*1.5, 1/5, 1]) #bounds=([0,0,0.5],[np.inf,np.inf,3.5]))
			if verbose:
				print ('fit parameter: ', popt)

			def integrand1(x):
				return me/(qe) * popt[0] * np.exp(-popt[1]*x**popt[2]) / np.sqrt(qe*x)
			yne, err = np.integrate.quad(integrand1, 0, np.inf)
			break

		except RuntimeWarning:
			max_ind -= 1

	Vfake = np.linspace(0,np.max(V),length)
	f = me/(qe**2) * func(Vfake, *popt)

	return dIdV_sub, Vnew_sub, Vp, Vfake, f, yne, popt
#---------------------------------------------------------------
def Vrfune(vin):
	"""  f(vin) = P(vin) * exp(k0*vin) + k1
	where k0 = a[0], k1 = a[1], and P = np.poly1d(a[2:])
	"""

	a = np.array([ 9.99999765e-01, -1.02341555e-04,  3.86436545e-04, -1.50072604e-04, -1.65457794e-04, -1.92827148e-04,  5.39873500e-04])
	V1 = -0.1
	V2 = 2.6  # = fit_V2 nearest value

	p = np.poly1d(a[1:])
	dpdv = np.polyder(p)

	lo_sel = vin < V1
	hi_sel = vin > V2
	mid_sel = (vin >= V1) & (vin <= V2)

	mid = mid_sel * p(vin)*(np.exp(a[0]*vin)-1)

	Vrfun_V1 = p(V1) * (math.exp(a[0]*V1)-1)
	slope_V1 = dpdv(V1) * (math.exp(a[0]*V1)-1) + p(V1)*a[0]*math.exp(a[0]*V1)
	lo = lo_sel * (Vrfun_V1 + slope_V1 * (vin-V1))

	Vrfun_V2 = p(V2) * (math.exp(a[0]*V2)-1)
	slope_V2 = dpdv(V2) * (math.exp(a[0]*V2)-1) + p(V2)*a[0]*math.exp(a[0]*V2)
	hi = hi_sel * (Vrfun_V2 + slope_V2 * (vin - V2))

	return lo + mid + hi
#----------------------------------------------------------------------------------------------------
def particle_number(popt, lower_bound, upper_boud):

	def integrand1(x):
		return me/(qe*area) * popt[0] * np.exp(-popt[1]*x**popt[2]) / np.sqrt(qe*x)

	yne, err = integrate.quad(integrand1, lower_bound, upper_boud) #, weight='alg', wvar=(-1/2,0))

	return math.sqrt(2/me) * yne


#===========================================================================================================
#===========================================================================================================

def temperature(phi, I, Vp_ndx, plot=False):
	'''
	find temperature by fitting a straight line to semi-log plot of IV curve (Isat already subtracted)
	'''

	lnI = np.log(I[:Vp_ndx])

	Vnew = phi[:Vp_ndx][~np.isnan(lnI)]
	lnI = lnI[~np.isnan(lnI)]

	if plot:
		plt.figure()
		plt.plot(Vnew, lnI)

	a1 = np.max(lnI)
	a2 = np.min(lnI)

	a3 = (lnI < a1 - (a1-a2)*0.1) & (lnI > a1 - (a1-a2)*0.3)
	a4 = lnI*a3

	if plot:
		plt.plot(Vnew, a4)

	p = np.poly1d(np.polyfit(Vnew[np.nonzero(a4)], a4[np.nonzero(a4)], 1))

	if plot:
		plt.plot(Vnew, p(Vnew))

	return p


def EEPF(I, V, smooth=True, plot=False):
	'''
	Electron probability function for cylindrical probe geometry => found from second derivitvie of IV curve
	see https://pdfs.semanticscholar.org/502e/3fab71d85c9163c9ee0599f64e65c3d99aa2.pdf
	'''

	if plot:
		plt.figure()
		plt.plot(V, I)
		plt.title('IV curve')

	# Fit straight line to Isat
	dif1 = [(max(I) - min(I))*0.005 + min(I), (max(I) - min(I))*0 + min(I)]
	vals = np.argwhere(np.logical_and(I < dif1[0], I > dif1[1]))

	cropped_voltage = []
	cropped_current = []
	for i in range(0, len(vals)):
			idx = vals[i][0]
			cropped_voltage.append(V[idx])
			cropped_current.append(I[idx])

	c = np.polyfit(cropped_voltage, cropped_current, 1)
	y = c[0] * V + c[1]

	Inew = I - y # New current subtracts Isat

	if plot: # Plots the fitted Isat straight line
		plt.plot(V, y)


	# compute first derivative to find plasma potential 
	dIdV = np.gradient(Inew, V, edge_order=2)

	Vp_ndx = np.argmax(dIdV)
	Vp = V[Vp_ndx]
	Vnew = Vp - V # new probe voltage with respect to plasma potential

	if plot:
		plt.figure()
		plt.title('dIdV')
		plt.plot(Vnew, dIdV)

	# compute second derivative to find EEPF
	d2IdV2 = np.gradient(dIdV, Vnew, edge_order = 2)

	if plot:
		plt.figure()
		plt.title('d2IdV2')
		plt.plot(Vnew, d2IdV2)

#	if smooth:
#		d2IdV2 = general.smooth(d2IdV2, 500)
#		if plot:
#			plt.plot(Vnew, d2IdV2)
	
	f = 2/(qe) * np.sqrt(2*me/qe) * d2IdV2 # EEPF
	Vp_ndx_1 = np.argmax(f)
	print('Vp(old) = %.3f  Vp(new) = %.3f' %(V[Vp_ndx], V[Vp_ndx_1]))

	ne = trapezoid(f[:Vp_ndx_1], Vnew[:Vp_ndx_1])	# density = area under EEPF
	print('density = %.3e' %(ne))

	return Vnew[:Vp_ndx_1], f[:Vp_ndx_1], ne, Vp

#===========================================================================================================
#===========================================================================================================
def find_sweep_indices(V, padding=10):
    """
    Extracts start and stop indices for pulsed voltage sweeps of any size, 
    duration, baseline, or polarity.
    """
    # 0. FORCE 1D ARRAY: This strips out any hidden dimensions (like (N, 1) -> (N,))
    V = np.asarray(V).flatten()
    
    # 1. Dynamically find the resting baseline.
    baseline = np.median(V)
    
    # 2. "Rectify" the signal. By taking the absolute difference from the baseline,
    # all sweeps become positive spikes starting from 0.
    rectified_V = np.abs(V - baseline)
    
    # 3. Dynamically set a noise floor.
    # FORCE FLOAT: Wrapping this in float() guarantees SciPy reads it as a single scalar number,
    # preventing the "interval border must match x" ValueError.
    noise_floor = float(np.max(rectified_V) * 0.10)
    
    # 4. Find the tips of the triangles
    peaks, _ = find_peaks(rectified_V, prominence=noise_floor, distance=10)
    
    if len(peaks) == 0:
        print("No prominent sweeps found in this data.")
        return [], []
    
    # 5. Find the base of each peak (98% of the way down from the tip)
    widths, width_heights, left_ips, right_ips = peak_widths(rectified_V, peaks, rel_height=0.98)
    
    # 6. Extract the start (left) and stop (right) indices.
    start_t_ls = np.maximum(0, np.floor(left_ips) - padding).astype(int).tolist()
    stop_t_ls = np.minimum(len(V) - 1, np.ceil(right_ips) + padding).astype(int).tolist()
    
    return start_t_ls, stop_t_ls

def reshape_IV(Vsweep_arr, Isweep_arr, start_t_ls, stop_t_ls, trim_percent=1.0):
    """
    Slices raw arrays into individual sweeps, standardizes their length, 
    and trims a percentage off the edges to remove switching noise.
    """
    # 1. Calculate the lengths of all detected sweeps
    lengths = [stop - start for start, stop in zip(start_t_ls, stop_t_ls)]
    
    # 2. Find the raw minimum length
    min_len = min(lengths)
    
    # 3. Calculate how many points equal the requested percentage
    trim_points = int(min_len * (trim_percent / 100.0))
    final_len = min_len - (2 * trim_points)
    
    print(f"Standardizing raw sweep length: {min_len} points.")
    print(f"Trimming {trim_percent}% ({trim_points} points) from both the start and end.")
    print(f"Final sweep length stacked: {final_len} points.")

    # Initialize an empty list to store the chunks
    I_chunks = []
    V_chunks = []

    # 4. Loop through the starts, applying the trim and the uniform final length
    for start in start_t_ls:
        # Shift the start index forward by the trim amount
        actual_start = start + trim_points
        
        # Ensure the chunk is exactly the final_len to avoid dimension mismatch
        actual_stop = actual_start + final_len

        I_chunks.append(Isweep_arr[:, :, actual_start:actual_stop])
        V_chunks.append(Vsweep_arr[:, actual_start:actual_stop])

    # Stack the list of chunks into a new array
    Isweep_reshaped = np.stack(I_chunks, axis=2)
    Vsweep_reshaped = np.stack(V_chunks, axis=1)

    return Vsweep_reshaped, -Isweep_reshaped

#=== from lp_iv_analysis.py: canonical IV-curve analyzer =====================
# --- Configuration constants ---
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
        # Any exception raised by analyze_IV (e.g. unreachable amplitude
        # threshold, failed exponential fit) is caught here. Print the id and
        # the specific error message. (Note: an out-of-range Te per TE_MAX_EV is
        # flagged as NaN inside analyze_IV, not raised, so it returns normally.)
        if verbose:
            print(f"[{file_name}] Analysis failed: {e}")
        
        # Return NaNs so the main loop can store them and safely move on
        return np.nan, np.nan, np.nan
