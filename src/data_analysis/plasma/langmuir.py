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

The batch-pipeline section at the bottom (``prepare_sweep_data``,
``process_iv_and_save``, ``sweep_npz_paths`` and the ``load_*`` npz loaders) is
the campaign-independent half of the Mar-2026 / Jun-2026 sweep workflow: each
experiment script keeps only its raw-read step and hands the arrays here.

Authors: Jia Han (orig. 2018), Google Gemini (IV analyzer, 2026)
"""

import datetime
import math
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy import integrate, constants
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from data_analysis.signal import line_average

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
    duration, baseline, or polarity.  Emits one window per *physical* sweep:
    duplicate peaks on one ramp are merged and partial edge sweeps dropped
    here, so windows, timestamps and reshaped arrays always describe the same
    sweeps.  Raises ``ValueError`` when no sweep is found.
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
        raise ValueError("No prominent sweeps found in this data.")

    # 5. Find the base of each peak (98% of the way down from the tip)
    _, _, left_ips, right_ips = peak_widths(rectified_V, peaks, rel_height=0.98)

    # 6. Merge overlapping windows.  distance=10 above is far smaller than a
    # sweep, so a noisy ramp apex can register two peaks whose 98%-height bases
    # coincide -- the same physical sweep twice.  Overlap is judged on the
    # unpadded ips so +/-padding cannot glue genuinely adjacent pulses.
    windows = sorted(zip(left_ips, right_ips))
    merged = [list(windows[0])]
    for lo, hi in windows[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    if len(merged) < len(windows):
        print(f"Merged {len(windows) - len(merged)} duplicate sweep window(s).")

    # 7. Pad into integer windows, then drop partial sweeps.  All pulses share
    # one length, so a window much shorter than the longest is a ramp cut off
    # at the record edge.  Dropping it here (not in reshape_IV) keeps every
    # downstream per-sweep axis -- timestamps, calibration windows, reshaped
    # arrays -- in step.
    lo, hi = np.array(merged).T
    start = np.maximum(0, np.floor(lo).astype(int) - padding)
    stop = np.minimum(len(V) - 1, np.ceil(hi).astype(int) + padding)
    full = (stop - start) >= 0.5 * (stop - start).max()
    if not full.all():
        print(f"Dropped {(~full).sum()} partial sweep(s) at the record edge.")

    return start[full].tolist(), stop[full].tolist()

def reshape_IV(Vsweep_arr, Isweep_arr, start_t_ls, stop_t_ls, trim_percent=1.0):
    """
    Slices the raw arrays into individual sweeps and trims a percentage off each
    edge to remove switching noise.

    One output sweep per input window: ``find_sweep_indices`` already merged
    duplicates and dropped partial edge sweeps, so nothing is dropped here --
    the sweep axis stays aligned with the windows (and any timestamps derived
    from them).  Windows are sliced to the common minimum length so they stack
    into one array.
    """
    lengths = np.array([stop - start for start, stop in zip(start_t_ls, stop_t_ls)])
    sweep_len = int(lengths.min())
    trim_points = int(sweep_len * (trim_percent / 100.0))
    final_len = sweep_len - (2 * trim_points)

    print(f"Sweep length: {sweep_len} points.")
    print(f"Trimming {trim_percent}% ({trim_points} points) from each end.")
    print(f"Final sweep length stacked: {final_len} points.")

    I_chunks = []
    V_chunks = []
    for start in start_t_ls:
        a = start + trim_points
        b = a + final_len
        I_chunks.append(Isweep_arr[:, :, a:b])
        V_chunks.append(Vsweep_arr[:, a:b])

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
VP_MAX_V = 100  # Flag Vp as unreasonable if |Vp| > 100 V (Vp itself may be negative)

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


def ne_from_esat(I_esat, Te):
    """Density [cm^-3] from an electron-saturation current density and Te.

    ``I_esat`` [A/cm^2] is the current where ``analyze_IV``'s transition and Esat
    fit lines cross; ``Te`` [eV].  Both may be arrays.
    ``ne = I_esat / (e * vth)``, ``vth = sqrt(e*Te/m_e)`` -- the LAPD LP recipe
    this pipeline has always used.  Non-positive or non-finite Te -> ``nan``
    (never a placeholder Te: a fabricated value yields a real-looking density).

    Absolute scale is only as good as ``Aprobe`` and the ``vth`` convention:
    this omits the standard flux-average factor sqrt(2*pi)/4 ~ 0.63, so it is a
    consistent relative measure, not an absolute density.  Prefer
    :func:`calibrate_plasma_npz` when interferometer data exists.
    """
    Te = np.asarray(Te, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        vth_cm = np.sqrt(constants.e * Te / constants.m_e) * 100  # m/s -> cm/s
        return np.where(Te > 0, np.asarray(I_esat, dtype=float) / (vth_cm * constants.e),
                        np.nan)


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

def _plot_iv_diagnostics(V, I_raw, I_baseline, I_sub, isat_idx, V_fit, I_fit,
                         lower_idx, upper_idx, popt, c_trans, d_esat,
                         trans_voltage, trans_current, esat_volt, esat_curr,
                         Vp, Te, label):
    """``analyze_IV``'s three diagnostic panels in one shared-x figure; blocks on show.

    Takes every intermediate it draws because none of them survive in
    ``analyze_IV``'s return value -- that is the point of the panels.
    """
    fig, (ax_isat, ax_exp, ax_vp) = plt.subplots(3, 1, figsize=(11, 13), sharex=True)

    ax_isat.plot(V, I_raw, label='Input (User Smoothed)', color='tab:blue', linewidth=2)
    ax_isat.plot(V, I_baseline, '--', label='Isat Baseline (Linear)', color='tab:orange')
    ax_isat.plot(V[:isat_idx], I_raw[:isat_idx], color='tab:orange',
                 linewidth=4.5, alpha=0.5, zorder=4,
                 label=f'Isat Fit Region (to {V[isat_idx]:.1f} V)')
    ax_isat.set_title('1. Isat Baseline Fit')
    ax_isat.set_ylabel('Current density (A/cm²)')
    ax_isat.legend(fontsize=8)
    ax_isat.grid(True, alpha=0.3)

    ax_exp.plot(V, I_raw, label='Input (User Smoothed)', color='tab:blue', linewidth=2)
    ax_exp.plot(V, I_baseline, '--', label='Isat Baseline (Linear)', color='tab:orange')
    ax_exp.plot(V, I_sub, label='Isat Subtracted', color='tab:green', linewidth=2)
    ax_exp.plot(V_fit, I_fit, color='red', label='Region Sent to Fitter', linewidth=4.5, zorder=5)

    V_ext = V[max(0, lower_idx - BOUNDARY_PAD_POINTS) : min(len(V), upper_idx + BOUNDARY_PAD_POINTS)]
    ax_exp.plot(V_ext, exponential_func(V_ext, *popt), '--', color='purple',
                label=f'Exp Fit (Te={Te:.2f} eV)', linewidth=2.5, zorder=6)

    ax_exp.set_ylim(np.min(I_raw)*1.1, np.max(I_raw)*1.1)
    ax_exp.set_title('2. Exponential Fit -> Te')
    ax_exp.set_ylabel('Current density (A/cm²)')
    ax_exp.legend(fontsize=8)
    ax_exp.grid(True, alpha=0.3)

    ax_vp.plot(V, I_sub, label='Isat Subtracted Signal', color='tab:green', linewidth=2)
    ax_vp.plot(V, _eval_polyfit(c_trans, V), '--', color='tab:red',
               label='Transition Linear Fit', linewidth=2)
    ax_vp.plot(V, _eval_polyfit(d_esat, V), '--', color='tab:purple',
               label='Esat Linear Fit', linewidth=2)
    ax_vp.plot(trans_voltage, trans_current, 'o', color='tab:red',
               label='Transition Data Points', markersize=5)
    ax_vp.plot(esat_volt, esat_curr, 'o', color='tab:purple',
               label='Esat Data Points', markersize=5)

    ax_vp.set_ylim(np.min(I_sub) - 0.1 * np.max(I_sub), np.max(I_sub) * 1.1)
    ax_vp.set_xlabel('Voltage (V)')
    ax_vp.set_ylabel('Current density (A/cm²)')
    ax_vp.set_title('3. Transition/Esat Crossing -> Vp, I_esat')
    ax_vp.legend(fontsize=8)
    ax_vp.grid(True, alpha=0.3)

    fig.suptitle(f'{label}\nVp = {Vp:.2f} V,  Te = {Te:.2f} eV' if label
                 else f'Langmuir IV Analysis:  Vp = {Vp:.2f} V,  Te = {Te:.2f} eV')
    fig.tight_layout()
    plt.show()


def analyze_IV(voltage, current, plot=False, calibrated=True, label=""):
    """Analyze one IV curve; returns ``(Vp, Te, ne)``.

    ``calibrated`` refers to against interferometer

    Both are the same current density up to the Te factor; the units differ, so
    callers that mix runs must not plot them on one axis.

    ``plot=True`` draws the three diagnostic panels into one figure and calls
    ``plt.show()``; ``label`` names the trace in its title (:func:`show_iv_fit`
    passes run/position/sweep, which these two bare arrays cannot carry).
    """
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

    Te_unclamped = Te            # panel 2 labels the fitted Te, not 4b's NaN

    # ==========================================
    # 4b. Check for Unreasonable Te
    # ==========================================
    if Te > TE_MAX_EV or Te <= 0:  # Hard threshold
        # NaN propagates into ne on the uncalibrated path (ne_from_esat divides
        # by vth(Te)); the calibrated path takes I_esat alone and is unaffected.
        Te = np.nan

    # ==========================================
    # 5. Cross point of the transition and Esat linear fits -> I_esat, Vp
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

    # c[0]*V + c[1] = d[0]*V + d[1]  =>  V = (d[1] - c[1]) / (c[0] - d[0]).
    # polyfit returns [slope, intercept], so the slope difference is c[0]-d[0];
    # the reverse order negates every crossing (a prior abs() hid this).
    denom = c_trans[0] - d_esat[0]
    if abs(denom) < DENOM_THRESHOLD:
        V_cross = np.nan
        I_esat = np.nan
    else:
        V_cross = (d_esat[1] - c_trans[1]) / denom
        I_esat = _eval_polyfit(d_esat, np.array([V_cross]))[0]

        if I_esat <= 0:
            I_esat = np.nan

    Vp = np.nan if abs(V_cross) >= VP_MAX_V else V_cross

    # Drawn in one place, after every fit exists: a raise in section 5 would
    # otherwise leave a half-filled figure open that analyze_IV_safe swallows.
    if plot:
        _plot_iv_diagnostics(V, I_raw, I_baseline, I_sub, isat_idx, V_fit, I_fit,
                             lower_idx, upper_idx, popt, c_trans, d_esat,
                             trans_voltage, trans_current, esat_volt, esat_curr,
                             Vp, Te_unclamped, label)

    # ==========================================
    # 6. Density
    # ==========================================
    # Calibrated path keeps Te out: the absolute scale comes from the
    # interferometer, and dividing by a per-shot vth(Te) would only inject the
    # Te fit's scatter into the profile.
    ne = I_esat if calibrated else float(ne_from_esat(I_esat, Te))

    return (Vp, Te, ne)

def analyze_IV_safe(voltage, current, file_name="", verbose=False, calibrated=True):
    """
    Wrapper function to safely execute analyze_IV.
    Catches any fitting errors or data quality exceptions, logs them,
    and returns NaNs to prevent the batch loop from crashing.

    ``calibrated`` selects the ``ne`` convention -- see :func:`analyze_IV`.
    """
    try:
        return analyze_IV(voltage, current, calibrated=calibrated)

    except Exception as e:
        if verbose:
            print(f"[{file_name}] Analysis failed: {e}")

        # Return NaNs so the main loop can store them and safely move on
        return np.nan, np.nan, np.nan


def show_iv_fit(V_trace, I_trace, label="", calibrated=True):
    """One trace -> ``analyze_IV``'s diagnostic panels in a GUI window.

    Returns ``(Vp, Te, ne)``.  Takes arrays, not a path: the sweep-npz layout is
    campaign-specific (see :func:`load_sweep_trace`), the panels are not.

    Switches to an interactive matplotlib backend, so call it only from an
    interactive session -- never on the batch path, which must stay headless.
    """
    import matplotlib
    # Agg is both the headless default and a deliberate batch choice --
    # indistinguishable from here -- so switch and say so rather than drawing a
    # figure that silently goes nowhere.
    if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
        print(f"  (show_iv_fit: matplotlib backend "
              f"{matplotlib.get_backend()} cannot display; switching to qtagg)")
        matplotlib.use("qtagg")

    Vp, Te, ne = analyze_IV(V_trace, I_trace, plot=True, calibrated=calibrated,
                            label=label)
    print(f"Vp = {Vp:.3f} V | Te = {Te:.3f} eV | ne = {ne:.4g}")
    return Vp, Te, ne


#=== batch pipeline: campaign-independent sweep processing + npz I/O =========
# The raw read (which board/scope/channel, scaling) stays in each experiment
# script; everything from sweep detection onwards is shared here.

def prepare_sweep_data(tarr, Vswp_arr, Iswp_arr, padding=10, trim_percent=10,
                       smooth_sigma=10):
    """Detect sweeps in the voltage trace, reshape both arrays, smooth the current.

    ``find_sweep_indices`` on the first position's voltage -> per-sweep middle
    timestamps -> ``reshape_IV`` -> Gaussian smoothing of the current along the
    sample axis.  ``Vswp_arr`` is ``(npos, nsamples)`` (shot-averaged) and
    ``Iswp_arr`` is ``(npos, nshot, nsamples)``.

    """
    start_t_ls, stop_t_ls = find_sweep_indices(Vswp_arr[0], padding=padding)

    mid_indices = [(start + stop) // 2 for start, stop in zip(start_t_ls, stop_t_ls)]
    data_timestamp = tarr[mid_indices]
    if not np.all(np.diff(data_timestamp) > 0):
        raise ValueError(
            "data_timestamp is not strictly increasing -- non-monotonic tarr, "
            "or sweep detection emitted duplicate/out-of-order windows.")
    sweep_t_start = tarr[start_t_ls]
    sweep_t_stop = tarr[stop_t_ls]
    print(f"Number of sweeps: {len(data_timestamp)}")

    Vswp_arr_rs, Iswp_arr_rs = reshape_IV(Vswp_arr, Iswp_arr,
                                          start_t_ls, stop_t_ls, trim_percent)

    print("Applying smoothing to current array...")
    Iswp_arr_rs = gaussian_filter1d(Iswp_arr_rs, smooth_sigma, axis=-1)
    return Vswp_arr_rs, Iswp_arr_rs, data_timestamp, sweep_t_start, sweep_t_stop


def mean_sem(vals):
    """Mean and standard error of the valid (non-NaN) entries of ``vals``.

    Returns ``(nan, nan)`` when nothing is valid and a ``nan`` SEM for a single
    sample (no spread to estimate).  Shared by the Vp/Te/ne averaging in
    :func:`process_iv_and_save` so the three quantities are reduced identically.
    """
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan, np.nan
    sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
    return np.mean(vals), sem


def process_iv_and_save(voltage_data, current_data, save_path, calibrated=True):
    """
    Loops through the multi-dimensional Langmuir probe dataset, extracting
    plasma parameters. Averages the valid shots for each location/sweep combination,
    calculates the standard error, and outputs 2D arrays.
    Saves progress incrementally to prevent data loss.

    ``voltage_data`` is ``(n_locs, n_sweeps, nsamples)`` and ``current_data`` is
    ``(n_locs, n_shots, n_sweeps, nsamples)`` -- the reshaped arrays from
    :func:`prepare_sweep_data`.  Returns
    ``(Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err)``.

    ``calibrated`` selects the ``ne`` convention (:func:`analyze_IV`) and is
    saved into the npz as ``ne_is_proxy``, so loaders know the units of
    ``ne_arr`` without being told.
    """
    n_locs, n_shots, n_sweeps, _ = current_data.shape

    # Pre-allocate the (mean, err) output pair per quantity, NaN-filled
    # (locs, sweeps); everything downstream (averaging, incremental save,
    # return) is driven off this dict so adding a quantity is one key.
    arrs = {k: (np.full((n_locs, n_sweeps), np.nan),
                np.full((n_locs, n_sweeps), np.nan))
            for k in ("Vp", "Te", "ne")}

    total_traces = n_locs * n_shots * n_sweeps
    print(f"Starting batch processing of {total_traces} traces across {n_locs} locations...")
    print(f"Averaging {n_shots} shots per sweep...")

    start_time = time.time()
    fail_count = 0

    # Progress bar over locations; the per-trace fail rate is shown in the postfix
    # and refreshed each location.  tqdm provides the %, elapsed, ETA and rate.
    pbar = tqdm(range(n_locs), desc="Analyzing", unit="loc")
    for loc in pbar:
        for swp in range(n_sweeps):
            # The voltage trace applies to all shots at this location/sweep
            V_trace = voltage_data[loc, swp, :]

            # Per-shot results for this sweep, keyed like `arrs`.
            temp = {"Vp": [], "Te": [], "ne": []}
            for sht in range(n_shots):
                I_trace = current_data[loc, sht, swp, :]
                trace_id = f"Loc:{loc}|Shot:{sht}|Swp:{swp}"

                # Analyze trace
                Vp, Te, ne = analyze_IV_safe(V_trace, I_trace, file_name=trace_id,
                                             calibrated=calibrated)
                if np.isnan(ne):
                    fail_count += 1
                temp["Vp"].append(Vp)
                temp["Te"].append(Te)
                temp["ne"].append(ne)

            # Mean + standard error of the valid shots, identically for each quantity.
            for key, (arr, err) in arrs.items():
                arr[loc, swp], err[loc, swp] = mean_sem(temp[key])

        # Incremental save of all 6 arrays after every location.
        np.savez(save_path,
                 **{f"{k}_arr": arr for k, (arr, _) in arrs.items()},
                 **{f"{k}_err": err for k, (_, err) in arrs.items()},
                 ne_is_proxy=calibrated)

        # Live running fail rate alongside the progress bar
        traces_done = (loc + 1) * n_sweeps * n_shots
        pbar.set_postfix(fails=f"{fail_count} ({fail_count / traces_done * 100:.1f}%)")

    total_time = time.time() - start_time
    final_time_str = str(datetime.timedelta(seconds=int(total_time)))
    final_fail_rate = (fail_count / total_traces) * 100

    print("\n" + "=" * 55)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 55)
    print(f"Total Time:    {final_time_str}")
    print(f"Total Traces:  {total_traces}")
    print(f"Total Fails:   {fail_count}")
    print(f"Fail Rate:     {final_fail_rate:.2f}%")
    print(f"Data saved to: {save_path}")
    print("=" * 55)

    (Vp_arr, Vp_err), (Te_arr, Te_err), (ne_arr, ne_err) = (
        arrs[k] for k in ("Vp", "Te", "ne"))
    return Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err


def tip_tag(tip):
    """Filename fragment for a probe tip (``"-tipR"``); empty for the untagged
    single-tip case.  Used in the npz names (:func:`sweep_npz_paths`) and in
    figure names built from them."""
    return f"-tip{tip}" if tip else ""


def _require_npz_key(npz, key, path, hint):
    """Membership check for an expected npz key, with a uniform 'regenerate it'
    error.  ``npz`` is an open ``NpzFile`` (``in`` reads the zip directory, not
    the array); raises ``KeyError`` naming ``path`` and how to produce ``key``.
    """
    if key not in npz:
        raise KeyError(f"{path} has no {key!r} -- {hint}")


def sweep_npz_paths(data_dir, run_num, tip=None):
    """The saved-npz pair for one run/tip: ``(sweep_path, plasma_path)``.

    The one home of the ``<run>[-tip<T>]-sweep-data.npz`` /
    ``<run>[-tip<T>]-plasma-data.npz`` co-located filename convention (the npz
    sit beside the raw HDF5).  ``tip=None`` gives the untagged names used by
    single-probe campaigns.
    """
    tag = tip_tag(tip)
    return (os.path.join(data_dir, f"{run_num}{tag}-sweep-data.npz"),
            os.path.join(data_dir, f"{run_num}{tag}-plasma-data.npz"))


def load_sweep_data(data_dir, run_num, tip=None):
    """Load the reshaped sweep arrays + axes saved by the campaign's save step.

    Expects the single-current-array key layout (``Vswp_arr_rs`` /
    ``Iswp_arr_rs``); campaigns that store several current arrays per npz keep
    their own loader.
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(sweep_path) as data:
        return (data["Vswp_arr_rs"], data["Iswp_arr_rs"], data["data_timestamp"],
                data["xpos"], data["ypos"], int(data["npos"]), int(data["nshot"]))


def load_sweep_axes(data_dir, run_num, tip=None):
    """Just the position axes + shot layout from a saved sweep npz.

    ``np.load`` reads npz entries lazily, so this skips the multi-hundred-MB
    sweep arrays -- for plot drivers that only need ``(xpos, ypos, npos, nshot)``.
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(sweep_path) as data:
        return data["xpos"], data["ypos"], int(data["npos"]), int(data["nshot"])


def load_plasma_data(data_dir, run_num, tip=None):
    """Load saved plasma parameters + sweep timestamps for plotting.

    Returns ``(Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls)``.
    """
    sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(sweep_path) as data:
        t_ls = data["data_timestamp"]

    with np.load(plasma_path) as ps_data:
        return (ps_data["Vp_arr"], ps_data["Te_arr"], ps_data["ne_arr"],
                ps_data["Vp_err"], ps_data["Te_err"], ps_data["ne_err"], t_ls)


def load_sweep_trace(data_dir, run_num, loc, sweep, shot=0, tip=None,
                     current_key="Iswp_arr_rs"):
    """One trace out of a saved sweep npz: ``(V_trace, I_trace, x_cm, t_mid_ms)``.

    ``current_key`` names the npz's current array, defaulting to the
    single-current-array layout :func:`load_sweep_data` documents.  A campaign
    holding both tips in one npz passes its own key -- Mar-2026 writes
    ``IswpL_arr_rs`` / ``IswpR_arr_rs``, where the tip lives in the *key* and its
    files are untagged, so such a caller passes ``tip=None`` and supplies the tip
    to ``label`` itself.

    Reads the whole current array: ``np.load`` ignores ``mmap_mode`` for npz, and
    a deflated zip member cannot be sliced without inflating it anyway.  Fine for
    one interactive lookup, too slow to sit in a loop.
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(sweep_path) as data:
        _require_npz_key(data, current_key, sweep_path,
                         f"this npz holds {sorted(k for k in data.files if 'swp' in k)}; "
                         "pass current_key= for a per-tip layout")
        V_all, I_all = data["Vswp_arr_rs"], data[current_key]
        # numpy would raise on the index anyway; this names which axis and what
        # the layout is, which the bare IndexError does not.
        for axis, idx, n, shape_note in (
                ("loc", loc, V_all.shape[0], f"Vswp_arr_rs {V_all.shape} (npos, n_sweeps, nt)"),
                ("sweep", sweep, V_all.shape[1], f"Vswp_arr_rs {V_all.shape} (npos, n_sweeps, nt)"),
                ("shot", shot, I_all.shape[-3], f"{current_key} {I_all.shape} (npos, nshot, n_sweeps, nt)")):
            if not -n <= idx < n:
                raise IndexError(f"{axis}={idx} outside 0..{n-1} in "
                                 f"{sweep_path}: {shape_note}")
        # Iswp has a shot axis and Vswp does not: every position/sweep shares one
        # programmed voltage ramp, only the collected current differs per shot.
        return (np.asarray(V_all[loc, sweep, :]),
                np.asarray(I_all[loc, shot, sweep, :]),
                float(np.asarray(data["xpos"])[loc]),
                float(np.asarray(data["data_timestamp"])[sweep]) * 1e3)


def load_ne_calibrated(data_dir, run_num, tip=None):
    """The interferometer-calibrated density written by :func:`calibrate_plasma_npz`.

    Returns ``(ne_cal_arr (n_locs, n_sweeps) [cm^-3], cal_factor (n_sweeps,))``.
    Raises ``KeyError`` if the plasma npz has not been calibrated yet -- run
    :func:`calibrate_plasma_npz` first.
    """
    _, plasma_path = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(plasma_path) as ps_data:
        _require_npz_key(ps_data, "ne_cal_arr", plasma_path,
                         "run calibrate_plasma_npz(ifn, interf_chan) first.")
        return ps_data["ne_cal_arr"], ps_data["cal_factor"]


def load_ne(data_dir, run_num, raw_ne, tip=None, prefer_calibrated=True):
    """The ne to use for one run/tip, with its unit: ``(ne, is_density)``.

    ``prefer_calibrated`` True swaps in the interferometer-calibrated
    ``ne_cal_arr`` when :func:`calibrate_plasma_npz` has written it, falling
    back to ``raw_ne`` with a printed note otherwise.

    ``is_density`` reports the unit of what is *returned*, not what was asked
    for -- True means [cm^-3], False means :func:`analyze_IV`'s proxy [A/cm^2].
    So an uncalibrated run whose npz was written with ``calibrated=False``
    (``ne_arr`` already a density via :func:`ne_from_esat`) still reports True:
    the fallback is a real density, just not interferometer-scaled.
    """
    if prefer_calibrated:
        try:
            return load_ne_calibrated(data_dir, run_num, tip=tip)[0], True
        except KeyError:
            print(f"  (ne: run {run_num}{tip_tag(tip)} not calibrated yet -- "
                  "using raw ne; run calibrate_plasma_npz to calibrate)")
    return raw_ne, not ne_arr_is_proxy(data_dir, run_num, tip=tip)


def ne_arr_is_proxy(data_dir, run_num, tip=None):
    """Is this run's saved ``ne_arr`` the [A/cm^2] proxy rather than a density?

    Reads the ``ne_is_proxy`` flag :func:`process_iv_and_save` writes.  npz from
    before that flag existed are proxies (it was the only convention then), so a
    missing key reads as True.
    """
    _, plasma_path = sweep_npz_paths(data_dir, run_num, tip)
    with np.load(plasma_path) as ps:
        return bool(ps["ne_is_proxy"]) if "ne_is_proxy" in ps else True


def interferometer_calibration(profile_arr, x, t_start, t_stop, interf_t,
                               interf_ne_line, t_offset=0.0):
    """Per-sweep calibration of a probe profile against interferometer density.

    Programmatic version of the legacy recipe (LP_analysis Langmuir_Iisat.ipynb):
    line-average the probe profile along the interferometer chord and ratio it
    against the interferometer line-averaged ne.  Works for any probe quantity
    proportional to ne -- the IV-derived ne proxy from :func:`analyze_IV` or
    raw Isat (both [A/cm^2]; the factor converts to cm^-3).

    profile_arr : (n_locs, n_sweeps) probe quantity proportional to ne
    x           : (n_locs,) positions [cm] along the interferometer chord
    t_start, t_stop : (n_sweeps,) sweep time windows [s]
    interf_t    : (nt,) interferometer time [s] (file stores ms -- caller converts)
    interf_ne_line : (nt,) line-averaged ne [cm^-3], already shot-averaged
    t_offset    : scope trigger time relative to the interferometer's t=0
                  (plasma breakdown), [s].  Added to ``t_start``/``t_stop``
                  here to express the sweep windows on the interferometer time
                  base; the caller's scope-timed arrays stay untouched.

    For each sweep ``k``::

        factor[k] = mean(interf_ne_line over the window
                         [t_start[k], t_stop[k]] + t_offset)
                    / line_average(profile_arr[:, k], x)

    A sweep with an empty window, or whose line average is not finite and
    positive, gets a ``nan`` factor (no exception), and the count of calibrated
    sweeps is printed.  Failed-fit ``nan`` positions are excluded from the line
    average, not treated as zeros.  But
    if *no* sweep window overlaps the interferometer trace at all -- the usual
    symptom of a wrong ``t_offset`` -- this raises ``ValueError`` rather than
    silently returning an all-``nan`` calibration.  Returns
    ``(factor, profile_arr * factor, chord_avg)`` -- per-sweep factors
    ``(n_sweeps,)`` broadcast across locations, plus the probe chord averages
    ``(n_sweeps,)`` used in the ratio (for plotting against the interferometer
    trace).

    Caveats: the probe chord average only spans the measured ``x`` range while
    the interferometer averages its full beam path (through the 40 cm plasma
    length baked into the phase->ne factor); any trigger-time difference
    between the two diagnostics must be supplied via ``t_offset``; and the
    merged interferometer traces come from only the first/last shots of the
    run, so plasma conditions are assumed stationary across it.
    """
    profile_arr = np.asarray(profile_arr, dtype=float)
    interf_t = np.asarray(interf_t, dtype=float)
    interf_ne_line = np.asarray(interf_ne_line, dtype=float)
    t_start = np.asarray(t_start, dtype=float) + t_offset
    t_stop = np.asarray(t_stop, dtype=float) + t_offset

    n_sweeps = profile_arr.shape[1]
    factor = np.full(n_sweeps, np.nan)
    # NaN positions are failed fits; line_average integrates the finite subset,
    # so they are excluded rather than pulling the chord average toward 0.
    chord_avg = np.array([line_average(profile_arr[:, k], x) for k in range(n_sweeps)])
    n_hit = 0
    for k in range(n_sweeps):
        in_win = (interf_t >= t_start[k]) & (interf_t <= t_stop[k])
        # > 0, not != 0: a non-positive chord average is unphysical for a
        # density-proportional quantity, and a negative one would flip the sign
        # of the whole calibrated profile.
        if in_win.any() and np.isfinite(chord_avg[k]) and chord_avg[k] > 0:
            factor[k] = np.nanmean(interf_ne_line[in_win]) / chord_avg[k]
            n_hit += 1
    if n_hit == 0:
        raise ValueError(
            f"No sweep window overlapped the interferometer trace: sweeps span "
            f"{t_start.min():.4g}..{t_stop.max():.4g} s (after t_offset={t_offset:g}), "
            f"interferometer spans {interf_t.min():.4g}..{interf_t.max():.4g} s. "
            "Check t_offset (interferometer t=0 is plasma breakdown).")
    print(f"Calibrated {n_hit}/{n_sweeps} sweeps ({n_sweeps - n_hit} had no "
          "interferometer overlap or a non-finite chord average).")
    return factor, profile_arr * factor, chord_avg


def calibrate_plasma_npz(ifn, interf_chan, tip=None, t_offset=0.0):
    """Calibrate a run's batch IV density against the interferometer, in place.

    Wires the three pieces together for one run + tip: loads the batch
    ``ne_arr`` and probe positions (from the co-located sweep/plasma npz written
    by :func:`process_iv_and_save`), reads the merged interferometer chord
    ``interf_chan`` (:func:`data_analysis.io.read_interferometer`), runs
    :func:`interferometer_calibration` over each sweep's true time window, and
    **writes the calibrated density back into the plasma npz** so downstream
    line-data analysis loads an absolutely-scaled ``ne`` without recomputing.

    ``ifn``          : raw datarun HDF5 (its directory + run number locate the npz).
    ``interf_chan``  : interferometer channel to calibrate against, e.g. ``"phase_p29"``.
    ``tip``          : probe tip whose npz pair to calibrate (``None`` = untagged).
    ``t_offset``     : scope-vs-interferometer trigger offset [s], forwarded to
                       :func:`interferometer_calibration` (interferometer t=0 is
                       plasma breakdown; the scope's sweep windows are shifted
                       onto that base, the stored arrays are not).

    The plasma npz gains two keys alongside the existing ``*_arr``/``*_err``:
    ``ne_cal_arr`` (``(n_locs, n_sweeps)`` calibrated density [cm^-3]) and
    ``cal_factor`` (``(n_sweeps,)`` per-sweep factor; ``nan`` where a sweep
    window caught no interferometer samples or the probe chord average was
    non-finite).  Raises ``ValueError`` if ``interf_chan`` has no usable shots,
    or (via :func:`interferometer_calibration`) if *no* sweep window overlaps
    the interferometer trace -- the usual sign of a wrong ``t_offset``.  The 6
    original arrays are re-saved unchanged.  Returns
    ``(cal_factor, ne_cal_arr, chord_avg)``.

    Requires the sweep npz to carry ``sweep_t_start``/``sweep_t_stop`` (added by
    :func:`prepare_sweep_data`); npz written before that must be regenerated
    with :func:`process_iv_and_save`'s driver.
    """
    from data_analysis.io import read_interferometer
    from data_analysis.utils import run_num_of

    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)
    sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num, tip)

    with np.load(sweep_path) as sw:
        xpos = sw["xpos"]
        _require_npz_key(sw, "sweep_t_start", sweep_path,
                         "regenerate it (prepare_sweep_data now saves the "
                         "per-sweep windows sweep_t_start/sweep_t_stop).")
        t_start, t_stop = sw["sweep_t_start"], sw["sweep_t_stop"]

    with np.load(plasma_path) as ps:
        saved = dict(ps)             # keep the 6 arrays to re-save alongside
    ne_arr = saved["ne_arr"]
    if not bool(saved.get("ne_is_proxy", True)):
        raise ValueError(
            f"{plasma_path} was written with calibrated=False -- ne_arr is "
            "already a density [cm^-3] from ne_from_esat, so scaling it against "
            "the interferometer would double-apply a density scale. Regenerate "
            "with process_iv_and_save(..., calibrated=True) to calibrate.")

    ch = read_interferometer(ifn, channels=[interf_chan])[interf_chan]
    if ch.phase.shape[0] == 0:
        raise ValueError(
            f"interferometer channel {interf_chan!r} in {ifn} has no usable "
            f"shots (all skipped: {ch.skipped}); cannot calibrate.")
    if ch.phase.shape[0] >= 2:
        # First/last merged shots bracket the run; a large gap flags that the
        # stationarity assumption behind the two-shot average is shaky.
        spread = np.nanmax(np.abs(ch.ne_line_cm3[0] - ch.ne_line_cm3[-1]))
        print(f"  interferometer first/last-shot line-ne differ by up to "
              f"{spread:.3g} cm^-3 (two-shot stationarity assumption).")

    factor, ne_cal_arr, chord_avg = interferometer_calibration(
        ne_arr, xpos, t_start, t_stop,
        ch.t_ms * 1e-3, ch.ne_line_avg_cm3(), t_offset=t_offset)

    np.savez(plasma_path, **saved, ne_cal_arr=ne_cal_arr, cal_factor=factor)
    print(f"Calibrated ne against {interf_chan} -> {plasma_path}")
    return factor, ne_cal_arr, chord_avg
