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

``process_sweep_run`` at the bottom is the campaign-agnostic driver: a campaign
script parses its own file format into ``SweepRecord``s -- which board/scope
channel, which resistor and probe area -- and everything after (sweep detection,
batch ``analyze_IV``, the npz layout, interferometer calibration) happens here,
so it cannot drift between campaigns.

One npz pair per run holds every channel, each under a ``<prefix>/`` key
namespace listed in ``channels`` and versioned by ``schema``.  ``ne`` is always
a density [cm^-3]; the ``calibrated`` flag says whether an interferometer scale
was applied, and nothing else changes meaning.

Authors: Jia Han (orig. 2018), Google Gemini (IV analyzer, 2026)
"""

import datetime
import math
import os
import time
import warnings
from contextlib import contextmanager
from typing import NamedTuple

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


def analyze_IV(voltage, current, plot=False, label=""):
    """Analyze one IV curve; returns ``(Vp [V], Te [eV], ne [cm^-3])``.

    ``ne`` is always a density (:func:`ne_from_esat`), never the raw Esat
    current density -- interferometer calibration scales this by a dimensionless
    per-sweep factor, so both calibrated and uncalibrated runs carry the same
    unit and can share an axis.  Uncalibrated, the absolute scale is only as
    good as ``Aprobe`` and the vth convention: see :func:`ne_from_esat`.

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
    # Dividing Esat by vth(Te) injects the Te fit's scatter into ne, but a
    # density is what every caller wants and what interferometer calibration
    # scales; storing bare I_esat here is what used to make ne_arr's unit
    # depend on whether calibration had run yet.
    ne = float(ne_from_esat(I_esat, Te))

    return (Vp, Te, ne)

def analyze_IV_safe(voltage, current, file_name="", verbose=False):
    """
    Wrapper function to safely execute analyze_IV.
    Catches any fitting errors or data quality exceptions, logs them,
    and returns NaNs to prevent the batch loop from crashing.
    """
    try:
        return analyze_IV(voltage, current)

    except Exception as e:
        if verbose:
            print(f"[{file_name}] Analysis failed: {e}")

        # Return NaNs so the main loop can store them and safely move on
        return np.nan, np.nan, np.nan


def show_iv_fit(V_trace, I_trace, label=""):
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

    Vp, Te, ne = analyze_IV(V_trace, I_trace, plot=True, label=label)
    print(f"Vp = {Vp:.3f} V | Te = {Te:.3f} eV | ne = {ne:.4g} cm^-3")
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


def process_iv_and_save(voltage_data, current_data, save_path, prefix):
    """
    Loops through the multi-dimensional Langmuir probe dataset, extracting
    plasma parameters. Averages the valid shots for each location/sweep combination,
    calculates the standard error, and outputs 2D arrays.
    Saves progress incrementally to prevent data loss.

    ``voltage_data`` is ``(n_locs, n_sweeps, nsamples)`` and ``current_data`` is
    ``(n_locs, n_shots, n_sweeps, nsamples)`` -- the reshaped arrays from
    :func:`prepare_sweep_data`.  Returns
    ``(Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err)``.

    ``prefix`` is the channel's npz key namespace; every array is written under
    ``<prefix>/``, so one file holds every channel of a run.  ``ne_arr`` is a
    density [cm^-3], saved again as ``<prefix>/ne_uncal_arr`` (with
    ``ne_uncal_err``) so :func:`calibrate_plasma_npz` always scales from the
    unscaled pair and stays idempotent.
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
                Vp, Te, ne = analyze_IV_safe(V_trace, I_trace, file_name=trace_id)
                if np.isnan(ne):
                    fail_count += 1
                temp["Vp"].append(Vp)
                temp["Te"].append(Te)
                temp["ne"].append(ne)

            # Mean + standard error of the valid shots, identically for each quantity.
            for key, (arr, err) in arrs.items():
                arr[loc, swp], err[loc, swp] = mean_sem(temp[key])

        # Incremental save after every location.  Every channel of a run shares
        # this file, so the other channels' keys must be carried through: a bare
        # savez here would delete them once per location.  Cheap -- plasma npz
        # hold six small arrays per channel.
        # .copy(): these are the arrays the loop keeps filling, so storing them
        # by reference would leave ne_uncal_* tracking ne_* -- and an
        # interrupted run would later calibrate a half-filled profile.
        np.savez(save_path, schema=SCHEMA_VERSION,
                 **_other_channel_keys(save_path, prefix),
                 **{f"{prefix}/{k}_arr": arr for k, (arr, _) in arrs.items()},
                 **{f"{prefix}/{k}_err": err for k, (_, err) in arrs.items()},
                 **{f"{prefix}/ne_uncal_arr": arrs["ne"][0].copy(),
                    f"{prefix}/ne_uncal_err": arrs["ne"][1].copy(),
                    f"{prefix}/calibrated": False})

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


#: npz layout version.  Bumped when the key layout changes in a way older
#: readers would misinterpret rather than fail on; loaders refuse anything else.
SCHEMA_VERSION = 3

#: Run-level npz keys, owned by the writer rather than by any one channel.
#: Excluded from the carry-through so a re-save cannot resurrect a stale value
#: (and cannot collide with the keyword the writer passes).
_RUN_LEVEL_KEYS = ("schema", "channels")

#: Channel label when neither probe port nor tip is known (the campaign's
#: I_CHAN/V_CHAN override).  A real tip label never collides with it.
OVERRIDE_PREFIX = "override"


def channel_prefix(port, tip):
    """One channel's identity, and its npz key namespace: ``'P29-R'``.

    The single owner of the prefix format, because :func:`resolve_prefix`
    depends on the ``-<tip>`` suffix to resolve a bare tip label -- a campaign
    spelling it out itself would drift from the matcher.  Falls back to the bare
    tip when the file names no port, and to :data:`OVERRIDE_PREFIX` when it
    names neither: a prefix is never empty, since every npz key is
    ``<prefix>/<name>``.
    """
    if not tip:
        return OVERRIDE_PREFIX
    return f"P{port}-{tip}" if port else tip


def match_tip(candidates, tip, where):
    """Resolve ``tip`` against ``candidates``: ``'R'`` -> ``'P29-R'``.

    ``candidates`` is channel labels, or any container keyed by them (callers
    pass both a list and a channel-name -> channels dict).

    An exact label wins; otherwise a bare tip resolves when exactly one probe
    carries it.  Raises when several do -- two probes each with an R make
    ``tip='R'`` genuinely ambiguous, and picking either would pair a profile
    with the wrong probe.  ``where`` names the file or run in that error.
    """
    if tip in candidates:
        return tip
    matches = [c for c in candidates if c.endswith(f"-{tip}")]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"tip {tip!r} " + (f"matches {matches} in" if matches else "is not in")
        + f" {where}; channels are {sorted(candidates)}.")


def _other_channel_keys(path, prefix):
    """Every *channel* array in ``path`` that does NOT belong to ``prefix``.

    The read half of the read-modify-write that lets several channels share one
    npz: ``np.savez`` replaces a file wholesale, so a writer must carry the
    other channels' keys through or destroy them.  Empty when the file does not
    exist yet (the first channel of a run).
    """
    if not os.path.exists(path):
        return {}
    with np.load(path, allow_pickle=False) as prev:
        return {k: prev[k] for k in prev.files
                if not k.startswith(f"{prefix}/") and k not in _RUN_LEVEL_KEYS}


def sweep_npz_paths(data_dir, run_num):
    """The saved-npz pair for one run: ``(sweep_path, plasma_path)``.

    The one home of the ``<run>-sweep-data.npz`` / ``<run>-plasma-data.npz``
    co-located filename convention (the npz sit beside the raw HDF5).  One pair
    per run regardless of how many probe tips it recorded: the tip lives in the
    npz keys, not the filename.
    """
    return (os.path.join(data_dir, f"{run_num}-sweep-data.npz"),
            os.path.join(data_dir, f"{run_num}-plasma-data.npz"))


def discover_channels(ifn):
    """The channel labels a run has saved sweep data for: ``['P29-L', 'P29-R']``.

    Read from the sweep npz's ``channels`` index, not from filenames or by
    re-running channel discovery against the raw multi-GB HDF5: the saved file
    *is* what can be plotted, so a tip whose processing failed or was skipped is
    simply absent.

    Raises ``FileNotFoundError`` when the run has no sweep npz at all, so a
    batch caller can skip that run and *name* it rather than failing opaquely.
    """
    from data_analysis.utils import run_num_of

    data_dir, run_num = os.path.dirname(ifn), run_num_of(ifn)
    sweep_path, _ = sweep_npz_paths(data_dir, run_num)
    if not os.path.isfile(sweep_path):
        raise FileNotFoundError(
            f"no saved sweep npz for run {run_num} in {data_dir}; run the "
            "campaign's process_run(ifn) first")
    with np.load(sweep_path) as d:
        _check_schema(d, sweep_path)
        return [str(c) for c in d["channels"]]


def _check_schema(npz, path):
    """Refuse an npz written before :data:`SCHEMA_VERSION`.

    Pre-v3 files store one channel's arrays at the top level, so a reader would
    find ``Vswp_arr_rs`` and return one tip's data as if it were the run's --
    wrong rather than absent.  Fail by name instead.
    """
    found = int(npz["schema"]) if "schema" in npz else None
    if found != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is npz schema {found if found else '<3 (untagged)'}, not "
            f"{SCHEMA_VERSION}: regenerate it with the campaign's process_run "
            "-- older files hold one tip per file and would be misread as the "
            "whole run.")


def resolve_only(channels, path):
    """The sole channel in ``channels``; raises when a run has several.

    ``tip=None`` has no sensible default on a multi-channel run: picking one
    would silently return a different probe's profile than the caller meant.
    """
    if len(channels) != 1:
        raise ValueError(f"{path} holds {channels}; pass tip= to choose one.")
    return channels[0]


def resolve_prefix(npz, tip, path):
    """The key prefix for ``tip`` in an open sweep npz: ``'P29-R'``.

    Reads the run-level ``channels`` index, so it sees every channel the run
    recorded -- including one whose analysis has not been written yet.
    """
    channels = [str(c) for c in npz["channels"]]
    return (resolve_only(channels, path) if tip is None
            else match_tip(channels, tip, path))


@contextmanager
def open_channel(path, tip):
    """Open an npz and resolve ``tip``: yields ``(npz, prefix)``.

    The one place open + schema check + prefix resolution happen together, so no
    loader can skip a step -- ``np.load`` reads entries lazily, so this is cheap
    even on the multi-hundred-MB sweep npz.
    """
    with np.load(path) as d:
        _check_schema(d, path)
        yield d, resolve_prefix(d, tip, path)


def load_sweep_data(data_dir, run_num, tip=None):
    """Load one channel's reshaped sweep arrays + axes from the run's sweep npz.

    Returns ``(Vswp_arr_rs, Iswp_arr_rs, data_timestamp, xpos, ypos, npos,
    nshot)`` for ``tip`` (see :func:`resolve_prefix`).
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num)
    with open_channel(sweep_path, tip) as (d, p):
        return (d[f"{p}/Vswp_arr_rs"], d[f"{p}/Iswp_arr_rs"],
                d[f"{p}/data_timestamp"], d[f"{p}/xpos"], d[f"{p}/ypos"],
                int(d[f"{p}/npos"]), int(d[f"{p}/nshot"]))


def load_sweep_axes(data_dir, run_num, tip=None):
    """Just one channel's position axes + shot layout from the run's sweep npz.

    ``np.load`` reads npz entries lazily, so this skips the multi-hundred-MB
    sweep arrays -- for plot drivers that only need ``(xpos, ypos, npos, nshot)``.
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num)
    with open_channel(sweep_path, tip) as (d, p):
        return (d[f"{p}/xpos"], d[f"{p}/ypos"],
                int(d[f"{p}/npos"]), int(d[f"{p}/nshot"]))


def load_plasma_data(data_dir, run_num, tip=None):
    """Load one channel's plasma parameters + sweep timestamps for plotting.

    Returns ``(Vp_arr, Te_arr, ne_arr, Vp_err, Te_err, ne_err, t_ls)``.
    ``ne_arr`` is a density [cm^-3] whether or not the run was calibrated
    against the interferometer; read ``<prefix>/calibrated`` from the plasma npz
    to tell which (:func:`ne_is_calibrated`).
    """
    sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num)
    with open_channel(sweep_path, tip) as (d, p):
        t_ls = d[f"{p}/data_timestamp"]

    with np.load(plasma_path) as ps:
        _check_schema(ps, plasma_path)
        return (ps[f"{p}/Vp_arr"], ps[f"{p}/Te_arr"], ps[f"{p}/ne_arr"],
                ps[f"{p}/Vp_err"], ps[f"{p}/Te_err"], ps[f"{p}/ne_err"], t_ls)


def ne_is_calibrated(data_dir, run_num, tip=None):
    """Was this channel's ``ne_arr`` scaled against the interferometer?

    Both cases are a density [cm^-3]; uncalibrated means the absolute scale
    rests on ``Aprobe`` and the vth convention (:func:`ne_from_esat`) rather
    than on a measured line density.

    Reads the plasma npz alone: the flag is written beside the ne it describes
    (``process_iv_and_save``, flipped by :func:`calibrate_plasma_npz`), so this
    must not fail on a run whose large sweep npz was pruned.
    """
    _, plasma_path = sweep_npz_paths(data_dir, run_num)
    with np.load(plasma_path) as ps:
        _check_schema(ps, plasma_path)
        channels = sorted({k.split("/")[0] for k in ps.files if "/" in k})
        p = match_tip(channels, tip, plasma_path) if tip is not None \
            else resolve_only(channels, plasma_path)
        return bool(ps[f"{p}/calibrated"])


def load_sweep_trace(data_dir, run_num, loc, sweep, shot=0, tip=None):
    """One trace out of a saved sweep npz: ``(V_trace, I_trace, x_cm, t_mid_ms)``.

    Reads the whole current array: ``np.load`` ignores ``mmap_mode`` for npz, and
    a deflated zip member cannot be sliced without inflating it anyway.  Fine for
    one interactive lookup, too slow to sit in a loop.
    """
    sweep_path, _ = sweep_npz_paths(data_dir, run_num)
    with open_channel(sweep_path, tip) as (data, p):
        V_all, I_all = data[f"{p}/Vswp_arr_rs"], data[f"{p}/Iswp_arr_rs"]
        # numpy would raise on the index anyway; this names which axis and what
        # the layout is, which the bare IndexError does not.
        v_note = f"{p}/Vswp_arr_rs {V_all.shape} (npos, n_sweeps, nt)"
        for axis, idx, n, shape_note in (
                ("loc", loc, V_all.shape[0], v_note),
                ("sweep", sweep, V_all.shape[1], v_note),
                ("shot", shot, I_all.shape[-3],
                 f"{p}/Iswp_arr_rs {I_all.shape} (npos, nshot, n_sweeps, nt)")):
            if not -n <= idx < n:
                raise IndexError(f"{axis}={idx} outside 0..{n-1} in "
                                 f"{sweep_path}: {shape_note}")
        # Iswp has a shot axis and Vswp does not: every position/sweep shares one
        # programmed voltage ramp, only the collected current differs per shot.
        return (np.asarray(V_all[loc, sweep, :]),
                np.asarray(I_all[loc, shot, sweep, :]),
                float(np.asarray(data[f"{p}/xpos"])[loc]),
                float(np.asarray(data[f"{p}/data_timestamp"])[sweep]) * 1e3)


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
    ``tip``          : probe tip whose channel to calibrate (see :func:`resolve_prefix`).
    ``t_offset``     : scope-vs-interferometer trigger offset [s], forwarded to
                       :func:`interferometer_calibration` (interferometer t=0 is
                       plasma breakdown; the scope's sweep windows are shifted
                       onto that base, the stored arrays are not).

    Writes ``<prefix>/ne_arr`` and ``<prefix>/ne_err`` (both scaled by the same
    per-sweep factor), ``<prefix>/cal_factor`` (``(n_sweeps,)``; ``nan`` where a
    sweep window caught no interferometer samples or the probe chord average was
    non-finite), ``<prefix>/cal_chord`` and ``<prefix>/calibrated = True``;
    every other channel's keys are carried through unchanged.  Raises
    ``ValueError`` if ``interf_chan`` has no usable shots, or (via
    :func:`interferometer_calibration`) if *no* sweep window overlaps the
    interferometer trace -- the usual sign of a wrong ``t_offset``.  Returns
    ``(cal_factor, ne_cal_arr, chord_avg)``.

    Idempotent: the scale is always applied to ``ne_uncal_arr`` /
    ``ne_uncal_err``, never to whatever ``ne_arr`` currently holds, so
    re-running with a different chord or ``t_offset`` replaces the calibration
    instead of compounding it.
    """
    from data_analysis.io import read_interferometer
    from data_analysis.utils import run_num_of

    data_dir = os.path.dirname(ifn)
    run_num = run_num_of(ifn)
    sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num)

    with open_channel(sweep_path, tip) as (sw, prefix):
        xpos = sw[f"{prefix}/xpos"]
        t_start = sw[f"{prefix}/sweep_t_start"]
        t_stop = sw[f"{prefix}/sweep_t_stop"]

    # One read of the plasma npz: this channel's keys are rewritten below, every
    # other channel's are carried through untouched.
    with np.load(plasma_path) as ps:
        _check_schema(ps, plasma_path)
        keep = {k: ps[k] for k in ps.files if k not in _RUN_LEVEL_KEYS}
    ne_uncal = keep[f"{prefix}/ne_uncal_arr"]
    ne_uncal_err = keep[f"{prefix}/ne_uncal_err"]

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
        ne_uncal, xpos, t_start, t_stop,
        ch.t_ms * 1e-3, ch.ne_line_avg_cm3(), t_offset=t_offset)

    keep.update({f"{prefix}/ne_arr": ne_cal_arr,
                 # The same per-sweep factor scales the error: ne_err annotates
                 # ne_arr, and leaving it at the uncalibrated scale would make
                 # it the error bar of a quantity ~1e12 times smaller.
                 f"{prefix}/ne_err": ne_uncal_err * factor,
                 f"{prefix}/cal_factor": factor,
                 f"{prefix}/cal_chord": np.str_(interf_chan),
                 f"{prefix}/calibrated": True})
    np.savez(plasma_path, schema=SCHEMA_VERSION, **keep)
    print(f"Calibrated {prefix} ne against {interf_chan} -> {plasma_path}")
    return factor, ne_cal_arr, chord_avg


#=== the campaign-agnostic driver ============================================
# One entry point every campaign calls.  A campaign script's job is to parse
# its own file format into SweepRecords -- which board/scope/channel, which
# resistor and probe area -- and nothing else; detection, analysis, storage and
# calibration all live here so they cannot drift between campaigns.

class SweepRecord(NamedTuple):
    """One probe tip's raw sweep data, in the units the shared pipeline expects.

    ``Vswp_arr`` [V] is ``(npos, nsamples)``, shot-averaged; ``Iswp_arr``
    [A/cm^2] is ``(npos, nshot, nsamples)``.  The current must already carry the
    campaign's resistor and probe-area division and its sign convention: that
    is the one place a campaign's electronics are allowed to matter.

    ``prefix`` is the npz key namespace and the channel's identity
    (``'P29-R'``).  ``port`` is the LAPD port the probe sat on, used to derive
    the interferometer chord; ``None`` when the file does not say, which forces
    an explicit chord at calibration time.
    """
    prefix: str
    tarr: np.ndarray
    Vswp_arr: np.ndarray
    Iswp_arr: np.ndarray
    xpos: np.ndarray
    ypos: np.ndarray
    npos: int
    nshot: int
    port: str | None = None
    #: Extra string metadata (I_chan / V_chan / motion_group), saved verbatim.
    meta: dict | None = None



def interferometer_chords(ifn):
    """The ``phase_*`` chords a run carries, or ``[]`` when it has none.

    Presence of merged interferometer data is what decides whether a run's ne
    can be calibrated, and it varies run to run *within* a campaign -- so this
    asks the file rather than the campaign.  Backend-independent: pydaq and
    bapsflib runs both carry the merged group.
    """
    import h5py
    with h5py.File(ifn, "r") as f:
        g = f.get("diagnostics/interferometer")
        return sorted(k for k in g if k.startswith("phase_")) if g else []


#: How far along the machine an interferometer chord may sit from the probe and
#: still measure the same plasma, in LAPD ports.  Ports are on a fixed axial
#: pitch, so a port-number difference is proportional to distance.  4 admits the
#: p29 chord for a probe at p33; beyond that the chord is sampling a different
#: axial region and calibrating against it would scale the profile by a density
#: the probe never saw.
MAX_CHORD_PORT_DISTANCE = 4


def chord_for_port(chords, port, ifn):
    """Nearest interferometer chord to ``port``: ``(chord_name, ports_away)``.

    Exact match wins; otherwise the closest chord within
    :data:`MAX_CHORD_PORT_DISTANCE`.  Raises naming the available chords when
    nothing is near enough -- an out-of-range chord measures a different part of
    the column, so scaling by it would be a fabricated absolute density rather
    than a slightly worse one.  Ties (a probe exactly between two chords) take
    the lower port, deterministically.
    """
    if port is None:
        raise ValueError(
            f"{ifn} names no port for this channel, so its interferometer chord "
            f"cannot be derived; available chords are {chords}. Pass interf_chan= "
            "explicitly, or set calibrate=False to store an uncalibrated density.")

    # Tie-break on the port *number*, not the chord name: 'phase_p9' sorts after
    # 'phase_p21' as a string, so a name sort would silently prefer the higher
    # port for a single-digit chord.
    by_distance = sorted((abs(int(p) - int(port)), int(p), c) for c in chords
                         if (p := c.removeprefix("phase_p")).isdigit())
    if not by_distance or by_distance[0][0] > MAX_CHORD_PORT_DISTANCE:
        raise ValueError(
            f"probe is on port {port} but {ifn} has no interferometer chord "
            f"within {MAX_CHORD_PORT_DISTANCE} ports; available chords are "
            f"{chords}. Pass interf_chan= to choose one explicitly, or set "
            "calibrate=False to store an uncalibrated density.")
    distance, _, chord = by_distance[0]
    return chord, distance


def process_sweep_run(ifn, records, t_offset=0.0, calibrate=None,
                      interf_chan=None, store_dtype=None, **prep_kwargs):
    """Detect, analyze, save and calibrate every channel of one run.

    ``records`` are :class:`SweepRecord` -- see there for the units contract.
    ``prep_kwargs`` (``padding`` / ``trim_percent`` / ``smooth_sigma``) forward
    to :func:`prepare_sweep_data`, which owns the defaults.

    ``calibrate`` ``None`` decides per run from the presence of interferometer
    data in ``ifn``; ``True`` forces it and **raises** where impossible rather
    than silently storing an uncalibrated density under a calibrated name;
    ``False`` skips it.  ``interf_chan`` overrides the per-channel chord that is
    otherwise derived from each probe's own port.

    ``store_dtype`` (e.g. ``np.float32``) casts the stored sweep arrays; ``None``
    keeps the source dtype.  Analysis always runs in the source precision --
    this affects what is written, not what is computed.

    Returns ``(sweep_path, plasma_path)``.

    Two phases, because they differ in cost by orders of magnitude: sweep
    preparation is seconds per channel and its arrays are 70 MB-1 GB, so every
    channel is prepared and the sweep npz written **once**; the per-trace
    analysis is the hours, so it loops per channel and checkpoints into the
    plasma npz after every position.
    """
    from data_analysis.utils import run_num_of

    data_dir = os.path.dirname(ifn)
    sweep_path, plasma_path = sweep_npz_paths(data_dir, run_num_of(ifn))

    chords = interferometer_chords(ifn)
    if calibrate is None:
        calibrate = bool(chords)
    elif calibrate and not chords:
        raise ValueError(
            f"CALIBRATE=True but {ifn} carries no diagnostics/interferometer "
            "group, so there is nothing to calibrate against. Merge the "
            "interferometer traces first, or set CALIBRATE=None/False.")

    # --- phase 1: prepare every channel, write the sweep npz once ------------
    # One savez for the whole run: these arrays are the large ones, and a
    # per-channel write would have to re-read and re-compress the file each time
    # to avoid deleting the channels already in it.
    cast = (lambda a: a.astype(store_dtype)) if store_dtype else (lambda a: a)
    prepared, store = {}, {"schema": SCHEMA_VERSION,
                           "channels": np.array([r.prefix for r in records])}
    for rec in records:
        print(f"\n--- preparing {rec.prefix}")
        V_rs, I_rs, t_mid, t_start, t_stop = prepare_sweep_data(
            rec.tarr, rec.Vswp_arr, rec.Iswp_arr, **prep_kwargs)
        prepared[rec.prefix] = (V_rs, I_rs)
        store.update({
            f"{rec.prefix}/Vswp_arr_rs": cast(V_rs),
            f"{rec.prefix}/Iswp_arr_rs": cast(I_rs),
            f"{rec.prefix}/data_timestamp": t_mid,
            f"{rec.prefix}/sweep_t_start": t_start,
            f"{rec.prefix}/sweep_t_stop": t_stop,
            f"{rec.prefix}/xpos": rec.xpos, f"{rec.prefix}/ypos": rec.ypos,
            f"{rec.prefix}/npos": rec.npos, f"{rec.prefix}/nshot": rec.nshot,
            f"{rec.prefix}/port": np.str_(rec.port or ""),
            **{f"{rec.prefix}/{k}": np.str_(v) for k, v in (rec.meta or {}).items()},
        })
    np.savez(sweep_path, **store)
    print(f"\nSaved sweep arrays for {len(records)} channel(s) -> {sweep_path}")
    # These are the run's largest arrays and phase 2 never reads them again: a
    # float32 store copy is ~0.5 GB per channel, and `records` still holds the
    # untrimmed originals.  Freeing here keeps peak RAM at roughly one channel
    # rather than 1.5x the channel count.
    store.clear()
    records = [rec._replace(tarr=None, Vswp_arr=None, Iswp_arr=None)
               for rec in records]

    # --- phase 2: analyze and calibrate each channel -------------------------
    for rec in records:
        print("\n" + "=" * 70)
        print(f"ANALYZING {rec.prefix}")
        print("=" * 70)
        # pop, not index: this channel's reshaped arrays are the largest thing
        # in memory and nothing reads them after its own analysis.
        V_rs, I_rs = prepared.pop(rec.prefix)
        process_iv_and_save(V_rs, I_rs, plasma_path, rec.prefix)
        del V_rs, I_rs

        if calibrate:
            if interf_chan:
                chan = interf_chan
            else:
                chan, away = chord_for_port(chords, rec.port, ifn)
                # A chord on a different port than the probe is legitimate but
                # worth seeing: it is the difference between "measured here" and
                # "measured a metre down the machine".
                print(f"  {rec.prefix}: probe on port {rec.port} -> {chan}"
                      + (f" ({away} ports away)" if away else " (same port)"))
            calibrate_plasma_npz(ifn, chan, tip=rec.prefix, t_offset=t_offset)

    return sweep_path, plasma_path
