# -*- coding: utf-8 -*-

"""
Generic functions for langmuir probe analysis

Originally Created on Wed Sep 19 11:26:21 2018
Copied from old repo LP_analysis

@author: Jia Han

1. Find linear fitting for Isat and subtract from data
2. Find exponential fitting to Isat
3. Find the crossing point between linear fitting of the transition region and Esat

Update Aug. 2024:
-- Removed area; all function now takes current/area as input instead of current
-- TODO: Check result has correct unit after removing area
"""

import math
import numpy as np
import warnings
from scipy.signal import savgol_filter, medfilt
from scipy.optimize import curve_fit
from scipy import integrate, interpolate, ndimage, constants, optimize
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


#===============================================================================================================================================

qe = constants.e # electron charge (C)
me = constants.electron_mass # electron mass (kg)
mi = constants.proton_mass # proton mass (kg)

epsilon = constants.epsilon_0 #permittivity (F/m)

#===============================================================================================================================================
def ion_sound_speed(Te, Ti, mi=mi):
	'''
	Compute ion sound speed in cm/s
	'''
	gamma = 5/3 # adiabatic index; monoatomic gas is 5/3
	cs = np.sqrt((qe * (Te + gamma*Ti)) / (mi))
	return cs*1e2

def collision(n):

	n0 = 3.3e20 #neutral density (Van der Waal calculation ~10^23)
	xs = 8.6e-20 #e-n collision cross section for elastic collision in m^-3
	vth = math.sqrt(qe*Te/(math.pi*me)) #electron thermal speed 
	ve = n0 * xs * vth 	# collision frequency (equation from NRL)
	sigma = n * qe**2 / (me * ve)

	return ve, sigma

def collision_ee(Te, ne):
	return 2.91 * 10**-6 * ne/Te**(3/2) * 10

def collision_en(Te, ng, tfn = r"C:\data\cross-section\num.txt"):
	'''
	return collision frequency from electron neutral collision cross section
	Cross section data comes from LXCAT
	Neutral gas density calculated from ideal gas law (see collision-Oct2019.xlsx)
	'''    
	
	data = np.loadtxt(tfn) # Read cross section info from txt file

	f = interpolate.interp1d(data[:,0], data[:,1]) # Interpolation return f(Te)
	sig = f(Te) # collision cross section in m^2
	mfp = 1/(ng * sig) # unit is m if ng is m^-3

	Vth = 4.19e5 * np.sqrt(Te) # Thermal velocity (unit:m/s, see NRL)

	return sig, mfp, Vth/mfp

def conductivity(ne, nu, w): # conductivity from e-n collision

	wpe = 5.64e4 * np.sqrt(ne) # ne in cm^-3, wpe in rad/sec

	nume = epsilon * wpe**2 * nu # in rad/sec
	deno = w**2 + nu**2

	return  nume / deno
#===============================================================================================================================================

def analyze_Isat(Iisat, cs): #Iisat in A/cm^2, cs in cm/s

	n = Iisat / (cs * qe) #Iisat = density*ion sound speed*q*area

	return n

def analyze_Esat(Iesat, area, Te): # Esat in mA, Te in eV 
	'''
	Iesat = A*ne*sqrt(Te/(2*pi*me))
	'''
	return 1.49e9 * Iesat /(area * np.sqrt(Te))

def exponential_func(x, a, b):
	return a * np.exp(b * x)

def analyze_IV(voltage, current, plot = False):
	"""
	Analyze the IV curve.
	Input:
		voltage: the probe sweep voltage in V
		current: the measured current from the probe in A
		plot: return three plots of the fitting
		value: prints Vp, Te, ne
	Output:
		Vp, Te, ne
	"""
	if plot:
		plt.figure()
		plt.plot(voltage,current, label='Original')

	# Take XX% of the current and fit a linear line
	dif1 = [(np.max(current) - np.min(current))*0.01 + np.min(current), (np.max(current) - np.min(current))*0 + np.min(current)]
	vals = np.argwhere(np.logical_and(current < dif1[0], current > dif1[1]))

	cropped_voltage = []
	cropped_current = []
	for i in range(0, len(vals)):
		idx = vals[i][0]
		cropped_voltage.append(voltage[idx])
		cropped_current.append(current[idx])

	c = np.polyfit(cropped_voltage, cropped_current, 1)
	y = c[0] * voltage + c[1]

	if plot:
		plt.plot(voltage,y, label='Isat Fit')

	current -= y
	if plot:
		plt.plot(voltage, current, label='Subtracted')

	# Define the portion of the signal to fit
	portion_start = int(len(current) * 0.1)  # Start at 20% of the signal
	portion_end = int(len(current) * 0.5)    # End at 80% of the signal
	Inew_cropped = current[portion_start:portion_end]
	Vnew_cropped = voltage[portion_start:portion_end]

	# Define initial guesses for exponential fitting
	initial_guesses = [
		[1, 0.1],
		[1, 0.01],
		[1, 0.001]
	]

	best_fit_params = None
	lowest_error = float('inf')

	# Perform exponential fitting for each initial guess
	for guess in initial_guesses:
		try:
			popt, _ = curve_fit(exponential_func, Vnew_cropped, Inew_cropped, p0=guess)
			fitted_curve = exponential_func(Vnew_cropped, *popt)
			error = np.sum((Inew_cropped - fitted_curve) ** 2)
			if error < lowest_error:
				lowest_error = error
				best_fit_params = popt
		except RuntimeError:
			continue

	if best_fit_params is None:
		raise Exception('No fitting function could be applied successfully.')

	if plot:
		plt.plot(voltage, exponential_func(voltage, *best_fit_params), label='Exponential Fit')
		plt.legend()
		plt.ylim(top = np.max(current) * 1.1, bottom = np.min(current) * 1.1)
		plt.xlabel('Voltage (V)')
		plt.ylabel('Current (A)')
		plt.show()
	
	# print(f"Exponential fit parameters: a={best_fit_params[0]}, b={best_fit_params[1]}")

	# Example output: Te calculation for exponential fit
	Te = 1 / best_fit_params[1]
	# print(f"Te = {Te:.2f} eV")
	# if Te > 10:
		# raise Exception('Te is very high')


	# Defines which region is the transition
	dif3 = (np.max(current) - np.min(current))* 6/10 + np.min(current) # Upper limit
	dif4 = (np.max(current) - np.min(current))*4/10 + np.min(current) # Lower limit

	lower_bound = np.argwhere(current > dif4)
	start_idx = lower_bound[0][0]
	upper_bound = np.argwhere(current < dif3)
	stop_idx = upper_bound[len(upper_bound)-1][0]

	trans_voltage = []
	trans_current = []
	for i in range(start_idx, stop_idx):
		trans_voltage.append(voltage[i])
		trans_current.append(current[i])
	c = np.polyfit(trans_voltage, trans_current, 1)
	y = c[0] * voltage + c[1]


	# Finds linear fitting to Esat
	dif5 = np.min(current) + (np.max(current) - np.min(current)) * 0.8
	esat_pos = np.argwhere(current > dif5)

	esat_volt = []
	esat_curr = []

	for i in esat_pos[:,0]:
		esat_volt.append(voltage[i])
		esat_curr.append(current[i])
	d = np.polyfit(esat_volt, esat_curr, 1)
	z = d[0] * voltage + d[1]

	if plot:
		plt.figure()
		plt.plot(voltage, current)
		plt.plot(voltage,y)
		plt.plot(esat_volt,esat_curr)
		plt.plot(voltage, z)

	# Find the crossing point of transition and Esat linear fit to produce ne and Vp
	Vp = abs((d[1]-c[1]) / (d[0] - c[0])) #plasma potential in V
	I = d[0] * Vp + d[1]                  #electron current in A


	if Te > 0:
		vth = math.sqrt(qe*Te / me)
								# electron thermal velocity in cm/s
		ne = I/(vth * qe)
	else:
		ne = 0
		raise Exception('Te is negative')

	# print ('Esat=%.2g'%(I), 'A/cm^2')
	# print ('ne=%.2g'%(ne*1e-6), 'cm^3')
	# print ('Plasma potential=%.2f'%(Vp), 'V \n')

	return (Vp, Te, ne)


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

def derivative(V, I, sigma=15, th=0.8, smth=True):
	'''
	sigma: smoothing factor for gaussian filter
	threshold: threshold for finding max dIdV
	'''
	max_value = np.max(I)
	threshold = th * max_value
	exceed_index = np.where(I > threshold)[0][0]

	dIdV = np.gradient(I, V)
	ss_dIdV = gaussian_filter1d(dIdV, sigma=sigma)
	max_ind = np.argmax(abs(ss_dIdV[:exceed_index])) # Find max dIdV before threshold

	if smth:
		return ss_dIdV, max_ind
	else:
		return dIdV, max_ind

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

	ne = np.trapz(f[:Vp_ndx_1], Vnew[:Vp_ndx_1])	# density = area under EEPF
	print('density = %.3e' %(ne))

	return Vnew[:Vp_ndx_1], f[:Vp_ndx_1], ne, Vp


#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================

if __name__ == '__main__':

	Iisat = 0.0003
	Te = 0.3

	ne, sigma = analyze_Isat(Iisat, area, Te)

	print('density: %.2e   sigma: %.2f' %(ne, sigma))