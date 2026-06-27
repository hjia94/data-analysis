'''
useful functions that calculates various plasma parameters
'''

import numpy as np
import math
from scipy import constants as const
from scipy import interpolate

#===========================================================================================================
#===========================================================================================================
qe = const.e # electron charge (C)
me = const.electron_mass # electron mass (kg)
mi = const.proton_mass # proton mass (kg)
epsilon = const.epsilon_0 #permittivity (F/m)
K_B = const.Boltzmann # Boltzmann constant (J/K)
#===========================================================================================================
#===========================================================================================================
def neutral_density(P_Torr: float, T_K: float) -> float:
	'''
	calculates neutral density of a given pressure
	return units: m^-3
	'''
	TORR_TO_PA = 133.322368 # 1 Torr = 133.322368 Pa
	return (P_Torr * TORR_TO_PA) / (K_B * T_K)

def ion_sound_speed(T_e, mu, gamma=5/3, Z=1):
    '''
    Compute ion sound speed in cm/s
    input:
    T_e: electron temperature in eV
    mu: ion mass ratio (mi/mp)
    gamma: adiabatic index; monoatomic gas is 5/3
    Z: ion charge state
	'''
    return 9.79e5 * np.sqrt(gamma * Z * T_e/mu)

def v_therm(T_i, T_e, mu=1):
    
    v_the = 4.19e7 * np.sqrt(T_e) # e thermal speed,
    v_thi = 9.79e5 * np.sqrt(T_i/mu)  # i thermal speed
    return v_thi, v_the

def eV_to_K(eV):
	'''
	converts electron volts to kelvin
	'''
	return eV * 11604.525

def collision_ee(Te, ne):
	return 2.91 * 10**-6 * ne/Te**(3/2) * 10

def collision_en(Te, ng, tfn = r"D:\data\cross-section\num.txt"):
	'''
	return collision frequency from electron neutral collision cross section
	Cross section data comes from LXCAT
	Neutral gas density calculated from ideal gas law (see collision-Oct2019.xlsx)
	'''    
	
	data = np.loadtxt(tfn) # Read cross section info from txt file

	f = interpolate.interp1d(data[:,0], data[:,1]) # Interpolation return f(Te)
	sig = f(Te) # collision cross section in m^2
	mfp = 1/(ng * sig) # unit is m if ng is m^-3

	return sig, mfp

def conductivity(ne, nu, w): # conductivity from e-n collision

	wpe = 5.64e4 * np.sqrt(ne) # ne in cm^-3, wpe in rad/sec

	nume = epsilon * wpe**2 * nu # in rad/sec
	deno = w**2 + nu**2

	return  nume / deno