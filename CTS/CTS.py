# -*- coding: utf-8 -*-

"""
Thomson scattering parameters calculation using plasmapy
reference: https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson.html
        Sheffield, Plasma Scattering of Electromagnetic Radiation 2nd ed.

Author: Jia Han
Date: 2025-03-19
"""

import numpy as np
from plasmapy.diagnostics import thomson
from astropy.units import Quantity
import astropy.units as astro_units
from scipy.signal import find_peaks
from scipy import integrate
from scipy.constants import *

#===========================================================================================================
r_0 = 2.8179403262e-13 # classical electron radius in cm
#===========================================================================================================

def compute_alpha(lambda_i, theta, n_e, T_e):
    """
    Compute alpha based on the given equation.

    Parameters:
    lambda_i (float): Wavelength in cm
    theta (float): Scattering angle in degrees
    n_e (float): Electron density in cm^-3
    T_e (float): Electron temperature in eV

    Returns:
    float: Value of alpha
    """
    theta_rad = np.radians(theta)  # Convert degrees to radians
    alpha = (1.08e-4 * lambda_i / np.sin(theta_rad / 2)) * np.sqrt(n_e / T_e)
    return alpha

def compute_theta(lambda_i, alpha, n_e, T_e):
    """
    Rearrange the alpha equation to solve for theta

    Parameters:
    lambda_i (float): Wavelength in cm
    alpha (float): Scattering parameter
    n_e (float): Electron density in cm^-3
    T_e (float): Electron temperature in eV

    Returns:
    float: Scattering angle in degrees
    """
    # Solve for sin(theta_rad / 2)
    sin_half_theta = (1.08e-4 * lambda_i * np.sqrt(n_e / T_e)) / alpha
    
    # Calculate theta_rad
    theta_rad = 2 * np.arcsin(sin_half_theta)
    
    # Convert radians to degrees
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg


def generate_spectral_density(probe_wavelength, T_e, T_i, n_e, scattering_angle=180, delta_lam=20, num_points=10000, ions="He+"):
    """
    Generate the spectral density function (Skw) for Thomson scattering.
    
    Parameters:
    -----------
    probe_wavelength: in nm
    delta_lam : Wavelength range to calculate around center wavelength
    scattering_angle : in degrees
    ne :  cm^-3
    T_e : eV
    T_i : eV
    ions : Ion species
    num_points : Number of points in wavelength array (default: 2000)
        
    Returns:
    --------
    Skw: spectral density (rad/s)
    """

    # Convert inputs to astropy units if they aren't already
    if not isinstance(probe_wavelength, Quantity):
        probe_wavelength = Quantity(probe_wavelength, astro_units.nm)
    if not isinstance(n_e, Quantity):
        n_e = Quantity(n_e, astro_units.cm**-3)
    if not isinstance(T_e, Quantity):
        T_e = Quantity(T_e, astro_units.eV)
    if not isinstance(T_i, Quantity):
        T_i = Quantity(T_i, astro_units.eV)
    
    # Define probe and scattering vectors
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(scattering_angle)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])
    
    # Generate wavelength range
    lambda_arr = Quantity(np.linspace(probe_wavelength.value - delta_lam, probe_wavelength.value + delta_lam, num_points), astro_units.nm)
    
    # Calculate spectral density
    alpha, Skw = thomson.spectral_density(
        lambda_arr,
        probe_wavelength,
        n_e,
        T_e=T_e,
        T_i=T_i,
        ions=ions,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
    )
    # Convert wavelength to angular frequency (omega)
    # omega = 2πc/λ where c is speed of light
    # Using astropy units to ensure dimensional consistency
    omega_arr = 2*np.pi*c/lambda_arr.to(astro_units.m)
    omega_in = 2*np.pi*c/probe_wavelength.to(astro_units.m)
    
    return alpha, omega_arr, omega_in, Skw

def power_ratio(n_e, omega_arr, Skw, scattering_angle=180, L=0.1):
    '''
    calculate the ratio scattered power/incident power
    n_e: electron density in cm^-3
    omega_arr: angular frequency array in rad/s
    Skw: spectral density in rad/s
    scattering_angle: scattering angle in degrees
    L: length of the plasma in cm
    '''
    
    # Convert scattering_angle to radians if it's not already
    scattering_angle_rad = np.radians(scattering_angle)
    
    # Ensure n_e has proper units if it's not already a Quantity
    if not isinstance(n_e, Quantity):
        n_e = Quantity(n_e, astro_units.cm**-3)
    
    # Calculate the function with proper unit handling
    func = r_0**2*L/(4*np.pi) * (1 + np.cos(scattering_angle_rad)**2) * n_e * Skw

    # Integrate over angular frequency
    result = integrate.simpson(func, omega_arr)
        
    return -result

def faraday_rotation_angle(freq_in, ne, B, L):
    """
    Computes the Faraday rotation angle in degrees using CGS units.
    
    Parameters:
    - freq_in : float or np.array
        Frequency of the wave in Hz
    - ne : float
        Electron density in cm^-3
    - B : float
        Magnetic field (parallel to propagation direction) in Gauss
    - L : float
        Length of plasma path in cm

    Returns:
    - theta_deg : float
        Faraday rotation angle in degrees
    """
    coeff = e**3 / (8*pi**2*epsilon_0*m_e**2*c**3)
    
    wavelength = c / freq_in

    # Compute angle in radians
    theta_rad = coeff * (wavelength ** 2) * ne*1e6 * B*1e-4 * L*1e2
    
    # Convert to degrees
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg