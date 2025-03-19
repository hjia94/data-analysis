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


def generate_spectral_density(probe_wavelength, T_e, T_i, n_e, scattering_angle=180, delta_lam=20, num_points=2000, ions="He+"):
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
    omega_arr = 2*np.pi*astro_units.rad*c/(lambda_arr*1e-9*astro_units.m)
    
    return alpha, omega_arr, Skw

def power_ratio(n_e, omega_arr, Skw, scattering_angle=180, L=0.1):
    ''' calculate the ratio scattered power/incident power '''
    
    # Ensure lambda_in has proper units
    if not isinstance(lambda_in, Quantity):
        lambda_in = Quantity(lambda_in, astro_units.nm)
    
    # Convert scattering_angle to radians if it's not already
    scattering_angle_rad = np.radians(scattering_angle)
    
    # Ensure n_e has proper units if it's not already a Quantity
    if not isinstance(n_e, Quantity):
        n_e = Quantity(n_e, astro_units.cm**-3)
    

    # Calculate the function with proper unit handling
    func = (L/(4*np.pi)) * 3*(1 + np.cos(scattering_angle_rad)**2) * n_e * Skw
    
    # Integrate over angular frequency
    result = integrate.simpson(func, omega_arr)
        
    return result