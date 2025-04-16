# -*- coding: utf-8 -*-

"""
Thomson scattering parameters calculation using plasmapy
reference: https://docs.plasmapy.org/en/stable/notebooks/diagnostics/thomson.html
        Sheffield, Plasma Scattering of Electromagnetic Radiation 2nd ed.

Author: Jia Han
Date: 2025-03-19
"""

import numpy as np
from scipy.signal import find_peaks
from scipy import integrate
from scipy.constants import *
import scipy.fft as fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from plasmapy.dispersion import plasma_dispersion_func
from plasmapy.formulary.frequencies import plasma_frequency
from plasmapy.particles import Particle
from plasmapy.diagnostics import thomson
from astropy.units import Quantity
import astropy.units as astro_units

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
    probe_wavelength: float
        Probe wavelength in nm
    T_e: float
        Electron temperature in eV
    T_i: float
        Ion temperature in eV
    n_e: float
        Electron density in cm^-3
    scattering_angle: float, optional
        Scattering angle in degrees (default: 180)
    delta_lam: float, optional
        Wavelength range to calculate around center wavelength (default: 20)
    num_points: int, optional
        Number of points in wavelength array (default: 10000)
    ions: str, optional
        Ion species (default: "He+")
        
    Returns:
    --------
    tuple
        - alpha: float
            Scattering parameter
        - omega_arr: ndarray
            Angular frequency array in rad/s
        - omega_in: float
            Incident angular frequency in rad/s
        - Skw: ndarray
            Spectral density function
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

def power_ratio(n_e, L=0.1, scattering_angle=180):
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
    # if not isinstance(n_e, Quantity):
        # n_e = Quantity(n_e, astro_units.cm**-3)
    
    # Calculate the function with proper unit handling
    func = r_0**2*L/(4*np.pi) * (1 + np.cos(scattering_angle_rad)**2) * n_e #* Skw

    # Integrate over angular frequency
    result = func * 2.5 #integrate.simpson(func, omega_arr)
        
    return result

def faraday_rotation_angle(omega, ne, B, L):
    """
    Computes the Faraday rotation with uniform magnetic field
    
    Parameters:
    omega: angular frequency in rad/s
    ne: electron density in m^-3
    B: magnetic field in Tesla
    L: length of plasma in m

    Returns:
    - theta_deg : Faraday rotation angle in degrees
    """
    coeff = e**3 / (2*epsilon_0*m_e**2*c)
    theta_rad = coeff/omega**2 * ne * B * L
    
    # Convert to degrees
    theta_deg = np.degrees(theta_rad)
    
    # If angle exceeds 360 degrees, set it to zero
    if isinstance(theta_deg, np.ndarray):
        theta_deg[theta_deg > 360] = 0
    else:
        if theta_deg > 360:
            theta_deg = 0
    
    return theta_deg


#===========================================================================================================
def generate_thz_waveform(f0_THz, sigma_t, npulses, dt, pulse_offset=0, npd=1000):
    """
    Generate a THz waveform pulse train.
    
    Parameters:
    -----------
    f0_THz: float
        Center frequency in THz (defines pulse spacing)
    sigma_t: float
        Width parameter of the single-cycle pulse in ps
    npulses: int
        Number of pulses in the train
    dt: float
        Sampling interval in ps
    pulse_offset: float, optional
        Time offset before the first pulse (default: 0 ps)
    npd: int, optional
        Number of padding points 
    
    Returns:
    --------
    tuple
        - tarr: ndarray
            Time array in ps
        - waveform: ndarray
            Time-domain waveform
        - freqs: ndarray
            Frequency array in Hz
        - signal_fft: ndarray
            Frequency-domain representation of the waveform
        - x: float
            Wave covered distance in meters
    """
    # Define the single-cycle THz field
    def ETHz(t):
        # Normalize to ensure amplitude = 1
        normalization = (sigma_t/np.sqrt(2)) * np.exp(-0.5)
        return (t/normalization) * np.exp(-t**2 / sigma_t**2)

    deltat = 1 / f0_THz  # pulse period in ps
    pulse_duration = npulses * deltat  # Duration of the pulse train
    padding = npd * pulse_duration  # Equal padding before and after
    
    # Create time array with equal padding
    t_start = -padding
    t_end = pulse_duration + pulse_offset + padding
    tarr = np.arange(t_start, t_end, dt)
    
    # Create the waveform
    waveform = np.zeros_like(tarr)
    for i in range(npulses):
        t_pulse = tarr - pulse_offset - i * deltat
        waveform += ETHz(t_pulse)
    
    # Check if imaginary parts are just numerical noise
    if np.max(np.abs(np.imag(waveform))) < 1e-10:
        # Safe to discard - they're just numerical artifacts
        waveform = np.real(waveform)
    else:
        # Log a warning - might be physically meaningful
        print("Warning: Significant imaginary components detected in waveform")

    # Transform signal to frequency domain
    signal_fft = np.fft.rfft(waveform)
    freqs = np.fft.rfftfreq(len(tarr), dt*1e-12)  # Frequency in Hz

    x = c*sigma_t*npulses*1e-12
    
    return tarr, waveform, freqs, signal_fft, x


def generate_n_cycle_wave_packet(f0_THz, sigma_t, npulses=1, dt=0.01, pulse_offset=0, npd=1000):
    """
    Generate a THz waveform with a Gaussian envelope.
    
    Parameters:
    -----------
    f0_THz: float
        Center frequency in THz

    npulses: int, optional
        Number of pulses in the train (default: 1)
    dt: float, optional
        Sampling interval in ps (default: 0.01)
    pulse_offset: float, optional
        Time offset before the first pulse (default: 0 ps)
    npd: int, optional
        Number of padding points (default: 1000)
    
    Returns:
    --------
    tuple
        - tarr: ndarray
            Time array in ps
        - waveform: ndarray
            Time-domain waveform with Gaussian envelope
        - freqs: ndarray
            Frequency array in Hz
        - signal_fft: ndarray
            Frequency-domain representation of the waveform
        - x: float
            Wave covered distance in meters
    """
    # First generate the base THz waveform
    tarr, waveform, freqs, signal_fft, x = generate_thz_waveform(
        f0_THz, sigma_t, npulses, dt, pulse_offset, npd
    )
    
    # Create a Gaussian envelope centered at the middle of the time array
    t_center = (tarr[-1] + tarr[0]) / 2
    # Width of the Gaussian envelope (adjust as needed)
    tau = 1/f0_THz * npulses
    
    # Create the Gaussian envelope
    envelope = np.exp(-((tarr - t_center) ** 2) / (tau ** 2))
    
    # Normalize the envelope to have a maximum of 1
    envelope = envelope / np.max(envelope)
    
    # Apply the envelope to the waveform
    modulated_waveform = waveform * envelope
    
    # Recalculate the FFT for the modulated waveform
    modulated_signal_fft = np.fft.rfft(modulated_waveform)
    
    return tarr, modulated_waveform, freqs, modulated_signal_fft, x

def plasma_dispersion_relation(omega, wpe, debug=False):
    """
    Calculate the wave number k(omega) for a cold plasma.

    Parameters:
    -----------
    omega: ndarray
        Angular frequency array in rad/s
    wpe: float
        Plasma frequency in rad/s
    debug: bool, optional
        If True, print debug information (default: False)

    Returns:
    --------
    tuple
        - k: ndarray
            Wavenumber array in rad/m
        - wpe: float
            Plasma frequency in rad/s
        - vgarr: ndarray
            Group velocity array in m/s
    """
    if debug:
        print(f"Plasma frequency: {wpe/(2*np.pi)/1e9:.2f} GHz")

    # Initialize k array with zeros
    k = np.zeros_like(omega)
    
    # Only compute k for frequencies above plasma frequency
    propagating_mask = omega > wpe
    k[propagating_mask] = np.sqrt((omega[propagating_mask]**2 - wpe**2) / c**2)

    # Calculate group velocity, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        vgarr = c*2 * k / omega  # Group velocity in m/s
        # Replace NaN values with zeros
        vgarr = np.nan_to_num(vgarr, nan=0.0)

    return k, wpe, vgarr

def propagate_through_dispersive_medium(NT, freqs, signal_fft, n_e, L, debug=False):
    """
    Propagate a wave packet through a dispersive plasma medium.
    
    Parameters:
    -----------
    freqs: ndarray
        Frequency array in Hz
    signal_fft: ndarray
        Input signal waveform in frequency domain
    n_e: float
        Electron density in cm^-3
    L: float
        Propagation distance in meters
    debug: bool, optional
        If True, print debug information (default: False)
        
    Returns:
    --------
    tuple
        - signal_propagated: ndarray
            Propagated signal in time domain
        - fft_propagated: ndarray
            Propagated signal in frequency domain
        - vgarr: ndarray
            Group velocity array in m/s
    """
    
    # Calculate angular frequencies and plasma frequency
    omega = 2 * np.pi * freqs  # Angular frequency in rad/s

    wpe = 5.64e4 * np.sqrt(n_e)  # Plasma frequency in rad/s
    
    k, wpe, vgarr = plasma_dispersion_relation(omega, wpe, debug=debug)

    # Calculate phase shift for propagation
    phase_shift = np.exp(-1j * k * L)
    
    # Apply phase shift to frequency components
    fft_propagated = signal_fft * phase_shift
    
    # Transform back to time domain
    signal_propagated = np.fft.irfft(fft_propagated, n=NT)

    return signal_propagated, fft_propagated, vgarr


def total_propagation(NT, freqs, signal_fft, n_e, L_arr, debug=False):
    """
    Simulate wave propagation through multiple layers of plasma and calculate total field.
    
    Parameters:
    -----------
    freqs: ndarray
        Frequency array in Hz
    signal_fft: ndarray
        Input signal in frequency domain
    n_e: float
        Electron density in cm^-3
    L_arr: ndarray
        Array of propagation distances in meters
    debug: bool, optional
        If True, print debug information (default: False)
        
    Returns:
    --------
    tuple
        - freqs: ndarray
            Frequency array in Hz
        - tot_wave: ndarray
            Total propagated wave, summed over all distances with 1/L² attenuation
    """
    tot_wave, _, _ = propagate_through_dispersive_medium(NT, freqs, signal_fft, n_e, L_arr[0], debug=debug)

    for L in L_arr[1:]:

        signal_propagated, _, _ = propagate_through_dispersive_medium(NT, freqs, signal_fft, n_e, L, debug=debug)
        tot_wave += signal_propagated / L**2
    
    return tot_wave

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================
if __name__ == "__main__":
    print('speed of light = ', c, 'm/s')

