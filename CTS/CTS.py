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

#===========================================================================================================
    """
    Generate a pulse wave packet with exactly n cycles,
    with zero padding before and after the wave packet, making the total signal length 
    3 times the wave packet duration. The pulse amplitude goes from 0 to 1.

    Parameters:
    - f0 : float
        Carrier frequency in Hz.
    - n_cycles : int
        Number of cycles in the wave packet.
    - num_points : int
        Number of time samples for the wave packet portion.
        Total points will be 3 * num_points.
    - return_envelope : bool
        If True, also return a rectangular envelope.

    Returns:
    - t : ndarray
        Time array in seconds.
    - signal : ndarray
        Pulse wave with zero padding.
    - freqs : ndarray
        Frequency array in Hz (matches FFT output).
    - fft_signal : ndarray
        FFT of the waveform (complex).
    - envelope (optional) : ndarray
        Rectangular envelope of the pulse (if return_envelope is True).
    """
    T = 1 / f0
    duration = n_cycles * T
    
    # Create a time array 3 times longer (zeros before, wave, zeros after)
    total_points = 3 * num_points
    total_duration = 3 * duration
    t = np.linspace(0, total_duration, total_points)
    dt = t[1] - t[0]
    
    # Create signal with zeros
    signal = np.zeros(total_points)
    
    # The wavepacket is in the middle third
    start_idx = num_points
    end_idx = 2 * num_points
    
    # Time values for the wavepacket portion
    t_packet = t[start_idx:end_idx]
    
    # Create the pulse wave in the middle section only
    # Get time values modulo one period
    phase = (t_packet % T) / T
    # Create pulse wave (1 for first half of period, 0 for second half)
    pulse_wave = np.where(phase < 0.5, 1.0, 0.0)
    
    # Apply to the signal
    signal[start_idx:end_idx] = pulse_wave
    
    # Create rectangular envelope (all ones in middle section)
    envelope = np.zeros(total_points)
    envelope[start_idx:end_idx] = 1.0
    
    # Calculate FFT of the full signal (including zeros)
    fft_signal = fft.fft(signal)
    freqs = fft.fftfreq(total_points, dt)

    if return_envelope:
        return t, signal, freqs, fft_signal, envelope
    else:
        return t, signal, freqs, fft_signal

def generate_n_cycle_wave_packet(f0, n_cycles, num_points=4096, return_envelope=False):
    """
    Generate a Gaussian-modulated sinusoidal wave packet with exactly n cycles,
    ensuring the signal starts and ends near zero. The function adds zero padding
    before and after the wave packet, making the total signal length 3 times the 
    wave packet duration.

    Parameters:
    - f0 : float
        Carrier frequency in Hz.
    - n_cycles : int
        Number of cycles in the wave packet.
    - num_points : int
        Number of time samples for the wave packet portion.
        Total points will be 3 * num_points.
    - return_envelope : bool
        If True, also return the Gaussian envelope.

    Returns:
    - t : ndarray
        Time array in seconds.
    - signal : ndarray
        Gaussian-modulated sinusoidal waveform with zero padding.
    - freqs : ndarray
        Frequency array in Hz (matches FFT output).
    - fft_signal : ndarray
        FFT of the waveform (complex).
    - envelope (optional) : ndarray
        Envelope of the pulse (if return_envelope is True).
    """
    T = 1 / f0
    duration = n_cycles * T
    
    # Create a time array 3 times longer (zeros before, wavepacket, zeros after)
    total_points = 3 * num_points
    total_duration = 3 * duration
    t = np.linspace(0, total_duration, total_points)
    dt = t[1] - t[0]
    
    # Create signal with zeros
    signal = np.zeros(total_points)
    
    # The wavepacket is in the middle third
    t_packet = np.linspace(duration, 2*duration, num_points)
    t0 = 1.5 * duration  # Center of the time array
    tau = duration / 6   # Set width so envelope is ~0 at pulse edges
    
    # Calculate the wavepacket
    packet_center_idx = num_points // 2
    start_idx = num_points
    end_idx = 2 * num_points
    
    # Time values for the wavepacket portion
    t_packet = t[start_idx:end_idx]
    
    # Create the wavepacket in the middle section only
    envelope_packet = np.exp(-((t_packet - t0) ** 2) / (2 * tau ** 2))
    carrier_packet = np.cos(2 * np.pi * f0 * (t_packet - t0))
    signal[start_idx:end_idx] = envelope_packet * carrier_packet
    
    # Create full envelope (including zeros) for return if needed
    envelope = np.zeros(total_points)
    envelope[start_idx:end_idx] = envelope_packet
    
    # Calculate FFT of the full signal (including zeros)
    fft_signal = fft.fft(signal)
    freqs = fft.fftfreq(total_points, dt)

    if return_envelope:
        return t, signal, freqs, fft_signal, envelope
    else:
        return t, signal, freqs, fft_signal

#===========================================================================================================
def generate_thz_waveform(f0_THz, sigma_t, npulses, Ts, pulse_offset=0, npd=1000):
    """
    Generate a THz waveform pulse train and its spectrum.
    
    Parameters:
        f0_THz (float): Center frequency in THz (defines pulse spacing).
        sigma_t (float): Width parameter of the single-cycle pulse in ps.
        npulses (int): Number of pulses in the train.
        Ts (float): Sampling interval in ps.
        pulse_offset (float): Time offset before the first pulse (default 0 ps).
    
    Returns:
        tarr (np.ndarray): Time array in ps.
        waveform (np.ndarray): Time-domain waveform.
        freqs (np.ndarray): Frequency array in THz.
        fft_signal (np.ndarray): FFT of the waveform (complex).
        envelope (np.ndarray): Envelope of the waveform.
    """
    # Define the single-cycle THz field
    def ETHz(t):
        return t * np.exp(-t**2 / sigma_t**2)

    deltat = 1 / f0_THz  # pulse period in ps
    pulse_duration = npulses * deltat  # Duration of the pulse train
    padding = npd * pulse_duration  # Equal padding before and after
    
    # Create time array with equal padding
    t_start = -padding
    t_end = pulse_duration + pulse_offset + padding
    tarr = np.arange(t_start, t_end, Ts)
    
    # Create the waveform
    waveform = np.zeros_like(tarr)
    for i in range(npulses):
        t_pulse = tarr - pulse_offset - i * deltat
        waveform += ETHz(t_pulse)
    
    # Calculate the envelope using the Hilbert transform
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(waveform))

    x = c*sigma_t*npulses*1e-12
    
    return tarr, waveform, envelope, x


def plasma_dispersion_relation(omega, wpe, debug=False):  # e.g. 0.5 THz plasma frequency
    """
    Returns the wave number k(omega) for a cold plasma.

    Parameters:
    - omega : ndarray
        Angular frequency array [rad/s]
    - n_e : electron density in cm^-3

    Returns:
    - k : ndarray
        Wavenumber array [rad/m]
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

def propagate_through_dispersive_medium(tarr, signal, L, n_e, debug=False):
    """
    Propagate a wave packet through a dispersive plasma medium.

    """
    # Calculate time step and total points
    dt = (tarr[1] - tarr[0]) *1e-12  # seconds
    NT = len(tarr)
    
    # Transform signal to frequency domain
    signal_fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(NT, dt)  # Frequency in Hz
    
    # Calculate angular frequencies and plasma frequency
    omega = 2 * np.pi * freqs  # Angular frequency in rad/s

    wpe = 5.64e4 * np.sqrt(n_e)  # Plasma frequency in rad/s
    
    k, wpe, vgarr = plasma_dispersion_relation(omega, wpe, debug=False)

    # Calculate phase shift for propagation
    phase_shift = np.exp(-1j * k * L)
    
    # Apply phase shift to frequency components
    fft_propagated = signal_fft * phase_shift
    
    # Transform back to time domain
    signal_propagated = np.fft.irfft(fft_propagated)

    return signal_propagated, fft_propagated, vgarr


def total_propagation(f0, n_cycles, n_e, L_arr):
    tarr, signal, envelope, x = generate_thz_waveform(f0/1e12, 0.9, n_cycles, 0.1, 5, 100)
    
    tot_wave = np.zeros_like(tarr)
    for L in L_arr:
        signal_propagated, fft_propagated, vgarr = propagate_through_dispersive_medium(tarr, signal, L, n_e)   
        tot_wave += signal_propagated / L**2
    
    return tot_wave

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================
if __name__ == "__main__":
    print('speed of light = ', c, 'm/s')

