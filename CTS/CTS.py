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

def generate_pulse_wave_packet(f0, n_cycles, num_points=4096, return_envelope=False):
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
def generate_thz_waveform(f0_THz, sigma_t, npulses, Ts, pulse_offset=10.0):
    """
    Generate a THz waveform pulse train and its spectrum.
    
    Parameters:
        f0_THz (float): Center frequency in THz (defines pulse spacing).
        sigma_t (float): Width parameter of the single-cycle pulse in ps.
        npulses (int): Number of pulses in the train.
        Ts (float): Sampling interval in ps.
        pulse_offset (float): Time offset before the first pulse (default 10 ps).
    
    Returns:
        t (np.ndarray): Time array in ps.
        waveform (np.ndarray): Time-domain waveform.
        freqs (np.ndarray): Frequency array in THz.
        fft_signal (np.ndarray): FFT of the waveform (complex).
        envelope (np.ndarray): Envelope of the waveform.
    """
    # Define the single-cycle THz field
    def ETHz(t):
        return t * np.exp(-t**2 / sigma_t**2)

    deltat = 1 / f0_THz  # pulse period in ps
    t_max = 3 * npulses * deltat
    t = np.arange(0, t_max, Ts)

    waveform = np.zeros_like(t)
    for i in range(npulses):
        waveform += ETHz(t - pulse_offset - i * deltat)

    # Compute FFT and frequency axis
    n = len(waveform)
    fft_signal = np.fft.fft(waveform)
    freqs = np.fft.fftfreq(n, d=Ts)  # in THz

    # Calculate the envelope using the Hilbert transform
    from scipy.signal import hilbert
    envelope = np.abs(hilbert(waveform))

    return t, waveform, freqs, fft_signal, envelope


def plasma_dispersion_relation(omega, n_e):  # e.g. 0.5 THz plasma frequency
    """
    Returns the wave number k(omega) for a cold plasma.

    Parameters:
    - omega : ndarray
        Angular frequency array [rad/s]
    - n_e : float
        Electron density [m^-3]

    Returns:
    - k : ndarray
        Wavenumber array [rad/m]
    """
    mass = Particle("e-").mass.value
    omega_p = plasma_frequency.lite(n_e, mass, Z=-1)

    k_squared = (omega**2 - omega_p**2) / c**2
    k = np.where(k_squared >= 0, np.sqrt(k_squared), 0.0)  # Avoid evanescent modes
    
    return k

def propagate_through_dispersive_medium(omega, fft_signal, L, dispersion_relation):
    """
    Propagate a wave packet through a dispersive medium.

    Parameters:
    - omega : ndarray
        Angular frequency array [rad/s].
    - fft_signal : ndarray
        FFT of the initial wave packet (complex).
    - L : float
        Propagation distance in meters.
    - dispersion_relation : callable
        Function that takes omega [rad/s] and returns k [rad/m].

    Returns:
    - fft_propagated : ndarray
        Modified FFT after propagation.
    """

    # Compute wavenumber from dispersion relation (rad/m)
    k = dispersion_relation(omega)

    # Apply dispersion: each spectral component gets a phase shift exp(i * k * L)
    # This properly accounts for both phase velocity and group velocity effects
    phase_shift = np.exp(1j * k * L)

    # Propagated spectrum
    fft_propagated = fft_signal * phase_shift

    return fft_propagated

#===========================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===========================================================================================================
if __name__ == "__main__":
    # Example usage of the wave packet generator with focused FFT
    
    # Parameters
    f0 = 300e9 # GHz
    n_cycles = 2
    n_e = 1e19  # m^-3
    L = 0.001 # meter

    # Generate the THz waveform
    t, signal, freqs, signal_fft, envelope = generate_thz_waveform(f0/1e12, 0.9, n_cycles, 0.001)
    
    # Create a wrapper function that can be passed as the dispersion relation
    dispersion_func = lambda omega: plasma_dispersion_relation(omega, n_e)

    
    # Calculate angular frequency
    omega = 2 * np.pi * freqs*1e12
    
    # Propagate the signal through the plasma
    fft_propagated = propagate_through_dispersive_medium(omega, signal_fft, L, dispersion_func)
    
    # Convert back to time domain using inverse FFT
    signal_propagated = np.fft.ifft(fft_propagated)
    
    # Calculate the expected time delay due to propagation
    # Get the wavenumber at the center frequency
    omega_center = 2 * np.pi * f0
    k_center = dispersion_func(omega_center)
    
    # Calculate group velocity (dω/dk) numerically at the center frequency
    delta_omega = omega_center * 0.01  # Small delta for numerical derivative
    k_plus = dispersion_func(omega_center + delta_omega)
    k_minus = dispersion_func(omega_center - delta_omega)
    
    # Group velocity = dω/dk ≈ Δω/Δk
    v_group = (2 * delta_omega) / (k_plus - k_minus) if (k_plus - k_minus) != 0 else c
    
    # Calculate propagation delay
    delay_time = L / v_group  # seconds
    
    # Convert to ps for plotting
    delay_time_ps = delay_time * 1e12
    
    # Print information for verification
    print(f"Center frequency: {f0/1e9:.2f} GHz")
    print(f"Wavenumber at center frequency: {k_center:.2f} rad/m")
    print(f"Group velocity: {v_group/1e6:.2f} x 10^6 m/s")
    print(f"Expected time delay: {delay_time_ps:.2f} ps")
    
    # Define frequency range of interest around f0
    bandwidth_factor = 0.8  # Show frequencies within 80% of f0
    f_min = f0 * (1 - bandwidth_factor)
    f_max = f0 * (1 + bandwidth_factor)
    
    # Create masks for frequencies within the range of interest
    # First for positive frequencies
    pos_mask = (freqs >= f_min/1e12) & (freqs <= f_max/1e12)
    
    # Create figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Frequency domain (FFT magnitude) - only around f0
    axs[0].plot(freqs[pos_mask]*1e12/1e9, np.abs(signal_fft[pos_mask]), 'b-', alpha=0.6, label='Generator (L=0)')
    axs[0].plot(freqs[pos_mask]*1e12/1e9, np.abs(fft_propagated[pos_mask]), 'r-', alpha=0.6, label='Receiver (L={} m)'.format(L))
    axs[0].set_xlabel('Frequency (GHz)')
    axs[0].set_ylabel('FFT Magnitude')
    axs[0].set_title(f'Wave Packet Spectrum around {f0/1e9:.1f} GHz')
    axs[0].legend()
    axs[0].grid(True)
    
    # Add vertical line at center frequency
    axs[0].axvline(x=f0/1e9, color='k', linestyle='--', label=f'f0 = {f0/1e9:.2f} GHz')
    
    # Plot 2: Time domain signals
    # First plot: the original signal at the generator
    axs[1].plot(t, signal.real, 'b-', label='Generator (L=0)')
    
    # Create a copy of the time array shifted by the propagation delay
    # This represents the actual time when signal reaches the receiver
    t_shifted = t + delay_time_ps
    axs[1].plot(t_shifted, signal_propagated.real, 'r-', label=f'Receiver (L={L} m, delay={delay_time_ps:.2f} ps)')
    
    # Mark the expected arrival time of the pulse center
    # Find the time at which the original signal has its maximum amplitude
    max_idx = np.argmax(np.abs(signal))
    t_peak = t[max_idx]
    t_arrival = t_peak + delay_time_ps
    axs[1].axvline(x=t_arrival, color='g', linestyle='--', 
                  label=f'Expected arrival time: {t_arrival:.2f} ps')
    
    axs[1].set_xlabel('Time (ps)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title(f'Wave Packet Time Domain (n_e={n_e:.1e} $m^{-3}$, L={L} m)')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

