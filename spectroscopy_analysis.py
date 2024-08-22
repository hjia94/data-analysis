from scipy import constants
from data_analysis_utils import ion_sound_speed

#===============================================================================================================================================
# Constants
m_He = 4 * constants.m_p  # mass of helium in kg
c = constants.c  # speed of light in m/s
#===============================================================================================================================================

# Input parameters
Te = 8  # electron temperature in eV
Ti = 1  # ion temperature in eV

M = 0.5  # Mach number
lambda_0 = 468.58  # wavelength of the emitted light in nm
lambda_0 *= 1e-9  # converting from nm to meters
#===============================================================================================================================================

cs = ion_sound_speed(Te, Ti, m_He)  # ion sound speed in cm/s
print(f"Ion sound speed: {cs*1e-3:.2f} km/s")

# Calculate the plasma flow velocity
v = M * cs

# Calculate the Doppler shift (delta lambda)
delta_lambda = (v / c) * lambda_0  # wavelength shift

# Calculate the observed wavelength
delta_lambda *= 1e9  # converting from meters to nm

print(f"Wavelength shift: {delta_lambda:.4f} nm")
