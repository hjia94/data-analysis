# -*- coding: utf-8 -*-
"""
Magnetic Field Calculator for Coils
==================================

This module provides comprehensive magnetic field calculations for coils using
elliptic integral solutions. It combines ring current physics with elliptic
integral calculations for accurate magnetic field computation.

Key Features:
- Ring current magnetic field calculations using exact elliptic integrals
- Vector potential and magnetic flux calculations
- Mutual inductance calculations
- Normalized and actual field components
- Optimized for small k² values

Initial version Uses PP's ellipticB function in pyfields.dll, 16-10-19
Python wrapper created 2016 by PP

Python only version ellipticB_Ver2.py created on 2021 by JH
Used for calculating field from ET coils

This file merges the base calculations into a single file, and adds some utility functions.
Created on 2025-07-30 by JH
"""

import math
import numpy
import scipy.special as ss

# Physical constants
mu0 = math.pi * 4.e-7  # Permeability of free space (H/m)
K2MIN = numpy.sqrt(numpy.finfo(float).eps)  # Minimum k² for numerical stability

# =============================================================================
# ELLIPTIC INTEGRAL CALCULATIONS
# =============================================================================

def BRk2(E, K, k2):
    """
    Worker function for elliptic B calculations.
    Computes BR component for small k² values using optimized series expansion.
    
    Parameters:
    -----------
    E : float
        Complete elliptic integral of second kind
    K : float
        Complete elliptic integral of first kind  
    k2 : float
        k² parameter (must be > 0.00017 for accuracy)
    
    Returns:
    --------
    float : BR component for small k²
    """
    if k2 > 0.00017:  # .00017^4 < 1e-15
        return (E - (1 - k2/2 - k2*k2/2)*K) / (k2 * (1-k2))

    # Computes BR for small R using a combination of elliptic integrals
    # that does not require an intermediate result of 1/R
    return (math.pi*k2/16) / (1-k2) *\
           (7/2 + 9/16 * k2 *\
            (10/9 + 25/36 * k2 *\
             (69/100 + 49/64 * k2 *\
              (124/245 + 81/100 * k2 *\
               (65/162)))))


def B(R, z):
    """
    Calculate normalized magnetic field components from a ring current.
    
    Parameters:
    -----------
    R : float
        Normalized radial position (R_obs/R_coil)
    z : float
        Normalized axial position (z_obs/R_coil)
    
    Returns:
    --------
    tuple : (BR, Bz) normalized to B0 = mu0 * I / (2 * R_coil)
    """
    Q = (1+R)*(1+R) + z*z
    k2 = 4*R/Q
    sqrtQ = numpy.sqrt(Q)

    K = ss.ellipk(k2)  # Complete elliptic integral of first kind
    E = ss.ellipe(k2)  # Complete elliptic integral of second kind

    Bz = (E * (1-R*R-z*z) / (Q-4*R) + K) / (math.pi*sqrtQ)
    BR = (4*(1+R*R+z*z)/Q * BRk2(E, K, k2) - k2*K) * z/(math.pi*Q*sqrtQ)

    return BR, Bz


def A(R, z):
    """
    Calculate normalized vector potential Aφ from a ring current.
    
    Parameters:
    -----------
    R : float
        Normalized radial position (R_obs/R_coil)
    z : float
        Normalized axial position (z_obs/R_coil)
    
    Returns:
    --------
    float : Aφ normalized to A0 = mu0 * I
    """
    Q = (1+R)*(1+R) + z*z
    k2 = 4*R/Q
    sqrtQ = numpy.sqrt(Q)

    K = ss.ellipk(k2)
    E = ss.ellipe(k2)

    if k2 < K2MIN:
        return (k2/8) / (2*sqrtQ)
    return (((2-k2)*K - 2*E)/k2 * 2/numpy.pi) / (2*sqrtQ)


# =============================================================================
# RING CURRENT CLASS
# =============================================================================

class RingCurrent:
    """
    Ring current descriptor class for magnetic field calculations.
    
    This class provides methods to calculate magnetic fields, vector potentials,
    magnetic flux, and mutual inductance for a single ring current.
    
    Parameters:
    -----------
    R : float
        Radial position of the ring current (m)
    z : float  
        Axial position of the ring current (m)
    cur : float, optional
        Current in the ring (A), default=1.0
    d : float, optional
        Conductor diameter (m), default=0.05*0.0254 (0.1")
    """
    
    def __init__(self, R, z, cur=1.0, d=0.05*0.0254):
        self.R = R
        self.z = z
        self.cur = cur
        self.d = d

    def __repr__(self):
        return f"RingCurrent(R={self.R}, z={self.z}, cur={self.cur}, d={self.d})"

    def Aphi(self, R, z):
        """
        Calculate the vector potential Aφ(R,z) from this ring current.
        
        Parameters:
        -----------
        R : float
            Radial position where to calculate Aφ (m)
        z : float
            Axial position where to calculate Aφ (m)
            
        Returns:
        --------
        float : Vector potential Aφ (Wb/m)
        """
        return mu0 * self.cur * A(R/self.R, (z-self.z)/self.R)

    def Psi(self, R, z):
        """
        Calculate the magnetic flux linked by a circle at (R,z).
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : Magnetic flux (Wb)
        """
        if (R-self.R)**2 + (z-self.z)**2 < self.d**2:
            # Approximately the same location - avoid singularity
            return 2 * math.pi * (R-self.d) * self.Aphi(R-self.d, z)
        return 2 * math.pi * R * self.Aphi(R, z)

    def Mutual(self, arg, z=None):
        """
        Calculate mutual inductance between this ring current and another.
        
        Parameters:
        -----------
        arg : RingCurrent or float
            Either another RingCurrent object or radial position
        z : float, optional
            Axial position (if arg is radial position)
            
        Returns:
        --------
        float : Mutual inductance (H)
        """
        if z is None:
            # arg is another RingCurrent object
            return self.Psi(arg.R, arg.z) / self.cur
        # arg is R, z are separate arguments
        return self.Psi(arg, z) / self.cur

    def nB(self, R, z):
        """
        Calculate normalized magnetic field components.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        tuple : (nBR, nBz) where nB = B(R,z) / I
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * br / (2 * self.R), mu0 * bz / (2 * self.R)

    def nBR(self, R, z):
        """
        Calculate normalized radial magnetic field component.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : Normalized BR = BR(R,z) / I
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * br / (2 * self.R)

    def nBz(self, R, z):
        """
        Calculate normalized axial magnetic field component.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : Normalized Bz = Bz(R,z) / I
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * bz / (2 * self.R)

    def B(self, R, z):
        """
        Calculate actual magnetic field components in Tesla.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        tuple : (BR, Bz) in Tesla
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * self.cur * br / (2 * self.R), mu0 * self.cur * bz / (2 * self.R)

    def BR(self, R, z):
        """
        Calculate actual radial magnetic field component.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : BR in Tesla
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * self.cur * br / (2 * self.R)

    def Bz(self, R, z):
        """
        Calculate actual axial magnetic field component.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : Bz in Tesla
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * self.cur * bz / (2 * self.R)

    def magB(self, R, z):
        """
        Calculate magnitude of magnetic field.
        
        Parameters:
        -----------
        R : float
            Radial position (m)
        z : float
            Axial position (m)
            
        Returns:
        --------
        float : |B| = sqrt(Bz² + BR²) in Tesla
        """
        br, bz = B(R/self.R, (z-self.z)/self.R)
        return mu0 * self.cur * math.sqrt(br**2 + bz**2) / (2 * self.R)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_field_from_ring_currents(ring_currents, R, z):
    """
    Calculate total magnetic field from multiple ring currents.
    
    Parameters:
    -----------
    ring_currents : list
        List of RingCurrent objects
    R : float
        Radial position where to calculate field (m)
    z : float
        Axial position where to calculate field (m)
        
    Returns:
    --------
    tuple : (BR_total, Bz_total, Bmag_total) in Tesla
    """
    BR_total = 0.0
    Bz_total = 0.0
    
    for rc in ring_currents:
        BR, Bz = rc.B(R, z)
        BR_total += BR
        Bz_total += Bz
    
    Bmag_total = math.sqrt(BR_total**2 + Bz_total**2)
    return BR_total, Bz_total, Bmag_total


def calculate_flux_from_ring_currents(ring_currents, R, z):
    """
    Calculate total magnetic flux from multiple ring currents.
    
    Parameters:
    -----------
    ring_currents : list
        List of RingCurrent objects
    R : float
        Radial position where to calculate flux (m)
    z : float
        Axial position where to calculate flux (m)
        
    Returns:
    --------
    float : Total magnetic flux (Wb)
    """
    psi_total = 0.0
    for rc in ring_currents:
        psi_total += rc.Psi(R, z)
    return psi_total


# =============================================================================
# TESTING AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    print("Testing magnetic field calculations...")
    
    # Test reciprocity of mutual inductance
    ca = RingCurrent(1, 2)
    cb = RingCurrent(3, 4)
    print(f'Reciprocity test 1: {ca.Mutual(cb):.6e} should equal {cb.Mutual(ca):.6e}')
    print(f'Difference: {ca.Mutual(cb) - cb.Mutual(ca):.2e}')
    
    ca = RingCurrent(1, 2)
    cb = RingCurrent(1, 4)
    print(f'Reciprocity test 2: {ca.Mutual(cb):.6e} should equal {cb.Mutual(ca):.6e}')
    print(f'Difference: {ca.Mutual(cb) - cb.Mutual(ca):.2e}')
    
    ca = RingCurrent(1, 2)
    cb = RingCurrent(3, 2)
    print(f'Reciprocity test 3: {ca.Mutual(cb):.6e} should equal {cb.Mutual(ca):.6e}')
    print(f'Difference: {ca.Mutual(cb) - cb.Mutual(ca):.2e}')
    
    # Test solenoid approximation
    a = 1
    dz = 20
    N = 709
    
    ring_currents = []
    for i in range(N):
        z = (i + 0.5) * dz/N
        ring_currents.append(RingCurrent(a, z))
    
    L = 0
    for ca in ring_currents:
        for cb in ring_currents:
            L += ca.Mutual(cb)
    
    approx_L = mu0 * N**2 * math.pi * a**2 / dz
    print(f'Computed L = {L:.6e}')
    print(f'Solenoid approximation = {approx_L:.6e}')
    print(f'Relative error = {abs(L - approx_L)/approx_L:.2e}')
    
    print("All tests completed!") 