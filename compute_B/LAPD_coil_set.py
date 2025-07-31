# -*- coding: utf-8 -*-
"""
LAPD Coil Set - Python Implementation
====================================

This module provides a Python implementation of the LAPD (Large Plasma Device) 
coil set for magnetic field calculations. It uses the magnetic_field_calculator 
module for accurate field computations using elliptic integrals.

The LAPD consists of multiple coil sets:
- BaO coil set: Main coils with Yellow and Purple color coding
- LaB6 coil set: Additional Black coils plus BaO coils

Key Features:
- Accurate coil positioning and parameters
- Power supply management (12 supplies)
- Uniform field configuration
- Magnetic field computation at arbitrary points
- Vector potential and flux calculations

Based on LAPD_coil_set.h by PP (coil locations from SK)
Python version created on 2025-07-30 by JH
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .magnetic_field_calculator import (RingCurrent, calculate_field_from_ring_currents, 
                                         calculate_flux_from_ring_currents, calculate_vector_potential_from_ring_currents)

# Color constants for coil identification
YELLOW = 1
PURPLE = 2
BLACK = 3

# Physical constant
mu0 = math.pi * 4.e-7  # Permeability of free space (H/m)

@dataclass
class CoilData:
    """
    Data structure for individual coil parameters.
    
    Attributes:
    -----------
    color : int
        Coil color code (YELLOW, PURPLE, or BLACK)
    supply_number : int
        Power supply number (1-12)
    z : float
        Coil mean z location in meters (0 is at far end of machine from cathode)
    eff_port_number : float
        Effective port number for the coil
    a : float
        Coil mean radius in meters
    num_turns : int
        Number of turns in the coil
    current : float
        Current in Amperes
    """
    color: int
    supply_number: int
    z: float
    eff_port_number: float
    a: float
    num_turns: int
    current: float = 0.0


def create_BaO_coil_set() -> List[CoilData]:
    """
    Create the BaO (Barium Oxide) coil set configuration.
    
    This represents the main LAPD coil configuration with Yellow and Purple coils
    distributed along the machine length.
    
    Returns:
    --------
    List[CoilData] : List of coil data objects
    """
    coils = []
    
    # Yellow coils - Power Supply 1
    coils.extend([
        CoilData(YELLOW, 1, 16.93350, 0.0, 0.67842, 10, 0.0),
        CoilData(YELLOW, 1, 16.77375, 0.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 1, 16.45425, 1.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 1, 16.13475, 2.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 1, 15.81525, 3.5, 0.67842, 10, 0.0),
    ])
    
    # Yellow coils - Power Supply 2
    coils.extend([
        CoilData(YELLOW, 2, 15.49575, 4.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 2, 15.17625, 5.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 2, 14.85675, 6.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 2, 14.53725, 7.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 2, 14.21775, 8.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 2, 13.89825, 9.5, 0.67842, 10, 0.0),
    ])
    
    # Purple coils - Power Supply 5
    coils.extend([
        CoilData(PURPLE, 5, 13.62175, 10.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 13.53575, 10.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 13.30225, 11.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 13.21625, 11.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.98275, 12.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.89675, 12.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.66325, 13.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.57725, 13.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.34375, 14.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.25775, 14.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 12.02425, 15.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 5, 11.93825, 15.635, 0.67348, 14, 0.0),
    ])
    
    # Purple coils - Power Supply 6
    coils.extend([
        CoilData(PURPLE, 6, 11.70475, 16.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 11.61875, 16.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 11.38525, 17.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 11.29925, 17.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 11.06575, 18.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.97975, 18.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.74625, 19.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.66025, 19.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.42675, 20.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.34075, 20.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 6, 10.10725, 21.365, 0.67348, 14, 0.0),  # z location corrected 22-07-28
    ])
    
    # Purple coils - Power Supply 7
    coils.extend([
        CoilData(PURPLE, 7, 10.02125, 21.65, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.78775, 22.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.70175, 22.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.46825, 23.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.38225, 23.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.14875, 24.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 9.06275, 24.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 8.82925, 25.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 8.74325, 25.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 8.50975, 26.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 7, 8.42375, 26.635, 0.67348, 14, 0.0),
    ])
    
    # Purple coils - Power Supply 8
    coils.extend([
        CoilData(PURPLE, 8, 8.19025, 27.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 8.10425, 27.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.87075, 28.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.78475, 28.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.55125, 29.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.46525, 29.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.23175, 30.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 7.14575, 30.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 6.91225, 31.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 6.82625, 31.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 8, 6.59275, 32.635, 0.67348, 14, 0.0),  # z location corrected 22-07-28
    ])
    
    # Purple coils - Power Supply 9
    coils.extend([
        CoilData(PURPLE, 9, 6.50675, 32.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 6.27325, 33.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 6.18725, 33.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.95375, 34.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.86775, 34.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.63425, 35.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.54825, 35.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.31475, 36.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 5.22875, 36.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 4.99525, 37.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 9, 4.90925, 37.635, 0.67348, 14, 0.0),
    ])
    
    # Purple coils - Power Supply 10
    coils.extend([
        CoilData(PURPLE, 10, 4.67575, 38.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 4.58975, 38.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 4.35625, 39.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 4.27025, 39.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 4.03675, 40.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.95075, 40.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.71725, 41.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.63125, 41.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.39775, 42.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.31175, 42.635, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 3.07825, 43.365, 0.67348, 14, 0.0),
        CoilData(PURPLE, 10, 2.99225, 43.635, 0.67348, 14, 0.0),
    ])
    
    # Yellow coils - Power Supply 3
    coils.extend([
        CoilData(YELLOW, 3, 2.71575, 44.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 3, 2.39625, 45.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 3, 2.07675, 46.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 3, 1.75725, 47.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 3, 1.43775, 48.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 3, 1.11825, 49.5, 0.67842, 10, 0.0),
    ])
    
    # Yellow coils - Power Supply 4
    coils.extend([
        CoilData(YELLOW, 4, 0.79875, 50.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 4, 0.47925, 51.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 4, 0.15975, 52.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 4, -0.15975, 53.5, 0.67842, 10, 0.0),
        CoilData(YELLOW, 4, -0.31950, 54.0, 0.67842, 10, 0.0),
    ])
    
    return coils


def create_LaB6_coil_set() -> List[CoilData]:
    """
    Create the LaB6 (Lanthanum Hexaboride) coil set configuration.
    
    This includes additional Black coils plus the complete BaO coil set.
    The coils must be sorted by decreasing z-coordinate for proper port number interpolation.
    
    Returns:
    --------
    List[CoilData] : List of coil data objects sorted by decreasing z-coordinate
    """
    coils = []
    
    # Black coils - Power Supply 12
    coils.extend([
        CoilData(BLACK, 12, 19.12413 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 19.03475 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.94538 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.85600 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.76663 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.67725 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.58788 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 12, 18.49850 - 0.4, -1, 0.55, 16, 0.0),
    ])
    
    # Black coils - Power Supply 11
    coils.extend([
        CoilData(BLACK, 11, 18.40913 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 18.31975 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 18.23038 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 18.14100 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 18.05163 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 17.96225 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 17.87288 - 0.4, -1, 0.55, 16, 0.0),
        CoilData(BLACK, 11, 17.78350 - 0.4, -1, 0.55, 16, 0.0),
    ])
    
    # Add BaO coil set
    coils.extend(create_BaO_coil_set())
    
    # Note: Don't sort here - the C++ version concatenates in specific order
    # The port number interpolation should work with the natural ordering
    
    return coils


class LAPDCoilSet:
    """
    LAPD (Large Plasma Device) Coil Set for magnetic field calculations.
    
    This class manages the complete LAPD coil configuration, including:
    - Coil positioning and parameters
    - Power supply management (12 supplies)
    - Magnetic field computation using elliptic integrals
    - Uniform field configuration
    
    Attributes:
    -----------
    coils : List[CoilData]
        List of all coil data
    ring_currents : List[RingCurrent]
        Ring current representations for field calculations
    supply1-supply12 : float
        Current values for each of the 12 power supplies in Amperes
    """
    
    def __init__(self):
        """Initialize LAPD coil set with LaB6 configuration and uniform field."""
        self.coils = create_LaB6_coil_set()
        self.ring_currents = []
        
        # Initialize the 12 power supply currents
        self.supply1 = 0.0
        self.supply2 = 0.0
        self.supply3 = 0.0
        self.supply4 = 0.0
        self.supply5 = 0.0
        self.supply6 = 0.0
        self.supply7 = 0.0
        self.supply8 = 0.0
        self.supply9 = 0.0
        self.supply10 = 0.0
        self.supply11 = 0.0
        self.supply12 = 0.0
        
        self.set_uniform_field(0.1)  # Default 0.1 Tesla uniform field
        self._update_ring_currents()
    
    # =============================================================================
    # SUPPLY CURRENT MANAGEMENT
    # =============================================================================
    
    def get_supply_currents(self) -> List[float]:
        """
        Get current values for all 12 power supplies.
        
        Returns:
        --------
        List[float] : List of 12 supply currents in Amperes [I1, I2, ..., I12]
        """
        return [
            self.supply1, self.supply2, self.supply3, self.supply4,
            self.supply5, self.supply6, self.supply7, self.supply8,
            self.supply9, self.supply10, self.supply11, self.supply12
        ]
    
    def get_supply_current(self, supply_number: int) -> float:
        """
        Get current for a specific power supply.
        
        Parameters:
        -----------
        supply_number : int
            Supply number (1-12)
            
        Returns:
        --------
        float : Current in Amperes
        """
        self._validate_supply_number(supply_number)
        return getattr(self, f'supply{supply_number}')
    
    def set_individual_supply_current(self, supply_number: int, current: float):
        """
        Set current for a single power supply.
        
        Parameters:
        -----------
        supply_number : int
            Supply number (1-12)
        current : float
            Current in Amperes
        """
        self._validate_supply_number(supply_number)
        # Update the supply attribute
        setattr(self, f'supply{supply_number}', current)
        
        # Update all coils connected to this supply
        for coil in self.coils:
            if coil.supply_number == supply_number:
                coil.current = current
    
    def set_supply_current(self, current: float, *supply_numbers: int):
        """
        Set current for specified power supplies.
        
        Parameters:
        -----------
        current : float
            Current in Amperes
        supply_numbers : int
            Variable number of supply numbers to set
        """
        for supply_num in supply_numbers:
            self.set_individual_supply_current(supply_num, current)
    
    def set_supply_currents(self, i1: float, i2: float, i3: float, i4: float,
                           i5: float, i6: float, i7: float, i8: float,
                           i9: float, i10: float, i11: float, i12: float):
        """
        Set individual currents for all 12 power supplies.
        
        Parameters:
        -----------
        i1-i12 : float
            Current for each power supply in Amperes
        """
        currents = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12]
        for supply_num, current in enumerate(currents, 1):
            self.set_individual_supply_current(supply_num, current)
    
    # =============================================================================
    # FIELD CONFIGURATION
    # =============================================================================
    
    def set_uniform_field(self, B0: float):
        """
        Set currents for uniform magnetic field configuration.
        
        Parameters:
        -----------
        B0 : float
            Desired magnetic field strength in Tesla
        """
        m = B0 / 0.1  # Scaling factor relative to 0.1 T
        
        # Set power supply currents for uniform field
        self.set_supply_current(2600.0 * m, 1, 2, 3, 4)        # 2600 A for supplies 1-4
        self.set_supply_current(910.0 * m, 5, 6, 7, 8, 9, 10)  # 910 A for supplies 5-10
        self.set_supply_current(555.0 * m, 11, 12)             # 555 A for supplies 11-12
    
    # =============================================================================
    # MAGNETIC FIELD COMPUTATION
    # =============================================================================
    
    def compute_B(self, x: float, y: float, z: float) -> Tuple[float, float, float, float]:
        """
        Compute magnetic field at a Cartesian point.
        
        Parameters:
        -----------
        x, y, z : float
            Cartesian coordinates in meters
            z = 0 is at far end of machine from cathode
            z is positive toward cathode
        
        Returns:
        --------
        tuple : (Bx, By, Bz, Bmag) magnetic field components and magnitude in Tesla
        """
        # Convert to cylindrical coordinates
        R = math.sqrt(x**2 + y**2)
        phi = math.atan2(y, x)
        
        # Update ring currents if needed
        self._update_ring_currents()
        
        # Calculate cylindrical field components
        BR_total, Bz_total, Bmag_total = calculate_field_from_ring_currents(
            self.ring_currents, R, z)
        
        # Convert back to Cartesian coordinates
        if R > 0:
            cos_phi = x / R
            sin_phi = y / R
            Bx = BR_total * cos_phi
            By = BR_total * sin_phi
        else:
            Bx = 0.0
            By = 0.0
        
        return Bx, By, Bz_total, Bmag_total
    
    def compute_B_cylindrical(self, R: float, z: float) -> Tuple[float, float, float]:
        """
        Compute magnetic field at a cylindrical point.
        
        Parameters:
        -----------
        R : float
            Radial coordinate in meters
        z : float
            Axial coordinate in meters
        
        Returns:
        --------
        tuple : (BR, Bz, Bmag) magnetic field components and magnitude in Tesla
        """
        self._update_ring_currents()
        return calculate_field_from_ring_currents(self.ring_currents, R, z)
    
    def compute_Aphi(self, R: float, z: float) -> float:
        """
        Compute azimuthal vector potential at a cylindrical point.
        
        Parameters:
        -----------
        R : float
            Radial coordinate in meters
        z : float
            Axial coordinate in meters
        
        Returns:
        --------
        float : Azimuthal vector potential in Wb/m
        """
        self._update_ring_currents()
        return calculate_vector_potential_from_ring_currents(self.ring_currents, R, z)
    
    def compute_Psi(self, R: float, z: float) -> float:
        """
        Compute magnetic flux through a circle at (R, z).
        
        Parameters:
        -----------
        R : float
            Radial coordinate in meters
        z : float
            Axial coordinate in meters
        
        Returns:
        --------
        float : Magnetic flux in Wb
        """
        self._update_ring_currents()
        return calculate_flux_from_ring_currents(self.ring_currents, R, z)
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def z_to_eff_port_number(self, z: float) -> float:
        """
        Convert z-coordinate to effective port number by interpolation.
        
        Parameters:
        -----------
        z : float
            Axial coordinate in meters
        
        Returns:
        --------
        float : Effective port number
        """
        # Find coils with valid port numbers (not -1)
        valid_coils = [coil for coil in self.coils if coil.eff_port_number >= 0]
        
        if not valid_coils:
            return 0.0
            
        # Sort valid coils by z-coordinate (decreasing)
        valid_coils.sort(key=lambda c: c.z, reverse=True)
        
        # Boundary checks
        if z > valid_coils[0].z:
            return valid_coils[0].eff_port_number
        if z < valid_coils[-1].z:
            return valid_coils[-1].eff_port_number
        
        # Find interpolation interval
        for i in range(len(valid_coils) - 1):
            if valid_coils[i].z >= z >= valid_coils[i+1].z:
                # Linear interpolation
                dz = valid_coils[i].z - valid_coils[i+1].z
                if dz == 0:
                    return valid_coils[i].eff_port_number
                f = (z - valid_coils[i+1].z) / dz
                epn = (valid_coils[i+1].eff_port_number + 
                       f * (valid_coils[i].eff_port_number - valid_coils[i+1].eff_port_number))
                return epn
        
        return 0.0
    
    # =============================================================================
    # SPECIAL METHODS (DUNDER METHODS)
    # =============================================================================
    
    def __getitem__(self, index: int) -> CoilData:
        """Get coil data by index."""
        return self.coils[index]
    
    def __len__(self) -> int:
        """Get number of coils."""
        return len(self.coils)
    
    def __str__(self) -> str:
        """String representation of coil set."""
        header = f"{'z':>10} {'a_eff':>10} {'#turns':>10} {'current':>10} {'supply#':>10}\n"
        lines = [header]
        for coil in self.coils:
            line = (f"{coil.z:10.5f} {coil.a:10.3f} {coil.num_turns:10d} "
                   f"{coil.current:10.4f} {coil.supply_number:10d}\n")
            lines.append(line)
        return ''.join(lines)
    
    # =============================================================================
    # PRIVATE METHODS
    # =============================================================================
    
    def _validate_supply_number(self, supply_number: int):
        """Validate supply number is in valid range."""
        if not (1 <= supply_number <= 12):
            raise ValueError(f"Supply number must be between 1 and 12, got {supply_number}")
    
    def _update_ring_currents(self):
        """Update ring current representations for field calculations."""
        self.ring_currents = []
        for coil in self.coils:
            if coil.current != 0.0:  # Only include coils with current
                # Create ring current with total current = coil.current * num_turns
                total_current = coil.current * coil.num_turns
                ring_current = RingCurrent(R=coil.a, z=coil.z, cur=total_current)
                self.ring_currents.append(ring_current)


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("LAPD Coil Set - Python Implementation")
    print("=" * 50)
    
    # Create LAPD coil set
    lapd = LAPDCoilSet()
    
    print(f"Total number of coils: {len(lapd)}")
    print(f"Number of active ring currents: {len(lapd.ring_currents)}")
    
    # Test magnetic field calculation
    print("\nTesting magnetic field calculations...")
    
    # Calculate field at center of machine
    x, y, z = 0.0, 0.0, 8.5  # Center of machine
    Bx, By, Bz, Bmag = lapd.compute_B(x, y, z)
    
    print(f"Magnetic field at center (x={x}, y={y}, z={z}):")
    print(f"  Bx = {Bx:.6f} T")
    print(f"  By = {By:.6f} T") 
    print(f"  Bz = {Bz:.6f} T")
    print(f"  |B| = {Bmag:.6f} T")
    
    # Calculate field at off-axis point
    x, y, z = 0.1, 0.0, 8.5  # 10 cm off-axis
    Bx, By, Bz, Bmag = lapd.compute_B(x, y, z)
    
    print(f"\nMagnetic field off-axis (x={x}, y={y}, z={z}):")
    print(f"  Bx = {Bx:.6f} T")
    print(f"  By = {By:.6f} T")
    print(f"  Bz = {Bz:.6f} T") 
    print(f"  |B| = {Bmag:.6f} T")
    
    # Test different field strengths
    print("\nTesting different field strengths...")
    for B0 in [0.05, 0.1, 0.2]:
        lapd.set_uniform_field(B0)
        _, _, Bz, Bmag = lapd.compute_B(0.0, 0.0, 8.5)
        print(f"  B0 = {B0:.2f} T → |B| = {Bmag:.6f} T (Bz = {Bz:.6f} T)")
    
    # Test new supply current interface
    print("\nTesting supply current interface...")
    lapd.set_uniform_field(0.1)  # Reset to uniform field
    
    print("Supply currents after uniform field setup:")
    currents = lapd.get_supply_currents()
    for i, current in enumerate(currents, 1):
        print(f"  Supply {i:2d}: {current:8.1f} A")
    
    # Test individual supply access
    print(f"\nIndividual supply access:")
    print(f"  Supply 1: {lapd.get_supply_current(1):.1f} A")
    print(f"  Supply 5: {lapd.get_supply_current(5):.1f} A")
    print(f"  Supply 12: {lapd.get_supply_current(12):.1f} A")
    
    # Test individual supply setting
    print(f"\nSetting supply 1 to 3000 A...")
    lapd.set_individual_supply_current(1, 3000.0)
    print(f"  Supply 1 after change: {lapd.get_supply_current(1):.1f} A")
    print(f"  Direct access: lapd.supply1 = {lapd.supply1:.1f} A")
    
