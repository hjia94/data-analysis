# -*- coding: utf-8 -*-
"""
LAPD Magnetic Field Computation Package
========================================

This package provides comprehensive magnetic field calculations for the LAPD
(Large Plasma Device) using accurate elliptic integral methods.

Main Components:
----------------

High-Level Interface:
- LAPDCoilSet: Main class for LAPD magnetic field calculations with predefined coil configurations
- CoilData: Individual coil parameter representation
- create_BaO_coil_set, create_LaB6_coil_set: Predefined coil configurations

Low-Level Physics:
- RingCurrent: Fundamental ring current representation for magnetic field calculations
- calculate_field_from_ring_currents: Calculate magnetic field from multiple ring currents
- calculate_flux_from_ring_currents: Calculate magnetic flux from multiple ring currents

Example Usage:
--------------
```python
import compute_B

# High-level usage for LAPD calculations
lapd = compute_B.LAPDCoilSet()
lapd.set_uniform_field(0.1)  # 0.1 Tesla uniform field
Bx, By, Bz, Bmag = lapd.compute_B(0.0, 0.0, 8.5)  # Field at point (0,0,8.5)

# Low-level usage for custom calculations
ring = compute_B.RingCurrent(R=1.0, z=0.0, cur=1000.0)  # 1m radius, 1000A current
BR, Bz, Bmag = compute_B.calculate_field_from_ring_currents([ring], 0.5, 0.0)
```

Authors: PP (original C++), JH (Python implementation)
Created: 2025-07-30
"""

# Import main classes and functions from submodules
from .LAPD_coil_set import (
    LAPDCoilSet,
    CoilData, 
    create_BaO_coil_set,
    create_LaB6_coil_set,
    YELLOW,
    PURPLE, 
    BLACK
)

from .magnetic_field_calculator import (
    RingCurrent,
    calculate_field_from_ring_currents,
    calculate_flux_from_ring_currents,
    B,
    A,
    BRk2,
    mu0
)

# Define what gets imported with "from compute_B import *"
__all__ = [
    # High-level LAPD interface
    'LAPDCoilSet',
    'CoilData',
    'create_BaO_coil_set', 
    'create_LaB6_coil_set',
    
    # Low-level physics calculations
    'RingCurrent',
    'calculate_field_from_ring_currents',
    'calculate_flux_from_ring_currents',
    'B',
    'A',
    'BRk2',
    
    # Constants
    'YELLOW',
    'PURPLE', 
    'BLACK',
    'mu0'
]

# Package metadata
__version__ = "1.0.0"
__author__ = "JH (based on PP's work)"
__description__ = "LAPD Magnetic Field Computation Package"