# LAPD Magnetic Field Computation Package

This package provides comprehensive magnetic field calculations for the LAPD (Large Plasma Device) using accurate elliptic integral methods.

## Overview

The `compute_B` package contains two main modules:

- **`LAPD_coil_set.py`**: High-level interface for LAPD-specific calculations
- **`magnetic_field_calculator.py`**: Low-level physics calculations using elliptic integrals

## Installation

Since this is part of the LAPD_DAQ repository, you can import it directly:

```python
import compute_B
```

Or if you're in a different directory:

```python
import sys
sys.path.append('/path/to/LAPD_DAQ')
import compute_B
```

## Quick Start

### High-Level Usage (Recommended)

For most LAPD applications, use the `LAPDCoilSet` class:

```python
import compute_B

# Create LAPD coil set
lapd = compute_B.LAPDCoilSet()

# Set uniform magnetic field
lapd.set_uniform_field(0.1)  # 0.1 Tesla

# Calculate field at a point (x, y, z in meters)
Bx, By, Bz, Bmag = lapd.compute_B(0.0, 0.0, 8.5)
print(f"Magnetic field: |B| = {Bmag:.6f} T")

# Calculate field in cylindrical coordinates  
BR, Bz, Bmag = lapd.compute_B_cylindrical(0.1, 8.5)
```

### Custom Current Settings

```python
# Set individual power supply currents (supplies 1-12)
lapd.set_supply_currents(2600, 2600, 2600, 2600,  # supplies 1-4
                        910, 910, 910, 910, 910, 910,  # supplies 5-10
                        555, 555)  # supplies 11-12

# Or set specific supplies
lapd.set_supply_current(1000.0, 1, 2, 3)  # 1000A to supplies 1, 2, 3
```

### Low-Level Usage

For custom coil configurations or detailed physics calculations:

```python
import compute_B

# Create a ring current (R=1m, z=0, I=1000A)
ring = compute_B.RingCurrent(R=1.0, z=0.0, cur=1000.0)

# Calculate field from multiple rings
rings = [ring]  # List of RingCurrent objects
BR, Bz, Bmag = compute_B.calculate_field_from_ring_currents(rings, R=0.5, z=0.0)

# Calculate magnetic flux
flux = compute_B.calculate_flux_from_ring_currents(rings, R=0.5, z=0.0)
```

## API Reference

### Main Classes

#### `LAPDCoilSet`
High-level interface for LAPD magnetic field calculations.

**Methods:**
- `__init__()`: Initialize with LaB6 coil configuration
- `set_uniform_field(B0)`: Set uniform field strength in Tesla
- `set_supply_current(current, *supplies)`: Set current for specific power supplies
- `set_supply_currents(i1, i2, ..., i12)`: Set all 12 supply currents
- `compute_B(x, y, z)`: Calculate field at Cartesian coordinates
- `compute_B_cylindrical(R, z)`: Calculate field at cylindrical coordinates
- `compute_Aphi(R, z)`: Calculate azimuthal vector potential

#### `RingCurrent`
Represents a single ring current for magnetic field calculations.

**Constructor:**
- `RingCurrent(R, z, cur=1.0)`: Ring at radius R, position z, with current cur

**Methods:**
- `B(R, z)`: Magnetic field at point (R, z)
- `A(R, z)`: Vector potential at point (R, z)
- `Mutual(other)`: Mutual inductance with another ring

#### `CoilData`
Data structure for individual coil parameters.

**Attributes:**
- `color`: Coil color (YELLOW, PURPLE, BLACK)
- `supply_number`: Power supply number (1-12)
- `z`: Axial position (m)
- `a`: Coil radius (m)
- `num_turns`: Number of turns
- `current`: Current (A)

### Main Functions

- `create_BaO_coil_set()`: Create BaO coil configuration
- `create_LaB6_coil_set()`: Create LaB6 coil configuration  
- `calculate_field_from_ring_currents(rings, R, z)`: Calculate total field from ring list
- `calculate_flux_from_ring_currents(rings, R, z)`: Calculate total flux from ring list

### Constants

- `YELLOW`, `PURPLE`, `BLACK`: Coil color codes
- `mu0`: Permeability of free space (4π × 10⁻⁷ H/m)

## Coordinate System

- **Cartesian (x, y, z)**: Standard right-handed coordinates
- **Cylindrical (R, φ, z)**: R = √(x² + y²), φ = atan2(y, x)
- **z = 0**: Far end of machine from cathode
- **z positive**: Toward cathode

## Examples

See `example.py` for comprehensive usage examples including:
- Basic field calculations
- Custom coil configurations
- Low-level ring current calculations
- Different coordinate systems

## Physical Background

The calculations use exact elliptic integral solutions for the magnetic field of current rings. This provides high accuracy for:

- Magnetic field components (BR, Bz)
- Vector potential (Aφ)
- Magnetic flux
- Mutual inductance

The LAPD coil system consists of:
- **BaO coil set**: Yellow and Purple coils (main field coils)
- **LaB6 coil set**: BaO coils plus Black coils (full system)
- **12 power supplies**: Independent current control

## Authors

- **PP**: Original C++ implementation
- **JH**: Python implementation (2025-07-30)

Based on LAPD_coil_set.h with coil locations from SK.