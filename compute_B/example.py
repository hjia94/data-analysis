#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the compute_B package for LAPD magnetic field calculations.

This script demonstrates both high-level and low-level usage of the compute_B package.
"""

import sys
import os

# Add the parent directory to the path so we can import compute_B
# This would not be needed if the package is properly installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the compute_B package
import compute_B

def example_high_level():
    """Demonstrate high-level usage with LAPDCoilSet."""
    print("=" * 50)
    print("HIGH-LEVEL USAGE: LAPDCoilSet")
    print("=" * 50)
    
    # Create LAPD coil set instance
    lapd = compute_B.LAPDCoilSet()
    
    # Set uniform field of 0.1 Tesla
    lapd.set_uniform_field(0.1)
    print("Set uniform field to 0.1 Tesla")
    
    # Calculate field at center of machine
    x, y, z = 0.0, 0.0, 8.5  # Center point
    Bx, By, Bz, Bmag = lapd.compute_B(x, y, z)
    
    print(f"\nMagnetic field at center (x={x}, y={y}, z={z}):")
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

def example_low_level():
    """Demonstrate low-level usage with RingCurrent."""
    print("\n" + "=" * 50)
    print("LOW-LEVEL USAGE: RingCurrent")
    print("=" * 50)
    
    # Create a simple ring current
    ring = compute_B.RingCurrent(R=1.0, z=0.0, cur=1000.0)  # 1m radius, 1000A
    print(f"Created ring current: R={ring.R}m, z={ring.z}m, I={ring.cur}A")
    
    # Calculate field along the axis
    print("\nMagnetic field along axis:")
    for z in [0.0, 0.5, 1.0, 2.0]:
        BR, Bz, Bmag = compute_B.calculate_field_from_ring_currents([ring], 0.0, z)
        print(f"  z={z:3.1f}m: BR={BR:.6f}T, Bz={Bz:.6f}T, |B|={Bmag:.6f}T")
    
    # Calculate field off-axis
    print("\nMagnetic field off-axis (z=0):")
    for R in [0.0, 0.5, 1.0, 1.5]:
        BR, Bz, Bmag = compute_B.calculate_field_from_ring_currents([ring], R, 0.0)
        print(f"  R={R:3.1f}m: BR={BR:.6f}T, Bz={Bz:.6f}T, |B|={Bmag:.6f}T")

def example_coil_configurations():
    """Demonstrate different coil configurations."""
    print("\n" + "=" * 50)
    print("COIL CONFIGURATIONS")
    print("=" * 50)
    
    # Show available coil sets
    bao_coils = compute_B.create_BaO_coil_set()
    lab6_coils = compute_B.create_LaB6_coil_set()
    
    print(f"BaO coil set: {len(bao_coils)} coils")
    print(f"LaB6 coil set: {len(lab6_coils)} coils")
    
    # Show coil colors
    color_names = {compute_B.YELLOW: "YELLOW", compute_B.PURPLE: "PURPLE", compute_B.BLACK: "BLACK"}
    
    print(f"\nCoil color distribution in LaB6 set:")
    color_counts = {}
    for coil in lab6_coils:
        color = color_names[coil.color]
        color_counts[color] = color_counts.get(color, 0) + 1
    
    for color, count in color_counts.items():
        print(f"  {color}: {count} coils")

def main():
    """Run all examples."""
    print("COMPUTE_B PACKAGE EXAMPLE")
    print("Package version:", getattr(compute_B, '__version__', 'unknown'))
    print("Package description:", getattr(compute_B, '__description__', 'N/A'))
    
    try:
        example_high_level()
        example_low_level() 
        example_coil_configurations()
        
        print("\n" + "=" * 50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())