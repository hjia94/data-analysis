#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the compute_B package for LAPD magnetic field calculations.

This script demonstrates comprehensive usage of the compute_B package with a connected workflow:

1. uniform_field_to_currents(): Configure uniform magnetic field and extract supply currents
2. plot_Bz_onAxis(): Use those currents to compute and plot the Bz field along the machine axis

Features demonstrated:
- Uniform field configuration and supply current management
- Magnetic field computation and visualization  
- Individual supply current access and validation
- Multiple API access patterns (direct attributes, getters/setters, bulk operations)
- Connected workflow passing data between functions

Updated to showcase the latest enhancements to the LAPDCoilSet class.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path so we can import compute_B
# This would not be needed if the package is properly installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the compute_B package
from compute_B import LAPDCoilSet


def uniform_field_to_currents(B0=0.1):
    """
    Example 1: Given a magnetic field (set_uniform_field), print a list of supply currents.
    
    This example demonstrates how to:
    1. Create an LAPD coil set
    2. Set a uniform magnetic field
    3. Extract and display the resulting supply currents using multiple methods
    4. Demonstrate individual supply access
    
    Parameters:
    -----------
    B0 : float, optional
        Desired uniform magnetic field strength in Tesla (default: 0.1 T)
        
    Returns:
    --------
    list : List of 12 supply currents in Amperes for the uniform field
    """
    
    # Create LAPD coil set
    lapd = LAPDCoilSet()
    
    print(f"\nSetting uniform field: B0 = {B0:.3f} T")
    
    # Set uniform field
    lapd.set_uniform_field(B0)

    currents = lapd.get_supply_currents()
    print("All supply currents (A):")
    for i, current in enumerate(currents, 1):
        print(f"  Supply {i:2d}: {current:8.1f} A")

    # Reset to desired field and return currents
    lapd.set_uniform_field(B0)
    final_currents = lapd.get_supply_currents()
    
    return final_currents


def plot_Bz_onAxis(supply_currents):
    """
    Example 2: With a given list of supply currents, plot the magnetic field B_z on Z axis.
    
    This example demonstrates how to:
    1. Set specific supply currents
    2. Compute Bz along the z-axis (x=y=0)
    3. Plot Bz vs distance and port number
    
    Parameters:
    -----------
    supply_currents : list
        List of 12 supply currents in Amperes [I1, I2, ..., I12]
    """

    lapd = LAPDCoilSet()
    
    lapd.set_supply_currents(*supply_currents)
    
    # Define z-axis range for plotting (along the machine length)
    z_min, z_max = 0.0, 17.0  # Full machine length in meters
    z_points = np.linspace(z_min, z_max, 500)
    
    # Compute Bz along the z-axis
    Bz_values = []
    port_numbers = []
    
    print("\nComputing magnetic field along z-axis...")
    for z in z_points:
        # Compute field at (x=0, y=0, z)
        _, _, Bz, _ = lapd.compute_B(0.0, 0.0, z)
        Bz_values.append(Bz)
        
        # Convert z to port number
        port_num = lapd.z_to_eff_port_number(z)
        port_numbers.append(port_num)
    
    # Convert to numpy arrays
    z_points = np.array(z_points)
    Bz_values = np.array(Bz_values)
    port_numbers = np.array(port_numbers)
    
    # Convert Bz from Tesla to kiloGauss
    Bz_kG = Bz_values * 10.0  # 1 Tesla = 10 kiloGauss
    
    # Create the plot with two x-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot Bz vs port number (main axis)
    color = 'tab:blue'
    ax1.set_xlabel('Port Number', fontsize=12)
    ax1.set_ylabel('Bz (kiloGauss)', color=color, fontsize=12)
    line1 = ax1.plot(port_numbers, Bz_kG, color=color, linewidth=2, label='Bz field')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Set port number axis to start from 0 on the left
    port_min, port_max = np.min(port_numbers), np.max(port_numbers)
    ax1.set_xlim(0, port_max)  # Start from 0, go to maximum port number
    
    # Set nice port number ticks
    port_ticks = np.arange(0, int(port_max) + 1, 10)  # Every 10 ports
    ax1.set_xticks(port_ticks)
    
    # Create second x-axis for distance
    ax2 = ax1.twiny()
    color = 'tab:red'
    ax2.set_xlabel('Distance (m)', color=color, fontsize=12)
    ax2.tick_params(axis='x', labelcolor=color)
    
    # Both axes must represent the same physical space
    # Find distance values that correspond to round port numbers for tick placement
    distance_tick_values = np.arange(0, int(np.max(z_points)) + 1, 2)  # Every 2 meters
    tick_positions = []
    tick_labels = []
    
    for target_distance in distance_tick_values:
        # Find the port number where distance is closest to target_distance
        idx = np.argmin(np.abs(z_points - target_distance))
        port_pos = port_numbers[idx]
        actual_distance = z_points[idx]
        
        # Only add if the distance is reasonably close to our target
        if abs(actual_distance - target_distance) < 0.5:  # Within 0.5 meters
            tick_positions.append(port_pos)
            tick_labels.append(f'{target_distance:.0f}')
    
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(ax1.get_xlim())  # Same limits as port axis
    
    plt.tight_layout()
    plt.show()
    
    return z_points, Bz_kG, port_numbers



if __name__ == "__main__":

    uniform_currents = uniform_field_to_currents(B0=0.1)  # Get currents for 0.1 T uniform field

    z_points, Bz_kG, port_numbers = plot_Bz_onAxis(uniform_currents)
    


