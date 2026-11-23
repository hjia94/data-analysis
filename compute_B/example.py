#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the compute_B package for LAPD magnetic field calculations.

Set the 12 supply currents -- directly, or via the uniform-field preset -- and
plot the resulting Bz on the machine axis.

Edit CONFIGS in __main__ to try your own supply settings.
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

# Supplies grouped by coil color; z increases toward the cathode, so 11-12
# (black) sit at the source end and 3-4 (yellow) at the far end.
SUPPLY_GROUPS = [
    ("yellow", (1, 2, 3, 4)),
    ("purple", (5, 6, 7, 8, 9, 10)),
    ("black", (11, 12)),
]


def uniform_field_currents(B0=0.1):
    """Supply currents (A, 12-list) for a uniform field of B0 Tesla."""
    lapd = LAPDCoilSet()
    lapd.set_uniform_field(B0)
    return lapd.get_supply_currents()


def currents_from_groups(yellow=0.0, purple=0.0, black=0.0):
    """Supply currents (A, 12-list) from one current per coil-color group."""
    by_group = {"yellow": yellow, "purple": purple, "black": black}
    currents = [0.0] * 12
    for name, supplies in SUPPLY_GROUPS:
        for supply in supplies:
            currents[supply - 1] = float(by_group[name])
    return currents


def print_supply_currents(currents, label=""):
    """Print the 12 supply currents, grouped by coil color."""
    print(f"\nSupply currents{' -- ' + label if label else ''}:")
    for name, supplies in SUPPLY_GROUPS:
        listed = ", ".join(f"{s}:{currents[s - 1]:.0f}" for s in supplies)
        print(f"  {name:6s} {listed} A")


def compute_Bz_onAxis(supply_currents, z_min=0.0, z_max=17.0, n_points=500):
    """Bz on axis for the given supply currents -> (z [m], Bz [kG], port number)."""
    lapd = LAPDCoilSet()
    lapd.set_supply_currents(*supply_currents)

    z_points = np.linspace(z_min, z_max, n_points)
    Bz = np.array([lapd.compute_B(0.0, 0.0, z)[2] for z in z_points])
    ports = np.array([lapd.z_to_eff_port_number(z) for z in z_points])

    return z_points, Bz * 10.0, ports  # 1 T = 10 kG


def plot_Bz_onAxis(configs, z_min=0.0, z_max=17.0, n_points=500):
    """
    Plot Bz on axis for one or more supply-current configurations.

    configs : list of (label, supply_currents) pairs.
    Returns the list of (label, z, Bz_kG, ports) tuples that were plotted.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    results = []

    for label, currents in configs:
        z_points, Bz_kG, ports = compute_Bz_onAxis(currents, z_min, z_max, n_points)
        ax1.plot(ports, Bz_kG, linewidth=2, label=label)
        results.append((label, z_points, Bz_kG, ports))

    ax1.set_xlabel('Port Number', fontsize=12)
    ax1.set_ylabel('Bz (kiloGauss)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.axhline(0.0, color='gray', linewidth=0.8)

    # All configs share the z grid, so any one of them sets the port axis.
    _, z_points, _, ports = results[0]
    ax1.set_xlim(0, np.max(ports))
    ax1.set_xticks(np.arange(0, int(np.max(ports)) + 1, 10))

    # Second x-axis labels the same port positions by distance instead.
    ax2 = ax1.twiny()
    ax2.set_xlabel('Distance (m)', fontsize=12)
    tick_positions, tick_labels = [], []
    for target_distance in np.arange(0, int(np.max(z_points)) + 1, 2):
        idx = np.argmin(np.abs(z_points - target_distance))
        if abs(z_points[idx] - target_distance) < 0.5:
            tick_positions.append(ports[idx])
            tick_labels.append(f'{target_distance:.0f}')
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()
    plt.show()

    return results


if __name__ == "__main__":

    CONFIGS = [
        ("uniform 0.1 T", uniform_field_currents(B0=0.1)),
        ("tandem mirror", [900, 900, 900, 900,
                           3600, 150, 300, 300, 150, 3600,
                           555, 555]),
    ]

    for label, currents in CONFIGS:
        print_supply_currents(currents, label)

    results = plot_Bz_onAxis(CONFIGS)

    print("\nBz on axis (kG):")
    for label, _, Bz_kG, _ in results:
        print(f"  {label:16s} min {Bz_kG.min():7.3f}   max {Bz_kG.max():7.3f}")
