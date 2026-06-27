'''
Read Network Analyzer (VNA) S-parameter data from an Agilent/Keysight CSV export.

Moved here from ``read/read_network_analyzer_data.py`` during the package reorg;
the reader for a ``.csv`` of complex S-parameters belongs with the other
format parsers in ``data_analysis.io``.

Usage example:
data_dict = read_NA_data("your_file.csv")  # Normal usage
data_dict = read_NA_data("your_file.csv", verbose=True)  # Debug mode

Verified against the real Agilent N5230C exports under
``D:/data/LAPD/Bdot_calibration/...`` on pandas 3.0 / numpy 2.x (the previous
"doesn't work with current pandas/numpy" TODO was stale and has been removed).
'''

import pandas as pd
import numpy as np


def read_NA_data(filepath, verbose=False):
    """
    Read Network Analyzer data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        verbose (bool): If True, print detailed debug information
    
    Returns:
        dict: Dictionary containing metadata and complex S-parameters
    """
    # Read metadata
    metadata = []
    with open(filepath, 'r') as file:
        for i in range(5):
            line = file.readline().strip()
            metadata.append(line)
    
    # Read data + column names in one pass: row 8 (0-based index 7) is the header.
    data = pd.read_csv(filepath, skiprows=7, header=0)
    column_names = data.columns.tolist()

    if verbose:
        print("\nAll column names:")
        for i, col in enumerate(column_names):
            print(f"{i}: {repr(col)}")
    
    # Create dictionary for spectral data
    spectral_dict = {
        'metadata': {
            'ICSV_version': metadata[0],
            'instrument': metadata[1],
            'model': metadata[2],
            'datetime': metadata[3],
            'source': metadata[4]
        }
    }
    
    # Find the 'END' marker in frequency data to determine valid data length
    freq_data = data['Freq(Hz)']
    end_idx = freq_data[freq_data == 'END'].index
    if len(end_idx) == 0:
        raise ValueError("No 'END' marker found in frequency data")
    end_idx = end_idx[0]
    
    # Add frequency data as float values up to END marker
    freq_data = freq_data[:end_idx]
    spectral_dict['Freq(Hz)'] = freq_data.astype(float).values
    
    # Process S-parameters and combine real/imaginary parts
    for col in column_names:
        if 'REAL' in col and col.startswith('S'):
            if verbose:
                print(f"\nProcessing column: {repr(col)}")
            
            # Get base name (S-parameter number)
            base_name = col.split('REAL')[0].strip(' ()')
            if verbose:
                print(f"Base name: {repr(base_name)}")
            
            # Look for matching imaginary part
            imag_col = None
            for potential_imag in column_names:
                if potential_imag.startswith(base_name) and 'IMAG' in potential_imag:
                    imag_col = potential_imag
                    break
            
            if imag_col:
                if verbose:
                    print(f"Found matching pair:")
                    print(f"  Real: {repr(col)}")
                    print(f"  Imag: {repr(imag_col)}")
                
                # Combine real and imaginary into single complex entry, using same end index
                real_data = data[col].values[:end_idx]
                imag_data = data[imag_col].values[:end_idx]
                spectral_dict[base_name] = real_data + 1j * imag_data
                    
                if verbose:
                    print(f"Added {base_name} to dictionary")
            elif verbose:
                print(f"No matching imaginary column found for {repr(col)}")
    
    if verbose:
        print("\nDictionary keys:", spectral_dict.keys())

    return spectral_dict

