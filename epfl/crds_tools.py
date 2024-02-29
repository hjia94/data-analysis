import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import calibrate_power as cbp
import data_analysis_utils as utils
from read_scope_data import read_trc_data

def read_excel_sheets(filepath):
    """
    Read an Excel file and extract columns with non-zero values from each sheet.

    Parameters:
    filepath (str): The path to the Excel file.

    Returns:
    dict: A dictionary containing the sheet names as keys and the columns with non-zero values as values.
    """
    data_dict = {}

    # Read the Excel file
    xls = pd.ExcelFile(filepath)
    print(xls.sheet_names)

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        col_list = df.columns.tolist()
        # # Get the selected columns as a NumPy array
        selected_columns = df[['time, s - Plot 0', '1/c.tau, 1/cm - Plot 0']].values

        # Add the selected columns to the data dictionary
        data_dict[sheet_name] = selected_columns

    xls.close()

    return data_dict