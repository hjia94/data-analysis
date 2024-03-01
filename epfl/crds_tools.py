import sys
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis")
sys.path.append(r"C:\Users\hjia9\Documents\GitHub\data-analysis\read")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import calibrate_power as cbp
import data_analysis_utils as utils
from read_scope_data import read_trc_data


def get_sheet_names(filepath):
    xls = pd.ExcelFile(filepath)
    sheetname_ls = xls.sheet_names
    xls.close()
    return sheetname_ls

def read_excel_sheets(filepath, sheet_name, col_num):

    # Read the Excel file
    xls = pd.ExcelFile(filepath)

    # Read the sheet into a DataFrame
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    # # Get the selected columns as a NumPy array
    selected_columns = df.iloc[:, col_num].values

    xls.close()

    return selected_columns