'''
General unitility functions used for all data analysis.

Arthur: Jia Han
Date: 2024-02-28

Included functions:
- get_files_in_folder
- get_number_before_keyword
- save_to_npy
- read_from_npy
'''

import os
import re
import numpy as np
from datetime import datetime

def get_files_in_folder(folder_path, modified_date=None, omit_keyword=None):
    """
    Get a list of all files in a given folder and its subfolders.

    Args:
        folder_path (str): The path to the folder.
        modified_date (str, optional): The specific date in the format 'YYYY-MM-DD'. Defaults to None.
        omit_keyword (str, optional): The keyword to omit files. Defaults to None.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if omit_keyword is not None and omit_keyword in file:
                continue
            if modified_date is None:
                file_list.append(file_path)
            else:
                last_modified = os.path.getmtime(file_path)
                last_modified_date = datetime.fromtimestamp(last_modified).date()
                if last_modified_date == datetime.strptime(modified_date, '%Y-%m-%d').date():
                    file_list.append(file_path)
    return file_list
    
def get_number_before_keyword(string, keyword, verbose=False):
    """
    Extracts the number before a given keyword in a string.

    Parameters:
    string (str): The input string.
    keyword (str): The keyword to search for.
    verbose (bool, optional): If True, prints the number found. Defaults to False.

    Returns:
    float or None: The number found before the keyword, or None if no match is found.
    """
    match = re.search(r'(\d+)'+keyword, string)
    if match:
        if verbose:
            print("Number found:", match.group(1))
        return float(match.group(1))
    else:
        if verbose:
            print("No match found.")
        return None
    
def save_to_npy(data, npy_file_path):
    """
    Save data to a .npy file.

    Parameters:
    data (numpy.ndarray): The data to be saved.
    npy_file_path (str): The file path to save the .npy file.

    Returns:
    None
    """
    if os.path.exists(npy_file_path):
        user_input = input("The file already exists. Do you want to continue and overwrite it? (y/n): ")
        if user_input.lower() == 'n':
            return None
        if user_input.lower() != 'y' and user_input.lower() != 'n':
            print('Not the correct response. Please type "y" or "n"')

    np.save(npy_file_path, data)
    print(f"Data saved to {npy_file_path}")

def read_from_npy(npy_file_path):
    """
    Read data from a NumPy file (.npy).

    Args:
        npy_file_path (str): The path to the .npy file.

    Returns:
        numpy.ndarray or None: The loaded data if the file exists, None otherwise.
    """
    if not os.path.exists(npy_file_path):
        print(f"The file {npy_file_path} does not exist.")
        return None
    data = np.load(npy_file_path, allow_pickle=True)
    return data