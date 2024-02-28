'''
General unitility functions used for all data analysis.

Arthur: Jia Han
Date: 2024-02-28
'''

import os
import re

def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list
    
def get_number_before_keyword(string, keyword, verbose=False):
    match = re.search(r'(\d+)'+keyword, string)
    if match:
        if verbose:
            print("Number found:", match.group(1))
        return float(match.group(1))
    else:
        if verbose:
            print("No match found.")
        return None