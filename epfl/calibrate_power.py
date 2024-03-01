import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import data_analysis_utils as utils
from read_scope_data import read_trc_data

#======================================================================================
'''
Since CRDS data was taken in pulsed mode, RF power could not be read directly as shown on the power supply.
A photodiode was used to record light signal. That, together with RF power is recorded in CW operations.
The same photodiode records traces from pulsed mode, and that can be used as an indicator of RF power.

Data in this file was extracted from written record during the experiment.
'''
#======================================================================================

# power from RF generator 
pwrarr_a = np.array([200, 300, 400, 500, 550, 600])
pwrarr_b = np.array([200, 300, 400, 500, 600])
pwrarr_c = np.array([300, 400, 500, 600])
pwrarr_d = np.array([400, 500, 600])

pwr_dic = [pwrarr_a, pwrarr_b, pwrarr_c, pwrarr_d]

def func(x, a, b, c):
    return a*x**2 + b*x + c

#======================================================================================
# light signal unit in mV

def ltop_birdcage(light_level, pres, rt_data=False):
    '''
    input light signal level
    output corresponding RF generator power (approximate)
    pres -- pressure in mTorr
    if rt_data, returns CW data array taken at that pressure
    '''

    calibration_data = {
        1: np.array([85, 200, 350, 540, 640, 750]), # 1mTorr flow 0.013
        2: np.array([60, 160, 290, 450, 530, 630]), # 2mTorr flow 0.024
        5: np.array([30, 110, 210, 350, 520]),      # 5mTorr flow 0.06
        10: np.array([70, 170, 290, 430]),          # 10mTorr flow 0.11
        15: np.array([75, 160, 270, 420]),          # 15mTorr flow ?
        25: np.array([100, 200, 310]),              # 25mTorr flow 0.22
        35: np.array([85, 170, 290]),               # 35mTorr flow 0.22
        50: np.array([80, 170, 260])                # 50mTorr flow 0.22
    }

    if pres not in calibration_data:
        print('Invalid pressure value')
        return None

    ydata = calibration_data[pres]

    for pwrarr in pwr_dic:
        if len(ydata) == len(pwrarr):
            xdata = pwrarr

    popt, pcov = curve_fit(func, ydata, xdata)

    if True:
        plt.figure()
        plt.scatter(ydata, xdata)
        fxarr = np.linspace(ydata[0],ydata[2],100)
        plt.plot(fxarr, func(fxarr, *popt))
        plt.show()

    if light_level < np.min(ydata):
        print('Input value lower than calibration data range')
        return 0
    if light_level > np.max(ydata):
        print('Input value higher than calibration data range')
        return 999

    if rt_data:
        return func(light_level, *popt), ydata
    else:
        return func(light_level, *popt)
    
#======================================================================================

def ltop_solenoid(light_level, pres, rt_data=False):
    '''
    Same as above for solenoid antenna
    '''
    calibration_data = {
        0.5: np.array([70, 160, 270, 400, 530]),  # 0.5mTorr flow 0.007
        1: np.array([80, 150, 250, 370, 510]),    # 1 mTorr flow 0.012
        5: np.array([40, 110, 200, 320, 450]),    # 5 mTorr flow 0.06
        10: np.array([80, 160, 250, 380]),        # 10 mTorr flow 0.12
        15: np.array([60, 150, 250, 380]),        # 15 mTorr flow 0.16
        20: np.array([120, 210, 320]),            # 20 mTorr flow 0.19
        25: np.array([110, 200, 310]),            # 25 mTorr flow 0.21
        35: np.array([100, 190, 290])             # 35 mTorr flow 0.25
    }

    if pres not in calibration_data:
        print('Invalid pressure value')
        return None

    ydata = calibration_data[pres]

    for pwrarr in pwr_dic:
        if len(ydata) == len(pwrarr):
            xdata = pwrarr

    popt, pcov = curve_fit(func, ydata, xdata)

    if False:
        plt.figure()
        plt.scatter(ydata, xdata)
        fxarr = np.linspace(ydata[0], ydata[2], 100)
        plt.plot(fxarr, func(fxarr, *popt))
        plt.show()

    if light_level < np.min(ydata):
        print('Warning: Input value lower than calibration data range')

    if light_level > np.max(ydata):
        print('Warning: Input value higher than calibration data range')


    if rt_data:
        return func(light_level, *popt), ydata
    else:
        return func(light_level, *popt)
    
#======================================================================================

def get_power_solenoid(pressure, start, stop):
    path = r"C:\data\epfl\diagnostic-source\CRDS\scope-trc\solAnt"
    file_ls = utils.get_files_in_folder(path, modified_date='2023-06-01', omit_keyword='water')
    pwr_arr = np.empty((0, 2))

    for file in file_ls:
        if "C3" in file and '-'+str(pressure)+'mT' in file:
            power = utils.get_number_before_keyword(file, "W")
            
            light, tarr = read_trc_data(file)
            lavg = np.average(light[start:stop]) * 1e3 # take average light signal conver V to mV
            
            # Calibrated power according to CW light signal
            if pressure == 26: # 26mTorr is not in calibration data
                pwr = ltop_solenoid(lavg, 25)
            elif pressure == 36: # 36mTorr is not in calibration data
                pwr = ltop_solenoid(lavg, 35)
            else:
                pwr = ltop_solenoid(lavg, pressure)

            print(f'Listed power = {power}W  cal power = {pwr}W')
            pwr_arr = np.append(pwr_arr, np.array([[power,pwr]]), axis=0)

    return pwr_arr

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':

    print(ltop_birdcage(100, 25))