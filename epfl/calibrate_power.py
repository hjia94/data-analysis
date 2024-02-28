import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

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

def ltop_solenoid(light_level, pres, ver=0, rt_data=False):
    '''
    same as above for solenoid antenna
    ver -- The photodiode was moved on Thursday Jun.1st before CRDS data aqcuisition because camera was reinstalled to check laser profile
    Therefore, data taken on May.31 should use ver=1 
    '''
    if pres == 0.5:  # 0.5mTorr flow 0.007
        ydata = np.array([70, 160, 270, 400, 530])

    if pres == 1:    # 1 mTorr flow 0.012
        ydata = np.array([80, 150, 250, 370, 510])

    if pres == 5:    # 5 mTorr flow 0.06
        if ver == 0:
            ydata = np.array([40, 110, 200, 320, 450])
        else:
            ydata = np.array([45, 120, 230, 360, 520])

    if pres == 10:  # 10 mTorr flow 0.12
        ydata = np.array([80, 160, 250, 380])

    if pres == 15:  # 15 mTorr flow 0.16
        ydata = np.array([60, 150, 250, 380])

    if pres == 20:   # 20 mTor flow 0.19
        ydata = np.array([120, 210, 320])

    if pres == 25:   # 25 mTorr flow 0.21
        ydata = np.array([110, 200, 310])

    if pres == 35:  # 35 mTorr flow 0.25
        ydata = np.array([100, 190, 290])

    for pwrarr in pwr_dic:
        if len(ydata) == len(pwrarr):
            xdata = pwrarr

    popt, pcov = curve_fit(func, ydata, xdata)

    if False:
        plt.figure()
        plt.scatter(ydata, xdata)

        fxarr = numpy.linspace(ydata[0],ydata[2],100)
        plt.plot(fxarr, func(fxarr, *popt))
        plt.show()
    
    if light_level < np.min(ydata):
        print ('Input value lower than calibration data range')
        return 0
    if light_level > np.max(ydata):
        print ('Input value higher than calibration data range')
        return 999

    if rt_data:
        return func(light_level, *popt), ydata
    else:
        return func(light_level, *popt)
    
#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':

    print(ltop_birdcage(100, 25))