# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:48:34 2023

@author: coenberns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, timedelta
import os
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import seaborn as sns
from numpy.polynomial.polynomial import Polynomial



#file = r'/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/RF readings miniPC desk animal facility/Battery Tests/10232023_2series_incasing_n5_1s840ms_withbatmeas_FULL'

# file = r'C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF Readings miniPC/Battery testing/Mode 2 recs - 1s840ms sleep/10232023_n5_1s840ms_1per5_battmeas_increasingV.txt'

# %%
def plot_battery_temp(file, plot = False, bat = True, temp = True):
    _, filename = os.path.split(file)
    # extract date from the filename
    date = filename.split('_')[0]
    # Creating datetime object
    # which takes in "MMDDYYYY" like only US people write date order
    date = datetime.strptime(date, '%m%d%Y')

    dat = pd.read_csv(file, header=0, dtype=str, delimiter='|', names=[
                    'realtime', 'misc', 'packet', 'msg', 'rssi'])
    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('06')]
    dat = dat.reset_index(drop=True)

    counter = dat.packet.astype(int)
    new_counter = [0]
    for j, ele in enumerate(counter[1:]):
        step = counter[j+1]-counter[j]
        if step > 0:
            new_counter.append(step+new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])

    abscounterseries = pd.Series(new_counter, name='packet_re_idx')

    dat = pd.concat((dat, abscounterseries), axis=1)

    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    # datetime_counter = time.time()
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary, with additional glitch values excluded
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    dat['rollover'] = dat['time_diff'] < 0
    dat['glitch'] = (dat['time_diff'] > -5) & (dat['rollover'])

    n_glitch = 50
    dat['glitch'] = dat['glitch'].rolling(window=n_glitch, min_periods=1).max().astype(bool)
    dat.loc[dat['glitch'], 'realtime'] = np.nan
    dat['correct_rollover'] = dat['rollover'] & ~dat['glitch'] 
    dat['days_to_add'] = dat['correct_rollover'].cumsum()
    dat['corrected_realtime'] = dat['realtime'] + pd.to_timedelta(dat['days_to_add'], unit='D')
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    dat['time'] = dat['elapsed_t'].dt.total_seconds()

    # Split the 'msg' data and calculate battery and temperature
    dat['new_msg'] = dat['msg'].str.strip().str.split(' ')
    bat_temp_data = dat['new_msg'].apply(lambda x: pd.Series({
        'V_bat': int(''.join(x[0:2]), 16),  # most significant byte is first
        'Temp (C)': int(''.join(x[2:4]), 16)
    }))
    # Merge the new data into the original DataFrame
    dat = pd.concat([dat, bat_temp_data], axis=1)

    # Your data is now in 'dat' DataFrame and you can proceed with your analysis
    datcols = ['time', 'V_bat', 'Temp (C)']
    battempdat = dat[datcols]

    if plot == True: 
        # print(battempdat.head())
        if bat == True: 
            x=battempdat['time']
            y= battempdat['V_bat']
            plt.plot(x,y, label = "Mode 1 recording")
            plt.title('Battery voltage vs time')
            plt.xlabel('Time [s]')
            plt.ylabel('Battery voltage [mV]')
            plt.legend()
            plt.show()

        else: 
            print("No battery plot wanted?")

        if temp == True: 
            x=battempdat['time']
            z=battempdat['Temp (C)']
            plt.plot(x,z)
            plt.xlabel('Time [s]')
            plt.ylabel('Temperature [C]')
            plt.show()


        else: 
            print("No temperature plot wanted?")

    return battempdat


#%%
def define_lasthigh(df):
    # Start with the first data point
    last_high = [df.iloc[0]]
    
    # Iterate through the dataframe
    for i in range(1, len(df)):
        # If the voltage goes up again after going down, keep the last high point
        if df['V_bat'].iloc[i] > df['V_bat'].iloc[i - 1] and \
            all(df['V_bat'].iloc[i] >= df['V_bat'].iloc[j] for j in range(i, len(df))):
            last_high.append(df.iloc[i])
        # If we reach the end, we always include the last point
        elif i == len(df) - 1:
            last_high.append(df.iloc[i])
        # If the voltage goes down, keep the last point before it goes down
        elif df['V_bat'].iloc[i] < df['V_bat'].iloc[i - 1]:
            last_high.append(df.iloc[i - 1])

    return pd.DataFrame(last_high)

#%%
def exp_decreasing(x, a, b, c):
    return a * np.exp(-b * x) + c
# %% 
#file = r'/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/RF readings miniPC desk animal facility/Battery Tests/10232023_2series_incasing_n5_1s840ms_withbatmeas_FULL'
#full data for 13+ days 2||2 continuous experiment

filepaths = []
#Initial analysis folder!!
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 1 new")
files=list(dir.glob('*'))
filepaths.extend(files)

battery_data_mode1 = {}
times1={}
for i in range(len(filepaths)):
    file = filepaths[i]
    batmeas = plot_battery_temp(file, plot=False)
    # batmeas=batmeas[batmeas['V_bat']]
    batmeas['time'] = batmeas['time'] / (3600)
    #batmeas_smoothed = lowess(batmeas['V_bat'], batmeas['time'], frac=0.2)
    # params, cov = curve_fit(exp_decreasing, batmeas['time'], batmeas['V_bat'], maxfev=5000)
    # smoothed = exp_decreasing(batmeas['time'], *params)
    high_values = define_lasthigh(batmeas)
    params,cov = curve_fit(exp_decreasing, high_values['time'], high_values['V_bat'], maxfev=5000)
    smoothed = exp_decreasing(high_values['time'],*params)
    battery_data_mode1[i] = smoothed
    times1[i] = high_values['time']

#%%

filepaths2 = []
#Initial analysis folder!!
dir2 = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 2 - 1840ms sleep")
files2=list(dir2.glob('*'))

battery_data_mode2 = {}
times2={}
for i in range(len(files2)):
    file2 = files2[i]
    batmeas2 = plot_battery_temp(file2, plot=False)
    batmeas2=batmeas2[batmeas2['V_bat']<3070]
    batmeas2['time'] = batmeas2['time'] / (3600)
    #batmeas_smoothed = lowess(batmeas['V_bat'], batmeas['time'], frac=0.2)
    # params, cov = curve_fit(exp_decreasing, batmeas2['time'], batmeas2['V_bat'], maxfev=5000)
    # smoothed = exp_decreasing(batmeas2['time'], *params)
    high_values2 = define_lasthigh(batmeas2)
    params,cov = curve_fit(exp_decreasing, high_values2['time'], high_values2['V_bat'], maxfev=10000)
    smoothed2 = exp_decreasing(high_values2['time'],*params)
    battery_data_mode2[i] = smoothed2
    times2[i] = high_values2['time']



#%%

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,6), sharey=True)
y_limits = [2.35,3.05]
ax1.set_ylim(y_limits)
ax2.set_ylim(y_limits)

for i in range(len(battery_data_mode1)-2):
    # initial_time = 
    time = times1[i]-0.4
    battery = (battery_data_mode1[i]+150)/1000
    if i == 0:
        ax1.plot(time, battery, label = f'2-2 parallel')
        ax1.axhline(battery.min(), color = '#1f77b4', linestyle='dashed', label = r'$V_{dead,2/2}=' + f'{battery.min():.2f}' + '$')
    if i ==1:
        ax1.plot(time, battery, label = f'2 series')
        ax1.axhline(battery.min(), color = '#ff7f0e', linestyle='dashed', label = r'$V_{dead,2s}=' + f'{battery.min():.2f}' + '$')
    #Poly regression - does not function properly
    # degree=2
    # coefs = Polynomial.fit(time, battery, degree).convert().coef
    # p = np.poly1d(coefs[::-1])
    # smoothed_battery = p(initial_time)
    # ax1.plot(initial_time, smoothed_battery,color='blue', label='Pol regression')
    # #Lowess smoothing - could work
    # smoothed = lowess(battery, time, frac=0.45)
    # ax1.plot(smoothed[:,0],smoothed[:, 1], label=f'Lowess {i+1}')
ax1.set_xlim([0,12])
ax1.set_xlabel('Time [hrs]', size=16)
ax1.set_title('Always-on mode', size=18)
ax1.set_ylabel('Battery [V]', size=16) 
ax1.legend(loc='center')   

for i in range(len(battery_data_mode2)):
    time = times2[i]
    battery = battery_data_mode2[i]/1000
    if i == 0:
        ax2.plot(time, battery, label = f'2-2 parallel')
        ax2.axhline(battery.min(), color = '#1f77b4', linestyle='dashed', label = r'$V_{dead,2/2}=' + f'{battery.min():.2f}' + '$')
    if i ==1:
        ax2.plot(time, battery, label = f'2 series')
        ax2.axhline(battery.min(), color = '#ff7f0e', linestyle='dashed', label = r'$V_{dead,2s}=' + f'{battery.min():.2f}' + '$')
    # smoothed = lowess(battery, time, frac=0.3)
    # ax2.plot(smoothed[:,0],smoothed[:, 1], label=f'Measurement {i+1}')
ax2.set_title('Duty cycling mode ', size=18)
ax2.set_xlabel('Time [days]', size=16)
ax2.set_xlim([0,15])
ax2.legend(loc='center')

plt.show()

#%%
fig, ax = plt.subplots(figsize=(10,5))
y_limits = [2.35,3.05]
ax.set_ylim(y_limits)

for i in range(len(battery_data_mode1)-2):
    # initial_time = 
    time = (np.array(times1[i])-0.4)/24
    battery = (np.array(battery_data_mode1[i])+150)/1000
    if i == 0:
        ax.plot(time, battery, label = f'2-2 parallel', color = '#4c72b0', ls='dashed')
        min_idx=np.argmin(battery)
        min_time=time[min_idx]
        min_bat=battery[min_idx]
        ax.scatter(min_time, min_bat, color = '#4c72b0', marker='x',s=100)
        ax.text(min_time+0.2, min_bat, f'{min_bat:.2f} V', color='#4c72b0', verticalalignment='top', horizontalalignment='left')
    if i ==1:
        ax.plot(time, battery, label = f'2 series', color = '#dd8452', ls='dashed')
        min_idx=np.argmin(battery)
        min_time=time[min_idx]
        min_bat=battery[min_idx]
        ax.scatter(min_time, min_bat, color = '#dd8452', marker='x',s=100)
        ax.text(min_time-0.2, min_bat+0.025, f'{min_bat:.2f} V', color='#dd8452', verticalalignment='top', horizontalalignment='right')
    #Poly regression - does not function properly
    # degree=2
    # coefs = Polynomial.fit(time, battery, degree).convert().coef
    # p = np.poly1d(coefs[::-1])
    # smoothed_battery = p(initial_time)
    # ax1.plot(initial_time, smoothed_battery,color='blue', label='Pol regression')
    # #Lowess smoothing - could work
    # smoothed = lowess(battery, time, frac=0.45)
    # ax1.plot(smoothed[:,0],smoothed[:, 1], label=f'Lowess {i+1}')
ax.set_ylabel('Battery [V]', size=16) 

for i in range(len(battery_data_mode2)):
    time = np.array(times2[i])/24
    battery = np.array(battery_data_mode2[i])/1000
    if i == 0:
        ax.plot(time, battery, label = f'2-2 parallel', color = '#4c72b0')
        min_idx=np.argmin(battery)
        min_time=time[min_idx]
        min_bat=battery[min_idx]
        ax.scatter(min_time, min_bat, color = '#4c72b0', marker='x',s=100)
        ax.text(min_time-0.2, min_bat, f'{min_bat:.2f} V', color='#4c72b0', verticalalignment='top', horizontalalignment='right')
    if i ==1:
        ax.plot(time, battery, label = f'2 series', color = '#dd8452')
        min_idx=np.argmin(battery)
        min_time=time[min_idx]
        min_bat=battery[min_idx]
        ax.scatter(min_time, min_bat, color = '#dd8452', marker='x', s=100)
        ax.text(min_time+0.2, min_bat, f'{min_bat:.2f} V', color='#dd8452', verticalalignment='top', horizontalalignment='left')
    # smoothed = lowess(battery, time, frac=0.3)
    # ax2.plot(smoothed[:,0],smoothed[:, 1], label=f'Measurement {i+1}')
ax.set_xlabel('Time [days]', size=16)
ax.set_xlim([0,14])

patch1 = mpatches.Patch(color='#4c72b0', label='2-2 parallel')
patch2 = mpatches.Patch(color='#dd8452', label='2 series')
ax.legend(handles=[patch1, patch2], loc='center')
fig.show()

#%%
# Load and process data for both modes
filepaths = list(Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 1 new").glob('*'))
filepaths2 = list(Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 2 - 1840ms sleep").glob('*'))

battery_data = {}
times = {}

for mode, paths in enumerate([filepaths, filepaths2], start=1):
    for i, file in enumerate(paths):
        batmeas = plot_battery_temp(file, plot=False)
        if mode == 1:
            batmeas['time'] = batmeas['time'] / (3600*24)  # convert to hours
        else:
            batmeas = batmeas[batmeas['V_bat'] < 3070]
            batmeas['time'] = batmeas['time'] / (3600 * 24)  # convert to days

        high_values = define_lasthigh(batmeas)
        params, cov = curve_fit(exp_decreasing, high_values['time'], high_values['V_bat'], maxfev=10000)
        smoothed = exp_decreasing(high_values['time'], *params)
        battery_data[(mode, i)] = smoothed
        times[(mode, i)] = high_values['time']

#%%
# Plotting
fig, ax = plt.subplots(figsize=(18, 10))
y_limits = [2.35, 3.05]
ax.set_ylim(y_limits)

# Plot for mode 1
for i in range(len(filepaths) - 2):
    time = times[(1, i)] - 0.4  # adjust time if necessary
    battery = (battery_data[(1, i)] + 150) / 1000
    ax.plot(time, battery, label=f'Mode 1 - Measurement {i+1}')

# Plot for mode 2
for i in range(len(filepaths2)):
    time = times[(2, i)] * 24  # convert days to hours if necessary
    battery = battery_data[(2, i)] / 1000
    ax.plot(time, battery, label=f'Mode 2 - Measurement {i+1}', linestyle='--')

ax.set_xlabel('Time [hrs]', size=16)
ax.set_ylabel('Battery [V]', size=16)
ax.legend(loc='center')
ax.set_xlim(0,14)
plt.show()


#%%
file = r"C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF Readings miniPC\Battery testing/Mode 2 recs - 1s840ms sleep/10232023_2-2config_n5_1s840ms_continuous_1per5batmeas_FULL_incl100min_sleep.txt"
battempdat = plot_battery_temp(file, True, False)




# %%
# FOR ONE PARTICULAR FILE 
battempdat_short = battempdat
# battempdat_short = battempdat[battempdat['time']>1e6]
battempdat_short['time'] = battempdat_short['time'] - battempdat_short['time'].iloc[0]

x=battempdat_short['time']/3600
y=battempdat_short['V_bat']
smoothed = lowess(y, x, frac=0.1)
# smoothed1 = lowess(y1, x1, frac=0.1)
# smoothed2 = lowess(y2, x2, frac=0.1)
# smoothed3 = lowess(y3, x3, frac=0.1)
# q=17150
# r=2695
# u=23300
# v=battempdat_short['V_bat'].max()
#plt.plot(x,y, label = 'Original data') 
plt.plot(smoothed[:,0], smoothed[:,1])
# plt.plot(smoothed1[:,0], smoothed1[:,1])
# plt.plot(smoothed2[:,0], smoothed2[:,1])
# plt.plot(smoothed2[:,0], smoothed2[:,1])
#plt.plot(q,r,'rx', label = 'start sleep', markersize = 8)
#plt.plot(u,v, 'ro', label = 'end sleep', markersize=5)
plt.title('Time vs Battery voltage plot - lowess smoothened')
plt.xlabel('Time [hrs]')
plt.ylabel('Battery voltage [mV]')
plt.legend()
plt.show()


# x1=battempdat_short['time']/3600
# y1=battempdat_short['V_bat']
# x2=battempdat_short['time']/3600
# y2=battempdat_short['V_bat']
# x3=battempdat_short['time']/3600
# y3=battempdat_short['V_bat']
# smoothed1 = lowess(y1, x1, frac=0.1)
# smoothed2 = lowess(y2, x2, frac=0.1)
# smoothed3 = lowess(y3, x3, frac=0.1)
# # q=17150
# # r=2695
# # u=23300
# # v=battempdat_short['V_bat'].max()
# #plt.plot(x,y, label = 'Original data') 
# plt.plot(smoothed1[:,0], smoothed1[:,1], label = "2S config (n=2)")
# plt.plot(smoothed2[:,0], smoothed2[:,1], label = "2S config (n=2)")
# plt.plot(smoothed3[:,0], smoothed3[:,1], label = "2||2 config")
# plt.title('Time vs Battery voltage - Lowess smoothened')
# plt.xlabel('Time [hrs]')
# plt.ylabel('Battery voltage [mV]')
# plt.legend()
# plt.show()

#%%
#MISCELANEUOUS - TRYING TO SMOOTHEN BATTERY DATA FOR EXPONENTIAL DECAY FITTING
# battempdat_fit = battempdat[battempdat['time']>1.1e6]
# battempdat_fit['time'] = battempdat_fit['time'] - battempdat_fit['time'].iloc[0]

# def moving_average(data, window_size):
#     return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# # Define the exponential decay function
# def exponential_decay(t, V0, k):
#     return V0 * np.exp(-k * t)

# # Your battery data
# t_points = np.array(battempdat_fit['time']) # assuming 'Time' is in seconds
# voltage = np.array(battempdat_fit['V_bat']) # assuming 'Voltage_mV' is the voltage in mV

# # Apply moving average filter
# window_size = 10  # for example, you might need to adjust this
# smoothed_voltage = moving_average(voltage, window_size)
# adjusted_t_points = t_points[:len(smoothed_voltage)]

# # Perform the curve fitting on the smoothed data
# params_smooth, covariance_smooth = curve_fit(exponential_decay, adjusted_t_points, smoothed_voltage, p0=[smoothed_voltage[0], 0.0001])

# # Extract the parameters
# V0_fitted_smooth, k_fitted_smooth = params_smooth

# # Generate fitted values
# fitted_voltage_smooth = exponential_decay(adjusted_t_points, V0_fitted_smooth, k_fitted_smooth)

# # Plot original data and the fitted curve
# plt.figure(figsize=(10, 6))
# plt.plot(t_points, voltage, label='Original data')
# plt.plot(adjusted_t_points, fitted_voltage_smooth, label='Exponential fit', color='orange')
# plt.title('Time vs Battery voltage plot - before dying')
# plt.xlabel('Time [s]')
# plt.ylabel('Battery Voltage [mV]')
# plt.legend()
# plt.show()
