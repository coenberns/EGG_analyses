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
from datetime import datetime, timedelta, time
import pathlib 
from Plot_EGG import*
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import UnivariateSpline as univsp
from scipy.signal import savgol_filter
from compaction_read_burst import*
from scipy.signal import butter, filtfilt

# %%
func = 3

working_mean = groupdats_mean['function3']

food_mean = working_mean[5995:7221]
food_mean_reset = food_mean.copy()
food_mean_reset['timestamps']=food_mean_reset[f'corrected_t_f{func}']-\
    food_mean_reset[f'corrected_t_f{func}'].iloc[0]
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
dat_mean = food_mean_reset[datcols]

# window = 5
# polynomial = 2

for i in range(8):
    dat_mean[f'Channel_interp{i}'] = dat_mean[f'Channel {i}'].interpolate(method='cubic')
    #dat_mean[f'Channel_interp_savgol{i}'] = savgol_filter(dat_mean[f'Channel_interp{i}'], window, polynomial)


#%%
smalldat_mean = dat_mean[0:500]

plt.figure(figsize=(14, 6))
plt.plot(smalldat_mean['timestamps'], smalldat_mean['Channel 4'], marker='o')
plt.title('Voltage Readings Over Time')
plt.ylabel('Voltage')
plt.xlabel('Time')
plt.show()
# %%
smalldat_mean = dat_mean[0:500]
plt.figure(figsize=(14, 6))
plt.plot(smalldat_mean['timestamps'], smalldat_mean['Channel 5'], marker='o', label='Original')
plt.plot(smalldat_mean['timestamps'], smalldat_mean['Channel_interp5'], label='Interpolated', linestyle=(0, (5, 5)))
# For mean data, savgol smoothening has no/negative effect
# plt.plot(smalldat['timestamps'], smalldat['Channel_interp_savgol4'], label='Interpolated + Savgol', linestyle='dashdot')
plt.title('Original vs Interpolated VT data')
plt.ylabel('Voltage')
plt.xlabel('Time')
plt.legend()
plt.show()

# %%
func = 3

groupdat_3 = groupdats['function3']

#37230:43668 -- first day around 18 eating

around_eating = groupdat_3[35970:43326]
around_eating_reset_t = around_eating.copy()
around_eating_reset_t['timestamps'] = around_eating_reset_t[f'corrected_t_f{func}'] - \
    around_eating_reset_t[f'corrected_t_f{func}'].iloc[0]
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
dat = around_eating_reset_t[datcols]
dat = dat[~dat['timestamps'].isna()]

# window = 10
# polynomial = 5

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

fs = 0.5
cutoff = 0.05
order = 6

for i in range(8):
    dat[f'Channel_interp{i}'] = dat[f'Channel {i}'].interpolate(method='cubic')
    dat[f'Channel_lowpass{i}'] = butter_lowpass_filter(dat[f'Channel_interp{i}'], cutoff, fs, order)
    #dat[f'Channel_interp_savgol{i}'] = savgol_filter(dat[f'Channel_interp{i}'], window, polynomial)

#%%
smalldat = dat[500:600]

plt.figure(figsize=(14, 6))
plt.plot(smalldat['timestamps'], smalldat['Channel 4'], marker='o')
plt.title('Voltage Readings Over Time')
plt.ylabel('Voltage')
plt.xlabel('Time')
plt.show()
# %%
smalldat = dat[500:600]
plt.figure(figsize=(14, 6))
plt.plot(smalldat['timestamps'], smalldat['Channel 5'], marker='o', label='Original')
plt.plot(smalldat['timestamps'], smalldat['Channel_interp5'], label='Interpolated', linestyle=(0, (5, 5)))
plt.plot(smalldat['timestamps'], smalldat['Channel_lowpass5'], label='Interpolated + filter', linestyle='dashdot')
# plt.plot(smalldat['timestamps'], smalldat['Channel_interp_savgol5'], label='Interpolated + Savgol', linestyle='dashdot')
plt.title('Original vs Interpolated vs Inter+filtered VT data')
plt.ylabel('Voltage')
plt.xlabel('Time')
plt.legend()
plt.show()
# %%
meansy = dat_mean
rawsy = dat
lb = 301.0165010654
ub = 450.528008217

same_mean = meansy[(meansy['timestamps']>lb) & (meansy['timestamps']<ub)]
same_raw = rawsy[(rawsy['timestamps']>lb) & (rawsy['timestamps']<ub)]


plt.figure(figsize=(14, 6))
plt.plot(same_raw['timestamps'], same_raw['Channel 5'], marker='o', label='Raw')
plt.plot(same_mean['timestamps'], same_mean['Channel 5'], marker='x', label='Mean', color = 'r')
plt.plot(same_mean['timestamps'], same_mean['Channel_interp5'], label='Interpolated mean', linestyle=(0, (5, 5)), color = 'g')
plt.plot(same_raw['timestamps'], same_raw['Channel_interp_savgol5'], label='Interpolated raw', linestyle='dashdot', color = 'k')
plt.title('Raw + interpolated vs Mean + Interpolated  VT data')
plt.ylabel('Voltage [mV]')
plt.xlabel('Time [s]')
plt.legend()
plt.show()


# %%
