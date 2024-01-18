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
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.signal import butter, filtfilt

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from filterpy.kalman import KalmanFilter
import pywt
from functions_read_bursts import*
from Plot_EGG_adaptation import*

#%%
#Mac
# meas_path = pathlib.Path("/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")

# #Windows
#Baatery measurements
#meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF readings miniPC desk animal facility/Battery Tests/Mode 1 new")
meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")
# # List all the files in the selected folder
in_folder = [f for f in meas_path.iterdir() if f.is_file()]

# Print a list of available files
for i, f in enumerate(in_folder, start=1):
    print(f"{i}. {f.name}")

# Ask the user to choose a file
while True:
    try:
        choice = int(input("Enter the number of the file you want to use (1,2, etc.): "))
        if 1 <= choice <= len(in_folder):
            break
        else:
            print("Invalid choice. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Assign the selected file to a variable
file_0828 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0828.name}")

#%%
#For the general read-in of data file
v_compact_0828, v_fulldat_0828, times_0828 =read_egg_v3_bursts(file_0828,
                                                header = None,
                                                rate = 62.5,
                                                scale=300,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)



#%%
v_fulldat2_0828 = v_fulldat_0828
burst_length = 6
channels = [f'Channel {i}' for i in range(8)]

# def nanmean(series):
#     return np.nanmean(series)

# Apply the custom function for averaging
for channel in channels:
    v_fulldat2_0828[channel] = v_fulldat2_0828.groupby('burst_group')[channel].transform('mean')

# Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
for col in ['elapsed_s', 'corrected_realtime']:
    v_fulldat2_0828[col] = v_fulldat2_0828.groupby('burst_group')[col].transform('first')

# Filtering for the first packet of each burst
v_mean_0828 = v_fulldat2_0828[v_fulldat2_0828['packet_miss_idx'] % burst_length == 0]

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0828 = interpolate_data(v_mean_0828, cycle_time=times_0828['t_cycle'])
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0828 = savgol_filt(interp_mean_0828)


#%%
seg_vmean_0828 = {}
seg_vmean_0828 = segment_data(v_mean_0828, gap_size=30, seg_length=1800, window=100, min_frac=0.8, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0828)

#%%
seg_interp_0828={}
seg_filtered_0828={}
seg_savgol_0828={}
# seg_smooth={}
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
fs=times_0828['effective_rate']
t_cycle = times_0828['t_cycle']


for i in range(len(seg_vmean_0828)):
        seg_interp_0828[i] = interpolate_data(seg_vmean_0828[i],cycle_time=t_cycle, max_gap=14, pchip=True)
        seg_filtered_0828[i]=butter_filter(seg_interp_0828[i], low_freq=0.02, high_freq=0.2, fs=fs)
        seg_savgol_0828[i]=savgol_filt(seg_interp_0828[i], window=3,polyorder=1,deriv=0,delta=1)
        # seg_smooth[i]=smooth_signal_moving_average(seg_filtered[i], window_size=5)

#%%
fig1, _,_ = signalplot(seg_savgol_0828[0],xlim=(0,5000),spacer=200,vline=[],freq=[0.02,0.2],order=3,
            rate=fs, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
#%% Feeding/non-feeding calculations
abs_mean_foodz=[]
abs_mean_partz=[]
abs_mean_nonz=[]
filtered = butter_filter(seg_savgol_0828[0], fs=fs)
for i in range(8):
    abs_mean_food=np.abs(filtered[f'Channel {i}'][0:750]).mean()
    abs_mean_partial= np.abs(filtered[f'Channel {i}'][750:2500]).mean()
    # abs_mean_non=np.abs(filtered[f'Channel {i}'][:4000]).mean()
    abs_mean_foodz.append(abs_mean_food)
    abs_mean_partz.append(abs_mean_partial)
    # abs_mean_nonz.append(abs_mean_non)

print("During intensive eating: ", np.mean(abs_mean_foodz), np.std(abs_mean_foodz))
print("During partial eating: ", np.mean(abs_mean_partz), np.std(abs_mean_partz))
# print("During resting: ", np.mean(abs_mean_nonz), np.std(abs_mean_nonz))

#%%
fig2, _, activ = heatplot(filtered,xlim=(0,5000),spacer=0,vline=[],freq=1,order=3,rate=fs, title='',skip_chan=[],
                            figsize=(10,10),textsize=16,vrange=[0,25],interpolation='bilinear',norm=True)

#%%
# for i in range(len(seg_filtered)):
for i in range(len(seg_filtered_0828)):

    signalplot(seg_interp_0828[i],xlim=(0,0,0),spacer=0,vline=[],freq=[0.0005,0.01],order=3,
                rate=times_0828['effective_rate'], title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# 'Channel 0','Channel 1','Channel 2','Channel 3','Channel 4','Channel 5','Channel 6'
egg_freq_heatplot_v2(savgol_mean_0828, rate=fs, xlim=[0,6400],seg_length=400,freq=[0.03,0.15],freqlim=[1,7], order=6,
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.4,norm=True,time='timestamps',
                            skip_chan=[])
#%%
# skip_chan=['Channel 2','Channel 4', 'Channel 5', 'Channel 6']
egg_freq_heatplot_v2(seg_interp_0828[0], rate=fs, xlim=[0,4400],seg_length=400,freq=[0.03,0.15],freqlim=[1,7], order=3,
                            vrange=[0],figsize=(10,10),interpolation='bilinear', n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                             skip_chan=['Channel 4','Channel 5','Channel 6'])


#%%
egg_freq_heatplot_v2(seg_interp_0828[1], rate=fs, xlim=[0,3500],seg_length=250,freq=[0.02,0.2],freqlim=[2,7],
                            vrange=[0],figsize=(10,14),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])

#%%
egg_freq_heatplot_v2(seg_interp_0828[2], rate=fs, xlim=[0,4000],seg_length=400,freq=[0.02,0.2],freqlim=[1,8],
                            vrange=[0],figsize=(10,5),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=['Channel 2','Channel 3','Channel 4','Channel 5', 'Channel 6', 'Channel 7'])

#%%
filtered_compact = seg_filtered_0828[0][datcols]

egg_signalfreq(filtered_compact,rate=fs,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
               labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
savgol_mean_comp = savgol_mean_0828[datcols]
egg_signalfreq(savgol_mean_comp,rate=fs,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
               labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
heatplot(seg_interp_0828[0],xlim=(0,3000),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=62.5, title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,40],interpolation='bilinear',norm=True)


# %%