#
# Created on Tue Jan 09 2024
#
# Copyright (c) 2024 Berns&Co
#

#%% Importing packages
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
from sklearn.metrics import mean_absolute_error as mae

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
from Plot_EGG_adaptation import*

#%% Dir selection
meas_path = pathlib.Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Pig measurements\01042024_multiweek")
# # Make list of available files in dir
in_folder = [f for f in meas_path.iterdir() if f.is_file()]

# Print list of available files in directory
for i, f in enumerate(in_folder, start=1):
    print(f"{i}. {f.name}")

# Choice selection of file 
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
file_0111 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0111.name}")
#%%
#For the general read-in of data file
v_compact_0111, v_fulldat_0111, times_0111 =read_egg_v3_bursts(file_0111,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

#%%
v_fulldat2_0111 = v_fulldat_0111
burst_length = 6
channels = [f'Channel {i}' for i in range(8)]

def nanmean(series):
    return np.nanmean(series)

# Apply the custom function for averaging
for channel in channels:
    v_fulldat2_0111[channel] = v_fulldat2_0111.groupby('burst_group')[channel].transform(nanmean)

# Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
for col in ['elapsed_s', 'corrected_realtime']:
    v_fulldat2_0111[col] = v_fulldat2_0111.groupby('burst_group')[col].transform('first')

# Filtering for the first packet of each burst
v_mean_0111 = v_fulldat2_0111[v_fulldat2_0111['packet_miss_idx'] % burst_length == 0]
# v_mean_0111 = averaging_bursts(v_fulldat_0111,n_burst=5, sleep_ping=1)

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0111 = interpolate_data(v_mean_0111, cycle_time=times_0111['t_cycle'])
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0111 = savgol_filt(interp_mean_0111)

#%%
fs = times_0111['effective_rate']
signalplot(savgol_mean_0111,xlim=(),spacer=50,vline=[],freq=[0.02,0.2],order=3,
            rate=fs, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# egg_freq_heatplot_v2(savgol_mean_0105,rate=.5,xlim=[400,2000], freq=[0.02,0.2], seg_length=100, 
#                         freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

#%%
seg_vmean_0111 = {}
seg_vmean_0111 = segment_data(v_mean_0111, gap_size=30, seg_length=1000, window=100, min_frac=0.6, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0111)

#%%
seg_interp_0111={}
seg_filtered_0111={}
seg_savgol_0111={}
fs=times_0111['effective_rate']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean_0111)):
        seg_interp_0111[i] = interpolate_data(seg_vmean_0111[i], cycle_time=times_0111['t_cycle'])
        seg_filtered_0111[i]=butter_filter(seg_interp_0111[i], low_freq=0.02, high_freq=0.2, fs=fs)
        seg_savgol_0111[i]=savgol_filt(seg_interp_0111[i], window=3,polyorder=1,deriv=0,delta=1)


# %%
for i in range(len(seg_interp_0111)):
    print(i)
    signalplot(seg_interp_0111[i],xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
                rate=fs, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %%
signalplot(seg_savgol_0111[0],xlim=(),spacer=10,vline=[],freq=1,order=3,
            rate=fs, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %% Good result during the night of 10 january, very consistent ~ 4 cpm
egg_freq_heatplot_v2(seg_savgol_0111[2],rate=times_0111['effective_rate'],xlim=[0,4000], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Other segments part 3_2
#segment 4, could potentially be combined with segment 5, only 2 min difference, USE interpolate_data() without rescaling. 
egg_freq_heatplot_v2(seg_savgol_0111[3],rate=times_0111['effective_rate'],xlim=[0,2400], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

# %%
heatplot(seg_savgol_0111[2],xlim=(),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=times_0111['effective_rate'], 
            title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)

# %%
