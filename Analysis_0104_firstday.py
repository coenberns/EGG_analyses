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
import seaborn as sns
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
import Old_Plot_EGG as oldEGG
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
file_0104 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0104.name}")
#%%
#For the general read-in of data file
v_compact_0104, v_fulldat_0104, times_0104 =read_egg_v3_bursts(file_0104,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

#%%
v_fulldat2_0104 = v_fulldat_0104
burst_length = 6
channels = [f'Channel {i}' for i in range(8)]

# def nanmean(series):
#     return np.nanmean(series)

# Apply the custom function for averaging
for channel in channels:
    v_fulldat2_0104[channel] = v_fulldat2_0104.groupby('burst_group')[channel].transform('mean')

# Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
for col in ['elapsed_s', 'corrected_realtime']:
    v_fulldat2_0104[col] = v_fulldat2_0104.groupby('burst_group')[col].transform('first')

# Filtering for the first packet of each burst
v_mean_0104 = v_fulldat2_0104[v_fulldat2_0104['packet_miss_idx'] % burst_length == 0]

# v_mean_0104 = averaging_bursts(v_fulldat_0104,n_burst=5, sleep_ping=1)

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0104 = interpolate_data(v_mean_0104, cycle_time=times_0104['t_cycle'], pchip=True)
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0104 = savgol_filt(interp_mean_0104)

#%% 
sns.set_palette('tab10')

fs_0104 = times_0104['effective_rate']
signalplot(savgol_mean_0104,xlim=(),spacer=200,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0104, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%% Signal plot for potential MMC recordings? Looks interesting
a,b,c_0104 = signalplot_hrs(savgol_mean_0104,xlim=(),spacer=100,vline=[],freq=[0.0001,0.01],order=3,
            rate=fs_0104, title='',skip_chan=[3,4,5],
            figsize=(10,8),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%% Plotting its power density for low frequencies. Clear view of MMC 
a1,b1,c2_0104 = egg_signalfreq(c_0104, rate=fs_0104, freqlim=[0.001*60,0.08*60], mode='power', vline=[0.25,1],mmc=True,
                                figsize=(8,8))

#%% Looking at dominant frequencies?
oldEGG.egg_freq_heatplot_v2(savgol_mean_0104,rate=fs_0104,xlim=[0,36000], freq=[0.0001,0.01], seg_length=6000, 
                        freqlim=[0.00001,0.08],time='timestamps', max_scale=.8, n=10, norm=True, skip_chan=[], figsize=(10,15))
#%%
# egg_freq_heatplot_v2(savgol_mean_0105,rate=.5,xlim=[400,2000], freq=[0.02,0.2], seg_length=100, 
#                         freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

#%%
seg_vmean_0104 = {}
seg_vmean_0104 = segment_data(v_mean_0104, gap_size=15, seg_length=1000, window=100, min_frac=0.8, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0104)

#%%
seg_interp_0104={}
seg_filtered_0104={}
seg_savgol_0104={}
fs_0104=times_0104['effective_rate']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean_0104)):
        seg_interp_0104[i] = interpolate_egg_v3(seg_vmean_0104[i], method='cubicspline', time=False, rescale=True)
        seg_filtered_0104[i]=butter_filter(seg_interp_0104[i], low_freq=0.02, high_freq=0.2, fs=fs_0104)
        seg_savgol_0104[i]=savgol_filt(seg_interp_0104[i], window=3,polyorder=1,deriv=0,delta=1)


# %%
for i in range(0,7):
    print(i)
    signalplot(seg_interp_0104[i],xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
                rate=fs_0104, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %%
signalplot(seg_savgol_0104[3],xlim=(),spacer=50,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0104, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0104[0],rate=times_0104['effective_rate'],xlim=[0,3600], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0104[1],rate=times_0104['effective_rate'],xlim=[0,2400], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0104[2],rate=times_0104['effective_rate'],xlim=[0,3600], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0104[3],rate=times_0104['effective_rate'],xlim=[0,3600], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0104[4],rate=times_0104['effective_rate'],xlim=[0,8000], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)


# %%
heatplot(seg_savgol_0104[4],xlim=(),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=times_0104['effective_rate'], 
            title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)

# %%
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
segments = {}
for i in range(0,5):
    segments[i] = seg_savgol_0104[i][datcols]

#%% To show where the dominant frequency is located at different times (from during the day to night)
# During the night, the dominant frequency is more clearly in the preferred range, not a lot of lower freq
for i in range(0,5):
    egg_signalfreq(segments[i], rate=fs_0104, freqlim=[1,10], mode='power', vline=[3.2,4.4])



