# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri October 18 15:48:34 2023

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

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
import Old_Plot_EGG as oldEGG
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
file_0829 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0829.name}")

#%%
#For the general read-in of data file
v_compact_0829, v_fulldat_0829, times_0829 =read_egg_v3_bursts(file_0829,
                                                header = None,
                                                rate = 62.5,
                                                scale=300,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

v_mean_0829 = averaging_bursts(v_fulldat_0829)

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0829 = interpolate_data(v_mean_0829, cycle_time=times_0829['t_cycle'])
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0829 = savgol_filt(interp_mean_0829)

#%%
fs_0829=times_0829['effective_rate']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
a,b,c_0829 = signalplot_hrs(savgol_mean_0829,xlim=(0,3),spacer=200,vline=[],
           freq=[0.0001,0.02],order=3, rate=fs_0829, title='',skip_chan=[0,1,2],
            figsize=(10,8),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

a1,b1,c2_0814 = egg_signalfreq(c_0829, rate=fs_0829, freqlim=[0.001*60,0.1*60], mode='power', vline=[0.25,1.33],mmc=True,
                                figsize=(8,8))

#%%
seg_vmean_0829 = {}
seg_vmean_0829 = segment_data(v_mean_0829, gap_size=14, seg_length=1800, window=100, min_frac=0.8, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0829)

seg_interp_0829={}
seg_filtered_0829={}
seg_savgol_0829={}


for i in range(len(seg_vmean_0829)):
        seg_interp_0829[i] = interpolate_egg_v3(seg_vmean_0829[i], method='cubicspline', time=False)
        seg_filtered_0829[i]=butter_filter(seg_interp_0829[i], low_freq=0.02, high_freq=0.2, fs=fs)
        seg_savgol_0829[i]=savgol_filt(seg_interp_0829[i], window=3,polyorder=1,deriv=0,delta=1)

#%%
signalplot(savgol_mean_0829,xlim=(),spacer=200,vline=[],freq=[0.02,0.2],order=3,
            rate=times_0829['effective_rate'], title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
#%%
signalplot_hrs(savgol_mean_0829,xlim=(0,34.8),spacer=100,vline=[],freq=[0.02,0.2],order=3,
            rate=times_0829['effective_rate'], title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# for i in range(len(seg_filtered)):
for i in range(1, len(seg_interp_0829)):
    print(i)
    signalplot(seg_interp_0829[i],xlim=(),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                rate=fs, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%

egg_freq_heatplot_v2(savgol_mean_0829, rate=fs, xlim=[15000,20000],seg_length=500,freq=[0.03,0.15],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])

#%%
egg_freq_heatplot_v2(seg_savgol_0829[0], rate=fs, xlim=[0,4400],seg_length=400,freq=[0.02,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.7,norm=True,time='timestamps',
                            skip_chan=[])


#%% Quite okay segment with 4 cpm in this xlim range
egg_freq_heatplot_v2(seg_savgol_0829[1], rate=fs, xlim=[2000,3600],seg_length=400,freq=[0.03,0.2],freqlim=[1,8],
                            vrange=[0],figsize=(10,18),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=[4])

#%%
egg_freq_heatplot_v2(seg_savgol_0829[2], rate=fs, xlim=[0,4000],seg_length=400,freq=[0.02,0.2],freqlim=[1,8],
                            figsize=(10,15),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=[])

#%%
egg_freq_heatplot_v2(seg_savgol_0829[3], rate=fs, xlim=[0,4000],seg_length=400,freq=[0.03,0.15],freqlim=[1,8],
                            figsize=(10,20),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=[])

#%% Night segment, with a steady last 15 min of 4cpm
egg_freq_heatplot_v2(seg_savgol_0829[4], rate=fs, xlim=[2400,3600],seg_length=600,freq=[0.02,0.2],freqlim=[1,8],
                            figsize=(10,20),interpolation='bilinear',n=20, intermediate=False,
                            max_scale=.5,norm=True,time='timestamps',
                            skip_chan=[])
#%% Additional figures for night segment, pig wakes up
a4,b4,c4 = signalplot(seg_savgol_0829[4], xlim=(0,3600), freq=[0.02,0.2], rate=fs, vline=[], spacer=75)
#%%
egg_signalfreq(c4,rate=fs,freqlim=[1,10])
heatplot(seg_savgol_0829[4], rate=fs,freq=[0.02,0.2], vrange=[0,20])

#%%
signalplot(seg_savgol_0829[4],xlim=(3400,3600),spacer=40,vline=[],freq=[0.02,0.2],order=3,
            rate=fs, title='',skip_chan=[1,2,4,6],
            figsize=(10,12),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
egg_freq_heatplot_v2(seg_savgol_0829[5], rate=fs, xlim=[0,4800],seg_length=800,freq=[0.02,0.2],freqlim=[1,7],
                            figsize=(10,20),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=[])

#%%
filtered_compact = seg_filtered_0829[0][datcols]

egg_signalfreq(filtered_compact,rate=fs,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
               labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
figs={}
for i in np.arange(0,20000, 1000):
    i2=i+1000
    savgol_compact = savgol_mean_0829[datcols][i:i2]

    figs[i],_,_ = egg_signalfreq(savgol_compact,rate=fs,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
                labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={}, title=f'Index region {i}:{i2}')

#%%
fig_heat, _, _ = heatplot(seg_savgol_0829[4],xlim=(0,0,0),spacer=0,vline=[],freq=1,order=3,rate=times_0829['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)
fig_heat.show()
#%%
heatplot(savgol_mean_0829,xlim=(125000,125830),spacer=0,vline=[],freq=[0.02,0.15],order=3,rate=times_0829['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,50],interpolation='bilinear',norm=True)

# %%
