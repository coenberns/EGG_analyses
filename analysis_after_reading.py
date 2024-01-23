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
from Plot_EGG_adaptation import*
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns

from scipy.interpolate import UnivariateSpline as univsp
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
from functions_read_bursts import*


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
file = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file.name}")

#%%
#For the general read-in of data file
v_compact, v_fulldat, times =read_egg_v3_bursts(file,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)
# %%
#For the grouping of definitely true data
#grouped_fulldat=assign_groups(v_fulldat)

# %%
_, t_elapsed,s_elapsed = calculate_time(file)

#%%

def averaging_bursts(df, n_burst=5, sleep_ping=1):
    df2=df.copy()
    for i in range(8):
        channel = f'Channel {i}'
        # Convert to numpy
        data = df2[channel].values
        # Calculate padding
        remainder = len(data) % (n_burst+sleep_ping)
        padding_size = 0 if remainder == 0 else ((n_burst+sleep_ping) - remainder)
        # Pad with nan
        padded_data = np.pad(data, (0, padding_size), constant_values=np.nan)
        # Reshape the data to have n_burst+sleep_ping values per row
        reshaped_data = padded_data.reshape(-1, (n_burst+sleep_ping))
        # Compute the mean per row, ignoring nan's
        means = np.nanmean(reshaped_data, axis=1)
        # Repeat mean 6 times to get original shape back
        repeated_means = np.repeat(means, (n_burst+sleep_ping))
        # Trim back to old length
        trimmed_means = repeated_means[:len(data)]
        # Assign to the voltage channels
        df2[channel] = trimmed_means

    #Filter for 1st of burst only to shift averaging starting at t=0
    df2 = df2[df2['packet_miss_idx'] % (n_burst+sleep_ping) == 0]
    return df2

v_mean = averaging_bursts(v_fulldat)

#%% NEWER SEGMENTATION - OLDER AT BOTTOM
def rescale_time(segment):
    offset = segment['timestamps'].iloc[0]
    segment['timestamps'] = segment['timestamps'] - offset
    return segment

def filter_segment(segment, window, min_frac, window_frac):
    start = 0
    end = len(segment)
    threshold = window * min_frac
    total_windows = (end + window - 1) // window
    max_invalid_windows = int(total_windows * window_frac)
    invalid_windows_count = 0

    # Check all windows in the segment
    for start in range(0, end, window):
        end_window = min(start + window, end)
        window_segment = segment.iloc[start:end_window]
        non_nan_count = window_segment['Channel 0'].notna().sum()

        window_threshold = threshold if end_window - start == window else ((end_window - start) * min_frac)

        # Count invalid windows
        if non_nan_count < window_threshold:
            #Check for last ending windows - something weird? Not really so far
            # if len(window_segment) != 100:
                # print("Window size =", len(window_segment), "non_nan =", non_nan_count)
            #If want to know the other percentages of non-nan values
            # print("Percentage of non-nan =", ((non_nan_count/len(window_segment))*100), "%")
            invalid_windows_count += 1
            if invalid_windows_count > max_invalid_windows:
                return None  # Discard segment if too many windows are invalid

    # Handle leading/trailing NaN values
    segment = segment.iloc[0:end]
    while len(segment) > 0 and pd.isna(segment.iloc[-1]['Channel 0']):
        segment = segment.iloc[:-1]
    while len(segment) > 0 and pd.isna(segment.iloc[0]['Channel 0']):
        segment = segment.iloc[1:]

    return segment

def segment_data(df, gap_size, seg_length, window, min_frac, window_frac):
    segments = {}
    start_index=0 
    segment_id=0
    nan_count = 0
    time_interval = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]

    for i in range(start_index, len(df)):
        # Check if row is NaN
        if pd.isna(df.iloc[i, df.columns.get_loc("Channel 0")]):
            nan_count += 1
        else:
            if nan_count > 0:
                # Calculate time gap
                time_gap = nan_count * time_interval
                if time_gap > gap_size:
                    segment = df.iloc[start_index:i - nan_count]
                    filtered_segment = filter_segment(segment, window, min_frac, window_frac)
                    if filtered_segment is not None and len(filtered_segment) >= seg_length:
                        filtered_segment = rescale_time(filtered_segment)
                        segments[segment_id] = filtered_segment
                        segment_id += 1
                    else:
                        # If segment is too long and invalid, split it and process each half recursively
                        if len(segment) >= 2 * seg_length:
                            middle = (start_index + i - nan_count) // 2
                            segments.update(segment_data(df, gap_size, seg_length, window, min_frac, start_index, segment_id))
                            segments.update(segment_data(df, gap_size, seg_length, window, min_frac, middle, segment_id))
                    start_index = i
                nan_count = 0

    # Process the last segment
    segment = df.iloc[start_index:]
    filtered_segment = filter_segment(segment, window, min_frac, window_frac)
    if filtered_segment is not None and len(filtered_segment) >= seg_length:
        filtered_segment = rescale_time(filtered_segment)
        segments[segment_id] = filtered_segment
    print("Amount of segments =", segment_id)
    return segments

segmented_vmean = {}
segmented_vmean = segment_data(v_mean, gap_size=50, seg_length=2000, window=100, min_frac=0.6, window_frac=0.2)

for i in range(len(segmented_vmean)):
    print(len(segmented_vmean[i]))


#%%
def interpolate_egg_v3(df, method = 'cubicspline'):
    df2 = df.copy()
    while pd.isna(df2.iloc[0]['Channel 0']):
        df2 = df2.iloc[1:].reset_index(drop=True)

    # Reset timestamps to start from zero
    df2['timestamps'] -= df2['timestamps'].iloc[0]

    # Interpolate each channel
    for i in range(8):
        channel = f'Channel {i}'
        df2[channel] = df2[channel].interpolate(method=method)
    return df2

seg_interp = {}

for i in range(len(segmented_vmean)):
    seg_interp[i] = interpolate_egg_v3(segmented_vmean[i])
# v_mean_intp = interpolate_egg_v3(v_mean)
#%%
def kalman_filter(df):
    df2=df.copy()
    for i in range(8):
        channel = f'Channel {i}'

        #Kalman filter and then interpolating
        # all_values = df2[channel].tolist()
        measurements = df2[channel].dropna().tolist()        

        filtered_values = []

        #dt = times['t_cycle']
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = [df2[channel].iloc[0]]  # initial state
        kf.F = np.array([[1]])  # state transition matrix, 1x1 matrix since we have only one state.
        kf.H = np.array([[1]])  # measurement function, 1x1 matrix.
        kf.P *= 1000.  # covariance matrix
        kf.R = 10  # state uncertainty, adjust based on sensor accuracy
        kf.Q = 1  # process uncertainty, adjust based on how predictable the voltage evolution is

        # if nan_vals == True: 
        #     for z in all_values:
        #         kf.predict()
        #         if not np.isnan(z):  # If the measurement is not NaN, update the filter
        #             kf.update([z])
        #         filtered_values.append(kf.x[0])
            
        #     df2[channel] = filtered_values

        for z in measurements:
            kf.predict()
            kf.update([z])
            filtered_values.append(kf.x[0])

        # replace the original values with the filtered ones for non-NaN values
        valid_indices = df2[channel].dropna().index
        df2.loc[valid_indices, channel] = filtered_values

    return df2


kalman={}
for i in range(len(seg_interp)):
    kalman[i] = kalman_filter(seg_interp[i])

#%%Plotting of ambu_firstpart_noglitch: 1 segment with large overshoot, cutting out initial voltage stabilization part
#Only for this file!
segmented_vmean[0] = segmented_vmean[0][segmented_vmean[0]['timestamps']>3000]
seg_interp[0] = seg_interp[0][seg_interp[0]['timestamps']>3000]
kalman[0] = kalman[0][kalman[0]['timestamps']>3000]
segmented_vmean[0]=rescale_time(segmented_vmean[0])
seg_interp[0]=rescale_time(seg_interp[0])
kalman[0]=rescale_time(kalman[0])


#%%
for i in range(len(segmented_vmean)):

    for j in range(8):

        x1 = segmented_vmean2[i]['timestamps']
        x2 = seg_interp[i]['timestamps']
        x3 = kalman[i]['timestamps']
        y1 = segmented_vmean2[i][f'Channel {j}']
        y2 = seg_interp[i][f'Channel {j}']
        y3 = kalman[i][f'Channel {j}']
        

        plt.plot(x1,y1, marker = 'x', label="Averaged")
        plt.plot(x2,y2, alpha=0.5,linestyle='dashed',  label= "Interpolated")
        plt.plot(x3, y3, alpha=.5, label='Interpolated + Kalman',linestyle='dotted', color='g')
        plt.xlim(400,600)
        #plt.ylim(-40,-10)
        plt.title(f"Channel {j}")
        plt.xlabel('Time [s]')
        plt.ylabel('Biopotential [mV]')
        plt.legend()
        plt.show()


#%%
egg_signalfreq(seg0_short, rate = 1/1.9935, freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})
egg_freq_heatplot_v2(dat, rate=62.5, xlim=[0,1000],seg_length=500,freq=[0.02,0.2],freqlim=[1,10],interpolation='bilinear',n=10, intermediate=False,max_scale=.4,norm=True,time='timestamps',skip_chan=[])

#%%
for i in range(1):

    for j in range(8):

        x1 = segmented_vmean[i]['timestamps']
        x2 = seg_interp[i]['timestamps']
        y1 = segmented_vmean[i][f'Channel {j}']
        y2 = seg_interp[i][f'Channel {j}']

        plt.plot(x1,y1, linestyle='dashed', marker = 'x', label=f"Mean {i}")
        plt.plot(x2,y2, alpha=0.5, label= f"Interpolated {i}")
        #plt.xlim(500,1000)
        #plt.ylim(-40,-10)
        plt.title(f"Channel {j}")
        plt.xlabel('Time [s]')
        plt.ylabel('Biopotential [mV]')
        plt.legend()
        plt.show()


#%%
for i in range(3):
    print(i)
    signalplot(vmean_seg_interp[i], skip_chan=[0,1,2], freq=[0.01,0.5])
#%%
def kalman_filter(df, nan_vals=True, method='cubicspline'):
    df2=df.copy()
    for i in range(8):
        channel = f'Channel {i}'

        #Kalman filter and then interpolating
        all_values = df2[channel].tolist()
        measurements = df2[channel].dropna().tolist()        

        filtered_values = []

        #dt = times['t_cycle']
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = [df2[channel].iloc[0]]  # initial state
        kf.F = np.array([[1]])  # state transition matrix, 1x1 matrix since we have only one state.
        kf.H = np.array([[1]])  # measurement function, 1x1 matrix.
        kf.P *= 1000.  # covariance matrix
        kf.R = 10  # state uncertainty, adjust based on sensor accuracy
        kf.Q = 1  # process uncertainty, adjust based on how predictable the voltage evolution is

        if nan_vals == True: 
            for z in all_values:
                kf.predict()
                if not np.isnan(z):  # If the measurement is not NaN, update the filter
                    kf.update([z])
                filtered_values.append(kf.x[0])
            
            df2[channel] = filtered_values

        else: 
            for z in measurements:
                kf.predict()
                kf.update([z])
                filtered_values.append(kf.x[0])

            # replace the original values with the filtered ones for non-NaN values
            valid_indices = df2[channel].dropna().index
            df2.loc[valid_indices, channel] = filtered_values
            #Fill in nan-values using interpolation
            df2[channel] = df2[channel].interpolate(method=method)
    return df2

# vmean_kalman_intp = kalman_filter(v_mean, nan_vals=False)
# v_kalman = kalman_filter(v_mean, nan_vals=True)


#CONSIDER A KALMAN FILTER, OR CONSIDER DELETING LARGER GAPS COMPLETELY FROM DATA

#%%
# list_ticks = []
# list_labels = []
# for i in range(0,45,5):
#     list_ticks.append(i)
#     list_labels.append(str(i))

x = v_fulldat['timestamps']
r=seg_vmean[0]['timestamps']
channel = 'Channel 6'
pal=sns.color_palette('deep')
plt.figure(figsize=(14, 10))
# plt.scatter(x,v_fulldat[channel], marker='o', label='Raw data', color = pal[0], alpha=0.5)
# plt.plot(r, seg_vmean[0][channel], marker='x', label='Mean', color = pal[1], markersize=18, alpha=0.2)
plt.plot(r, seg_interp[0][channel], label='Interpolated', marker='x', markersize=18, color = pal[2], linestyle = 'dashed',alpha=1)
#plt.plot(r, seg_filtered[0][channel], label='Butter', color=pal[3], linestyle = 'dotted')
plt.plot(r, seg_savgol[0][channel], label='Filtered', color=pal[4], linestyle = 'dashed')
# plt.plot(r, seg_smooth[0][channel], label='Moving avg', color=pal[5], linestyle = 'dashed')

# plt.plot(x, v_kalman[channel], label='Kalman', linestyle='dashdot', color = 'g')
# plt.plot(x, v_kalman_intp[channel], label='Kalman + interpolation', linestyle='dotted', color = 'orange')
# plt.title('Mean vs Mean + Interpolation vs Mean + Kalman + Interpolation VT data')
plt.title('Interpolated and filtered voltage-time data', size=32)
plt.ylabel('Voltage [mV]', size=28)
plt.xlabel('Time [s]', size=28)
plt.xticks(size=24)
#plt.xticks(ticks=list_ticks, labels=list_labels,size=24)
plt.yticks(size=24)
plt.legend(loc='center left', fontsize=28)
plt.xlim(3020,3060)
# plt.xlim(520,560)
#plt.xlim(12300,13000)
# plt.ylim(100,130)
plt.ylim(-43,10)
plt.show()

# %%
# import seaborn

for i in range(len(vmean_seg_interp)):
    x = v_fulldat['timestamps']
    r=v_mean['timestamps']
    channel = f'Channel {i}'
    plt.figure(figsize=(14, 6))
    plt.plot(x,v_fulldat[channel], marker='o', label='Raw data', color = 'b')
    plt.plot(r, v_mean[channel], marker='x', label='Mean', color = 'r')
    # plt.plot(x, v_mean_intp[channel], label='Interpolation', color = 'r', linestyle = 'dashed')
    # plt.plot(x, v_kalman[channel], label='Kalman', linestyle='dashdot', color = 'g')
    # plt.plot(x, v_kalman_intp[channel], label='Kalman + interpolation', linestyle='dotted', color = 'orange')
    # plt.title('Mean vs Mean + Interpolation vs Mean + Kalman + Interpolation VT data')
    plt.title('Raw data Voltage Time plot')
    plt.ylabel('Voltage [mV]')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.xlim(25000,25050)
    #plt.xlim(12300,13000)
    plt.ylim(10,25)
    plt.show()


#%% OLDER SEGMENTATION
def filter_segment(segment, window, min_frac, seg_length):
    start = 0
    end = len(segment)
    threshold = window * min_frac

    # Check all windows in the segment
    while start < end:
        end_window = min(start + window, end)
        window_segment = segment.iloc[start:end_window]
        non_nan_count = window_segment['Channel 0'].notna().sum()

        # Adjust threshold for the last window if it's smaller than the window size
        window_threshold = threshold if end_window - start == window else ((end_window - start) * min_frac)
        
        if non_nan_count < window_threshold:
            return None  # Discard segment if any window does not meet the threshold
        
        start = end_window  # Move to the next window

    # Now, handle potential leading/trailing NaN values
    segment = segment.iloc[0:end]  # Reset the end of the segment
    while len(segment) > 0 and pd.isna(segment.iloc[-1]['Channel 0']):
        segment = segment.iloc[:-1]
    while len(segment) > 0 and pd.isna(segment.iloc[0]['Channel 0']):
        segment = segment.iloc[1:]

    return segment if len(segment) >= seg_length else None



def segment_data(df, gap_size, seg_length, window, min_frac):
    segments = {}
    segment_id = 0
    start_index = 0
    nan_count = 0
    time_interval = df['timestamps'].iloc[1]-df['timestamps'].iloc[0] #can probably leave iloc[0] but just to be sure

    for i in range(len(df)):
            # Check if row is NaN
            if pd.isna(df.iloc[i, df.columns.get_loc("Channel 0")]): 
                nan_count += 1
            else:
                if nan_count > 0:
                    # Calculate time gap
                    time_gap = nan_count * time_interval

                    if time_gap > gap_size:
                        segment = df.iloc[start_index:i - nan_count]
                        filtered_segment = filter_segment(segment, window, min_frac, seg_length)
                        # print(filtered_segment)
                        # if len(filtered_segment) >= seg_length:
                        if filtered_segment is not None:
                            filtered_segment = rescale_time(filtered_segment) #exclude if I want to know where stuff happens for analysis
                            segments[segment_id] = filtered_segment
                            segment_id += 1
                        start_index = i
                    nan_count = 0

    segment = df.iloc[start_index:]
    filtered_segment = filter_segment(segment, window, min_frac, seg_length)
    if filtered_segment is not None:
        filtered_segment = rescale_time(filtered_segment) #exclude if I want to know where stuff happens for analysis
        segments[segment_id] = filtered_segment
    print('Amount of segments = ', segment_id+1)
    return segments


    #                 if time_gap > gap_size:
    #                     # Check segment length
    #                     segment_length = (i - nan_count - start_index)
    #                     if segment_length >= seg_length:
    #                         segments[segment_id] = df.iloc[start_index:i - nan_count]
    #                         segment_id += 1
    #                     start_index = i
    #                 nan_count = 0

    # # Check last segment
    # segment_length = (len(df) - start_index)
    # if segment_length >= seg_length:
    #         segments[segment_id] = df.iloc[start_index:]
    # print('Amount of segments = ',segment_id)
    # return segments


segmented_vmean2 = {}
segmented_vmean2 = segment_data(v_mean, gap_size=50, seg_length=3000, window=500, min_frac=0.5)

for i in range(len(segmented_vmean2)):
    print(len(segmented_vmean2[i]))
# %%
#MISCELANEOUS FUNCTIONS
#v_fulldat2 = v_fulldat.copy()
# v_fulldat2['avg_groups'] = (v_fulldat2['packet_miss_idx']/(n_burst+sleep_ping)).apply(np.floor)

# takes_time=time.time()
# for i in range(8):
#     channel = f'Channel {i}'
#     new_channel = f'Channel_avg {i}'
#     # Calculate mean and store in new Channel_avg columns, overwriting old data
#     v_fulldat2[new_channel] = v_fulldat2.groupby('avg_groups')[channel].transform(lambda x: x.mean(skipna=True))
# takes_time2=time.time()
# print("Group averaging function: ", (takes_time2-takes_time))

# v_fulldat_f2 = v_fulldat2[v_fulldat2['packet_miss_idx'] % (n_burst+sleep_ping) == 0]

# OLDER SEGMENT FILTER
# def filter_segment(segment, window, threshold):
#     start = 0
#     end = len(segment)

#     while start < end:
#         start_window = segment.iloc[start:start + window]
#         end_window = segment.iloc[max(end - window, start):end]

#         if start_window['Channel 0'].notna().sum() < threshold and start + window <= end:
#             start += window
#         elif end_window['Channel 0'].notna().sum() < threshold and end - window >= start:
#             end -= window
#         else:
#             break
    
#     filtered_segment = segment.iloc[start:end]
#     while len(filtered_segment) > 0 and pd.isna(filtered_segment.iloc[-1]['Channel 0']):
#         filtered_segment = filtered_segment.iloc[:-1]

#     return filtered_segment