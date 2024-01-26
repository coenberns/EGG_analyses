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
from Old_Plot_EGG import*

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



#%%
def interpolate_egg_v3(df, method = 'cubicspline', order=3):
    df2 = df.copy()
    while pd.isna(df2.iloc[0]['Channel 0']):
        df2 = df2.iloc[1:].reset_index(drop=True)

    # Reset timestamps to start from zero
    df2['timestamps'] -= df2['timestamps'].iloc[0]
    if  (method == 'polynomial') or (method == 'spline'):
        # Interpolate each channel, using order
        for i in range(8):
            channel = f'Channel {i}'
            df2[channel] = df2[channel].interpolate(method=method, order=order)

    else:
        # Interpolate each channel
        for i in range(8):
            channel = f'Channel {i}'
            df2[channel] = df2[channel].interpolate(method=method)
    return df2

def butter_filter(df, low_freq, high_freq, fs, order=3):
    df_filtered = df.copy()
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')

    for column in df.columns:
        if column.startswith('Channel'):
            df_filtered[column] = filtfilt(b, a, df[column].values, padlen=150)
    return df_filtered

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
v_compact, v_fulldat_control, times =read_egg_v3_bursts(file,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

#%%
v_mean_control = averaging_bursts(v_fulldat_control)

segmented_vmean_control = segment_data(v_mean_control, gap_size=60, seg_length=200, window=50, min_frac=0.5, window_frac=0.2)

control_interp={}
filter_control={}
print(len(segmented_vmean_control[0]))

for i in range(len(segmented_vmean_control)):
    fs=times['effective_rate']
    segmented_vmean_control[i]=rescale_time(segmented_vmean_control[i])
    control_interp[i]=interpolate_egg_v3(segmented_vmean_control[i])
    filter_control[i]=butter_filter(control_interp[i], low_freq=0.02, high_freq=0.2, fs=fs)


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
v_compact, v_fulldat_test, times =read_egg_v3_bursts(file,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

#%%
v_mean_test = averaging_bursts(v_fulldat_test)

segmented_vmean_test = {}
segmented_vmean_test = segment_data(v_mean_test, gap_size=50, seg_length=200, window=100, min_frac=0.5, window_frac=0.2)
print(len(segmented_vmean_test[0]))


def perform_wavelet_analysis(signal, wavelet_name='db4'):
    # Perform Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet_name)
    
    # Plot the scalogram
    plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='PRGn', aspect='auto',
               vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.title("Wavelet Transform (Scalogram)")
    plt.xlabel("Time")
    plt.ylabel("Frequency scale")
    plt.show()
    
    return coefficients, frequencies

# for i in range(8):
#     # Extract the signal (voltage values) for the channel
#     signal = segmented_vmean_test[1][f'Channel {i}'].values
#     # Drop NaN values for wavelet analysis
#     signal = signal[~np.isnan(signal)]
#     # Perform wavelet analysis on the signal
#     perform_wavelet_analysis(signal, wavelet_name='db4')
#%%

# #8 sec gap xlims
# xlim = {
#     0: [27,36],
#     1: [133,141],
#     2: [183, 191],
#     3: [255, 263]
# }

#14 sec gap xlims
xlim = {
    0: [8,22],
    1: [56,70],
    2: [102,116],
    3: [193,207],
    4: [349,363],
    5: [480, 494]
}

seg_interp={}
kalman={}
filter_test={}
stats = []
methods = ['cubicspline', 'akima', ]
for method in methods:

    for i in range(len(segmented_vmean_test)):
        fs=times['effective_rate']
        seg_interp[i] = interpolate_egg_v3(segmented_vmean_test[i], method=method, order=3)
        kalman[i] = kalman_filter(seg_interp[i])
        filter_test[i]=butter_filter(seg_interp[i], low_freq=0.02, high_freq=0.2, fs=fs)

        for j in range(8):

            x = segmented_vmean_control[i]['timestamps']
            y1 = filter_control[i][f'Channel {j}']
            y2 = filter_test[i][f'Channel {j}']

            # for q,r in enumerate(xlim):
            #     xlim_low=xlim[q][0]
            #     xlim_up=xlim[q][1]

            #     plt.plot(x,y1, marker = 'x', label="Averaged", alpha=.5)
            #     # plt.plot(x,y2, alpha=0.5,linestyle='dashed',  label= f"Interpolated ({method})")
            #     # plt.plot(x, y3, alpha=.5, label=f'{method} + Kalman',linestyle='dotted', color='g')
            #     plt.plot(x, y2, alpha=.8, label=f'{method} + butter',linestyle='dotted', color='black')
            #     plt.xlim(xlim_low-20, xlim_up+20)
            #     plt.axvline(x=xlim_low, alpha = 0.2, linestyle='dashed', color='r', label=f'{xlim_low:.2f}')
            #     plt.axvline(x=xlim_up, alpha = 0.2, linestyle='dashed', color='r', label=f'{xlim_up:.2f}')
            #     #plt.ylim(-40,-10)
            #     plt.title(f"Channel {j}")
            #     plt.xlabel('Time [s]')
            #     plt.ylabel('Biopotential [mV]')
            #     plt.legend()
            #     plt.show()

            # Calculate MSE/RMSE/MAE only for non-NaN values
            mse_mean_filtered = mse(y1, y2)
            # mse_mean_kalman = mse(y1, y3)
            rmse_mean_filtered = np.sqrt(mse_mean_filtered)
            # rmse_mean_kalman = np.sqrt(mse_mean_kalman)
            mae_mean_filtered = mae(y1, y2)
            # mae_mean_kalman = mae(y1, y3)

            freq1,mag1=signal.periodogram(y1, fs=fs, scaling='spectrum')
            amplitudes1=np.sqrt(mag1)

            # Find dominant frequency
            dominant_freq1 = freq1[np.argmax(amplitudes1)]
            print(f"Dominant frequency 1: {dominant_freq1} Hz")

            # Convert to cycles/minute
            dominant_freq_cpm1 = dominant_freq1 * 60
            print(f"Dominant frequency 1: {dominant_freq_cpm1} cycles/minute")

            freq2,mag2=signal.periodogram(y2, fs=fs, scaling='spectrum')
            amplitudes2=np.sqrt(mag2)

            # Find dominant frequency
            dominant_freq2 = freq2[np.argmax(amplitudes2)]
            print(f"Dominant frequency 2: {dominant_freq2} Hz")

            # Convert to cycles/minute
            dominant_freq_cpm2 = dominant_freq2 * 60
            print(f"Dominant frequency 2: {dominant_freq_cpm2} cycles/minute")

            abs_diff_dfreq = np.abs(dominant_freq_cpm1-dominant_freq_cpm2)
            mse_amplitude = mse(amplitudes1, amplitudes2)


            # Plotting the spectrum
            freq1_cpm=freq1*60
            freq2_cpm=freq2*60
            plt.figure(figsize=(14,7))
            plt.stem(freq1_cpm, amplitudes1, label = 'filter control', linefmt='C0-', markerfmt='C0o', basefmt=" ")
            plt.stem(freq2_cpm, amplitudes2, label='filter test', linefmt='C1-', markerfmt='C1x', basefmt=" ")
            plt.title(f"Frequency Spectrum of Channel {j}, method = {method}")
            plt.xlabel("Frequency (cpm)")
            plt.ylabel("Amplitude")
            plt.xlim(0, 10)
            plt.legend()
            plt.show()

            # Append results to the list
            stats.append({
                'Channel': j,
                'Method': method,
                'MSE_Filtered': mse_mean_filtered,
                # 'MSE_Kalman': mse_mean_kalman,
                'RMSE_Interp': rmse_mean_filtered,
                # 'RMSE_Kalman': rmse_mean_kalman,
                'MAE_Interp': mae_mean_filtered,
                'Abs diff D Freq': abs_diff_dfreq, 
                'MSE Amplitude': mse_amplitude
                # 'MAE_Kalman': mae_mean_kalman
            })


stats_df = pd.DataFrame(stats)  
stats_df['Avg MSE Amplitude'] = stats_df.groupby('Method')['MSE Amplitude'].transform('mean')


# %%
