# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:21:42 2023

@author: Coen
"""
#%%
import numpy as np
from scipy.signal import correlate, coherence
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import seaborn as sns
import os
import re
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pathlib import Path
from dtw import *
from functions_read_bursts import *

#from compaction_read_burst import read_egg_v3_bursts
#%%

def extract_info_from_filename(file):
    # Extract the number of samples
    samples_match = re.search(r'(\d+)samps', file.stem)
    n_samples = int(samples_match.group(1)) if samples_match else None

    # Extract the sleep period
    sleep_match = re.search(r'(\d+)(?:_?)(\d*)sec', file.stem)
    sleep_period = None
    if sleep_match:
        seconds = sleep_match.group(1)
        fraction = sleep_match.group(2)
        sleep_period = float(f"{seconds}.{fraction}") if fraction else int(seconds)
    else:
        print(f"No match for sleep period in file: {file}")

    # Extract the skipped samples
    skipped_match = re.search(r'_(\d+)skipped', file.stem)
    skipped_samples = int(skipped_match.group(1)) if skipped_match else 0
    
    return n_samples, sleep_period, skipped_samples

# # Only necessary for Phase calculation 1, which is not used now
# def calculate_initial_phase(recorded_waveform, amplitude):
#     return np.arctan2(recorded_waveform[0], amplitude)

def sine_wave(t, amplitude, frequency, phase, offset):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

def generate_waveform(time_array, frequency, amplitude, offset, phase):
    """Generate a sinusoidal waveform for reference and initial phase+offset calculations"""
    waveform = amplitude * np.sin(2 * np.pi * frequency * time_array + phase) + offset
    return waveform

def cosine_similarity(x, y):
    """Calculate the cosine similarity between two signals x and y"""
    return np.dot(x - np.mean(x), y - np.mean(y)) / (np.sqrt(np.sum((x - np.mean(x)) ** 2)) * np.sqrt(np.sum((y - np.mean(y)) ** 2)))    

def calc_corr_stats(reference_signal, recorded_signal, fit_value):
    """Calculation of statistic measures of the correlation between the two waveforms"""    
    #To account for the curve_fitted data in reference; for fairness
    reference_waveform_nonfit = reference_signal[fit_value:] 
    recorded_waveform_nonfit = recorded_signal[fit_value:]
    
    #Global synchrony variable pearson's r 
    pear_corr,p_val = stats.pearsonr(reference_waveform_nonfit, recorded_waveform_nonfit)
    #Root mean squared error between the signals
    mse = mean_squared_error(reference_waveform_nonfit, recorded_waveform_nonfit)
    rmse = np.sqrt(mse)
    #Mean absolute error
    mae = mean_absolute_error(reference_waveform_nonfit, recorded_waveform_nonfit)
    #Cosine similarity
    cos_sim = cosine_similarity(reference_waveform_nonfit, recorded_waveform_nonfit)
    # #coefficient of determination(r^2 score)
    # r2 = r2_score(reference_waveform_nonfit, recorded_waveform_nonfit)
    
    # SNR
    signal_power = np.mean(reference_waveform_nonfit**2)
    noise_power = np.mean((reference_waveform_nonfit - recorded_waveform_nonfit)**2)
    snr = 10 * np.log10(signal_power / noise_power)

    #DTW
    #First query, then reference in using the dtw() function
    align = dtw(recorded_waveform_nonfit, reference_waveform_nonfit, keep_internals=True)
    dtw_dist = align.distance
    #norm_dtw_dist = dtw_dist / len(align.index1)

    align.plot(type='threeway', xlab='Recorded index')
    
    #Additional magnitude and phase calculations in Fourier domain
    # Compute the FFT of each signal
    fft_recorded = np.fft.rfft(recorded_waveform_nonfit)
    fft_reference = np.fft.rfft(reference_waveform_nonfit)
    
    # Compute the magnitudes and phases
    magnitudes_recorded = np.abs(fft_recorded)
    phases_recorded = np.angle(fft_recorded)
    magnitudes_reference = np.abs(fft_reference)
    phases_reference = np.angle(fft_reference)
    
    # Compute mean square error for magnitude
    mse_magni = mean_squared_error(magnitudes_reference, magnitudes_recorded)

    # MSE phases - without unwrapping
    mse_phase = mean_squared_error(phases_reference, phases_recorded)
    # # Compute mse for unwrapped phases
    # unwrapped_recorded = np.unwrap(phases_recorded)
    # unwrapped_reference = np.unwrap(phases_reference)
    # mse_unwrapped = np.mean((unwrapped_reference - unwrapped_recorded) ** 2)
    #print("Sample phase difference: ", (phases_recorded-phases_reference)[:10])
    sin_diff = np.sin(phases_reference-phases_recorded)
    ##print("Cosine differences: ",cos_diff[:10])
    circ_mse = np.mean(sin_diff**2)

    # Complex Correlation (pearson's r for complex part of signal)
    complex_corr = np.mean(np.conj(fft_reference) * fft_recorded) / \
        (np.sqrt(np.mean(np.abs(fft_reference)**2) * np.mean(np.abs(fft_recorded)**2)))
    #It's absolute value ranges between 0 and 1 for 1 perfectly correlated
    #It's angle, or the argument, gives phase diff between signals (0 is in phase)
    
    statistics = {"Pearsonr": pear_corr,
                "p-val": p_val,
                "RMSE": rmse,
                "MAE": mae,
                "SNR": snr,
                "DTW": dtw_dist, 
                "mse_magnitude": mse_magni,
                "mse_phase": mse_phase, 
                "circ_mse": circ_mse,
                "Complex corr mag": np.abs(complex_corr),
                "Complex corr phase": np.angle(complex_corr)
                }
            
    return statistics

def evaluate_recording(file,
                       channel=0, 
                       n_burst=5,
                       scale=300,
                       sleep_time = 1.84, 
                       frequency=50e-3, 
                       amplitude=2.5, 
                       offset=-37.5, 
                       t_cutoff_low=100, 
                       t_cutoff_up=500):
    """Creation of the recording waveform and reference waveform, based on input and curve fitting parameters """
    
    # load and retrieve necessary data
    _, recorded_data, _ = read_egg_v3_bursts(file,
                                             func=1, 
                                             scale=scale,
                                             n_burst=n_burst,
                                             sleep_ping=1,
                                             sleep_time=sleep_time,
                                             t_deviation=0.2)
    #recorded_data = recorded_data.dropna()
    recorded_data = averaging_bursts(recorded_data, n_burst=n_burst)
    #Make evenly long recording data for equality in analysis
    recorded_data = recorded_data[(recorded_data['elapsed_s']>t_cutoff_low) & (recorded_data['elapsed_s']<t_cutoff_up)]
    recorded_waveform = recorded_data[f"Channel {channel}"].values
    #print(recorded_waveform)
        
    timestamps = recorded_data["elapsed_s"].values
    
    fit_value = int(np.floor(len(recorded_data)/5))
    #Phase Calculation 2 
    # Fit the first x data points using non-linear least square error function curve_fit
    t = timestamps[:fit_value]
    initial_guess = [amplitude, frequency, 0, offset]  #phase = 0
    
    #Boundary setting for if the recorded signal is very diffent due to parsing; for correlation, otherwise corr values still too high
    lower_bound = [1, 20e-3, -90, -40]
    upper_bound = [50, 500e-3, 90, -35]
    bounds = (lower_bound, upper_bound)
    
    #Second returns pcov
    popt, _ = curve_fit(sine_wave, xdata=t, ydata=recorded_waveform[:fit_value], p0=initial_guess, bounds=bounds, maxfev = 5000) #maxfev can be lower probably, may save calc time
    
    # Optimized parameters for [amplitude, frequency, phase, offset]
    _, _, phase_opt, offset_opt = popt
    
    #For this particular batch of measurements, something weird happens with the optimal phase of n=3
    # if (n_samples == 3 & sleep_period==1):
    #     print(n_samples, sleep_period)
    #     phase_opt = 90
    # else: 
    #     print("No weird phase observed")
    
    print(f'Optimized phase: {phase_opt}')
    print(f'Optimized offset: {offset_opt}')
    
    # generate the reference waveform, correcting for the offset and the phase recording starts at
    reference_waveform = generate_waveform(timestamps, frequency, amplitude, offset_opt, phase_opt)

    t_plot = np.linspace(timestamps.min(), timestamps.max(), len(recorded_data))
    ref_plot = generate_waveform(t_plot, frequency, amplitude, offset_opt, phase_opt)

    #fs = timestamps[1]-timestamps[0]
    statistics =  calc_corr_stats(reference_waveform, recorded_waveform, fit_value=fit_value)
    
    # return recording data, timestamps, correlation, statistics, and the waveforms (returning them for inspection now, later can be altered)
    return recorded_data, timestamps, recorded_waveform, reference_waveform, t_plot, ref_plot, statistics 

def plot_signals_and_correlation(timesamps, t_plot, recorded_signal, reference_signal, n_burst, freq=50e-3):
    """Plotting the two signals in order to visualize correlation, """
    pal = sns.color_palette('deep')
    plt.figure(figsize=(10, 8))

    # Plot the recorded and reference signals
    plt.subplot(2, 1, 1)
    plt.scatter(timesamps, recorded_signal, label=f'TX per cycle: {n_burst}', marker='o',s=14, alpha=.7, color=pal[0])
    plt.plot(t_plot, reference_signal, label='Reference', color=pal[1])
    #plt.title("Recorded vs reference input ({:.0f} mHz)".format(freq*1000))
    plt.xlabel('Time (s)', size=16)
    plt.ylabel('Averaged potential [mV]', size=16)
    plt.xticks(size=14)
    plt.yticks(size=14)

    plt.legend(loc='upper center', fontsize=14)
    plt.ylim(-41,-34) #TODO: ADD Y-LIMIT TO PLOTS WHEN RECORDINGS ARE DONE AND GOOD!!
    #plt.xlim(150,175)

    plt.tight_layout()
    plt.show()


#%% JUST RUN AND PLOT 1 FILE

file1n_1s = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis\1 sec sleep signals\07182023_lowPmode_1samps_1secsleep_50mHz")
file7n_1s = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis\1 sec sleep signals\08082023_lowPmode_7samps_1secsleep_50mHz_second_5min")
file5n_184s = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis\1_84 sec sleep signals\11082023_inputwave_5samps_1_840sec_5min")

n_burst, sleep_period, skipped_samples = extract_info_from_filename(file5n_184s)

recorded_data, timestamps, \
            recorded_waveform, reference_waveform, \
                t_plot, ref_plot, statistics = evaluate_recording(file5n_184s,
                                                channel=0,
                                                n_burst=n_burst,
                                                scale=300,
                                                sleep_time=sleep_period,
                                                frequency=50e-3,
                                                amplitude=2.5,
                                                offset= -37.5, 
                                                t_cutoff_low=50, 
                                                t_cutoff_up=300)

plot_signals_and_correlation(timestamps, t_plot, recorded_waveform, reference_waveform, n_burst, freq=50e-3)

#%%
#General calculation of the statistics values and if necessary, plotting of the waveforms as well (comment out)

#folder_path = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF measurements\07182023_RF Readings_1-7samples_5min_lowP"


filepaths = []
#Initial analysis folder!!
# dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis\07182023 signals")
# #1s840ms sleep path
# #dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis\11082023 signals")
# files=list(dir.glob('*'))
# filepaths.extend(files)

#1s840ms folder path
#folder_path = r"C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF Readings miniPC/Battery testing/Slow_wave_input_1s840ms"

#Trying for both dirs
base_dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\RF Readings miniPC\Signal correlation analysis")
subdirs = [f for f in base_dir.iterdir() if f.is_dir()]

for subdir in subdirs:
    files=list(subdir.glob('*'))
    filepaths.extend(files)


statistics_meas = pd.DataFrame(columns=["Tx scheme", "Sleep"])
# statistics_meas = pd.DataFrame(columns=['Tx scheme',
#                                         "Sleep",
#                                         "Pearsonr", 
#                                         "MSE", 
#                                         "cosine", 
#                                         "R2", 
#                                         "mse_magnitude", 
#                                         "mse_phase"])

for i in range(len(filepaths)):
    file = filepaths[i]
    # packets_batch = extract_packets_batch(file)
    n_burst, sleep_period, skipped_samples = extract_info_from_filename(file)
    print(f"Transmissions per cycle = {n_burst} ")
    
    #Get back data for plotting, to compare and the statistics of the correlation between rec and ref
    if (n_burst == 7):
        recorded_data, timestamps, \
            recorded_waveform, reference_waveform, \
                t_plot, ref_plot, statistics = evaluate_recording(file,
                                                channel=0,
                                                n_burst=n_burst,
                                                scale=300,
                                                sleep_time=1,
                                                frequency=50e-3,
                                                amplitude=2.5,
                                                offset= -37.5, 
                                                t_cutoff_low=94, 
                                                t_cutoff_up=300)

    elif (sleep_period == 1.84):
        recorded_data, timestamps, \
            recorded_waveform, reference_waveform, \
                t_plot, ref_plot, statistics = evaluate_recording(file,
                                                channel=0,
                                                n_burst=n_burst,
                                                scale=300,
                                                sleep_time=1.84,
                                                frequency=50e-3,
                                                amplitude=2.5,
                                                offset= -37.5, 
                                                t_cutoff_low=94, 
                                                t_cutoff_up=300)

    elif (sleep_period == 1.88):
        recorded_data, timestamps, \
            recorded_waveform, reference_waveform, \
                t_plot, ref_plot, statistics = evaluate_recording(file,
                                                channel=0,
                                                n_burst=n_burst,
                                                scale=300,
                                                sleep_time=1.84,
                                                frequency=50e-3,
                                                amplitude=2.5,
                                                offset= -37.5, 
                                                t_cutoff_low=94, 
                                                t_cutoff_up=300)
    
    else:
        recorded_data, timestamps, \
            recorded_waveform, reference_waveform, \
                t_plot, ref_plot, statistics = evaluate_recording(file,
                                                channel=0,
                                                n_burst=n_burst,
                                                scale=600,
                                                sleep_time=1,
                                                frequency=50e-3,
                                                amplitude=2.5,
                                                offset= -37.5, 
                                                t_cutoff_low=94, 
                                                t_cutoff_up=300)
        
    
    
    statistics["Tx scheme"] = n_burst
    statistics["Sleep"] = sleep_period
    
    #Statistical measures for correlation       
    statistics_meas = pd.concat([statistics_meas, pd.DataFrame(statistics, index=[0])], ignore_index=True)    
    
    ## In order to plot both signals for comparison
    plot_signals_and_correlation(timestamps, t_plot, recorded_waveform, reference_waveform, n_burst, freq=50e-3)

#pear_corr, mse, cos_sim, r2, mse_magni, mse_phases

# %% MISCELANEOUS INCLUDING CROSS CORR. FUNCTION

# def cross_correlate(recorded_waveform, reference_waveform):
#     """Calculate the cross-correlation between the reference and recorded waveforms"""
#     # only use non-fitted data (curve-fit uses first 100 datapoints for optimal parameters)
#     recorded_waveform_nonfit = recorded_waveform[101:]
#     reference_waveform_nonfit = reference_waveform[101:]
    
#     #Defining the correlation using the scipy correlate function
#     correlation = correlate(recorded_waveform_nonfit, reference_waveform_nonfit, mode='same', method='direct')
#     #norm = np.sqrt(np.sum(recorded_waveform_nonfit ** 2) * np.sum(reference_waveform_nonfit ** 2))
#     # correlation = correlation / norm
#     #get rid of the padding, according to site: https://scicoding.com/cross-correlation-in-python-3-popular-packages/
    
#     return correlation

# def plot_signal(file,channel):
#     """Plotting of the recorded signal only, adding frequency and batch number"""    
#     #Preprocessing of the data
#     recorded_data = read_egg_v3_lowP(file)
#     time = recorded_data['timestamps']
#     recorded_signal = recorded_data['meanmV_Ch{}'.format(channel)]
#     frequency = extract_frequency_from_filename(file)
#     n_batch = extract_packets_batch(file)
    
    
#     #Plotting
#     plt.figure(figsize=(12, 8))
    
#     plt.plot(time, recorded_signal)
#     plt.title('Average mV of recorded waveform sent in batch of (n={}), input {:.0f} mHz'.format(n_batch, frequency*1000))
#     plt.xlabel('Time (s)')
#     plt.ylabel('mean Voltage [mV]')    
#     #plt.xlim(0:100)
    
       
