#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu December 7 12:06 2023

@author: coenberns
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, timedelta, time
import pathlib 
from Old_Plot_EGG import*
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
from Old_Plot_EGG import*

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
file1 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file1.name}")

#%%
#For the general read-in of data file
v_compact1, v_fulldat1, times1 =read_egg_v3_bursts(file1,
                                                header = None,
                                                rate = 62.5,
                                                scale=300,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

#%%
v_mean1 = averaging_bursts(v_fulldat1)
segment_lengths = []
seg_vmean1 = {}
seg_vmean1 = segment_data(v_mean1, gap_size=14, seg_length=1800, window=100, min_frac=0.8, window_frac=0.2)
for i in range(len(seg_vmean1)):
    segment_lengths.append(len(seg_vmean1[i]))
print_segment_info(seg_vmean1)

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
file2 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file2.name}")

#%%
#For the general read-in of data file
v_compact2, v_fulldat2, times2 =read_egg_v3_bursts(file2,
                                                header = None,
                                                rate = 62.5,
                                                scale=300,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)


#%%
v_mean2 = averaging_bursts(v_fulldat2)
seg_vmean2 = {}
seg_vmean2 = segment_data(v_mean2, gap_size=14, seg_length=1800, window=100, min_frac=0.8, window_frac=0.2)
for i in range(len(seg_vmean2)):
    segment_lengths.append(len(seg_vmean2[i]))
print_segment_info(seg_vmean2)


# %%
gaps_seg1 = {}
gaps_seg2 = {}

# gaps=get_gap_sizes(seg1,sec_gap=15)
#%%
for i in range(len(seg_vmean1)):
    gaps_seg1[i] = get_gap_sizes(seg_vmean1[i], sec_gap=15)

for i in range(len(seg_vmean2)):
    gaps_seg2[i] = get_gap_sizes(seg_vmean2[i], sec_gap=15)
# %%
gap_sizes_combined = []
nr_gaps_seg = []

# Iterate through gaps_seg1 and append to gap_sizes_combined
for seg_id, gaps in gaps_seg1.items():
    nr_gaps_seg.append(len(gaps))
    for gap in gaps:
        gap_sizes_combined.append(gap)

# Iterate through gaps_seg2 and append to gap_sizes_combined
for seg_id, gaps in gaps_seg2.items():
    nr_gaps_seg.append(len(gaps))
    for gap in gaps:
        gap_sizes_combined.append(gap)

# Convert to DataFrame
gap_df = pd.DataFrame(gap_sizes_combined, columns = ['gap size'])

# gap_df.to_pickle('gap_df_segments_14sec.pkl')
ratios =[nr_gaps / seg_len for seg_len,nr_gaps in zip(segment_lengths, nr_gaps_seg)]

avg_ratio = np.mean(ratios)

# %%

