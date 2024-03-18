#File for determination of gap sizes, including bar plots
#%%
"""
@author: coenberns
11/29/2023 
"""
import sys
sys.path.append('/Users/coenberns/Documents_lab/Thesis/Coding/Ephys/Thesis Python')

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
import seaborn as sns

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
from Old_Plot_EGG import*

#%%
file_pt1=r"/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2/08282023_firstpartambu_noglitch.txt"
file_pt2=r"/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2/08282023_Ambulation_secondpart.txt"

#For the general read-in of data file
df1, v_fulldat_pt1, times_pt1 =read_egg_v3_bursts(file_pt1,
                                                            header = None,
                                                            rate = 62.5,
                                                            scale=600,
                                                            n_burst=5,
                                                            sleep_ping=1,
                                                            sleep_time=1.84,
                                                            t_deviation=0.2)

df2, v_fulldat_pt2, times_pt2 =read_egg_v3_bursts(file_pt2,
                                                            header = None,
                                                            rate = 62.5,
                                                            scale=600,
                                                            n_burst=5,
                                                            sleep_ping=1,
                                                            sleep_time=1.84,
                                                            t_deviation=0.2)

#%%
# Calculate the number of segments for each dataframe
num_segs1 = int(np.floor(df1['timestamps'].max() / (3 * 3600)))
num_segs2 = int(np.floor(df2['timestamps'].max() / (3 * 3600)))

# Determine the size of each segment based on the total length of the dataframe
seg_size1 = len(df1) // num_segs1
seg_size2 = len(df2) // num_segs2

# Create dictionaries to store the segments
df_dict1 = {}
df_dict2 = {}

# Split df1 into num_segs1 parts
for i in range(num_segs1):
    df_dict1[f'seg1 {i}'] = df1.iloc[i * seg_size1: (i + 1) * seg_size1]

# Split df2 into num_segs2 parts
for i in range(num_segs2):
    df_dict2[f'seg2 {i}'] = df2.iloc[i * seg_size2: (i + 1) * seg_size2]

#%%
gaps_dict1 = {}
gaps_dict2 = {}
mean_list1 = []
mean_list2 = []
length_segs = []
nr_gaps = []
for i in range(len(df_dict1)):
    gaps_dict1[i]=get_gap_sizes(df_dict1[f'seg1 {i}'], sec_gap=5000)
    mean_list1.append(np.mean(gaps_dict1[i]))
    length_segs.append(len(df_dict1[f'seg1 {i}']))
    nr_gaps.append(len(gaps_dict1[i]))
for i in range(len(df_dict2)):
    gaps_dict2[i]=get_gap_sizes(df_dict2[f'seg2 {i}'], sec_gap=5000)
    mean_list2.append(np.mean(gaps_dict2[i]))
    length_segs.append(len(df_dict2[f'seg2 {i}']))
    nr_gaps.append(len(gaps_dict2[i]))

all_means = np.concatenate([mean_list1, mean_list2])
meanest_mean = np.mean(all_means)
print("List of means: ", all_means)
print("List of # of gaps per 3+ hour rec: ", nr_gaps)
print("Mean of means: ", meanest_mean)
print("Mean of # of gaps: ", np.mean(nr_gaps))
print("Segment lengths: ", np.unique(length_segs))

#%%
for i in range(len(gaps_dict1)):

    plt.figure(figsize=(8,6))
    sns.histplot(gaps_dict1[i])
    #plt.plot(np.mean(gaps_dict1), marker='x', markersize=14)
    # plt.ylim(-1,50)
    plt.show()

#%%
for i in range(len(gaps_dict2)):

    plt.figure(figsize=(8,6))
    sns.histplot(gaps_dict2[i])
    plt.show()

# %%
# Combine gap sizes into a single DataFrame
gap_sizes_combined = []
gaps_amount = []

for seg_id, gaps in gaps_dict1.items():
    gaps_amount.append(len(gaps))
    for gap in gaps:
        gap_sizes_combined.append({'gap size': gap, 'Part': '1'})

for seg_id, gaps in gaps_dict2.items():
    gaps_amount.append(len(gaps))
    for gap in gaps:
        gap_sizes_combined.append({'gap size': gap, 'Part': '2'})

gap_df = pd.DataFrame(gap_sizes_combined)
print(gaps_amount)

ratio_dos = [nr_gaps / seg_len for seg_len,nr_gaps in zip(length_segs, gaps_amount)]
avg_ratio_dos = np.mean(ratio_dos)
print(avg_ratio_dos)

#%%
# gap_df.to_pickle('gap_df_48hrs.pkl')

#%%
# Plotting the histogram
sns.color_palette('tab20')
plt.figure(figsize=(8,6))
sns.histplot(data=gap_df, x='gap size', hue='Part', element='step', bins=range(1, 16, 1),stat='probability',common_norm=False, palette='deep')  # Adjust bins as needed
# plt.title("Distribution of gap sizes")
plt.xlabel("Gap size in seconds")
plt.ylabel("Probability")
plt.ylim(0,0.6)
# plt.legend(title="Part of recording")
plt.show()

# %% Now in distribute gaps function --> how probability for different gap sizes is distributed
# Define bin edges and label them
bins = np.arange(1.8, 16, 2)
labels = np.arange(2, 16, 2).astype(str)

# Bin the data 
gap_df['binned'] = pd.cut(gap_df['gap size'], bins=bins, labels=labels, include_lowest=True)

# Calculate counts and proportions
binned_counts = gap_df['binned'].value_counts().sort_index()
total_gaps = binned_counts.sum()
probs = binned_counts / total_gaps
print(probs)

# %%
