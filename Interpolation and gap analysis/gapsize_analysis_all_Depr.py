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
gaps_1 = get_gap_sizes(df1, sec_gap=5000)
gaps_2 = get_gap_sizes(df2, sec_gap=5000)

gaps_round1 = [round(num) for num in gaps_1]
gaps_round2 = [round(num) for num in gaps_2]

rounded_gaps_df = pd.DataFrame(gaps_round1 + gaps_round2)

gap_count = pd.DataFrame(rounded_gaps_df.value_counts(sort=True))
gap_counts_df = gap_count.reset_index(names='gap size')
gap_counts_df = gap_counts_df.sort_values('gap size')

#%% Non-weighted CDF of gap sizes
col = sns.color_palette('deep')
cdf = gap_counts_df['count'].cumsum() / gap_counts_df['count'].sum()

plt.figure(figsize=(10, 6))
plt.plot(gap_counts_df['gap size'], cdf, marker='.', linestyle='--', markersize=8, color = col[0])
plt.axhline(y=0.95, color=col[5], alpha=.75, linestyle=':')
plt.axvline(x=22, color=col[5], alpha=.75, linestyle=':')
plt.xscale('log')  
plt.xlabel('Gap size [s]')
plt.ylabel(r'CDF [$\mathbb{P}$]')
plt.grid(True)
plt.show()


#%% Weighted CDF based on gap size and count
data = gap_counts_df.copy()
data['weighted'] = data['gap size'] * data['count']
data = data.sort_values('gap size')
data['weighted_cdf'] = data['weighted'].cumsum() / data['weighted'].sum()

plt.figure(figsize=(10, 6))
plt.plot(data['gap size'], data['weighted_cdf'], marker='.', linestyle='--', markersize=8, color = col[0])
plt.axhline(y=0.95, color=col[5], alpha=.75, linestyle=':')
# plt.axvline(x=22, color=col[5], alpha=.75, linestyle=':')
plt.xscale('log')
plt.xlabel('Gap size [s]')
plt.ylabel(r'Weighted CDF [$\mathbb{P}$]')
plt.show()
# %%
