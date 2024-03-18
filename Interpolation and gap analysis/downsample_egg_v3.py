#%% Downsampling old measurements
"""
@author: coenberns
11/29/2023 
"""
import sys
sys.path.append('/Users/coenberns/Documents_lab/Thesis/Coding/Ephys/Thesis Python')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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

# %% IMPORT OLD MEASUREMENT - LOOK AT SETTINGS ETC
# All commented out are non-significant frequencies

file0= r"2022.07.19_Full.txt"
file1= r"2022.07.27_Ambulate.txt"
file2= r"2023.08.09_Full.txt"
file3=r"2023.09.21_Terminal.txt"
file4=r"2022.06.29_Az.txt"
file5=r'2022.08.17_Full.txt'
file6=r"2022.05.02_combined.txt"
# file7=r"2022.05.20_Full_End_Clipped.txt"


# #%%

# fulldaty = read_egg_v3(file)

# a,b,c = signalplot(fulldaty, freq=[0.02,0.2], output='PD')

#%%
file = file5
dat = read_egg_v3(file5, rate=62.5, scale=150)
a,b,c_fulldat = signalplot(dat, output='PD')

#%%
resampled_fulldat, fulldat = downsample_from_signalplot(c_fulldat)

# resampled_fulldat, fulldat = downsample_to_burst(file, time_str='2S', scale=150, round=True)
# %%
col = sns.color_palette('deep')

resampled_R, fulldat_R = plot_signal_comparison(resampled_fulldat.copy(), fulldat.copy(), xlim=[4000,7500],fig_raw=(10,15),
                                                       freqlim=[0,10], raw=True, savgol=False)

resampled_F, fulldat_F = plot_signal_comparison(resampled_fulldat.copy(), fulldat.copy(), xlim=[4000,7500],
                                                    fig_spec=(25,20),freqlim=[0,10], raw=False, savgol=False)

#%%
diffs_R, chan_abs_avg_R, stats_R = calc_diff_resample(resampled_R, fulldat_R, freq=True, warp_plot=True, window=25)

#%%
diffs_F,chan_abs_avg_F, stats_F = calc_diff_resample(resampled_F, fulldat_F, freq=True, warp_plot=True, window=62)

#%% Plotting differences
#Preprocessing for plot
long_diffs = diffs_R.drop('timestamps', axis=1)
long_diffs = long_diffs.melt(var_name='Channels', value_name='delta V (mV)')
long_diffs['delta V [mV]'] = np.abs(long_diffs['delta V (mV)'])

#%%
plt.figure(figsize=(8, 8))  # Adjust the figure size as necessary
sns.boxplot(
    data=long_diffs,
    y='Channels',  # Channel names will be on the y-axis
    x='delta V [mV]',  # Difference values will be on the x-axis
    orient='h',  # Horizontal boxplots
    showfliers = False, 
    saturation=.8
)
# sns.pointplot(
#     data=long_diffs,
#     y='Channels',
#     x='delta A (mV)_2',
#     estimator='mean',
#     orient='h',
#     errorbar=None,  # No confidence interval is needed
#     join=True,  # Do not join the points with a line
#     markers='D',  # Use diamond as marker
#     color='darkred',  # Use dark red color to distinguish the mean
#     capsize=0.1  # Adds caps to the error bars (though error bars are not shown here)
# )
# Rename y-tick labels to only include the channel number
plt.yticks([0,1,2,3,4,5,6,7], labels=['0','1','2', '3', '4', '5', '6', '7'], fontsize=14)
plt.xticks(size=14)
plt.xlabel(r'$\Delta V$ (mV)', size=16)
plt.ylabel('Channels', size=16)
plt.tight_layout()
plt.show()

#%%
fig1,_,c_resampled = signalplot(resampled_fulldat,xlim=(0,8000), rate=0.5, time='timestamps', freq=[0.02,0.2], 
           skip_chan=[],hline=[], spacer=50, output='PD')

#%%
fig2,_,c_control = signalplot(fulldat, xlim=(0,8000), rate=62.5, time='timestamps', freq=[0.02,0.2],
           skip_chan=[], hline=[], spacer=50, output='PD')

# %%

fig3,_,_=egg_freq_heatplot_v2(resampled_interp, rate=0.5, xlim=[0,7000],seg_length=500,freq=[0.02,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,14),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])

# %%
fig4,_,_ = egg_freq_heatplot_v2(fulldat, rate=62.5, xlim=[0,7000],seg_length=500,freq=[0.02,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,16),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])

# %%
fullcols = ['timestamps']+[f'Channel {i}' for i in range(8)]
resatza = resampled_interp[fullcols]
fulldatza = fulldat[fullcols]

fig5,_, freqs_downs = egg_signalfreq(resatza, rate=0.5)

fig6,_, freqs_full = egg_signalfreq(fulldatza, rate=62.5)
# %%
