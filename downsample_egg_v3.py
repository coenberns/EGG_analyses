#%% Downsampling old measurements
"""
@author: coenberns
11/29/2023 
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
import seaborn as sns

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
from Old_Plot_EGG import*

# %% IMPORT OLD MEASUREMENT - LOOK AT SETTINGS ETC
file0= r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Old pig\2022.07.19_Full.txt"
file1= r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Old pig\2022.07.27_Ambulate.txt"
file2= r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Old pig\2023.08.09_Full.txt"
file3=r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Old pig\2023.09.21_Terminal.txt"
file4='2022.06.29_Az.txt'


#%% DO I HAVE TO INTERPOLATE? 
# fulldat_intp = fulldat.interpolate('cubicspline')

#%%
file = file4

fulldat, resampled_fulldat, resampled_interp = downsample_to_burst(file, time_str='2S', round=False)


# resampled_fulldat['packet_miss_idx'] = resampled_fulldat.index * 6
#%%
resampled_filtered = butter_filter(resampled_interp, fs=0.5, low_freq=0.02, high_freq=0.2)
resampled_savgol = savgol_filt(resampled_filtered, window=3, polyorder=1)

# %%

fig1,_,_ = signalplot(resampled_interp,xlim=(0,1000), rate=0.5, time='timestamps', freq=[0.02,0.2], 
           skip_chan=[],hline=[], spacer=50)

fig2,_,_ = signalplot(fulldat, xlim=(0,1000), rate=62.5, time='timestamps', freq=[0.02,0.2],
           skip_chan=[], hline=[], spacer=50)
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
