#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   Untitled-1
@Time    :   2024/03/21 11:09:42
@Author  :   Coen Berns 
@Version :   1.0
@Contact :   coenjpberns@gmail.com
@License :   (C)Copyright 2024-2025, Coen Berns
@Desc    :   None
'''
#%%
#import all 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import pathlib 
import seaborn as sns
from scipy import signal
from functions_read_bursts import*
import Old_Plot_EGG as oldEGG
from Plot_EGG_adaptation import*
#%% File with ±1Hz sampling of latest experiment
file_0313 = r"/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/03132024_multiday_1Hz/03132024_Full.txt"

vmean_0313, _, times_0313 = read_egg_v3_bursts(file_0313,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1,
                                                t_deviation=0.2)


# %%
vip_0313 = interpolate_data_optimized(vmean_0313, cycle_time=times_0313['t_cycle'], time_ip=False)

# %%
fs_0313 = times_0313['effective_rate']
sav_0313 = savgol_filt(vip_0313, window=5, polyorder=1)
_,_,c_0313_slow = signalplot(vip_0313, xlim=(),freq=[0.02,0.2], rate=fs_0313, 
                              time='timestamps', output='PD')
_,_,freqs_0313_slow = egg_signalfreq(c_0313_slow, rate=fs_0313)
# %%
egg_freq_heatplot_v2(vip_0313, rate=fs_0313, freq=[0.035,0.2], xlim=[45000,54000], seg_length=600, n=10, 
                     time='timestamps', freqlim=[1,8])

    
# %% 7 segments
segments = segment_data(vmean_0313, gap_size=15, seg_length=5000, window=100, min_frac=0.6, rescale=True)
print_segment_info(segments)
#%%
seg_ip = {}
tc = 1/fs_0313
for i in range(len(segments)):
    seg_ip[i] = interpolate_data_optimized(segments[i], cycle_time=tc, max_gap=15)

#%%
for i, value in enumerate(seg_ip):
    tlen = len(value) * tc
    slen = 600
    rounds = tlen // slen
    egg_freq_heatplot_v2(value, rate=fs_0313, xlim=[0,round*slen], seg_length=slen, freq=[0.02,0.2], 
                            freqlim=[1,8],time='timestamps', max_scale=0.6)
# %%
tlen_0105 = len(vip_0313)*1/fs_0313
slen = 600
roundlen = tlen_0105//slen
x1=22
x2=24
h = 3600
# x1*h,x2*h
#%%
_,_,c_sw_h = signalplot(vip_0313, xlim=(85400, 86000), freq=[0.035,0.2],rate=fs_0313, time='timestamps',
                             spacer=10, vline=[], line_params=['black',2,'dashed'],
                             skip_chan=[0,3,4,6,7], figsize=(10,8), x_rotation=45, output='PD')
_,_,c_mmc = signalplot(vip_0313, xlim=(86200, 86800),freq=[0.0001,0.01],rate=fs_0313, time='timestamps',
                spacer=200,vline=[], line_params=['black',2,'dashed'], output='PD', 
                skip_chan=[0,3,4,6,7], figsize=(10,8), x_rotation=45)
                

#%%
egg_signalfreq(c_sw_h, rate=fs_0313, mode='power', freqlim=[1,8], vline=[],figsize=(10,8))
# egg_signalfreq(c_mmc, rate=fs_0313, mode='power', mmc=True, freqlim=[0,5], vline=[0.25,1.33])

#%%
_,_, v_after = heatplot(vip_0313, xlim=(x1*h,x2*h),freq=[0.02,0.2], figsize=(10,8), vrange=[0,20], 
             rate=fs_0313, norm=True)
heatplot(vip_0313, xlim=(x1*h,x2*h),freq=[0.00001,0.01], figsize=(10,8), vrange=[0,70], 
             rate=fs_0313, norm=True)    

#%%
egg_freq_heatplot_v2(vip_0313,freq=[0.03,0.2], rate = fs_0313,freqlim=[1,8], max_scale=0.8
                     ,seg_length=slen,xlim = [(x1-1)*h, (x2+.5)*h], time='timestamps', n=10,
                     skip_chan = [], figsize=(10,12))


# %%
signalplot(vip_0313, xlim=(23.10*3600,23.25*3600),freq=[0.0001,0.2],rate=fs_0313, time='timestamps',
                spacer=150,vline=[], line_params=['black',2,'dashed'])


# %%
