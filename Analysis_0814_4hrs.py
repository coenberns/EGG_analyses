# Old First Measurement analysis

"""
@author: coenberns

"""
#%%
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
from functions_read_bursts import*
from Plot_EGG_adaptation import*

#%%
file = r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Pig measurements\08152023_First_short_pigmeasurement\08152023_PowerCycle.txt"

v_compact, v_fulldat1, times =read_egg_v3_bursts(file,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=2,
                                                t_deviation=0.2)

# %%
v_mean1 = averaging_bursts(v_fulldat1)

# %%
#Take into account larger gaps and pchip interpolation over them
interp_mean1 = interpolate_data(v_mean1, cycle_time=times['t_cycle'],pchip=True)
#%%
savgol_mean1 = savgol_filt(interp_mean1, window=3,polyorder=1)
seg_vmean1 = {}
seg_vmean1 = segment_data(v_mean1, gap_size=15, seg_length=1500, window=100, min_frac=0.8, window_frac=0.2, rescale=False)
print_segment_info(seg_vmean1)
# %%
seg_interp1={}
kalman1={}
seg_filtered1={}
seg_savgol1={}
seg_combi1={}
# seg_smooth={}
fs1=times['effective_rate']
t_cycle1 = times['t_cycle']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean1)):
        seg_interp1[i] = interpolate_egg_v3(seg_vmean1[i], method='cubicspline', order=3, rescale=True)
        kalman1[i] = kalman_filter(seg_interp1[i])
        seg_filtered1[i]=butter_filter(seg_interp1[i], low_freq=0.02, high_freq=0.2, fs=fs1)
        seg_savgol1[i]=savgol_filt(seg_interp1[i], window=3,polyorder=1,deriv=0,delta=1)
        seg_combi1[i]=savgol_filt(interpolate_egg_v3(seg_vmean1[i], method='cubicspline', order=3, rescale=False))

segments_combi = pd.concat([seg_combi1[0],seg_combi1[1]])


# %%
signalplot(segments_combi,xlim=(6120,6180),spacer=10,vline=[],freq=[0.02,0.2],order=3,
            rate=times['effective_rate'], title='',skip_chan=[1,2,3,4,5,6,7],
            figsize=(10,5),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
#%%
#'Channel 0','Channel 4', 'Channel 5'
heatfig_boths, _, freqdat_boths = egg_freq_heatplot_v2(segments_combi, rate=fs1, xlim=[800,8000],seg_length=400,freq=[0.03,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,12),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=['Channel 0','Channel 4', 'Channel 5'])
        
#%%
signalplot(savgol_mean1,xlim=(1500,1600),spacer=25,vline=[],freq=[0.02,0.2],order=3,
            rate=times['effective_rate'], title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
heatfig_all, _, freqdat_all = egg_freq_heatplot_v2(savgol_mean1, rate=fs1, xlim=[0,12000],seg_length=400,freq=[0.04,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])

# %%
mean_all = interp_mean1[datcols]
savgol_all = savgol_mean1[datcols]

freqfig_all1, _, freqs_all1 = egg_signalfreq(mean_all,rate=fs1,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

freqfig_all2, _, freqs_all2 = egg_signalfreq(savgol_all,rate=fs1,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
signalplot_hrs(segments_combi,xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
            rate=times['effective_rate'], title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# for i in range(len(seg_filtered)):
for i in range(len(seg_interp1)):
    signalplot(seg_savgol1[i],xlim=(1000,1240),spacer=30,vline=[],freq=[0.02,0.2],order=3,
                rate=fs1, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
    
for i in range(len(seg_interp1)):
    signalplot(seg_interp1[i],xlim=(1000,1240),spacer=30,vline=[],freq=[0.02,0.2],order=3,
                rate=fs1, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%

heatfig1, _, freqdat1 = egg_freq_heatplot_v2(seg_savgol1[0], rate=fs1, xlim=[0,3200],seg_length=400,freq=[0.03,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])


#%%
heatfig2, _, freqdat2 = egg_freq_heatplot_v2(seg_savgol1[1], rate=fs1, xlim=[0,3000],seg_length=250,freq=[0.03,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,12),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])

# %%
seg1 = seg_interp1[0][datcols]

freqfig1, _, freqs1 = egg_signalfreq(seg1,rate=fs1,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
seg1s = seg_savgol1[0][datcols]

freqfig1s, _, freqs1s = egg_signalfreq(seg1s,rate=fs1,freqlim=[2,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})


#%%
seg2 = seg_interp1[1][datcols]

freqfig2, _, freqs2 = egg_signalfreq(seg2,rate=fs1,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})
#%%
seg2s = seg_savgol1[1][datcols]

freqfig2s, _, freqs2s = egg_signalfreq(seg2s,rate=fs1,freqlim=[2,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})


# %%
seg1_filtered1=butter_filter(seg_savgol1[0],fs=fs1)
seg1_filtered2=butter_filter(seg_savgol1[1],fs=fs1)
savgol_filtered1=butter_filter(savgol_mean1,fs=fs1)

fig_hplot11,_,activ11 = heatplot(seg1_filtered1,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)
fig_hplot12,_,activ12 = heatplot(seg1_filtered2,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)
fig_hplot13,_,activ13 = heatplot(savgol_filtered1,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)


# %%
