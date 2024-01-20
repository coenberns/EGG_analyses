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

v_compact_0814, v_fulldat_0814, times_0814 =read_egg_v3_bursts(file,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=2,
                                                t_deviation=0.2)

# %%
# v_mean1 = averaging_bursts(v_fulldat_0814)
v_fulldat2_0814 = v_fulldat_0814
burst_length = 6
channels = [f'Channel {i}' for i in range(8)]

# def nanmean(series):
#     return np.nanmean(series)

# Apply the custom function for averaging
for channel in channels:
    v_fulldat2_0814[channel] = v_fulldat2_0814.groupby('burst_group')[channel].transform('mean')

# Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
for col in ['elapsed_s', 'corrected_realtime']:
    v_fulldat2_0814[col] = v_fulldat2_0814.groupby('burst_group')[col].transform('first')

# Filtering for the first packet of each burst
v_mean_0814 = v_fulldat2_0814[v_fulldat2_0814['packet_miss_idx'] % burst_length == 0]

# %%
#Take into account larger gaps and pchip interpolation over them
interp_mean_0814 = interpolate_data(v_mean_0814, cycle_time=times_0814['t_cycle'],pchip=True)
savgol_mean_0814 = savgol_filt(interp_mean_0814, window=3,polyorder=1)
#%%

seg_vmean_0814 = {}
seg_vmean_0814 = segment_data(v_mean_0814, gap_size=65, seg_length=1500, window=100, min_frac=0.8, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0814)
# %%
seg_interp_0814={}
seg_filtered_0814={}
seg_savgol_0814={}
seg_combi_0814={}
# seg_smooth={}
fs_0814=times_0814['effective_rate']
t_cycle_0814 = times_0814['t_cycle']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean_0814)):
        seg_interp_0814[i] = interpolate_data(seg_vmean_0814[i],cycle_time=t_cycle_0814, max_gap=15, rescale=True)
        seg_filtered_0814[i]=butter_filter(seg_interp_0814[i], low_freq=0.02, high_freq=0.2, fs=fs_0814)
        seg_savgol_0814[i]=savgol_filt(seg_interp_0814[i], window=3,polyorder=1,deriv=0,delta=1)
        # seg_combi_0814[i]=savgol_filt(interpolate_egg_v3(seg_vmean_0814[i], method='cubicspline', order=3, rescale=False))

# segments_combi = pd.concat([seg_combi_0814[0],seg_combi_0814[1]])


#%% With start of feeding line
an_segment_0814 = seg_savgol_0814[0]
signalplot(an_segment_0814,xlim=(4000,6000),spacer=50,vline=[4380],freq=[0.02,0.2],order=3,line_params=['black',3,'dashed'],
            rate=times_0814['effective_rate'], title='',skip_chan=[0,2,7],
            figsize=(10,15),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
#%% 
import seaborn as sns
q,r,v_abs_0814 = heatplot(an_segment_0814,xlim=(4000,6000),spacer=0,vline=[4300],freq=[0.02,0.2],order=3,
                    rate=fs_0814, title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)

v_abs_0814 = v_abs_0814.T
pd_vabs_0814 = pd.DataFrame(v_abs_0814, columns=[f'Channel {i}' for i in range(8)])

plt.figure(figsize=(12, 8))
sns.set_palette('tab20')
boxplot = sns.boxplot(data=pd_vabs_0814, palette='tab10', showfliers=False)
boxplot.set_title('')
boxplot.set_xlabel('')
boxplot.set_ylabel('Electrical Activity (mV)')

means = pd_vabs_0814.mean()
stds = pd_vabs_0814.std()
Q3 = pd_vabs_0814.quantile(0.75)
Q1 = pd_vabs_0814.quantile(0.25)
IQR = Q3 - Q1
whisker_top = Q3 + 1.5 * IQR

for i in range(pd_vabs_0814.shape[1]):
    # Find the maximum value within the whisker range for the current channel
    whisker_val = whisker_top[i]
    channel_data = pd_vabs_0814.iloc[:, i]
    max_within_whisker = channel_data[channel_data <= whisker_val].max()

    # Place the text above the top whisker or max value within the whisker range
    plt.text(i, max_within_whisker + 0.2, f'Mean: {means[i]:.2f}\nSTD: {stds[i]:.2f}',
             horizontalalignment='center', size='small', color='black', weight='semibold')
plt.ylim(0,12)
plt.show()

#%%
heatfig_0814, _, freqdat_0814 = egg_freq_heatplot_v2(seg_savgol_0814[0], rate=fs_0814, xlim=[0,12000],seg_length=600,freq=[0.02,0.2],freqlim=[1,8],
                            figsize=(10,10),interpolation='bilinear',n=10, intermediate=False, mmc=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=[0,1,4,6])

#%%     
signalplot(seg_savgol_0814[0],xlim=(6120,6180),spacer=10,vline=[],freq=[0.02,0.2],order=3,
            rate=times_0814['effective_rate'], title='',skip_chan=[1,2,3,4,5,6,7],
            figsize=(10,5),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %%
# signalplot(segments_combi,xlim=(6120,6180),spacer=10,vline=[],freq=[0.02,0.2],order=3,
#             rate=times_0814['effective_rate'], title='',skip_chan=[1,2,3,4,5,6,7],
#             figsize=(10,5),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
#             output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
# #%%
# #'Channel 0','Channel 4', 'Channel 5'
# heatfig_boths, _, freqdat_boths = egg_freq_heatplot_v2(segments_combi, rate=fs_0814, xlim=[800,8000],seg_length=400,freq=[0.03,0.2],freqlim=[1,7],
#                             vrange=[0],figsize=(10,12),interpolation='bilinear',n=10, intermediate=False,
#                             max_scale=.8,norm=True,time='timestamps',
#                             skip_chan=['Channel 0','Channel 4', 'Channel 5'])
        
#%% Traversing wave front and it's motions
signalplot(savgol_mean_0814,xlim=(1500,1600),spacer=40,vline=[],freq=[0.02,0.2],order=3,
            rate=times_0814['effective_rate'], title='',skip_chan=['Channel 3','Channel 4', 'Channel 5','Channel 6','Channel 7'],
            figsize=(10,8),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
heatfig_all, _, freqdat_all = egg_freq_heatplot_v2(savgol_mean_0814, rate=fs_0814, xlim=[0,12000],seg_length=400,freq=[0.04,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=5, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])

# %%
mean_all = interp_mean_0814[datcols]
savgol_all = savgol_mean_0814[datcols]

freqfig_all1, _, freqs_all1 = egg_signalfreq(mean_all,rate=fs_0814,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

freqfig_all2, _, freqs_all2 = egg_signalfreq(savgol_all,rate=fs_0814,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
signalplot_hrs(segments_combi,xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
            rate=times_0814['effective_rate'], title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# for i in range(len(seg_filtered)):
for i in range(len(seg_interp_0814)):
    signalplot(seg_savgol_0814[i],xlim=(1000,1240),spacer=30,vline=[],freq=[0.02,0.2],order=3,
                rate=fs_0814, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
    
for i in range(len(seg_interp_0814)):
    signalplot(seg_interp_0814[i],xlim=(1000,1240),spacer=30,vline=[],freq=[0.02,0.2],order=3,
                rate=fs_0814, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%

heatfig_0814, _, freqdat_0814 = egg_freq_heatplot_v2(seg_savgol_0814[0], rate=fs_0814, xlim=[0,3200],seg_length=400,freq=[0.03,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])


#%%
heatfig2, _, freqdat2 = egg_freq_heatplot_v2(seg_savgol_0814[1], rate=fs_0814, xlim=[0,3000],seg_length=250,freq=[0.03,0.2],freqlim=[1,7],
                            vrange=[0],figsize=(10,12),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.8,norm=True,time='timestamps',
                            skip_chan=['Channel 4', 'Channel 5'])

# %%
seg1 = seg_interp_0814[0][datcols]

freqfig1, _, freqs1 = egg_signalfreq(seg1,rate=fs_0814,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
seg1s = seg_savgol_0814[0][datcols]

freqfig1s, _, freqs1s = egg_signalfreq(seg1s,rate=fs_0814,freqlim=[2,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})


#%%
seg2 = seg_interp_0814[1][datcols]

freqfig2, _, freqs2 = egg_signalfreq(seg2,rate=fs_0814,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})
#%%
seg2s = seg_savgol_0814[1][datcols]

freqfig2s, _, freqs2s = egg_signalfreq(seg2s,rate=fs_0814,freqlim=[2,10],ylim=0,mode='power',ylog=False,xlog=False,
               clip=False,labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})


# %%
seg1_filtered1=butter_filter(seg_savgol_0814[0],fs=fs_0814)
seg1_filtered2=butter_filter(seg_savgol_0814[1],fs=fs_0814)
savgol_filtered1=butter_filter(savgol_mean_0814,fs=fs_0814)

fig_hplot11,_,activ11 = heatplot(seg1_filtered1,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times_0814['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)
fig_hplot12,_,activ12 = heatplot(seg1_filtered2,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times_0814['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)
fig_hplot13,_,activ13 = heatplot(savgol_filtered1,xlim=(),spacer=0,vline=[],freq=1,order=3,rate=times_0814['effective_rate'], title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)


# %%
