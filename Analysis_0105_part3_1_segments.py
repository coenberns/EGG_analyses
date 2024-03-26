#
# Created on Tue Jan 09 2024
#
# Copyright (c) 2024 Berns&Co
#

#%% Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
import pathlib 
import seaborn as sns
from functions_read_bursts import*
import Old_Plot_EGG as oldEGG
from Plot_EGG_adaptation import*

#%% Dir selection
meas_path = pathlib.Path(r"/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/01042024_multiweek")
# # Make list of available files in dir
in_folder = [f for f in meas_path.iterdir() if f.is_file()]

# Print list of available files in directory
for i, f in enumerate(in_folder, start=1):
    print(f"{i}. {f.name}")

# Choice selection of file 
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
file_0105 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0105.name}")
#%%
#For the general read-in of data file
v_mean_0105, v_fulldat_0105, times_0105 =read_egg_v3_bursts(file_0105,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

if times_0105['t_cycle'] < 2 and times_0105['t_cycle'] > 1.9:
    print('Cycling time is okay')
else:
    print('FUCK')
#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0105 = interpolate_data_optimized(v_mean_0105, cycle_time=times_0105['t_cycle'], pchip=True)
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
# savgol_mean_0105 = interp_mean_0105
savgol_mean_0105 = savgol_filt(interp_mean_0105)

#%% 
sns.set_palette('tab10')

fs_0105 = times_0105['effective_rate']
a,b,c_0105_slow = signalplot(interp_mean_0105,xlim=(),spacer=250,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0105, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

c_0105_slow.to_hdf('0105_all_slowwave.h5', key='0105')

#%% Calculate dominant frequency per time segment, with 4 overlap
domfreq_segs, smooth_domfreqs_segs = calculate_dominant_frequencies(c_0105_slow, fs_0105, seg_time=120,time='Synctime',n=4)
domfreq_segs['DF_avg'] = domfreq_segs[[f'DF_Channel {i}' for i in range(8)]].mean(axis=1)
smooth_domfreqs_segs['DF_avg'] = smooth_domfreqs_segs[[f'DF_Channel {i}' for i in range(8)]].mean(axis=1)

domfreq_segs['SW_bool'] = (domfreq_segs['DF_avg']>= 3.2)&(domfreq_segs['DF_avg']<= 4.7)
smooth_domfreqs_segs['SW_bool'] = (smooth_domfreqs_segs['DF_avg']>= 3.2)&(smooth_domfreqs_segs['DF_avg']<= 4.7)
domfreq_segs['SW_bool'] = domfreq_segs['SW_bool'].astype(int)
smooth_domfreqs_segs['SW_bool'] = smooth_domfreqs_segs['SW_bool'].astype(int)

domfreq_segs.to_hdf('0105_DomFreq_segs_of120s.h5', key='0105')
smooth_domfreqs_segs.to_hdf('0105_DomFreq_segs_of120s_sm_window_n4.h5', key='0105')

#%% Signal plot for potential MMC recordings? Looks interesting
a,b,c_0105_mmc = signalplot(savgol_mean_0105,xlim=(),spacer=200,vline=[],freq=[0.0001,0.01],order=3,
            rate=fs_0105, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#Plotting its power density for low frequencies. Clear view of MMC 
a1,b1,c2_0105_mmc = egg_signalfreq(c_0105_mmc, rate=fs_0105, freqlim=[0.001*60,0.1*60], mode='power', vline=[0.25,1.33],mmc=True,
                                figsize=(10,20))

mmc_df = pd.DataFrame(c2_0105_mmc.T, columns=['Freq']+[f'Channel {i}' for i in range(8)])                

d_freqs = {}

# Iterate over channel columns
for channel in [f'Channel {i}' for i in range(8)]:
    # Find the index of the maximum magnitude for the current channel
    max_magnitude_index = mmc_df[channel].idxmax()
    
    # Extract the frequency corresponding to this maximum magnitude
    d_freq = mmc_df.loc[max_magnitude_index, 'Freq']
    
    d_freqs[channel] = d_freq

avg_dfreq = sum(d_freqs.values()) / len(d_freqs)

for channel, frequency in d_freqs.items():
    print(f"{channel}: {frequency} cycles/hr")

print(f"Average Dominant Frequency: {avg_dfreq} cycles/hr")


#%% Looking at dominant frequencies?
oldEGG.egg_freq_heatplot_v2(savgol_mean_0105,rate=fs_0105,xlim=[0,36000], freq=[0.0001,0.05], seg_length=6000, 
                        freqlim=[0.00001,0.05],time='timestamps', max_scale=.8, n=10, norm=True, skip_chan=[], figsize=(10,15))
#%%
# egg_freq_heatplot_v2(savgol_mean_0105,rate=.5,xlim=[400,2000], freq=[0.02,0.2], seg_length=100, 
#                         freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

#%%
seg_vmean_0105 = {}
seg_vmean_0105 = segment_data(v_mean_0105, gap_size=600, seg_length=4000, window=1000, min_frac=0.3, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0105)

#%%
seg_interp_0105={}
seg_filtered_0105={}
seg_savgol_0105={}
fs_0105=times_0105['effective_rate']
t_cycle_0105 = times_0105['t_cycle']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean_0105)):
        seg_interp_0105[i] = interpolate_data(seg_vmean_0105[i], cycle_time=t_cycle_0105, max_gap=14)
        seg_filtered_0105[i]=butter_filter(seg_interp_0105[i], low_freq=0.02, high_freq=0.2, fs=fs_0105)
        seg_savgol_0105[i]=savgol_filt(seg_interp_0105[i], window=3,polyorder=1,deriv=0,delta=1)


# %%
for i in range(0,7):
    print(i)
    signalplot(seg_interp_0105[i],xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
                rate=fs_0105, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %%
signalplot(seg_savgol_0105[3],xlim=(),spacer=50,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0105, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0105[0],rate=times_0105['effective_rate'],xlim=[0,9600], freq=[0.02,0.2], seg_length=600, 
                        freqlim=[1,8],time='timestamps', max_scale=.7, n=10)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0105[1],rate=times_0105['effective_rate'],xlim=[0,9600], freq=[0.02,0.2], seg_length=600, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0105[2],rate=times_0105['effective_rate'],xlim=[0,8400], freq=[0.02,0.2], seg_length=600, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Segment for segment heatplotting
egg_freq_heatplot_v2(seg_savgol_0105[3],rate=times_0105['effective_rate'],xlim=[0,13200], freq=[0.02,0.2], seg_length=600, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %%
heatplot(seg_savgol_0105[4],xlim=(),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=times_0105['effective_rate'], 
            title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)

# %%
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
segments = {}
for i in range(0,5):
    segments[i] = seg_savgol_0105[i][datcols]

#%% To show where the dominant frequency is located at different times (from during the day to night)
# During the night, the dominant frequency is more clearly in the preferred range, not a lot of lower freq
for i in range(0,5):
    egg_signalfreq(segments[i], rate=fs_0105, freqlim=[1,10], mode='power', vline=[3.2,4.4])



