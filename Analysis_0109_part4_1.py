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
from functions_read_bursts import*
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
file_0109 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0109.name}")
#%%
#For the general read-in of data file
v_mean_0109, v_fulldat_0109, times_0109 =read_egg_v3_bursts(file_0109,
                                                header = None,
                                                rate = 62.5,
                                                scale=600,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)

if times_0109['t_cycle'] < 2:
    print('Cycling time is okay')

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0109 = interpolate_data(v_mean_0109, cycle_time=2,pchip=True)
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0109 = savgol_filt(interp_mean_0109)

#%%
# savgol_mean_0109 = savgol_filt(interp_mean_0109)

#%% General plot in slow wave frequencies
fs_0109 = times_0109['effective_rate']
signalplot(savgol_mean_0109,xlim=(),spacer=250,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0109, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%% Signal plot for potential MMC recordings? Looks interesting
a,b,c_0109 = signalplot_hrs(savgol_mean_0109,xlim=(0,30),spacer=300,vline=[],freq=[0.0001,0.01],order=3,
            rate=fs_0109, title='',skip_chan=[0,1,2],
            figsize=(10,8),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%% Plotting its power density for low frequencies. Clear view of MMC 
a1,b1,c2_0109_mmc = egg_signalfreq(c_0109, rate=fs_0109, freqlim=[0.001*60,0.08*60], mode='power', vline=[0.25,.5,1.33],mmc=True,
                                figsize=(8,8))

mmc_df = pd.DataFrame(c2_0109_mmc.T, columns=['Freq', 'Channel 3','Channel 4','Channel 5','Channel 6','Channel 7'])

d_freqs = {}

# Iterate over channel columns
for channel in ['Channel 3', 'Channel 4', 'Channel 5', 'Channel 6', 'Channel 7']:
    # Find the index of the maximum magnitude for the current channel
    max_magnitude_index = mmc_df[channel].idxmax()
    
    # Extract the frequency corresponding to this maximum magnitude
    d_freq = mmc_df.loc[max_magnitude_index, 'Freq']
    
    d_freqs[channel] = d_freq

avg_dfreq = sum(d_freqs.values()) / len(d_freqs)

for channel, frequency in d_freqs.items():
    print(f"{channel}: {frequency} cycles/hr")

print(f"Average Dominant Frequency: {avg_dfreq} cycles/hr")

#%%
seg_vmean_0109 = {}
seg_vmean_0109 = segment_data(v_mean_0109, gap_size=60, seg_length=1000, window=100, min_frac=0.6, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0109)

#%%
seg_interp_0109={}
seg_filtered_0109={}
seg_savgol_0109={}
fs=times_0109['effective_rate']
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]

for i in range(len(seg_vmean_0109)):
        seg_interp_0109[i] = interpolate_data(seg_vmean_0109[i], cycle_time=times_0109['t_cycle'])
        seg_filtered_0109[i]=butter_filter(seg_interp_0109[i], low_freq=0.02, high_freq=0.2, fs=fs)
        seg_savgol_0109[i]=savgol_filt(seg_interp_0109[i], window=3,polyorder=1,deriv=0,delta=1)


# %%
for i in range(0,4):
    print(i)
    signalplot(seg_interp_0109[i],xlim=(),spacer=100,vline=[],freq=[0.02,0.2],order=3,
                rate=fs, title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %%
signalplot(seg_savgol_0109[3],xlim=(),spacer=50,vline=[],freq=[0.02,0.2],order=3,
            rate=fs, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

# %% Good result during the night of 10 january, very consistent ~ 4 cpm
egg_freq_heatplot_v2(seg_savgol_0109[2],rate=times_0109['effective_rate'],xlim=[0,4000], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=10)

# %% Other segments part 3_2
#segment 4, could potentially be combined with segment 5, only 2 min difference, USE interpolate_data() without rescaling. 
egg_freq_heatplot_v2(seg_savgol_0109[3],rate=times_0109['effective_rate'],xlim=[0,2400], freq=[0.02,0.2], seg_length=400, 
                        freqlim=[1,8],time='timestamps', max_scale=.8, n=5)

# %%
heatplot(seg_savgol_0109[2],xlim=(),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=times_0109['effective_rate'], 
            title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,20],interpolation='bilinear',norm=True)

# %%
