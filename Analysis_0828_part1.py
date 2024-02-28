# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:48:34 2023

@author: coenberns
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
from scipy.signal import butter, filtfilt
import seaborn as sns


from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from filterpy.kalman import KalmanFilter
import pywt
from functions_read_bursts import*
from Plot_EGG_adaptation import*

#%%
#Mac
# meas_path = pathlib.Path("/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")

# #Windows
#Baatery measurements
#meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF readings miniPC desk animal facility/Battery Tests/Mode 1 new")
# meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")
meas_path = pathlib.Path("/Users/coenberns/Documents_lab/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")

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
file_0828 = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file_0828.name}")

#%%
#For the general read-in of data file
v_mean_0828, v_fulldat_0828, times_0828 =read_egg_v3_bursts(file_0828,
                                                header = None,
                                                rate = 62.5,
                                                scale=300,
                                                n_burst=5,
                                                sleep_ping=1,
                                                sleep_time=1.84,
                                                t_deviation=0.2)



#%%
# v_fulldat2_0828 = v_fulldat_0828
# burst_length = 6
# channels = [f'Channel {i}' for i in range(8)]

# # def nanmean(series):
# #     return np.nanmean(series)

# # Apply the custom function for averaging
# for channel in channels:
#     v_fulldat2_0828[channel] = v_fulldat2_0828.groupby('burst_group')[channel].transform('mean')

# # Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
# for col in ['elapsed_s', 'corrected_realtime']:
#     v_fulldat2_0828[col] = v_fulldat2_0828.groupby('burst_group')[col].transform('first')

# # Filtering for the first packet of each burst
# v_mean_0828 = v_fulldat2_0828[v_fulldat2_0828['packet_miss_idx'] % burst_length == 0]

#%%
#Custom interpolation function that does not interpolate large gaps using cubic spline but with pchip or with linear interp1d
#Does take a considerable amount of time....
interp_mean_0828 = interpolate_data(v_mean_0828, cycle_time=times_0828['t_cycle'])
#For quick dirty interpolation:
# interp_mean = interpolate_egg_v3(v_mean)
savgol_mean_0828 = savgol_filt(interp_mean_0828)
#%% For the behavioral label data
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
fs_0828=times_0828['effective_rate']
t_cycle_0828 = times_0828['t_cycle']
egg_data_0828 = butter_filter(savgol_mean_0828, fs=times_0828['effective_rate'], low_freq=0.02, high_freq=0.2)
datcols = ['datetime', 'timestamps'] + [f'Channel {i}' for i in range(8)]
egg_data_0828 = egg_data_0828[datcols]
# egg_data_0828.to_hdf('08282023_part1_filtered.h5', key='part1')

#%%Alternative way 
# a,b,c_0828_slow = signalplot(savgol_mean_0828,xlim=(),spacer=100,vline=[],
#            freq=[0.02,0.2],order=3, rate=fs_0828, title='',skip_chan=[],
#             figsize=(10,15),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
#             output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})
# c_0828_slow.rename(columns={'Synctime': 'timestamps'}, inplace=True)
# savgol_small_0828 = savgol_mean_0828[['timestamps']+['datetime']]
# egg_data_0828 = c_0828_slow.merge(savgol_small_0828, how='left', on='timestamps')

#%% Plotting the savgol_mean complete part of the recording with lines for ambulation and feeding/drinking
sns.set_palette('tab10')
#%% Figure EGG results - vlines are from dataframe
signalplot(savgol_mean_0828,xlim=(),spacer=80,vline=[50,4451,8340,9363,10976,13444,15100,15513,16922,17800,21212,21956],
           freq=[0.02,0.2],order=3, line_params=['black', 2,'dashed'],
            rate=fs_0828, title='',skip_chan=[],
            figsize=(10,10),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%% MMC First part larger recording S1
a,b,c_0828 = signalplot_hrs(savgol_mean_0828,xlim=(),spacer=200,vline=[],
           freq=[0.0001,0.1],order=3, rate=fs_0828, title='',skip_chan=[0,1,2],
            figsize=(10,8),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='PD',Normalize_channels=False,labels=[],color_dict={},name_dict={})

a1,b1,c2_0828 = egg_signalfreq(c_0828, rate=fs_0828, freqlim=[0.001*60,0.1*60], mode='power', vline=[0.25,1.33],mmc=True,
                                figsize=(8,8))


#%% HEATPLOT AND BOOXPLOTS FOR Figure 2
q,r,v_feeding_0828 = heatplot(savgol_mean_0828,xlim=(0,4451),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

q,r,v_active_0828 = heatplot(savgol_mean_0828,xlim=(10976,13444),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

q,r,v_rest_0828 = heatplot(savgol_mean_0828,xlim=(17800,21212),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

v_feeding_0828 = v_feeding_0828.T
v_active_0828 = v_active_0828.T
v_rest_0828 = v_rest_0828.T

pd_feeding_0828 = pd.DataFrame(v_feeding_0828, columns=[f'Channel {i}' for i in range(8)])
pd_active_0828 = pd.DataFrame(v_active_0828, columns=[f'Channel {i}' for i in range(8)])
pd_rest_0828 = pd.DataFrame(v_feeding_0828, columns=[f'Channel {i}' for i in range(8)])

plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = 16

sns.set_palette('tab20')
boxplot = sns.boxplot(data=pd_feeding_0828, palette='tab10', showfliers=False)
boxplot.set_title('')
boxplot.set_xlabel('')
boxplot.set_ylabel('Electrical Activity (mV)')

means = pd_feeding_0828.mean()
stds = pd_feeding_0828.std()
Q3 = pd_feeding_0828.quantile(0.75)
Q1 = pd_feeding_0828.quantile(0.25)
IQR = Q3 - Q1
whisker_top = Q3 + 1.5 * IQR

for i in range(pd_feeding_0828.shape[1]):
    # Find the maximum value within the whisker range for the current channel
    whisker_val = whisker_top[i]
    channel_data = pd_feeding_0828.iloc[:, i]
    max_within_whisker = channel_data[channel_data <= whisker_val].max()

    # Place the text above the top whisker or max value within the whisker range
    plt.text(i, max_within_whisker + 0.2, f'Mean: {means[i]:.2f}\nSTD: {stds[i]:.2f}',
             horizontalalignment='center', size='small', color='black', weight='semibold')
plt.ylim(0,12)
plt.show()
#%% FREQUENCY HEATPLOT FIGURE 2 OF EGG RESULTS
egg_freq_heatplot_v2(savgol_mean_0828, rate=fs_0828, xlim=[0,25200],seg_length=600,freq=[0.03,0.2],freqlim=[1,8], order=3,
                            vrange=[0],figsize=(8,2),interpolation='bilinear',n=10, intermediate=False, mmc=False,
                            max_scale=.8,norm=True,time='timestamps', textsize=12,
                            skip_chan=[1,2,3,4,5,6,7])
#%%
seg_vmean_0828 = {}
seg_vmean_0828 = segment_data(v_mean_0828, gap_size=30, seg_length=1800, window=100, min_frac=0.8, window_frac=0.2, rescale=True)
print_segment_info(seg_vmean_0828)

#%%
seg_interp_0828={}
seg_filtered_0828={}
seg_savgol_0828={}
# seg_smooth={}


for i in range(len(seg_vmean_0828)):
        seg_interp_0828[i] = interpolate_data(seg_vmean_0828[i],cycle_time=t_cycle_0828, max_gap=14, pchip=True)
        seg_filtered_0828[i]=butter_filter(seg_interp_0828[i], low_freq=0.02, high_freq=0.2, fs=fs_0828)
        seg_savgol_0828[i]=savgol_filt(seg_interp_0828[i], window=3,polyorder=1,deriv=0,delta=1)
        # seg_smooth[i]=smooth_signal_moving_average(seg_filtered[i], window_size=5)

#%%
fig1, _,_ = signalplot(seg_savgol_0828[0],xlim=(0,5000),spacer=200,vline=[],freq=[0.02,0.2],order=3,
            rate=fs_0828, title='',skip_chan=[],
            figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
            output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})
#%% Feeding/non-feeding calculations
abs_mean_foodz=[]
abs_mean_partz=[]
abs_mean_nonz=[]
filtered = butter_filter(seg_savgol_0828[0], fs=fs_0828)
for i in range(8):
    abs_mean_food=np.abs(filtered[f'Channel {i}'][0:750]).mean()
    abs_mean_partial= np.abs(filtered[f'Channel {i}'][750:2500]).mean()
    # abs_mean_non=np.abs(filtered[f'Channel {i}'][:4000]).mean()
    abs_mean_foodz.append(abs_mean_food)
    abs_mean_partz.append(abs_mean_partial)
    # abs_mean_nonz.append(abs_mean_non)

print("During intensive eating: ", np.mean(abs_mean_foodz), np.std(abs_mean_foodz))
print("During partial eating: ", np.mean(abs_mean_partz), np.std(abs_mean_partz))
# print("During resting: ", np.mean(abs_mean_nonz), np.std(abs_mean_nonz))

#%%
fig2, _, activ = heatplot(seg_savgol_0828[0],xlim=(0,5000),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=fs_0828, title='',skip_chan=[],
                            figsize=(10,10),textsize=16,vrange=[0,15],interpolation='bilinear',norm=True)

#%%
# for i in range(len(seg_filtered)):
for i in range(len(seg_filtered_0828)):

    signalplot(seg_interp_0828[i],xlim=(0,0,0),spacer=0,vline=[],freq=[0.0005,0.01],order=3,
                rate=times_0828['effective_rate'], title='',skip_chan=[],
                figsize=(10,20),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
# 'Channel 0','Channel 1','Channel 2','Channel 3','Channel 4','Channel 5','Channel 6'
egg_freq_heatplot_v2(savgol_mean_0828, rate=fs_0828, xlim=[0,6400],seg_length=400,freq=[0.02,0.2],freqlim=[1,8], order=3,
                            vrange=[0],figsize=(10,15),interpolation='bilinear',n=5, intermediate=False, mmc=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])
#%% Channel 7 4cpm frequency heatplot
# skip_chan=['Channel 2','Channel 4', 'Channel 5', 'Channel 6']
egg_freq_heatplot_v2(seg_savgol_0828[0], rate=fs_0828, xlim=[0,3000],seg_length=600,freq=[0.02,0.2],freqlim=[1,8], order=3,
                            figsize=(8,3),interpolation='bilinear', n=10, intermediate=False, mmc=False,
                            max_scale=.5,norm=True,time='timestamps',
                            skip_chan=['Channel 0','Channel 1', 'Channel 2', 'Channel 3','Channel 4', 'Channel 5', 'Channel 6'])

#%% Signalplot for same segment 0-6000
a,b,c =signalplot(seg_savgol_0828[0],xlim=(0,3000),spacer=100,vline=[],freq=1,order=3,
                rate=times_0828['effective_rate'], title='',skip_chan=[0,1,2,3,4,5,6],
                figsize=(8,3),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

egg_signalfreq(c,rate=fs_0828, freqlim=[1,10], figsize=(9,3),labels=['Channel 7'])

a1,b1,c1 = signalplot(seg_savgol_0828[0],xlim=(0,3000),spacer=50,vline=[380,560],freq=[0.02,0.2],line_params=['black',3,'dashed'],
                    order=3,rate=times_0828['effective_rate'], title='',skip_chan=[0,1,2,3,4,5,6],
                    figsize=(8,4),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                    output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

egg_signalfreq(c1,rate=fs_0828, freqlim=[1,10], figsize=(9,3),labels=['Channel 7'])
#%%
signalplot(seg_savgol_0828[0],xlim=(380,560),spacer=50,vline=[420,479],freq=[0.02,0.2],order=3, line_params=['#d62728', 3, 'dashed'],
                rate=times_0828['effective_rate'], title='',skip_chan=[0,1,2,3,4,5,6],
                figsize=(8,4),textsize=16,hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',
                output='np',Normalize_channels=False,labels=[],color_dict={},name_dict={})

#%%
egg_freq_heatplot_v2(seg_interp_0828[1], rate=fs_0828, xlim=[0,7800],seg_length=600,freq=[0.02,0.2],freqlim=[2,7],
                            vrange=[0],figsize=(10,14),interpolation='bilinear',n=5, intermediate=False,mmc=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=[])

#%%
egg_freq_heatplot_v2(seg_interp_0828[2], rate=fs_0828, xlim=[0,4000],seg_length=400,freq=[0.02,0.2],freqlim=[1,8],
                            vrange=[0],figsize=(10,5),interpolation='bilinear',n=10, intermediate=False,
                            max_scale=.6,norm=True,time='timestamps',
                            skip_chan=['Channel 2','Channel 3','Channel 4','Channel 5', 'Channel 6', 'Channel 7'])

#%%
filtered_compact = seg_filtered_0828[0][datcols]

egg_signalfreq(filtered_compact,rate=fs_0828,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
               labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
savgol_mean_comp = savgol_mean_0828[datcols]
egg_signalfreq(savgol_mean_comp,rate=fs_0828,freqlim=[1,10],ylim=0,mode='power',ylog=False,xlog=False,clip=False,
               labels=[],figsize=(10,20),vline=[],vline_color='black',textsize=12,name_dict={})

#%%
heatplot(seg_interp_0828[0],xlim=(0,3000),spacer=0,vline=[],freq=[0.02,0.2],order=3,rate=62.5, title='',skip_chan=[],figsize=(10,10),textsize=16,vrange=[0,40],interpolation='bilinear',norm=True)


# %% Different characteristic states 
q,r,v_feeding_0828 = heatplot(savgol_mean_0828,xlim=(0,4451),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

q,r,v_active_0828 = heatplot(savgol_mean_0828,xlim=(10976,13444),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

q,r,v_rest_0828 = heatplot(savgol_mean_0828,xlim=(17800,21212),spacer=0,vline=[],freq=[0.02,0.2],order=3,
                    rate=fs_0828, title='',skip_chan=[],figsize=(10,10),textsize=24,vrange=[0,15],interpolation='bilinear',norm=True)

v_feeding_0828 = v_feeding_0828.T
v_active_0828 = v_active_0828.T
v_rest_0828 = v_rest_0828.T

# Create a DataFrame for each category with a 'Category' column
pd_feeding_0828 = pd.DataFrame(v_feeding_0828, columns=[f'Channel {i}' for i in range(8)])
pd_feeding_0828['Category'] = 'Feeding'

pd_active_0828 = pd.DataFrame(v_active_0828, columns=[f'Channel {i}' for i in range(8)])
pd_active_0828['Category'] = 'Active'

pd_rest_0828 = pd.DataFrame(v_rest_0828, columns=[f'Channel {i}' for i in range(8)])
pd_rest_0828['Category'] = 'Inactive'

#  Combine the data into a single DataFrame
combined_data = pd.concat([pd_feeding_0828, pd_active_0828, pd_rest_0828], ignore_index=True)

# Melt the combined DataFrame so each row is a single observation
melted_data = combined_data.melt(id_vars='Category', var_name='Channel', value_name='Electrical Activity (mV)')

# Plot the boxplot for each category
plt.figure(figsize=(12, 6))
sns.set_palette('tab10')
boxplot = sns.boxplot(x='Category', y='Electrical Activity (mV)', data=melted_data, showfliers=False, showmeans=True)
boxplot.set_xlabel('Characteristic state', size=18)
boxplot.set_ylabel('Electrical Activity (mV)', size= 18)

# Calculate the quartiles and IQR for each category
category_stats = melted_data.groupby('Category')['Electrical Activity (mV)'].agg(['mean', 'std', lambda x: x.quantile(0.75), lambda x: x.quantile(0.25)])
category_stats['IQR'] = category_stats.apply(lambda row: row['<lambda_0>'] - row['<lambda_1>'], axis=1)

# Sort categories based on the mean to match the order in the boxplot
category_stats_sorted = category_stats.sort_values('mean', ascending=False)

# Annotate the overall mean and std for each category
for i, (idx, row) in enumerate(category_stats_sorted.iterrows()):
    # Calculate the whisker value based on Q3 + 1.5 * IQR
    whisker_cap = row['<lambda_0>'] + 1.5 * row['IQR']
    
    # Get the max data point within the whisker cap for the category
    max_within_whisker = melted_data[
        (melted_data['Category'] == idx) &
        (melted_data['Electrical Activity (mV)'] <= whisker_cap)
    ]['Electrical Activity (mV)'].max()

    # Position the text above the whisker or max value within the whisker range
    plt.text(i, max_within_whisker + 0.1, f'Mean: {row["mean"]:.2f}\nSTD: {row["std"]:.2f}',
             horizontalalignment='center', size='x-small', color='black', weight='semibold')
plt.yticks(size=16)
plt.xticks(size=16)
plt.ylim(0, 7)
plt.show()

# %%
categories = ['Category'] + [f'Channel {i}' for i in range(8)]
filtered_0828 = egg_beh_data
filtered_0828 = filtered_0828[categories]
for i in range(8):
    channel = [f'Channel {i}']
    filtered_0828[channel]=np.absolute(filtered_0828[channel])

#%% PLOTTING AVERAGE OVER ALL CHANNELS PER CATEGORY
filtered_0828['mean_abs_volt'] = filtered_0828[[f'Channel {i}' for i in range(8)]].mean(axis=1)

# Now, we can create a new DataFrame that contains these means per category
means_per_category = filtered_0828[['Category', 'mean_abs_volt']]
category_stats = filtered_0828.groupby('Category')['mean_abs_volt'].agg(['mean', 'std'])
category_stats = category_stats.sort_values('mean')

# Calculate the quartiles and IQR for the mean electrical activity within each category.
category_stats['Q3'] = filtered_0828.groupby('Category')['mean_abs_volt'].quantile(0.75)
category_stats['Q1'] = filtered_0828.groupby('Category')['mean_abs_volt'].quantile(0.25)
category_stats['IQR'] = category_stats['Q3'] - category_stats['Q1']

# Plot the boxplot
plt.figure(figsize=(8, 6))
boxplot = sns.boxplot(x='Category', y='Mean Electrical Activity (mV)', data=filtered_0828, showfliers=False,
                        order=['Inactive', 'Active', 'Feeding/Drinking'])
plt.xlabel('Category',size=18)
plt.ylabel('Average Electrical Activity (mV)',size=18)
plt.xticks(size=16)
plt.yticks(size=16)

#  Add the annotations above the highest whisker
for i, (category, row) in enumerate(category_stats.iterrows()):
    # Calculate the whisker top end
    whisker_top_end = row['Q3'] + 1.5 * row['IQR']
    # Make sure the whisker top end doesn't exceed the max value
    max_value = filtered_0828[filtered_0828['Category'] == category]['Mean Electrical Activity (mV)'].max()
    whisker_top_end = min(whisker_top_end, max_value)
    # Annotate with mean and standard deviation
    plt.text(i, whisker_top_end+0.05, f"Mean: {row['mean']:.2f}\nSTD: {row['std']:.2f}", weight='semibold', 
             horizontalalignment='center', size=14, verticalalignment='bottom', color='black')

plt.ylim(-.2, 5.7)
plt.show()

# %% 
short = ['Category'] + [f'Channel {i}' for i in range(8)]
egg_beh_data_short = egg_beh_data[short]
for i in range(8):
    channel = [f'Channel {i}']
    egg_beh_data_short[channel] = np.absolute(egg_beh_data_short[channel])
plottable = egg_beh_data_short.groupby('Category').mean()
plottable_rest=plottable.reset_index()
plottable_long = pd.melt(plottable_rest, id_vars='Category', var_name='Channel', value_name='Average Activity')
plottable_long['Channel'] = plottable_long['Channel'].str.extract('(\d+)').astype(int)


plottable_long.sort_values(['Category', 'Average Activity'])
# Create the barplot with sorted data
plt.figure(figsize=(12, 6))
sns.set_palette('tab10')
barplot = sns.barplot(x='Channel', y='Average Activity', hue='Category', data=plottable_long, hue_order=['Inactive','Active','Feeding/Drinking'], alpha=1)
barplot.set_xlabel('Channel', size=18)
barplot.set_ylabel('Average Electrical Activity (mV)', size=18)
barplot.set_xticklabels(barplot.get_xticklabels(), rotation=0)  # Remove rotation if you prefer

# Show the legend
plt.legend(title='Category')
plt.xticks(size=16)
plt.yticks(size=16)
plt.ylim(0,3 )
plt.legend(prop={'size': 16})

# Set the x-tick labels to be just the channel numbers

# %%
