
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
from Plot_EGG import*
import timeit
import time
import cProfile
import sklearn
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import UnivariateSpline as univsp
from scipy.signal import savgol_filter
# from Plot_EGG_adaptation import*

# %%
#Mac
# meas_path = pathlib.Path("/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")

# #Windows
meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")
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
file = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file.name}")

#%% INITIAL ASSIGN GROUPS FUNCTION
def assign_groups(df, n_burst=5, sleep_time=2, t_deviation=0.2, n_missing=1):
    """
    Standardized grouping of the low power transmission scheme using packet bursts / sleeping cycles

    n_burst: Number of recordings SENT to receiver per recording/sleep cycle
    sleep_time: amount of time the device sleeps in between recordings (in seconds)
    t_deviation: variable to define allowed deviation from sleep_time
    n_missing: amount of packets in burst that can have a NaN value after the first transmission

    Packets are grouped based on the following:
        - There are n_burst consecutive recordings of voltages without errors 
        - Sleep transmission in between
        - Consecutive value after sleep is a transmission of a packet containing voltages again
        - Exception to the rule: n_missing amount of packets in the recording contain a NaN value
        - Possibly, after the sleep transmission; next burst can have a "non-transmission" value (i.e. 'miss')
    """
    group = 0
    # List to hold group numbers
    groups = [np.nan] * len(df)
    sleep_time_min = sleep_time - t_deviation
    sleep_time_max = sleep_time + t_deviation

    for i in range(len(df) - n_burst-1):

        if df['misc'].iloc[i] == " 16 ":
            next_n_values = df['misc'].iloc[i+1:i+n_burst-1].tolist()

            # Check to see not last potential group of data
            if i <= len(df) - n_burst:
                sleep_value = df['misc'].iloc[i+n_burst]
                new_burst_value = df['misc'].iloc[i+n_burst+1]
                time_diff = df['time_diff'].iloc[i+n_burst+1]

                if all(val == " 16 " for val in next_n_values) and sleep_value == " 0a " and new_burst_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                    group += 1
                    groups[i:i+n_burst] = [group] * n_burst

                elif sum(val == "miss" for val in next_n_values) <= n_missing and sleep_value == " 0a " and new_burst_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                    group += 1
                    groups[i:i+n_burst] = [group] * n_burst

            # Check for potential last group in data
            elif i == len(df) - n_burst+1 and all(val == " 16 " for val in next_n_values):
                group += 1
                groups[i:i+(n_burst-1)] = [group] * n_burst

    df['group'] = groups
    return df

#%%
#INITIAL TIME DIFFERENCE CALCULATIONS
def avg_time_diffs(df, n_burst=5, sleep_time=2, t_deviation=0.2):
    """
    Calcutating the effective sampling rate, using the values of average time differences
    between i) bursts and ii) sleep packets (to get the most reliable average). 
    Average wake up time as additional output. For conversion to low power mode, due to 
    rec/sleep cycling the sampling rate is slightly altered. This needs to be taken into 
    account to make sure the received values of the times and voltages can be used for 
    further analysis. 

    Inputs:     
        n_burst: Number of recordings SENT to receiver per recording/sleep cycle
        sleep_time: amount of time the device sleeps in between recordings (in seconds)
        t_deviation: variable to define allowed deviation from sleep_time

    Outputs:
        avg_burst_diff: Average time between the average burst difference
        avg_sleep_time: Average time between sleep packets received
        avg_wake_up_time: Average time between the sleep packet and new transmission - sleep time 
        effective_rate: the effective sampling rate
        - 
    """

    df_burst = df.copy()
    df_sleep = df.copy()
    sleep_time_min = sleep_time - t_deviation
    sleep_time_max = sleep_time + t_deviation

    # Calculate the average wake up time based on the sleep time and the time difference between sleep and next packet
    sleep_times = []
    wake_up_times = []
    for i in range(len(df_sleep)-1):
        if df_sleep['misc'].iloc[i] == " 0a ":
            next_value = df_sleep['misc'].iloc[i+1]
            time_diff = df_sleep['time_diff'].iloc[i+1]
            if next_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                sleep_times.append(time_diff)
                wake_up_time = time_diff - sleep_time
                wake_up_times.append(wake_up_time)

    if sleep_times:
        avg_sleep_time = sum(sleep_times)/len(sleep_times)
    else:
        sleep_times = 0
    
    if wake_up_times:
        avg_wake_up_time = sum(wake_up_times)/len(wake_up_times)
    else:
        avg_wake_up_time = 0

    burst_times = []
    for i in range(len(df_burst)-n_burst-1):
        if (df_burst['misc'].iloc[i] == " 16 ") and (not np.isnan(df_burst['group'].iloc[i])):
            first_burst = df_burst['packet_re_idx'].iloc[i]
            last_burst = df_burst['packet_re_idx'].iloc[i+n_burst-1]

            # Check if all values in the range are "16"
            next_value_cat = df_burst['misc'].iloc[i+1:i+n_burst]
            if all(val == " 16 " for val in next_value_cat):
                
                next_values = df_burst['time_diff'].iloc[i+1:i+n_burst]
                if (last_burst - first_burst == n_burst - 1):
                    burst_times.append(next_values.sum())
    if burst_times:
        avg_burst_time = sum(burst_times)/len(burst_times)
    else:
        avg_burst_time = 0
    # Average time between sample bursts and effective sampling rate in SPS
    #avg_time_between = np.mean([avg_burst_diff, avg_sleep_time])
    avg_time_between = avg_burst_time+avg_sleep_time
    effective_rate = 1 / avg_time_between

    return avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

#%%
#MASTER FUNCTION
def read_egg_v3_lowerP(file,
                        header=0,
                        rate = 62.5,
                        scale=600,
                        error=0,
                        date=None,
                        n_burst=5,
                        sleep_ping=1,
                        sleep_time=2,
                        t_deviation=0.2,
                        n_missing=1,
                        func = 3,
                        mean = True):
    """
    This function reads in and preprocesses the data from a text file generated by smartRF from 
    EGG_V3 when the recording mode is set to low power mode (Mode 2 or 3). 
    Inputs: 
        file : filepath of the target txt file
        header : Number of lines to skip
        scale : +- scale in mV 
        error : returns data with CRC errors. Default is 0 so those are stripped
        date : The date of the recording (automatically fetched if in beginning of txt file if format: MMDDYYYY_textfilename.txt)
        n_burst: Number of recordings SENT to receiver per recording/sleep cycle
        sleep_time: amount of time the device sleeps in between recordings (in seconds)
        t_deviation: variable to define allowed deviation from sleep_time
        n_missing: amount of packets in burst that can have a NaN value after the first transmission

    Outputs:
        VT_data: Voltage time data, use timestamps for time (others are indication of functionality, first two can be left out)
            .avg_elapsed_t: datetime object which contains the average elapsed time of a group (1, just for indication of difference)
            .avg_elapsed_s: The average elapsed time of the group, from start, in seconds (2, just for indication of difference)
            .packet_re_idx: The re-indexed packet number (zero is first voltage packet, note: this can be renamed "counter" if necessary)
            .group: group numbers based on the assign_groups() function 
            .Channel {i}: mean voltage per group for all channels
            .timestamps: The evenly spaced time variable, based on avg_time_between and the packet number divided by 6
        grouped_fulldat: The full dataset, grouped based on the assign_groups() function
        volt_fulldat: The dataset including all voltages, average and raw and all other columns as in grouped_fulldat (sleep and missing are dropped)
        avg_time_between: The average time between the sample bursts as calculated by avg_time_diffs() 
        effective_rate: The effective sampling rate based on the avg_time_between(*) (=1/*)

    """
    complete_start = time.time()
    # Putting in the date information to create complete datetime object, if information available
    if date is None:
        # get only the filename from the file path
        file_path = pathlib.Path(file)
        filename = file_path.name
        # extract date from the filename
        date = filename.split('_')[0]

    # Creating datetime object
    # which takes in "MMDDYYYY" like only US people write date order
    date = datetime.strptime(date, '%m%d%Y')
    dat = pd.read_csv(file, header=0, dtype=str, delimiter='|', names=[
        'realtime', 'misc', 'packet', 'msg', 'rssi'])
    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('16') | dat.misc.str.contains('0a')]
    dat = dat.reset_index(drop=True)
    counter = dat.packet.astype(int)
    new_counter = [0]
    for j, ele in enumerate(counter[1:]):
        step = counter[j+1]-counter[j]
        if step > 0:
            new_counter.append(step+new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
    abscounterseries = pd.Series(new_counter, name='packet_re_idx')
    dat = pd.concat((dat, abscounterseries), axis=1)  
    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    datetime_counter = time.time()
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary, with additional glitch values excluded
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    dat['rollover'] = dat['time_diff'] < 0
    dat['glitch'] = (dat['time_diff'] > -5) & (dat['rollover'])

    n_glitch = 50
    dat['glitch'] = dat['glitch'].rolling(window=n_glitch, min_periods=1).max().astype(bool)
    dat.loc[dat['glitch'], 'realtime'] = np.nan
    dat['correct_rollover'] = dat['rollover'] & ~dat['glitch'] 
    dat['days_to_add'] = dat['correct_rollover'].cumsum()
    dat['corrected_realtime'] = dat['realtime'] + pd.to_timedelta(dat['days_to_add'], unit='D')
    # dat['corrected_realtime'].interpolate(method='linear', inplace=True)

    # probably delete this if timestamps values at end are close to elapsed_s
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    datetime_end_counter = time.time()
    print("Time for datetime conversion :", datetime_end_counter-datetime_counter)

    hexdat_counter = time.time()
    dat_col = dat.msg
    # Return list of splits based on spaces in msg
    hexdat = dat_col.str.split(' ')
    serieslist = []
    sleep_pings = []  # List to store sleep pings
    for k, ele in enumerate(hexdat):
        if len(ele) == 23:  # Only select those that have the correct length for voltage data
            vlist = []
            for i in range(0, 10):
                n = i*2 + 2
                value = ''.join(['0x', ele[n], ele[n-1]])
                hvalue = int(value, 16)
                if i == 0:
                    vlist.append(hvalue)  # append hex code
                else:
                    if hvalue < 2**15:
                        vlist.append(scale*float(hvalue)/(2**15))
                    else:
                        vlist.append(scale*(((float(hvalue)-2**16)/(2**15))))
        elif len(ele) == 11:  # This is a sleep packet
            # Extract the number of pings from the first two bytes of the message and convert to integer
            ping_value = int(''.join(['0x', ele[1], ele[0]]), 16)
            sleep_pings.append(ping_value)
            # Fill the remaining places with None to maintain vlist structure
            vlist = [ping_value] + [None]*9
        else:  # Error in packet structure
            print('Line Error!' + str(k))
            print(ele)
            vlist = []  # add empty list on error
        serieslist.append(vlist)
    hexdat_end_counter = time.time()
    print("Time for hexdat: ", hexdat_end_counter-hexdat_counter)
    collist = ['SPI']
    for i in range(8):
        collist.append('Channel '+str(i))  # make channel list name
    collist.append('CRC')
    datalist = pd.DataFrame(serieslist, columns=collist)
    # print(datalist)
    fulldat = pd.concat((dat, datalist), axis=1)
    print(fulldat)
    #DO WE ALREADY NEED A MISSING PACKET NUMERATOR HERE? NO RIGHT?
    start_time2=time.time()
    grouped_fulldat = assign_groups(fulldat,
                                    n_burst=n_burst,
                                    sleep_time=sleep_time,
                                    t_deviation=t_deviation,
                                    n_missing=n_missing)
    end_time2=time.time()
    print("Grouping function takes: ", end_time2-start_time2)
    start_avg_times = time.time()
    avg_time_between, effective_rate, avg_wake_up_time,\
        avg_sleep_time, avg_burst_time, burst_times = avg_time_diffs(df=fulldat,
                                                                    n_burst=n_burst,
                                                                    sleep_time=sleep_time,
                                                                    t_deviation=t_deviation
                                                                    )                                        
    end_avg_times = time.time()
    print("Calculating average times: ", end_avg_times-start_avg_times)
    #EXTRA MANIPULATIONS BEFORE AVERAGING OR NOT
    grouped_fulldat1 = grouped_fulldat[grouped_fulldat['misc'] == " 16 "]
    groupdat = grouped_fulldat1.copy()
    groupdat['packet_re_idx'] = groupdat['packet_re_idx']-groupdat['packet_re_idx'].iloc[0]
    groupdat['elapsed_t'] = groupdat['elapsed_t']-groupdat['elapsed_t'].iloc[0]
    groupdat['elapsed_s'] = groupdat['elapsed_t'].dt.total_seconds()
    groupdat['packet_miss_idx'] = groupdat['packet_re_idx']
    #Finding missing packets for nan values
    expected_packets = list(range(min(groupdat['packet_miss_idx']), max(groupdat['packet_miss_idx'])+1))
    missing_packets = list(set(expected_packets) - set(groupdat['packet_miss_idx'].to_list()))
    missing_rows = pd.DataFrame(
        [{'misc': 'miss', 'packet_miss_idx': re_idx} for re_idx in missing_packets])
    groupdat = pd.concat([groupdat, missing_rows], ignore_index=True)
    groupdat = groupdat.sort_values(by='packet_miss_idx').reset_index(drop=True)
    #Time propagation functions - choose one of 3 (generally 1,3 are best, 3 is standard)
    print("Elapsed s max: ",groupdat['elapsed_s'].max())
    print(groupdat['packet_miss_idx'].max()+1)
    t_cycle = (groupdat['elapsed_s'].max())/((groupdat['packet_miss_idx'].max()+1)/6)
    print(t_cycle)
    tarray=[]
    for number in groupdat['packet_miss_idx']:
        if func == 1:
            burst_time = np.floor((number)/(n_burst+sleep_ping))*avg_time_between
            packet_time = ((number) % (n_burst+sleep_ping))* (avg_burst_time/(n_burst-1))
        elif func == 2: 
            burst_time = np.floor((number)/(n_burst+sleep_ping))*t_cycle
            packet_time = ((number) % (n_burst+sleep_ping))*(1/rate)
        else:
            burst_time = np.floor((number)/(n_burst+sleep_ping))*t_cycle
            packet_time = ((number) % (n_burst+sleep_ping))*(avg_burst_time/(n_burst-1))
        tarray.append(float(burst_time)+packet_time)
    print(f"Function value before tseries creation: {func}")
    tseries = pd.Series(tarray, name=f'timestamps_f{func}')
    print(f"Name of tseries: {tseries.name}")
    groupdat = groupdat.reset_index(drop=True)
    groupdat = pd.concat((groupdat, tseries), axis=1)

    if mean == True:
        for i in range(8):
            channel = f'Channel {i}'
            # Calculate mean and directly store in the Channel columns, overwriting old data
            groupdat[channel] = groupdat.groupby('group')[channel].transform('mean')
        groupdat_mean = groupdat[groupdat['packet_miss_idx'] % (n_burst+sleep_ping) == 0]
        groupdat_mean = groupdat_mean.reset_index(drop=True)
        error_t = (groupdat_mean['elapsed_s'].max()-groupdat_mean[f'timestamps_f{func}'].max())/((groupdat_mean['packet_miss_idx'].max()+1)/6)
        # print(error_t)
        # print("Length of mean groupdat: ",len(groupdat_mean))
        error_series = pd.Series([error_t]*len(groupdat_mean))
        error_series = error_series.cumsum()
        error_series -= error_t
        # print('length error series: ', len(error_series))
        # print("Length of timestamps: ",len(groupdat[f'timestamps_f{func}']))
        # print("Length of mean timestamps: ",len(groupdat_mean[f'timestamps_f{func}']))
        groupdat_mean = groupdat_mean.copy()
        groupdat_mean[f'corrected_t_f{func}'] = groupdat_mean[f'timestamps_f{func}'] + error_series
        groupdat = 0

    else: 
        error_t = (groupdat['elapsed_s'].max()-groupdat[f'timestamps_f{func}'].max())/(groupdat['packet_miss_idx'].max()+1)
        print(error_t)
        print(len(groupdat))
        error_series = pd.Series([error_t]*len(groupdat))
        error_series = error_series.cumsum()
        error_series -= error_t
        groupdat[f'corrected_t_f{func}'] = groupdat[f'timestamps_f{func}'] + error_series
        groupdat_mean = 0
    
    complete_end=time.time()
    print("The total function took: ", complete_end-complete_start, " to run")

    return groupdat, groupdat_mean, avg_time_between,\
            error_series, avg_wake_up_time, avg_sleep_time,\
                    avg_burst_time, burst_times, effective_rate

# %%
groupdat, _, avg_time_between, error_series,\
    avg_wake_up_time, avg_sleep_time, \
        avg_burst_time, burst_times, effective_rate = read_egg_v3_lowerP(file,
                                                                        header=0,
                                                                        rate = 62.5,
                                                                        scale=600,
                                                                        error=0,
                                                                        date=None,
                                                                        n_burst=5,
                                                                        sleep_ping=1,
                                                                        sleep_time=2,
                                                                        t_deviation=0.2,
                                                                        n_missing=1,
                                                                        func=3,
                                                                        mean = False
                                                                        )

# %% Getting data for raw values 
groupdats = {}

for i in range(1, 4):
    result = read_egg_v3_lowerP(file, func=i, mean=False)
    groupdat = result[0]
    groupdats[f'function{i}'] = groupdat


# %% Mean value data

groupdats_mean = {}

for i in range(1, 4):
    result = read_egg_v3_lowerP(file, func=i, mean=True)
    groupdat_mean = result[1]
    groupdats_mean[f'function{i}'] = groupdat_mean

# %%
func = 3

working_mean = groupdats_mean['function3']

food_mean = working_mean[25628:27167]
food_mean_reset = food_mean.copy()
food_mean_reset['timestamps']=food_mean_reset[f'corrected_t_f{func}']-food_mean_reset[f'corrected_t_f{func}'].iloc[0]
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
dat_mean = food_mean_reset[datcols]

num_rows_na = dat_mean.isna().any(axis=1).sum()
perc_na = (num_rows_na/ len(dat_mean)) * 100
print(f"The percentage of rows with an na value is: {perc_na}%")

dat_mean_noint = dat_mean.dropna()


for i in range(8):
    dat_mean[f'Channel {i}'] = dat_mean[f'Channel {i}'].interpolate(method='cubic')
    #dat_mean[f'Channel {i}'] = savgol_filter(dat_mean[f'Channel {i}'], window, polynomial)

#signalplot(dat=dat_mean,xlim=(1200,1380),skip_chan=[0,1,2,4], freq=[0.025,0.5], rate=effective_rate)

egg_signalfreq(dat_mean, rate=effective_rate)
egg_signalfreq(dat_mean_noint, rate=effective_rate)



#%%
func = 3

groupdat_3 = groupdats['function3']

#37230:43668 -- first day around 18 eating


around_eating = groupdat_3[153768:163002]
around_eating_reset_t = around_eating.copy()
around_eating_reset_t['timestamps'] = around_eating_reset_t[f'corrected_t_f{func}'] - around_eating_reset_t[f'corrected_t_f{func}'].iloc[0]
datcols = ['timestamps'] + [f'Channel {i}' for i in range(8)]
dat = around_eating_reset_t[datcols]

num_rows_na = dat.isna().any(axis=1).sum()
perc_na = (num_rows_na/ len(dat)) * 100
print(f"The percentage of rows with an na value is: {perc_na}%")

dat_non = dat.dropna()

window = 50
polynomial = 2

for i in range(8):
    dat[f'Channel {i}'] = dat[f'Channel {i}'].interpolate(method='cubic')
    dat[f'Channel {i}'] = savgol_filter(dat[f'Channel {i}'], window, polynomial)

#signalplot(dat=dat_non,xlim=(0,1000),skip_chan=[1,2,4], freq=[0.02,0.2], rate=effective_rate)
# signalplot(dat=dat,xlim=(0,1000),skip_chan=[1,2,4], freq=[0.02,0.2], rate=effective_rate)
egg_signalfreq(dat, rate=effective_rate*5)

# %%

y=dat['interp_Ch5'] 
z=dat['Channel 5']
plt.plot(y, label ='interpolated Ch5 - raw data')
plt.plot(z, label ='Ch5 - raw data')


q=dat_mean['interp_Ch5']
r=dat_mean['Channel 5']
plt.plot(q, label ='interpolated Ch5 - mean data')
plt.plot(r,label ='Ch5 - mean data')
plt.xlim(500,650)
plt.legend()
plt.show()



# %%
