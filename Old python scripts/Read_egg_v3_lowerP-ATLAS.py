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
from datetime import datetime, timedelta
import os
from Old_Plot_EGG import*
# from Plot_EGG_adaptation import*

file = r"C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2/08282023_1plushours-daytime1.txt"
# %%
# Important external functions


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
            
    # Calculate the time difference between the different sleep transmissions
    # Dataframe with only sleep packets
    df_sleep = df_sleep[df_sleep['misc'] == " 0a "]
    # Calculate time difference in between sleep packets
    # df_sleep['time_diff'] = df_sleep['realtime'].diff().dt.total_seconds()
    # df_sleep = df_sleep[df_sleep['time_diff'] >= sleep_time_min]
    # df_sleep = df_sleep[df_sleep['time_diff'] <= sleep_time_max]
    # avg_sleep_time = df_sleep['time_diff'].mean()
    # Calculate the sum of time differences per group to average per groups, then calculate the time average between these
    # Filter out group = nan rows for burst groups
    df_burst.dropna(subset=['group'], inplace=True)
    # Calculate summed delta t per group
    group_time_diff = df_burst.groupby('group')['time_diff'].sum()
    # Identify valid groups where summed delta t
    valid_groups = group_time_diff[(group_time_diff >= sleep_time_min) & (
        group_time_diff <= sleep_time_max)].index.tolist()
    # Filter out invalid groups
    df_burst = df_burst[df_burst['group'].isin(valid_groups)]
    # Assign the  summed delta t per group back to df_burst for potential later use
    df_burst['group_time_diff'] = df_burst['group'].map(group_time_diff)
    # Calculate the average time difference across valid groups
    avg_burst_diff = group_time_diff[valid_groups].mean()

    # Average time between sample bursts and effective sampling rate in SPS
    avg_time_between = np.mean([avg_burst_diff, avg_sleep_time])
    effective_rate = 1 / avg_time_between

    return  avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

# %%
# MASTERFUNCTION for low power reading


def read_egg_v3_lowerP(file, header=0, scale=600, error=0, date=None, n_burst=5, sleep_time=1.84, t_deviation=0.2, n_missing=1):
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
    # Putting in the date information to create complete datetime object, if information available
    if date is None:
        # get only the filename from the file path
        _, filename = os.path.split(file)
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
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary
    dat["realtime"] = dat["realtime"].mask(dat["realtime"].diff(
    ).dt.total_seconds() < 0, dat["realtime"] + timedelta(days=1))
    while any(dat["realtime"] < dat["realtime"].shift(1)):
        mask = (dat["realtime"] < dat["realtime"].shift(1))
        dat.loc[mask, "realtime"] += pd.Timedelta(days=1)
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    # probably delete this if values at end are non-close
    dat['elapsed_t'] = dat['realtime'] - dat['realtime'].iloc[0]

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
    collist = ['SPI']
    for i in range(8):
        collist.append('Channel '+str(i))  # make channel list name
    collist.append('CRC')
    datalist = pd.DataFrame(serieslist, columns=collist)
    # print(datalist)
    # print(dat)
    fulldat = pd.concat((dat, datalist), axis=1)
    # print(fulldat)
    # packet counter with actual packet values, seeing if we miss packets
    expected_packets = list(
        range(min(fulldat['packet_re_idx']), max(fulldat['packet_re_idx'])+1))
    missing_packets = list(set(expected_packets) -
                           set(fulldat['packet_re_idx'].to_list()))
    missing_rows = pd.DataFrame(
        [{'misc': 'miss', 'packet_re_idx': re_idx} for re_idx in missing_packets])
    fulldat = pd.concat([fulldat, missing_rows], ignore_index=True)
    fulldat = fulldat.sort_values(by='packet_re_idx').reset_index(drop=True)
    # Assigning groups to fulldat valid bursts
    grouped_fulldat = assign_groups(
        fulldat, n_burst, sleep_time, t_deviation, n_missing)
    # Calculating the average time between bursts
    avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times = avg_time_diffs(
        grouped_fulldat, n_burst, sleep_time, t_deviation)
    # Final processing steps
    volt_fulldat = grouped_fulldat.copy()
    # Don't know how to change since timestamps dependent on re-indexing to first voltage value
    volt_fulldat = volt_fulldat.dropna()
    volt_fulldat['avg_elapsed_t'] = volt_fulldat.groupby(
        'group')['elapsed_t'].transform('mean')
    volt_fulldat['avg_elapsed_start'] = volt_fulldat['avg_elapsed_t'] - \
        volt_fulldat['avg_elapsed_t'].iloc[0]
    volt_fulldat['avg_elapsed_s'] = volt_fulldat['avg_elapsed_start'].dt.total_seconds()
    # Storing RAW voltage data just in case
    for i in range(8):
        volt_fulldat[f'RAW Channel {i}'] = volt_fulldat[f'Channel {i}']
    # Mean voltage calculations
    for i in range(8):
        channel = f'Channel {i}'
        # Calculate mean and directly store in the Channel columns, overwriting old data
        volt_fulldat[channel] = volt_fulldat.groupby(
            'group')[channel].transform('mean')
    # calculation of timestamps based on function, would like to put here
    volt_fulldat['packet_re_idx'] = volt_fulldat['packet_re_idx'] - \
        volt_fulldat['packet_re_idx'].iloc[0]
    volt_fulldat['timestamps'] = volt_fulldat['packet_re_idx'] / \
        ((n_burst+1))*avg_time_between
    datcols = ['avg_elapsed_s', 'timestamps',
               'packet_re_idx', 'group'] + [f'Channel {i}' for i in range(8)]
    # Calculating Voltage time data
    VT_data = volt_fulldat[datcols].drop_duplicates(
        subset='group').reset_index(drop=True)

    return VT_data, grouped_fulldat, volt_fulldat, avg_time_between, effective_rate, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

# %%
VT_data, grouped_fulldat, volt_fulldat, avg_time_between, effective_rate, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times = read_egg_v3_lowerP(
    file)
# VT_data still has to be interpolated:



# %%
# for i in range(8):
#     x = VT_data['timestamps']
#     y = VT_data[f'Channel {i}']
#     plt.xlim(2350, 2450)
#     plt.ylim(-50, 50)
#     # You can also add the label here for clarity
#     plt.plot(x, y, label=f'evenly spaced time')

#     q = VT_data['avg_elapsed_s']
#     r = VT_data[f'Channel {i}']
#     # Dotted line for RAW data
#     plt.plot(q, r, linestyle='--', label=f'averages of groups')

#     plt.legend()
#     plt.title(
#         f'Voltage plotted for Channel {i} based on evenly spaced timepoints and averages of groups')
#     plt.show()

# %%
# Plot directly using signalplot
signalplot(dat=VT_data, xlim=(1500, 1700), freq=[0.025, 0.5])



# %%