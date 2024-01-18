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

file = file = r'C:/Users/Coen/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08152023_First_short_pigmeasurement/08152023_PowerCycle.txt'
scale = 600

# %%
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
dat["realtime"] = dat["realtime"].mask(dat["realtime"].diff().dt.total_seconds() < 0,
                                       dat["realtime"] + timedelta(days=1))
dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
dat['elapsed_t'] = dat['realtime'] - \
    dat['realtime'].iloc[0]  # probably delete this

# %%
dat_col = dat.msg
hexdat = dat_col.str.split(' ')  # Return list of splits based on spaces in msg
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

# %%
# Generate the columns of
collist = ['SPI']
for i in range(8):
    collist.append('Channel '+str(i))  # make channel list name
collist.append('CRC')
datalist = pd.DataFrame(serieslist, columns=collist)
# print(datalist)
# print(dat)
fulldat = pd.concat((dat, datalist), axis=1)
print(fulldat)


# %%
# packet counter with actual packet values, seeing if we miss packets
expected_packets = list(
    range(min(fulldat['packet_re_idx']), max(fulldat['packet_re_idx'])+1))
missing_packets = list(set(expected_packets) -
                       set(fulldat['packet_re_idx'].to_list()))
missing_rows = pd.DataFrame(
    [{'misc': 'miss', 'packet_re_idx': re_idx} for re_idx in missing_packets])
fulldat = pd.concat([fulldat, missing_rows], ignore_index=True)
fulldat = fulldat.sort_values(by='packet_re_idx').reset_index(drop=True)

# %%

# Functions


def assign_groups(df, n_burst=5, sleep_time=2, t_deviation=0.2, n_missing=1):
    """
    Standardized grouping of the low power transmission scheme using packet bursts / sleeping cycles

    n_burst: amount of packets transmitted in one cycle
    sleep_time: sleep time in seconds (or milliseconds) between transmission bursts
    time_dev:
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


grouped_fulldat = assign_groups(
    fulldat, n_burst=5, sleep_time=2, t_deviation=0.1, n_missing=1)

# %%


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
        time_deviation: variable to define allowed deviation from sleep_time

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
    wake_up_times = []
    for i in range(len(df_sleep)-1):
        if df_sleep['misc'].iloc[i] == " 0a ":
            next_value = df_sleep['misc'].iloc[i+1]
            time_diff = df_sleep['time_diff'].iloc[i+1]

            # maybe let work in two directions?
            if next_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                wake_up_time = time_diff - sleep_time
                wake_up_times.append(wake_up_time)

    if wake_up_times:
        avg_wake_up_time = sum(wake_up_times)/len(wake_up_times)
    else:
        avg_wake_up_time = 0

    # Calculate the time difference between the different sleep transmissions
    # Dataframe with only sleep packets
    df_sleep = df_sleep[df_sleep['misc'] == " 0a "]
    # Calculate time difference in between sleep packets
    df_sleep['time_diff'] = df_sleep['realtime'].diff().dt.total_seconds()
    df_sleep = df_sleep[df_sleep['time_diff'] >= sleep_time_min]
    df_sleep = df_sleep[df_sleep['time_diff'] <= sleep_time_max]
    avg_sleep_time = df_sleep['time_diff'].mean()
    # Calculate the sum of time differences per group to average per groups, then calculate the time average between these
    # Filter out group = nan rows for burst groups
    df_burst.dropna(subset=['group'], inplace=True)
    # Calculate summed delta t per group
    group_time_diff = df_burst.groupby('group')['time_diff'].sum()
    # Identify valid groups where summed delta t
    valid_groups = group_time_diff[(group_time_diff >= sleep_time_min) & (
        group_time_diff <= sleep_time_max)].index.tolist()
    # valid_group2 = group_time_diff[group_time_diff <=
    #                                sleep_time + t_deviation].index.tolist()
    # valid_groups = list(set(valid_group1 + valid_group2))
    # Filter out invalid groups
    df_burst = df_burst[df_burst['group'].isin(valid_groups)]
    # Assign the  summed delta t per group back to df_burst for potential later use
    df_burst['group_time_diff'] = df_burst['group'].map(group_time_diff)
    # Calculate the average time difference across valid groups
    avg_burst_diff = group_time_diff[valid_groups].mean()

    # Average time between sample bursts and effective sampling rate in SPS
    avg_time_between = np.mean([avg_burst_diff, avg_sleep_time])
    effective_rate = 1 / avg_time_between

    return avg_time_between, effective_rate, avg_wake_up_time


avg_time_between, effective_rate, avg_wake_up_time = avg_time_diffs(
    grouped_fulldat)

# %%
volt_fulldat = grouped_fulldat.copy()
volt_fulldat = volt_fulldat.dropna()
volt_fulldat['avg_elapsed_t'] = volt_fulldat.groupby(
    'group')['elapsed_t'].transform('mean')
volt_fulldat['avg_elapsed_start'] = volt_fulldat['avg_elapsed_t'] - \
    volt_fulldat['avg_elapsed_t'].iloc[0]
volt_fulldat['avg_elapsed_s'] = volt_fulldat['avg_elapsed_start'].dt.total_seconds()
# Storing RAW voltage data just in case
for i in range(8):
    volt_fulldat[f'RAW Channel {i}'] = volt_fulldat[f'Channel {i}']

for i in range(8):
    channel = f'Channel {i}'
    # Calculate mean and directly store in the Channel columns, overwriting old data
    volt_fulldat[channel] = volt_fulldat.groupby(
        'group')[channel].transform('mean')

# for i in range(8):
#     good_fulldat[f'meanmV_Ch{i}'] = good_fulldat.groupby('group')[f'Channel {i}'].transform('mean')

columns_to_keep = ['avg_elapsed_t', 'avg_elapsed_s',
                   'packet_re_idx', 'group'] + [f'Channel {i}' for i in range(8)]

VT_data = volt_fulldat[columns_to_keep].drop_duplicates(
    subset='group').reset_index(drop=True)
# Rebase the packet index based on the first "actual" voltage datapoint
VT_data['packet_re_idx'] = VT_data['packet_re_idx'] - \
    VT_data['packet_re_idx'].iloc[0]
VT_data['timestamps'] = (VT_data['packet_re_idx']/(n_burst+1))*avg_time_between
