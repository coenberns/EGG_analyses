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
from Plot_EGG_adaptation import*
import timeit
import time
import cProfile
# from Plot_EGG_adaptation import*

# %%
# #Mac
# meas_path = pathlib.Path("/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")

#Windows
meas_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")
# List all the files in the selected folder
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
print(f"You selected: {file.name}")

# %%
# Important external functions

#WORKING ASSIGN GROUP FUNCTION
def assign_groups(df, n_burst=5, sleep_time=2, t_deviation=0.2, n_missing=1):
    # Part from assign_groups
    sleep_time_min = sleep_time - t_deviation
    sleep_time_max = sleep_time + t_deviation

    df['numeric_misc'] = np.select(
        [df['misc'] == " 16 ", df['misc'] == " 0a ", df['misc'] == "miss"],
        [1, 0, 2]
    )
    df['rolling_sum'] = df['numeric_misc'].rolling(n_burst).sum()
    df['next_sleep'] = df['numeric_misc'].shift(-n_burst)
    df['next_burst'] = df['numeric_misc'].shift(-n_burst-1)
    df['next_time_diff'] = df['time_diff'].shift(-n_burst-1)
    
    # Define conditions for grouping
    cond1 = df['numeric_misc'] == 1
    cond2 = df['next_sleep'] == 0
    cond3 = df['next_burst'] == 1
    cond4 = df['rolling_sum'] == n_burst
    cond5 = (df['next_time_diff'] >= sleep_time_min) & (df['next_time_diff'] <= sleep_time_max)
    cond6 = df['rolling_sum'] == n_burst + n_missing

    valid_indices = df[cond1 & cond2 & cond3 & ((cond4 & cond5) | (cond6 & cond5))].index
    groups = [np.nan] * len(df)
    print(valid_indices)
    for idx, start_idx in enumerate(valid_indices):
        end_idx = start_idx + n_burst
        groups[start_idx:end_idx] = [idx+1] * n_burst
    df['group'] = groups
    df.drop(['next_sleep', 'next_burst', 'next_time_diff', 'rolling_sum', 'numeric_misc'], axis=1, inplace=True)
    return df


def process_data(df, n_burst=5, sleep_time=1.84, t_deviation=0.2, n_missing=1):
    # Part from assign_groups
    sleep_time_min = sleep_time - t_deviation
    sleep_time_max = sleep_time + t_deviation

    df['numeric_misc'] = np.select(
        [df['misc'] == " 16 ", df['misc'] == " 0a ", df['misc'] == "miss"],
        [1, 0, 2]
    )
    df['rolling_sum'] = df['numeric_misc'].rolling(n_burst).sum()
    df['next_sleep'] = df['numeric_misc'].shift(-n_burst)
    df['next_burst'] = df['numeric_misc'].shift(-n_burst-1)
    df['next_time_diff'] = df['time_diff'].shift(-n_burst-1)
    
    # Define conditions for grouping
    cond1 = df['numeric_misc'] == 1
    cond2 = df['next_sleep'] == 0
    cond3 = df['next_burst'] == 1
    cond4 = df['rolling_sum'] == n_burst
    cond5 = (df['next_time_diff'] >= sleep_time_min) & (df['next_time_diff'] <= sleep_time_max)
    cond6 = df['rolling_sum'] == n_burst + n_missing

    valid_indices = df[cond1 & cond2 & cond3 & ((cond4 & cond5) | (cond6 & cond5))].index
    groups = [np.nan] * len(df)
    print(valid_indices)
    for idx, start_idx in enumerate(valid_indices):
        end_idx = start_idx + n_burst
        groups[start_idx:end_idx] = [idx+1] * n_burst
    df['group'] = groups

    # Calculating the average sleep time and wake up time
    mask_sleep = df['misc'] == " 0a "
    mask_next_rec = df['misc'].shift(-1) == " 16 "
    mask_time_diff = (df['time_diff'].shift(-1) >= sleep_time_min) & (df['time_diff'].shift(-1) <= sleep_time_max)

    valid_sleep_times = df['time_diff'].shift(-1)[mask_sleep & mask_next_rec & mask_time_diff]
    avg_sleep_time = valid_sleep_times.mean() if not valid_sleep_times.empty else 0
    avg_wake_up_time = avg_sleep_time - sleep_time if avg_sleep_time != 0 else 0

    #FIX THIS
    # Calculation for avg_burst_time
    filtered_bursts = df[(df['numeric_misc'] == 1) & df['group'].notnull()]
    
    #print(burst_mask)
    # grouped_burst_sum = burst_mask.groupby('group').apply(lambda x: x['time_diff'].iloc[1:n_burst].sum())
    # print(grouped_burst_sum)
    # print(set(type(item) for item in burst_times))
    burst_times=0
    avg_burst_time=0


    df_burst = df[df['group'].notnull()]
    group_time_diff = df_burst.groupby('group')['time_diff'].sum()
    valid_groups = group_time_diff[(group_time_diff >= sleep_time_min) & (group_time_diff <= sleep_time_max)].index.tolist()
    avg_burst_diff = group_time_diff[valid_groups].mean()

    avg_time_between = avg_burst_time + avg_sleep_time + avg_wake_up_time
    effective_rate = 1 / avg_time_between

    # Drop intermediate columns
    #'rolling_sum', 'numeric_misc', 'next_time_diff'
    df.drop(['next_sleep', 'next_burst'], axis=1, inplace=True)
    return df, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

# %%
# MASTERFUNCTION for low power reading


def read_egg_v3_lowerP(file, header=0, scale=600, error=0, date=None, n_burst=5, sleep_time=2, t_deviation=0.2, n_missing=1, mean = True):
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
    start_time1 = time.time()
    dat = pd.read_csv(file, header=0, dtype=str, delimiter='|', names=[
        'realtime', 'misc', 'packet', 'msg', 'rssi'])
    end_time1 = time.time()
    print("reading csv file time: ", end_time1-start_time1)

    start_timer_dat = time.time()
    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('16') | dat.misc.str.contains('0a')]
    dat = dat.reset_index(drop=True)
    end_timer_dat = time.time()
    print("Initial data conversion: ", end_timer_dat-start_timer_dat)
    counter_timer = time.time()
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
    counter_end_timer = time.time()
    print("Time for new counter: ", counter_end_timer-counter_timer)

    dat = pd.concat((dat, abscounterseries), axis=1)
    
    
    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    datetime_counter = time.time()
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary
    #############################################################
    # TO DO: CHANGE THIS SO SMALL NEGATIVE TIME WILL NOT BE INCLUDED AS ROLLOVER; fill forward value before that for those values?? OR NAN 
    # dat["realtime"] = dat["realtime"].mask(dat["realtime"].diff(
    # ).dt.total_seconds() < 0, dat["realtime"] + timedelta(days=1))
    # while any(dat["realtime"] < dat["realtime"].shift(1)):
    #     mask = (dat["realtime"] < dat["realtime"].shift(1))
    #     dat.loc[mask, "realtime"] += pd.Timedelta(days=1)
    #FIX THIS!
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    dat['rollover'] = dat['time_diff'] < 0
    dat['glitch'] = (dat['time_diff'] > -5) & (dat['rollover'])
    dat['correct_rollover'] = dat['rollover'] & ~dat['glitch'] 
    dat['days_to_add'] = dat['correct_rollover'].cumsum()
    dat['corrected_realtime'] = dat['realtime'] + pd.to_timedelta(dat['days_to_add'], unit='D')

    # probably delete this if values at end are non-close
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    datetime_end_counter = time.time()
    print(dat['elapsed_t'])
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
    print(dat)
    fulldat = pd.concat((dat, datalist), axis=1)
    # print(fulldat)
    missing_packets_counter = time.time()
    # packet counter with actual packet values, seeing if we miss packets
    expected_packets = list(
        range(min(fulldat['packet_re_idx']), max(fulldat['packet_re_idx'])+1))
    missing_packets = list(set(expected_packets) -
                           set(fulldat['packet_re_idx'].to_list()))
    missing_rows = pd.DataFrame(
        [{'misc': 'miss', 'packet_re_idx': re_idx} for re_idx in missing_packets])
    fulldat = pd.concat([fulldat, missing_rows], ignore_index=True)
    fulldat = fulldat.sort_values(by='packet_re_idx').reset_index(drop=True)
    missing_packet_end_counter = time.time()
    print("missing packets: ", missing_packet_end_counter-missing_packets_counter)
    # Assigning groups to fulldat valid bursts
    start_time2=time.time()
    grouped_fulldat, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times = process_data(df=fulldat)
    # grouped_fulldat = assign_groups(
    #     fulldat, n_burst, sleep_time, t_deviation, n_missing)
    end_time2=time.time()
    func1_time= end_time2-start_time2
    print(func1_time)
    #print(all(grouped_fulldat['group'].isnull()))
    # Calculating the average time between bursts
    # start_time2=time.time()
    # avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times = avg_time_diffs(
    #     grouped_fulldat, n_burst, sleep_time, t_deviation)
    # end_time2=time.time()
    # func2_time= end_time2-start_time2
    # print(func2_time)
    if mean == True:
        # Final processing steps
        volt_fulldat = grouped_fulldat.copy()
        # Don't know how to change since timestamps dependent on re-indexing to first voltage value
        volt_fulldat = volt_fulldat.dropna()
        volt_fulldat['avg_elapsed_t'] = volt_fulldat.groupby(
            'group')['elapsed_t'].transform('mean')
        print(volt_fulldat['avg_elapsed_t'])
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
    #################################
    #else function for non-mean values
    complete_end = time.time()
    print(complete_end-complete_start)

    return fulldat, VT_data, grouped_fulldat, volt_fulldat, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

# %%
fulldat, VT_data, grouped_fulldat, volt_fulldat, avg_burst_diff, avg_time_between, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times = read_egg_v3_lowerP(
    file)
# VT_data still has to be interpolated:



# %%

# import pstats
# import csv

# cProfile.run('read_egg_v3_lowerP(file)', 'profile_output.pstat')

# stats = pstats.Stats('profile_output.pstat')
# stats.strip_dirs()
# stats.sort_stats('cumulative')

# with open('profile_output.csv', 'w', newline='') as csvfile:
#     csvwriter=csv.writer(csvfile)
#     csvwriter.writerow(["Filename", "Line Number", "Function Name", "Call Count", "Cumulative Time", "Internal Time"])

#     for func, (cc, nc, tt, ct, callers) in stats.stats.items():
#         filename, line_num, func_name = func
#         csvwriter.writerow([filename, line_num, func_name, cc, ct, tt])


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
