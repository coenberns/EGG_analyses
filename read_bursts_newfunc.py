#
# Created on Wed Jan 10 2024
#
# Copyright (c) 2024 Berns&Co
#
#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, time
import pathlib 
from Plot_EGG import*
from scipy.signal import butter, filtfilt, savgol_filter
from filterpy.kalman import KalmanFilter
import pywt
import time
import os

def avg_time_diffs(df, n_burst=5, sleep_time=1.84, t_deviation=0.2):
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
        if n_burst == 1:
            sleep_times = 0
        else:
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
        avg_sleep_time = 0
    
    if wake_up_times:
        avg_wake_up_time = sum(wake_up_times)/len(wake_up_times)
    else:
        avg_wake_up_time = 0

    burst_times = []
    for i in range(len(df_burst)-n_burst-1):
        # and (not np.isnan(df_burst['group'].iloc[i])), below in if statement previously
        if (df_burst['misc'].iloc[i] == " 16 "):
            if n_burst==1:
                burst_times=0
            else:
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
    #avg_t_cycle = np.mean([avg_burst_diff, avg_sleep_time])
    if n_burst==1:
        avg_t_cycle=0
        effective_rate=0
    else:
        avg_t_cycle = avg_burst_time+avg_sleep_time
        effective_rate = 1 / avg_t_cycle

    return avg_t_cycle, effective_rate, avg_wake_up_time, avg_sleep_time, avg_burst_time, burst_times

def packet_counter(dat, n_burst=5, sleep_ping=1, cycle_time = 2, t_deviation = 1, rssi_threshold=-97):
    max_bitval = 65536
    burst_length = n_burst+sleep_ping  # Packet rate

    counter = dat.packet.astype(int)
    new_counter = [0]
    for j in range(1, len(dat)):
        step = counter[j] - counter[j-1]
        time_diff = (dat['corrected_realtime'][j] - dat['corrected_realtime'][j-1]).total_seconds()

        if step > 0:
            new_counter.append(step + new_counter[-1])
        elif dat['rssi'][j] > rssi_threshold:
            corrected_step = (max_bitval - counter[j-1]) + counter[j]
            expected_time_diff = (corrected_step / burst_length) * cycle_time

            if abs(time_diff - expected_time_diff) <= t_deviation:
                new_counter.append(corrected_step + new_counter[-1])
                print('flip', step, corrected_step)
            else:
                new_counter.append(np.nan)
                print(f'Assigning nan at index {j} due to large time deviation')
        else:
            new_counter.append(np.nan)
            print(f'Assigning NaN at index {j} due to low RSSI: {dat["rssi"][j]}')

    return pd.Series(new_counter, name='packet_re_idx')
#%%
#MASTER FUNCTION
def read_egg_v3_bursts(file,
                        header=None,
                        rate = 62.5,
                        scale=600,
                        error=0,
                        date=None,
                        n_burst=5,
                        sleep_ping=1,
                        sleep_time=1.84,
                        t_deviation=0.2,
                        func = 1):
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
            .timestamps: The evenly spaced time variable, based on avg_t_cycle and the packet number divided by 6
        grouped_fulldat: The full dataset, grouped based on the assign_groups() function
        volt_fulldat: The dataset including all voltages, average and raw and all other columns as in grouped_fulldat (sleep and missing are dropped)
        avg_t_cycle: The average time between the sample bursts as calculated by avg_time_diffs() 
        effective_rate: The effective sampling rate based on the avg_t_cycle(*) (=1/*)

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
    dat = pd.read_csv(file, header=header, dtype=str, delimiter='|', names=[
        'realtime', 'misc', 'packet', 'msg', 'rssi'])
    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('16')]
    dat_sleep = dat[dat.misc.str.contains('16') | dat.misc.str.contains('0a')]
    dat.packet = dat.packet.astype(int)
    dat.rssi = dat.rssi.astype(float)
    dat = dat.reset_index(drop=True)
    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    # datetime_counter = time.time()
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
    # datetime_end_counter = time.time()
    # print("Time for datetime conversion :", datetime_end_counter-datetime_counter)
    # hexdat_counter = time.time()
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
    # hexdat_end_counter = time.time()
    # print("Time for hexdat: ", hexdat_end_counter-hexdat_counter)
    collist = ['SPI']
    for i in range(8):
        collist.append('Channel '+str(i))  # make channel list name
    collist.append('CRC')
    datalist = pd.DataFrame(serieslist, columns=collist)
    # print(datalist)
    counter = dat.packet
    new_counter = [0]
    for j, ele in enumerate(counter[1:]):
        step = counter[j+1]-counter[j]
        if step > 0:
            new_counter.append(step+new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print(f'flip at df index {j+1}', step, 65536-counter[j]+counter[j+1])
            print('Original packet number:', counter[j+1])
    abscounterseries = pd.Series(new_counter, name='packet_re_idx')
    dat = pd.concat((dat, abscounterseries), axis=1)

    fulldat = pd.concat((dat, datalist), axis=1)
    # print(fulldat)

    avg_t_cycle, effective_rate, avg_wake_up_time,\
        avg_sleep_time, avg_burst_time, burst_times = avg_time_diffs(df=fulldat,
                                                                    n_burst=n_burst,
                                                                    sleep_time=sleep_time,
                                                                    t_deviation=t_deviation
                                                                    )                                        
    # end_avg_times = time.time()
    # print("Calculating average times: ", end_avg_times-start_avg_times)
    #EXTRA MANIPULATIONS BEFORE AVERAGING OR NOT
    fulldat1 = fulldat[fulldat['misc'] == " 16 "]
    v_fulldat = fulldat1.copy()
    v_fulldat['packet_re_idx'] = v_fulldat['packet_re_idx']-v_fulldat['packet_re_idx'].iloc[0]
    v_fulldat['elapsed_t'] = v_fulldat['elapsed_t']-v_fulldat['elapsed_t'].iloc[0]
    v_fulldat['elapsed_s'] = v_fulldat['elapsed_t'].dt.total_seconds()
    v_fulldat['packet_miss_idx'] = v_fulldat['packet_re_idx']
    #Finding missing packets for nan values
    expected_packets = list(range(min(v_fulldat['packet_miss_idx']), max(v_fulldat['packet_miss_idx'])+1))
    missing_packets = list(set(expected_packets) - set(v_fulldat['packet_miss_idx'].to_list()))
    missing_rows = pd.DataFrame(
        [{'misc': 'miss', 'packet_miss_idx': re_idx} for re_idx in missing_packets])
    v_fulldat = pd.concat([v_fulldat, missing_rows], ignore_index=True)
    v_fulldat = v_fulldat.sort_values(by='packet_miss_idx').reset_index(drop=True)
    # last_packet = (v_fulldat['packet_miss_idx'].max()+1)
    # t_fin = v_fulldat['elapsed_s'].max()
    last_time_index = v_fulldat['corrected_realtime'].last_valid_index()
    if last_time_index is not None: 
        first_realtime = v_fulldat['corrected_realtime'].iloc[0]
        last_realtime = v_fulldat.loc[last_time_index,'corrected_realtime']  # last non-NaN value
        t_fin = (last_realtime - first_realtime).total_seconds()
        last_packet = v_fulldat.loc[last_time_index, 'packet_miss_idx']
    else:
        last_packet = 0
    cycles_def = np.floor(last_packet/(n_burst+sleep_ping))
    rest = last_packet % (n_burst+sleep_ping)
    if (rest==0):  
        t_cycle = t_fin/cycles_def
    else:
        if n_burst==1:
            t_cycle=t_fin/cycles_def + (rest/cycles_def)*(1/rate)
        else:    
            #Just to be concise, it is not gonna differ
            t_cycle = t_fin/cycles_def + (rest/cycles_def)*(avg_burst_time/(n_burst-1))
    #Time propagation functions - choose one of 3
    tarray=[]
    for number in v_fulldat['packet_miss_idx']:
        #This can probably be deleted, depending on differences in data for different datasets (keep in now)
        if func == 1: #BEST FUNCTION
            burst_time = np.floor((number)/(n_burst+sleep_ping))*t_cycle
            if ((number % (n_burst+sleep_ping))< n_burst):
                if (n_burst ==1):            
                    packet_time = ((number) % (n_burst+sleep_ping))*(1/rate)
                else: 
                    packet_time = ((number) % (n_burst+sleep_ping))*(avg_burst_time/(n_burst-1))
            else:
                packet_time = np.nan            
        elif func == 2: 
            burst_time = np.floor((number)/(n_burst+sleep_ping))*t_cycle
            if ((number % (n_burst+sleep_ping))<n_burst):            
                packet_time = ((number) % (n_burst+sleep_ping))*(1/rate)
            else:
                packet_time = np.nan                
        else:
            burst_time = np.floor((number)/(n_burst+sleep_ping))*avg_t_cycle
            if ((number % (n_burst+sleep_ping))<n_burst):
                if (n_burst ==1):            
                    packet_time = ((number) % (n_burst+sleep_ping))*(avg_burst_time/(n_burst))
                else: 
                    packet_time = ((number) % (n_burst+sleep_ping))*(avg_burst_time/(n_burst-1))
            else:
                packet_time = np.nan                        
        tarray.append(float(burst_time)+packet_time)
    # print(f"Function value before tseries creation: {func}")
    tseries = pd.Series(tarray, name=f'timestamps_f{func}')
    # print(f"Name of tseries: {tseries.name}")
    v_fulldat = v_fulldat.reset_index(drop=True)
    v_fulldat = pd.concat((v_fulldat, tseries), axis=1)

    error_t = (t_fin - v_fulldat[f'timestamps_f{func}'][last_time_index]) / last_packet
    error_series = pd.Series([error_t] * (last_time_index + 1))
    error_series = error_series.cumsum()
    error_series -= error_t

    # Assign the error_series to the 'timestamps' column up to the last valid index
    v_fulldat.loc[:last_time_index, 'timestamps'] = v_fulldat.loc[:last_time_index, f'timestamps_f{func}'] + error_series
    v_fulldat.loc[last_time_index + 1:, 'timestamps'] = np.nan

    #Add right timestamps to corrected realtimes first value
    v_fulldat['datetime'] = v_fulldat['corrected_realtime'].iloc[0] + pd.to_timedelta(v_fulldat['timestamps'],unit='s')

    times = {
        't_cycle': t_cycle,
        'avg_t_cycle': avg_t_cycle,
        'effective_rate': 1/t_cycle,
        'avg_burst_time': avg_burst_time,
        'avg_wake_up_time': avg_wake_up_time,
        'avg_sleep_time': avg_sleep_time,
        'burst_times': burst_times
    }

    datcols = ['timestamps', 'elapsed_s', 'packet_re_idx', 'packet_miss_idx'] + [f'Channel {i}' for i in range(8)]
    v_compact = v_fulldat[datcols]

    complete_end=time.time()
    print("The total function took: ", complete_end-complete_start, " to run")

    return v_compact, v_fulldat, times
# %%
