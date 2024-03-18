
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
from datetime import datetime, time
import pathlib 
from Old_Plot_EGG import*
from scipy.signal import butter, filtfilt, savgol_filter
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from scipy.interpolate import CubicSpline, interp1d, PchipInterpolator
from filterpy.kalman import KalmanFilter
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import pywt
import time
import os

#%% ALL FUNCTIONS CONSIDERING READ-IN OF MODE 2/3 OR READ-IN FOR ALIGNING MODE 1 (OLD 62.5 Hz)

# ____ START OF MASTER FUNCTION WITH HELPER FUNCTIONS ____ 
# Assigning groups to also get a feel for MCU wake-up and going to sleep times 
# + integrating right "burst_time" value in read_egg_v3_bursts
def assign_groups(df, n_burst=5, sleep_time=1.84, t_deviation=0.2, n_missing=1):
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
    group_start = time.time()
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

                if all(val == " 16 " for val in next_n_values) and sleep_value == "miss" and new_burst_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                    group += 1
                    groups[i:i+n_burst] = [group] * n_burst

                elif sum(val == "miss" for val in next_n_values) <= n_missing and sleep_value == "miss" and new_burst_value == " 16 " and (sleep_time_min <= time_diff <= sleep_time_max):
                    group += 1
                    groups[i:i+n_burst] = [group] * n_burst

            # Check for potential last group in data
            elif i == len(df) - n_burst+1 and all(val == " 16 " for val in next_n_values):
                group += 1
                groups[i:i+(n_burst-1)] = [group] * n_burst

    df['group'] = groups
    group_end = time.time()
    print("The total time for grouping was: ", (group_end-group_start))
    return df

#Calculating times based on average of the groups provided above - may be able to delete due to burst_group addition later
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

# MASTER FUNCTION FOR READ-IN AND AVERAGING OF THE DATA RECORDED IN MODES 2-3 
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
        VT_data: Voltage time data, 

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
    dat = dat[dat.misc.str.contains('16') | dat.misc.str.contains('0a')]
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
    v_fulldat['burst_group'] = v_fulldat['packet_miss_idx'] // (n_burst + sleep_ping)
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

    #Time correction part
    error_t = (t_fin-v_fulldat[f'timestamps_f{func}'][last_time_index])/(last_packet)
    error_series = pd.Series([error_t]*len(v_fulldat))
    error_series = error_series.cumsum()
    error_series -= error_t
    v_fulldat['timestamps'] = v_fulldat[f'timestamps_f{func}'] + error_series

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

    v_mean = v_fulldat.copy()
    channels = [f'Channel {i}' for i in range(8)]

    # Averaging by burst_group
    for channel in channels:
        v_mean[channel] = v_mean.groupby('burst_group')[channel].transform('mean')

    # Replicating the first 'elapsed_s' and 'corrected_realtime' across the group
    for col in ['elapsed_s', 'corrected_realtime']:
        v_mean[col] = v_mean.groupby('burst_group')[col].transform('first')

    # Filtering for the first packet of each burst
    v_mean = v_mean[v_mean['packet_miss_idx'] % (n_burst+sleep_ping) == 0]
    complete_end=time.time()
    
    print("The total function took: ", complete_end-complete_start, " to run")
    return v_mean, v_fulldat, times

    # datcols = ['timestamps', 'elapsed_s', 'packet_re_idx', 'packet_miss_idx'] + [f'Channel {i}' for i in range(8)]
    # v_compact = v_fulldat[datcols]

# ________ END OF MASTER FUNCTION AND HELPER FUNCTIONS _______

# Calculate total time from file
def calculate_time(file, date=None):

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
    dat = dat[dat.misc.str.contains('0a')]
    dat = dat.reset_index(drop=True)

    counter = dat.packet.astype(int)
    new_counter = [0]
    for j, ele in enumerate(counter[1:]):
        step = counter[j+1]-counter[j]
        if step > 0:
            new_counter.append(step+new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            #print('flip', step, 65536-counter[j]+counter[j+1])
    abscounterseries = pd.Series(new_counter, name='packet_re_idx')

    dat = pd.concat((dat, abscounterseries), axis=1)
    
    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary, with additional glitch values excluded
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    dat['rollover'] = dat['time_diff'] < 0
    dat['glitch'] = (dat['time_diff'] > -5) & (dat['rollover'])
    dat['correct_rollover'] = dat['rollover'] & ~dat['glitch'] 
    dat['days_to_add'] = dat['correct_rollover'].cumsum()
    dat['corrected_realtime'] = dat['realtime'] + pd.to_timedelta(dat['days_to_add'], unit='D')
    # dat['corrected_realtime'].interpolate(method='linear', inplace=True)

    # probably delete this if timestamps values at end are close to elapsed_s
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    dat['elapsed_s'] = dat['elapsed_t'].dt.total_seconds()
    t_elapsed = dat['elapsed_t'].max()
    s_elapsed = dat['elapsed_s'].max()
    print("The total time elapsed was: ", t_elapsed)

    return dat, t_elapsed, s_elapsed

# Reading egg_v3 - with additional part for introducing pseudotime for alignment with downsampled version
def read_egg_v3_sync(file,header=0,rate=62.5,scale=150,error=0, date=None):
    """
    This is a function which uses pandas to read in data recorded from EGG V3 and transmitted to a board using
    RFStudio7. It adds pseudotime, to make sure we can use datetime obj. for down-sampling (resample)
    
    file : filepath of the target txt file
    header : Number of lines to skip
    rate : Sampling rate in samples/second per channel set on the ADS131m8
    scale : +- scale in mV 
    error : returns data with CRC errors. Default is 0 so those are stripped
    
    output: Pandas data frame with the following information:
        .pseudotime: the datetime object as index for downsampling
        .realtime : realtime from RFStudio when packet was received
        .misc : RF Studio output, not useful
        .packet : packet number, set from EGGv3, ranges from 0 to 65535 (unit16). Roll over if higher
        .msg : str of packet recieved
        .rssi : RSSI of packet, also includes CRC error
        'Channel n': Channels of recording data in mV, n is from 0 to 7
        .counter : absolute renumbered packets (without overflow)
        .timestamps : timesamples calculated from sampling rate and absolute timer
        .SPI : SPI Status (first packet of msg)
    
    """
    if date is None:
        # get only the filename from the file path
        file_path = pathlib.Path(file)
        filename = file_path.name
        # extract date from the filename
        date = filename.split('_')[0]

    # Creating datetime object
    base_date = datetime.strptime(date, '%Y.%m.%d')
    print(base_date)
    dat=pd.read_csv(file, header=header, dtype = str, delimiter='|', names=['realtime','misc','packet','msg','rssi'])
    dat=dat[~dat.rssi.str.contains('error')]
    dat=dat[dat.misc.str.contains('16')]
    dat=dat.reset_index(drop=True)
    dat_col=dat.msg
    hexdat=dat_col.str.split(' ') #Return list of splits based on spaces in msg
    serieslist=[]
    for k,ele in enumerate(hexdat):
        if len(ele) == 23: #Only select those that have the correct length
            vlist=[]
            for i in range(0,10):
                n=i*2+2
                value= ''.join(['0x',ele[n],ele[n-1]])
                hvalue=int(value,16)
                if i==0:
                    vlist.append(hvalue) #append hex code
                else:    
                    if hvalue<2**15:
                        vlist.append(scale*float(hvalue)/(2**15))
                    else:
                        vlist.append(scale*(((float(hvalue)-2**16)/(2**15))))
        else:
#            print('Line Error!'+str(k))
#            print(ele)
            vlist=[] #add empty list on error
        serieslist.append(vlist)
    collist=['SPI']
    for i in range(8): collist.append('Channel '+str(i)) #make channel list name
    collist.append('CRC')
    datalist=pd.DataFrame(serieslist,columns=collist)
    fulldat=pd.concat((dat,datalist),axis=1)
    counter=fulldat.packet.astype(int)
    new_counter=[0]
    for j,ele in enumerate(counter[1:]): #Renumbered counter - note this will give an error if you accidentally miss the 0/65535 packets
        step=counter[j+1]-counter[j]
#       if step != -65535:
        if step > 0:
            new_counter.append(step+new_counter[j])
#       elif step < 0:
#            new_counter.append(new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
#            new_counter.append(1+new_counter[j])
    tarray=np.array(new_counter)*1/rate
    abscounterseries=pd.Series(new_counter,name='counter')
    tseries=pd.Series(tarray,name='timestamps')
    fulldat=pd.concat((fulldat,abscounterseries,tseries),axis=1)
    
    fulldat['timedelta'] = pd.to_timedelta(fulldat['timestamps'], unit='s')
    fulldat['realtime'] = fulldat['realtime'].str.strip()
    fulldat['realtime'] = pd.to_timedelta(fulldat['realtime'])
    base_time = base_date + fulldat['realtime'].iloc[0]
    fulldat['realtime'] = pd.to_timedelta(fulldat['timestamps'], unit='s') + base_time
    noerror=~fulldat.rssi.str.contains('error') # Gives rows without crc error
    if error: 
        return fulldat # return non-crc error 
    else:
        return fulldat[noerror]
#    hexdat.dropna() #drop out of range NaNs without shifting indicies
#%% READ-IN FUNCTION FOR BATTERY AND TEMPERATURE MEASUREMENTS
def plot_battery_temp(file, plot = False, bat = True, temp = True):
    _, filename = os.path.split(file)
    # extract date from the filename
    date = filename.split('_')[0]
    # Creating datetime object
    # which takes in "MMDDYYYY" like only US people write date order
    date = datetime.strptime(date, '%m%d%Y')

    dat = pd.read_csv(file, header=0, dtype=str, delimiter='|', names=[
                    'realtime', 'misc', 'packet', 'msg', 'rssi'])
    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('06')]
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
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    dat['time'] = dat['elapsed_t'].dt.total_seconds()

    # Split the 'msg' data and calculate battery and temperature
    dat['new_msg'] = dat['msg'].str.strip().str.split(' ')
    bat_temp_data = dat['new_msg'].apply(lambda x: pd.Series({
        'V_bat': int(''.join(x[0:2]), 16),  # most significant byte is first
        'Temp (C)': int(''.join(x[2:4]), 16)
    }))
    # Merge the new data into the original DataFrame
    dat = pd.concat([dat, bat_temp_data], axis=1)

    # Your data is now in 'dat' DataFrame and you can proceed with your analysis
    datcols = ['time', 'V_bat', 'Temp (C)']
    battempdat = dat[datcols]

    if plot == True: 
        # print(battempdat.head())
        if bat == True: 
            x=battempdat['time']
            y= battempdat['V_bat']
            plt.plot(x,y, label = "Mode 1 recording")
            plt.title('Battery voltage vs time')
            plt.xlabel('Time [s]')
            plt.ylabel('Battery voltage [mV]')
            plt.legend()
            plt.show()

        else: 
            print("No battery plot wanted?")

        if temp == True: 
            x=battempdat['time']
            z=battempdat['Temp (C)']
            plt.plot(x,z)
            plt.xlabel('Time [s]')
            plt.ylabel('Temperature [C]')
            plt.show()

        else: 
            print("No temperature plot wanted?")

    return battempdat

#%% SEGMENTATION FUNCTIONS
def rescale_time(segment):
    offset = segment['timestamps'].iloc[0]
    segment['timestamps'] = segment['timestamps'] - offset
    return segment

def filter_segment(segment, window, min_frac, window_frac):
    start = 0
    end = len(segment)
    threshold = window * min_frac
    total_windows = (end + window - 1) // window
    max_invalid_windows = int(total_windows * window_frac)
    invalid_windows_count = 0

    # Check all windows in the segment
    for start in range(0, end, window):
        end_window = min(start + window, end)
        window_segment = segment.iloc[start:end_window]
        non_nan_count = window_segment['Channel 0'].notna().sum()

        window_threshold = threshold if end_window - start == window else ((end_window - start) * min_frac)

        # Count invalid windows
        if non_nan_count < window_threshold:
            #Check for last ending windows - something weird? Not really so far
            # if len(window_segment) != 100:
                # print("Window size =", len(window_segment), "non_nan =", non_nan_count)
            #If want to know the other percentages of non-nan values
            # print("Percentage of non-nan =", ((non_nan_count/len(window_segment))*100), "%")
            invalid_windows_count += 1
            if invalid_windows_count > max_invalid_windows:
                return None  # Discard segment if too many windows are invalid

    # Discard nan values at begin and end
    segment = segment.iloc[0:end]
    while len(segment) > 0 and pd.isna(segment.iloc[-1]['Channel 0']):
        segment = segment.iloc[:-1]
    while len(segment) > 0 and pd.isna(segment.iloc[0]['Channel 0']):
        segment = segment.iloc[1:]

    return segment

def segment_data(df, gap_size=14, seg_length=1500, window=100, min_frac=0.8, window_frac=0.2, rescale=False):
    segments = {}
    start_index=0 
    segment_id=0
    nan_count = 0
    time_interval = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]

    for i in range(start_index, len(df)):
        # Nan counting
        if pd.isna(df.iloc[i, df.columns.get_loc("Channel 0")]):
            nan_count += 1
        else:
            #If there is a gap, check that all the conditions are satisfied
            if nan_count > 0:
                time_gap = nan_count * time_interval
                if time_gap > gap_size:
                    segment = df.iloc[start_index:i - nan_count]
                    filtered_segment = filter_segment(segment, window, min_frac, window_frac)
                    if filtered_segment is not None and len(filtered_segment) >= seg_length:
                        if rescale == True:
                            filtered_segment = rescale_time(filtered_segment)
                        segments[segment_id] = filtered_segment
                        segment_id += 1
                    else:
                        # If segment is too long and invalid, split it and process each half again
                        if len(segment) >= 2 * seg_length:
                            middle = (start_index + i - nan_count) // 2
                            segments.update(segment_data(df, gap_size, seg_length, window, min_frac, start_index, segment_id))
                            segments.update(segment_data(df, gap_size, seg_length, window, min_frac, middle, segment_id))
                    start_index = i
                nan_count = 0

    # Last segment
    segment = df.iloc[start_index:]
    filtered_segment = filter_segment(segment, window, min_frac, window_frac)
    if filtered_segment is not None and len(filtered_segment) >= seg_length:
        if rescale == True: 
            filtered_segment = rescale_time(filtered_segment)
        segments[segment_id] = filtered_segment
    print("Amount of segments =", segment_id)
    return segments

def print_segment_info(segmented_data):
    for segment_number, segment in segmented_data.items():
        print(f"Segment {segment_number+1} length:", len(segment))
        
        if not segment.empty:
            first_timepoint = segment['corrected_realtime'].first_valid_index()
            if first_timepoint is not None:
                print(f"First time for Segment {segment_number+1}:", segment['corrected_realtime'].loc[first_timepoint])
            else:
                print(f"Segment {segment_number} has no valid time values at the start.")

            # Find the last non-nan 'corrected_realtime'
            last_timepoint = segment['corrected_realtime'].last_valid_index()
            if last_timepoint is not None:
                print(f"Last time for Segment {segment_number+1}:", segment['corrected_realtime'].loc[last_timepoint])
                print(f"Total duration for Segment {segment_number+1}:", segment['corrected_realtime'].loc[last_timepoint] - segment['corrected_realtime'].loc[first_timepoint])
            else:
                print(f"Segment {segment_number} has no valid time values at the end.")
        else:
            print(f"Segment {segment_number} is empty.")

#%% INTERPOLATION FUNCTIONS
# Interpolation function where only gaps smaller than max_gap are CubicSpline() interpolated
def interpolate_data(df, cycle_time=2, max_gap=14, rescale=False, time_ip=False, pchip=True):
    df2 = df.copy()
    
    if rescale:
        df2['timestamps'] -= df2['timestamps'].iloc[0]
    
    if time_ip:
        df2['timestamps'] = df2['timestamps'].interpolate(method='linear')

    # Preprocess 'Channel 0' to identify gaps applicable across all channels
    df2['Channel 0'] = pd.to_numeric(df2['Channel 0'], errors='coerce')
    consec_nan_counts = df2['Channel 0'].isna().astype(int).groupby(df2['Channel 0'].notna().astype(int).cumsum()).sum()
    gap_indices = consec_nan_counts[consec_nan_counts > 0].index
    # Translate gap indices to start and end positions
    gap_starts_ends = []
    for idx in gap_indices:
        gap_mask = df2['Channel 0'].notna().astype(int).cumsum() == idx
        start, end = gap_mask.idxmax(), gap_mask[::-1].idxmax()
        gap_starts_ends.append((start, end))
    
    max_nan_count = int(max_gap / cycle_time)

    for channel in df2.columns:
        if 'Channel' in channel:
            x = df2['timestamps'].values
            y = pd.to_numeric(df2[channel], errors='coerce').values
            non_nan_mask = ~np.isnan(y)
            
            for start, end in gap_starts_ends:
                # Calculate gap_size based on timestamps to handle non-integer cycle times
                gap_size = end - start
                if gap_size <= max_nan_count:

                    interp_func = CubicSpline(x[non_nan_mask], y[non_nan_mask], extrapolate=True)
                
                else:
                    if pchip: 
                        interp_func = PchipInterpolator(x[non_nan_mask], y[non_nan_mask], extrapolate=True)
                    else:
                        interp_func = interp1d(x[non_nan_mask], y[non_nan_mask], kind='linear', fill_value="extrapolate", bounds_error=False)
                
                df2.loc[start:end, channel] = interp_func(x[start:end+1])
    return df2

# Quick and dirty interpolation method
def interpolate_egg_v3(df, method = 'cubicspline', order=3, rescale=False, time=False):
    df2 = df.copy()
    # Ensure columns are of a numeric type
    # for col in df2.columns:
    #     if 'Channel' in col:  # Assuming channel data columns have 'Channel' in their names
    #         df2[col] = pd.to_numeric(df2[col], errors='coerce')
    while pd.isna(df2.iloc[0]['Channel 0']):
        df2 = df2.iloc[1:].reset_index(drop=True)

    # Reset timestamps to start from zero
    if rescale == True: 
        df2['timestamps'] -= df2['timestamps'].iloc[0]
    if time == True: 
        df2['timestamps'] = df['timestamps'].interpolate('linear') # only used when downsampling to fill time gaps
    if  (method == 'polynomial') or (method == 'spline'):
        # Interpolate each channel, using order
        for i in range(8):
            channel = f'Channel {i}'
            df2[channel] = df2[channel].interpolate(method=method, order=order)

    else:
        # Interpolate each channel
        for i in range(8):
            channel = f'Channel {i}'
            df2[channel] = df2[channel].interpolate(method=method)
    return df2

#%% FILTER FUNCTIONS 
#normal butterworth filter, with coefficients a,b
def butter_filter(df, fs, freq = [0.02,0.2], order=3):
    df_filtered = df.copy()
    nyquist = 0.5 * fs
    low = freq[0] / nyquist
    freq_adj = min(freq[1],nyquist*.99)
    high = freq_adj / nyquist
    b, a = butter(order, [low, high], btype='band')

    for column in df.columns:
        if column.startswith('Channel'):
            df_filtered[column] = filtfilt(b, a, df[column].values)
    return df_filtered

#sos output butter filter, using sos output with digital filter consistency
def butter_filter_sos(df, fs, freq=[0.02,0.2], order=3):
    df_filtered = df.copy()
    nyquist = 0.5 * fs
    low = freq[0] / nyquist
    freq_adj = min(freq[1],nyquist*.99)
    high = freq_adj / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')

    for column in df.columns:
        if column.startswith('Channel'):
            df_filtered[column] = sig.sosfiltfilt(sos, df[column].values)
    return df_filtered

#Savitzky-Golay filtering
def savgol_filt(df, window=3, polyorder=1, deriv=0, delta=1.0):
    df_filtered = df.copy()
    
    # window has to be odd and larger than polyorder
    if window % 2 == 0:
        window += 1
    
    for column in df.columns:
        if column.startswith('Channel'):
            # apply SG filter per channel
            df_filtered[column] = savgol_filter(df[column].values, window, polyorder, deriv=deriv, delta=delta)

    return df_filtered

def kalman_filter(df):
    df2=df.copy()
    for i in range(8):
        channel = f'Channel {i}'
        #Kalman filter and then interpolating
        # all_values = df2[channel].tolist()
        measurements = df2[channel].dropna().tolist()        
        filtered_values = []
        #dt = times['t_cycle']
        kf = KalmanFilter(dim_x=1, dim_z=1)
        kf.x = [df2[channel].iloc[0]]  # initial state
        kf.F = np.array([[1]])  # state transition matrix, 1x1 matrix since we have only one state
        kf.H = np.array([[1]])  # measurement function, 1x1 matrix.
        kf.P *= 1000.  # covariance matrix
        kf.R = 10  # state uncertainty, adjust based on sensor accuracy
        kf.Q = 1  # process uncertainty, adjust based on how predictable the voltage evolution is

        for z in measurements:
            kf.predict()
            kf.update([z])
            filtered_values.append(kf.x[0])
        
                # if nan_vals == True: 
        #     for z in all_values:
        #         kf.predict()
        #         if not np.isnan(z):  # If the measurement is not NaN, update the filter
        #             kf.update([z])
        #         filtered_values.append(kf.x[0])
            
        #     df2[channel] = filtered_values

        # replace the original values with the filtered ones for non-NaN values
        valid_indices = df2[channel].dropna().index
        df2.loc[valid_indices, channel] = filtered_values

    return df2

# Old egg filter, adapted
def egg_filter(dat,rate=62.5,freq=[0,0.2],order=3,ncomb=0,debug=0):
    """
    Function which filters data using a butterworth filter
    Parameters
    ----------
    dat : List of 2 np arrays
        List of 2 np arrays where first array are timestamps and 2nd array is values
    rate : sampling rate in seconds, optional
        Sampling rate in seconds, used for interpolation of data prior to filtering. The default is 32.
    freq : List, optional
        Bandpass filter frequency. The default is [0,0.1].
    order : int, optional
        Order of butterworth filter generated for filtering. The default is 3.
    ncomb : float
        frequency in hrz of notch comb filter
    Returns
    -------
    fdata: numpy array of 2xN.
        1st index is columns, 2nd is rows. 1st column are timestamps and 2nd column is filtered data.

    """
    fn=rate/2
    wn=np.array(freq)/fn
#    wn[0]=np.max([0,wn[0]])
    wn[1]=np.min([.99,wn[1]])
#    print(wn)
    f=interp1d(dat[0,:],dat[1,:])
#    print(f)
    start_value=dat[0,:].min()
    end_value=dat[0,:].max()
    tfixed=np.arange(start_value,end_value, 1/rate)
    sos=sig.butter(order,wn,btype='bandpass',output='sos')
    filtered=sig.sosfiltfilt(sos,f(tfixed))
    if ncomb!=0:
        if not isinstance(ncomb, list):
            ncomb=[ncomb]
        for ele in ncomb:
            c,d=sig.iircomb(ele/rate, 3)
            filtered=sig.filtfilt(c,d,filtered)
    
    fdata=np.array([tfixed,filtered])
    return fdata

#%% SMALL SIGNAL ANALYSES FUNCTIONS
# Wavelet analysis for signal, however, not properly functioning atm
def perform_wavelet_analysis(signal, wavelet_name='db4'):
    # Perform Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet_name)
    
    # Plot the scalogram
    plt.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='PRGn', aspect='auto',
               vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())
    plt.title("Wavelet Transform (Scalogram)")
    plt.xlabel("Time")
    plt.ylabel("Frequency scale")
    plt.show()
    
    return coefficients, frequencies

#Check slow wave frequency in a signal for a particular channel and plot
def egg_signal_check_slowwave(data,rate=62.5, xpoint=1000, slow_window=200,chan_select=0, close=True, s_freq=[0.02,0.25],figsize=(10,15),s_flim=[1,10],rncomb=0):
    
    a0,b0,c0=signalplot(data,rate=rate,xlim=[xpoint,xpoint+slow_window],freq=[0.001,1000])

    a1,b1,c1=signalplot(data,rate=rate,xlim=[xpoint,xpoint+slow_window],freq=s_freq)
    aa1,bb1,cc1=egg_signalfreq(c1,rate=rate,freqlim=s_flim,mode='power',clip=True)
    maxloc=cc1[chan_select+1,:].argmax()
    s_peakfreq=cc1[0,maxloc]
    print('Peak Slow Wave Frequency is ', s_peakfreq)
    
    cc1=cc1.T

    if close:
        plt.close(a0)
        plt.close(a1)
    fig,ax_n=plt.subplots(nrows=2,figsize=figsize)
    ax_n[0].plot(c0[:,0],c0[:,chan_select+1])
    ax_n[0].set_ylabel('Raw Data (mV)')
    
    ax_n[1].plot(c1[:,0],c1[:,chan_select+1])
    ax_n[1].set_ylabel('Slow Wave (mV)')

    return fig,ax_n  
#%% ADAPTED OLD FUNCTIONS FROM PLOT_EGG
#Plotting signal with hour scale
def signalplot_hrs(dat,xlim=(0,0,0),spacer=0,vline=[],line_params= ['black', 5, 'dashed'],
                   freq=1,order=3,rate=62.5, title='',skip_chan=[],figsize=(10,20),textsize=16,
                   hline=[],ncomb=0,hide_y=False,points=False,time='timestamps',output='np',
                   Normalize_channels=False,labels=[],color_dict={},name_dict={}):
    """
    Function to plot all channels in dataframe following data import using read_egg_v3

    Inputs:
        sig: Dataframe containing time data in "timestamps" column and "Channel n" where n is channel number
        xlim: list of 2 elements taking time range of interest. Default behavior is to full timescale
        spacer: Spacing between plots, and their scaling. Default behavior is spaced on max y value in a channel
        freq: frequency list in Hz, 2 element list, for bandpass filtering. Default is no filtering
        order: order of butter filter used in filtering
        rate: sampling rate of data, for filtering
        title: title label of plot, default is none 
        vline: list of float marking lines for xvalues, usually for fast visulation/measurement
        skip_chan: list of channels to skip, default none. 
        figsize: tuple of figure size dimensions passed to matplotlib.figure, default 10,20
        ncomb: comb frequency hz, passed to egg filter
        
    Outputs:
        fig_an: figure instance of plot (from matplotlib)
        ax_an: axis instance of plot
        Outarray.T: exported filtered data
    Changelog:
        2023.09.29 - Added color-dict functionality
        2023.08.18 - Added rescaling functionality
    """
    x=dat[time].to_numpy()/3600
    rate=rate*3600
    hrs_freq=[i*3600 for i in freq]

    outarray=[]
    if len(skip_chan)>0:
        skip_chan_type=list(map(type,skip_chan))
#        skip_chan_len=list(map(len,skip_chan))
        if int in skip_chan_type or (len(skip_chan[0])<5): #This line preserves easy behavior of calling 1,2,3,a,b etc for skip_chan when not using column names
            skip_chan=list(map(lambda x: "Channel "+str(x),skip_chan))
    if len(labels) == 0: labels=dat.columns
    if freq==1: outarray.append(x)
    plt.rcParams['font.size']=textsize
    fig_an, ax_an = plt.subplots(figsize=figsize) 
    # we make only 1 axis instance and we will manually displace the plots below
    if len(xlim)==2:
        ax_an.set_xlim(xlim[0],np.min([xlim[1],x.max()]))
    else:
        ax_an.set_xlim([x.min(),x.max()])
        xlim=[x.min(),x.max()]
    xloc=ax_an.get_xlim()[0]
    ax_an.spines['right'].set_visible(False)
    ax_an.spines['top'].set_visible(False)
    ax_an.spines['left'].set_visible(False)
    ax_an.xaxis.set_ticks_position('none')
    ax_an.xaxis.set_ticks_position('bottom')
    ax_an.set_yticks([])
    ax_an.set_xlabel('Time (hrs)')
    xsize=ax_an.get_xlim()[1]-ax_an.get_xlim()[0]   

    loc=np.logical_and(x>xlim[0],x<xlim[1])
    space=0
    if spacer == 0: #this is to automatically set the spacing we want between the 
        distarr=[]
        for i,column in enumerate(labels):
#            if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
            if column.startswith('Channel') and not(column in skip_chan):
                y=dat[column].to_numpy()                
                if freq == 1:
                    if Normalize_channels: y=y/(y[loc].max()-y[loc].min())
                    distance=y[loc].max()-y[loc].min()
                else:
                    mod=egg_filter(np.array([x,y]),freq=hrs_freq,rate=rate,order=order,ncomb=ncomb)

                    loc2=np.logical_and(mod[0,:]>xlim[0],mod[0,:]<xlim[1])
                    if Normalize_channels: mod[1,:]=mod[1,:]/(mod[1,loc2].max()-mod[1,loc2].min()) # Need to divide by whole range
                    distance=mod[1,loc2].max()-mod[1,loc2].min()
                
                distarr.append(distance)
        distarr=np.array(distarr)
#        print(distarr)
        spacer=distarr.max()*1.1    
    column_list=['Synctime']
    for i,column in enumerate(labels):
#        if column.startswith('Channel') and not(int(column[-2:]) in skip_chan):
        if column.startswith('Channel') and not(column in skip_chan):
            y=dat[column].to_numpy()
            column_list.append(column)

            if freq == 1:
                if Normalize_channels: y=y/(y[loc].max()-y[loc].min())
                if column in color_dict:
                    ax_an.plot(x, y-y[loc].mean()+space,color_dict[column])
                else:
                    ax_an.plot(x, y-y[loc].mean()+space)
                if points:
                    ax_an.plot(x, y-y[loc].mean()+space,'ro') 
                print('plotted!')
                outarray.append(y)
            else:
                mod=egg_filter(np.array([x,y]),freq=hrs_freq,rate=rate,order=order,ncomb=ncomb)
                loc2=np.logical_and(mod[0,:]>xlim[0],mod[0,:]<xlim[1])
                if Normalize_channels: 
                    mod[1,:]=mod[1,:]/(mod[1,loc2].max()-mod[1,loc2].min())
                if len(outarray)==0: outarray.append(mod[0,:].squeeze())
                if column in color_dict:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space,color_dict[column])
                else:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space)
                if points:
                    ax_an.plot(mod[0,loc2], mod[1,loc2]+space,'ro')
                outarray.append(mod[1,:].squeeze())
#            print(dat[column].name)
            if not hide_y:
                if dat[column].name in name_dict:
                    ax_an.text(ax_an.get_xlim()[0]-xsize/40,space,name_dict[dat[column].name],ha='right')
                else:
                    ax_an.text(ax_an.get_xlim()[0]-xsize/40,space,dat[column].name,ha='right')
            space+=spacer
#            print(space)
    if len(vline) != 0:
        ax_an.vlines(vline,ymin=0-spacer/2, ymax=space-spacer/2,color=line_params[0],linewidth=line_params[1],linestyle=line_params[2])
    if len(hline) != 0:
        ax_an.hlines(hline,xmin=xlim[0],xmax=xlim[1],linewidth=5,color='black',linestyle='dashed')
    ax_an.set_ylim(0-spacer,space)
    ax_an.set_title(title)

    if not Normalize_channels: #Only show voltage if everything is not rescaled
        ax_an.vlines(xlim[0],ymin=0-3*spacer/4,ymax=0-spacer/2,linewidth=10,color='black')
        ax_an.text(xlim[0]+xsize/40,0-5/8*spacer,str(np.round(spacer*1/4,decimals=2))+' mV',ha='left') 
        
#    add_scalebar(ax_an,hidex=False,matchy=True)
    outarray=np.array(outarray)
    loc_out=np.logical_and(outarray[0,:]>xlim[0],outarray[0,:]< xlim[1])
    outarray=outarray[:,loc_out]
    
    
    if output=="PD": 
        outarray=pd.DataFrame(outarray.T,columns=column_list)
    else: 
        outarray=outarray.T
    return fig_an,ax_an,outarray

#%% GAP SIZE DISTRIBUTION, INSERTING GAPS ETC. FOR INTERPOLATION VALIDATION
def get_gap_sizes(df, sec_gap):
    gap_sizes = []
    gap_size = 0
    t_cycle = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]

    for i in range(len(df)):
        if pd.isna(df['Channel 0'].iloc[i]): 
            gap_size += 1
        else:
            if gap_size > 0:
                gap_sec = gap_size * t_cycle
                if gap_sec < sec_gap:
                    gap_sizes.append(gap_sec)
                gap_size = 0  # Reset gap_size after processing a gap

    # Check for a gap at the end of the DataFrame
    if gap_size > 0:
        gap_sec = gap_size * t_cycle
        if gap_sec < sec_gap:
            gap_sizes.append(gap_sec)

    return gap_sizes

# Plot CDF of all gap sizes for a recording, including all recording files in 'path' as dir
def gap_distribution_plot(path):
    filepath = pathlib.Path(path)
    count_df = pd.DataFrame()

    for file in filepath.iterdir():
        if file.is_file() and file.suffix == '.txt':     
            #For the general read-in of data file
            df, _, _ =read_egg_v3_bursts(file,
                                        header = None,
                                        rate = 62.5,
                                        scale=600,
                                        n_burst=5,
                                        sleep_ping=1,
                                        sleep_time=1.84,
                                        t_deviation=0.2)
            
            gaps = get_gap_sizes(df, sec_gap=20000)
            gaps_rounded = [round(num) for num in gaps]
            rounded_gaps_df = pd.DataFrame(gaps_rounded, columns=['gap size'])
            gap_count = pd.DataFrame(rounded_gaps_df.value_counts().reset_index(name='count'))           
            count_df = pd.concat([count_df, gap_count], ignore_index=True)
            
    count_df = count_df.groupby('gap size').sum().reset_index()
    count_df = count_df.sort_values('gap size').reset_index(drop=True)

    #Non-weighted CDF of gap sizes
    data = count_df.copy()
    col = sns.color_palette('deep')
    data['cdf'] = data['count'].cumsum() / data['count'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(data['gap size'], data['cdf'], marker='.', linestyle='--', markersize=8, color = col[0])
    plt.axhline(y=0.925, color=col[5], alpha=.75, linestyle=':')
    plt.text(x=np.min(data['gap size']), y=0.925, s='93%', va='bottom', ha='center', size=12, fontweight='bold')
    plt.axvline(x=14, color=col[5], alpha=.75, linestyle=':')
    plt.text(x=14, y=plt.ylim()[0], s='14s', va='bottom', ha='left', size=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xscale('log')  
    plt.xlabel('Gap size [s]', size=16)
    plt.ylabel(r'CDF [$\mathbb{P}$]', size=16)
    plt.grid(True)
    plt.show()

    # Weighted CDF of gap sizes
    col = sns.color_palette('deep')
    data_w = count_df.copy()
    data_w['weighted'] = data_w['gap size'] * data_w['count']
    data_w = data_w.sort_values('gap size')
    data_w['weighted_cdf'] = data_w['weighted'].cumsum() / data_w['weighted'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(data_w['gap size'], data_w['weighted_cdf'], marker='.', linestyle='--', markersize=8, color = col[0])
    plt.axhline(y=0.465,color=col[5], alpha=.75, linestyle=':')
    plt.text(x=np.min(data['gap size']), y=0.465, s='47%', va='bottom', ha='center', size=12, fontweight='bold')
    plt.axvline(x=14, color=col[5], alpha=.75, linestyle=':')
    plt.text(x=14, y=plt.ylim()[0], s='14s', va='bottom', ha='left', size=12, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xscale('log')
    plt.xlabel('Gap size [s]', size=16)
    plt.ylabel(r'Weighted CDF [$\mathbb{P}$]', size=16)
    plt.grid(True)
    plt.show()

    return data, data_w
#%% FUNCTIONS USED IN DOWNSAMPLING OF EAO RECORDINGS
# Custom resampler used in downsampling 62.5 SPS recordings
def burst_resampler(group, n_burst=5, rate=62.5):
    """
    Custom resampling function to average over first n_burst points in a group

    :param group: Group defined by the resampling parameter 'time_str'
    :param n_burst: Amount of burst values to average over after grouping (first n_burst values)
    :param rate: Sampling rate of initial recording - 
    :return avg_vals: Averages of voltage data, first of timestamps and other columns --> if empty, nan is returned
    """
    if not group.empty:
        #can use index since timedelta is set as index for resampling
        t0 = group.index[0]
        dt = group.index - t0 
        include = group[dt < pd.Timedelta(seconds=n_burst*1/rate)]
        # Average only voltage channels, similar to read_egg_v3_bursts()
        avg_vals = pd.Series(index=group.columns)
        for col in group.columns:
            if 'Channel' in col:
                avg_vals[col] = include[col].mean()
            else:
                avg_vals[col]=group[col].iloc[0]
        return avg_vals
    else:
        return pd.Series([np.nan] * len(group.columns), index=group.columns)

# Main downsampling function for resampling the EAO datasets
def downsample_to_burst(file,time_str='2S',scale=300, date=None, round=False):
    """
    Sample down a 62.5 SPS recording to a predefined period

    :param file: raw recording file (.txt file) 
    :param time_str: The new period in down-sampling; '2S' is data every 2 seconds
    :param scale: +- scale in mV 
    :param date: The date of the recording, if None; it is gotten from the filename
    :param round: Round off the first datetime object to the time_str variable
    :return fulldat: Initial full dataframe, with pseudotime included
    :return resampled_interp: The resampled/downsampled full dataset interpolated for nan values
    : 
    """
    if date is None:
        # get only the filename from the file path
        file_path = pathlib.Path(file)
        filename = file_path.name
        # extract date from the filename
        date = filename.split('_')[0]

    # Creating datetime object
    # which takes in "MMDDYYYY" like only US people write date order
    base_date = datetime.strptime(date, '%Y.%m.%d')
    fulldat = read_egg_v3(file,scale=scale)
    # base_date2 = pd.Timestamp('2023-09-21')
    fulldat['realtime'] = fulldat['realtime'].str.strip()
    fulldat['realtime'] = pd.to_timedelta(fulldat['realtime'])  # Convert to Timedelta
    if round == True:
        base_time = base_date + fulldat['realtime'].iloc[0].round(time_str)  # Add first realtime value to base_date
    else:
        base_time = base_date + fulldat['realtime'].iloc[0]
    fulldat['pseudo_time'] = pd.to_timedelta(fulldat['timestamps'], unit='s') + base_time  # Add to base_time
    fulldat.set_index('pseudo_time', inplace=True)
    datcols = ['timestamps']+[f'Channel {i}' for i in range(8)]
    fulldat_short = fulldat[datcols]
    # Resample and apply the custom function - burst resampler relies on n_burst=5 and rate=62.5
    resampled_fulldat = fulldat_short.resample(time_str, label='left').apply(burst_resampler)
    # Reset index to return pseudotime
    fulldat_short.reset_index(inplace=True)
    resampled_fulldat.reset_index(inplace=True)
    fulldat_short['timestamps']=(fulldat_short['pseudo_time'] - fulldat_short['pseudo_time'].iloc[0]).dt.total_seconds()
    resampled_fulldat['timestamps']=(resampled_fulldat['pseudo_time'] - resampled_fulldat['pseudo_time'].iloc[0]).dt.total_seconds()
    resampled_interp = interpolate_data(resampled_fulldat, cycle_time=2, time_ip=True)

    return resampled_interp, fulldat_short

# Downsample from filtered with signalplot
def downsample_from_signalplot(df, time_str='2S'):
    df['timedelta'] = pd.to_timedelta(df['Synctime'], unit='s')
    df.set_index('timedelta', inplace=True)
    resampled_df = df.resample(time_str,label='left').apply(burst_resampler)
    df.reset_index(inplace=True)
    resampled_df.reset_index(inplace=True)
    df = df.rename(columns={'Synctime': 'timestamps'})
    resampled_df = resampled_df.rename(columns={'Synctime': 'timestamps'})

    return resampled_df, df


#%% DISTRIBUTE GAPS RANDOMLY OVER DATA
def distribute_gaps(df, gap_file, max_gap=14, t_cycle=2):
    """
    Introduce gaps into voltage data based on gap distribution from a separate dataframe

    :param df: DataFrame with time and voltage data
    :param gap_df: DataFrame with the gap size data of the recordings - presaved
    :param max_gap: The maximum gap size in seconds for binning the data
    :param avg_gaps: Average number of gaps based on previous calculations
    :return: DataFrame with gaps introduced in voltage data
    """
    # Load the gap_df 
    gap_df = pd.read_pickle(gap_file)

    # Bin the gap data and calculate probabilities
    bins = np.arange(t_cycle-0.2, max_gap + t_cycle, t_cycle)
    labels = np.arange(t_cycle, max_gap + t_cycle, t_cycle).astype(str)
    gap_df['binned'] = pd.cut(gap_df['gap size'], bins=bins, labels=labels, include_lowest=True)
    binned_counts = gap_df['binned'].value_counts().sort_index()
    total_gaps = binned_counts.sum()
    gap_prob = binned_counts / total_gaps

    print(gap_prob)

    # Generate gaps based on the calculated probabilities
    if gap_file == 'gap_df_48hrs_14sec.pkl':
        df2=df.copy()
        t_tot = df2['timestamps'].max()
        length = len(df2)
        avg_ratio = 0.06978
        num_gaps = int(length*avg_ratio)
        print(num_gaps)
        gap_sizes = np.arange(t_cycle, max_gap + t_cycle, t_cycle)
    
    else: 
        df2=df.copy()
        t_tot = df2['timestamps'].max()
        length = len(df2)
        avg_ratio = 0.03845
        num_gaps = int(length*avg_ratio)
        print(num_gaps)
        gap_sizes = np.arange(t_cycle, max_gap + t_cycle, t_cycle)

    t_s = []
    t_end = []
    for _ in range(num_gaps):
        valid_start_time_found = False
        while not valid_start_time_found:
            t_start = np.random.uniform(0, t_tot)
            # Generate a single gap size for this iteration
            chosen_gap = np.random.choice(gap_sizes, p=gap_prob)
            t_end_time = t_start + chosen_gap
            # Check if the gap overlaps with existing gaps
            if not any((t_start < et and t_end_time > st) for st, et in zip(t_s, t_end)):
                valid_start_time_found = True
                t_s.append(t_start)
                t_end.append(t_end_time)

    sort_idx = np.argsort(t_s)
    t_s = np.array(t_s)[sort_idx]
    t_end = np.array(t_end)[sort_idx]

    # Introduce gaps into the data
    for start, end in zip(t_s, t_end):
        start_idxs = df2[df2['timestamps'] >= start].index
        end_idxs = df2[df2['timestamps'] < end].index
        if not start_idxs.empty and not end_idxs.empty:
            start_idx = start_idxs[0]
            end_idx = end_idxs[-1]
            df2.loc[start_idx:end_idx, 'Channel 0':'Channel 7'] = np.nan

    return df2

def distribute_gaps_weighted(df, gap_file=None, max_gap=14, t_cycle=2, weighted=True, rand_state = None):
    """
    Introduce gaps into voltage data based on gap distribution from a separate dataframe or uniformly.

    :param df: DataFrame with time and voltage data
    :param gap_file: File path for DataFrame with the gap size data of the recordings - presaved
    :param max_gap: The maximum gap size in seconds for binning the data
    :param t_cycle: Time cycle in seconds for binning
    :param weighted: Boolean to choose weighted or uniform gap distribution
    :return: DataFrame with gaps introduced in voltage data
    """
    df2 = df.copy()
    t_tot = df2['timestamps'].max()
    length = len(df2)

    # Gaps should be put in the same position for analysis of (Lin)-Regression
    if rand_state is not None:
        np.random.seed(rand_state)

    if weighted:
        # Load the gap_df for weighted distribution
        gap_df = pd.read_pickle(gap_file)
        bins = np.arange(t_cycle-0.2, max_gap + t_cycle, t_cycle)
        labels = np.arange(t_cycle, max_gap + t_cycle, t_cycle).astype(str)
        gap_df['binned'] = pd.cut(gap_df['gap size'], bins=bins, labels=labels, include_lowest=True)
        binned_counts = gap_df['binned'].value_counts().sort_index()
        total_gaps = binned_counts.sum()
        gap_prob = binned_counts / total_gaps
        print(gap_prob)

        if gap_file == 'gap_df_48hrs_14sec.pkl':
            avg_ratio = 0.06978
        else: 
            avg_ratio = 0.03845

        num_gaps = int(length * avg_ratio)
        gap_sizes = np.arange(t_cycle, max_gap + t_cycle, t_cycle)
    else:
        
        max_gaps = int(length * 0.1) / max_gap
        num_gaps = int(max_gaps)
        gap_sizes = np.array([max_gap] * num_gaps)  
        gap_prob = None

    t_s = []
    t_end = []
    for _ in range(num_gaps):
        valid_start_time_found = False
        while not valid_start_time_found:
            t_start = np.random.uniform(0, t_tot)
            chosen_gap = np.random.choice(gap_sizes, gap_prob) if weighted else max_gap
            t_end_time = t_start + chosen_gap
            if not any((t_start < et and t_end_time > st) for st, et in zip(t_s, t_end)):
                valid_start_time_found = True
                t_s.append(t_start)
                t_end.append(t_end_time)

    sort_idx = np.argsort(t_s)
    t_s = np.array(t_s)[sort_idx]
    t_end = np.array(t_end)[sort_idx]

    # Introduce gaps into the data
    for start, end in zip(t_s, t_end):
        start_idxs = df2[df2['timestamps'] >= start].index
        end_idxs = df2[df2['timestamps'] < end].index
        if not start_idxs.empty and not end_idxs.empty:
            start_idx = start_idxs[0]
            end_idx = end_idxs[-1]
            df2.loc[start_idx:end_idx, 'Channel 0':'Channel 7'] = np.nan

    return df2

#%% SIGNAL COMPARISON BETWEEN DOWNSAMPLED AND ACTUAL 

# Plotting both the v-t plots and freq spectrum of control vs resampled
# If raw = True, plot only the raw v-t and get data 
def plot_signal_comparison(resampled, data, fs_c=62.5, fs_d=0.5, xlim=None, scaler=1.1, fig_raw=(15,10),
                           fig_spec = (20,25),freqlim=[0,10], freq=[0.02, 0.2], raw = False, savgol=True, size=18):
    """
    Plot voltage-time and frequency spectrum for control and downsampled data across all channels
    in a 3-column layout: v-t plot, frequency spectrum downsampled, frequency spectrum control.
    """
    x_c = data['timestamps']
    x_d = resampled['timestamps']
    
    if xlim: 
        data = data[(x_c >= xlim[0]) & (x_c <= xlim[1])]
        resampled = resampled[(x_d >= xlim[0]) & (x_d <= xlim[1])]

    if raw: 

        fig, axs = plt.subplots(8, 1, figsize=fig_raw)

        # axs.text(-0.1, 0.5, f'Channel {i}', va='center', ha='right', transform=ax.transAxes, fontsize=size)
        
        for i in range(8):
            max_y = 0
            x_c = data['timestamps']
            y_c = data[f'Channel {i}']
            x_d = resampled['timestamps']
            y_d = resampled[f'Channel {i}']

            # if xlim:
            #     mask_c = (x_c >= xlim[0]) & (x_c <= xlim[1])
            #     mask_d = (x_d >= xlim[0]) & (x_d <= xlim[1])
            #     x_c, y_c = x_c[mask_c], y_c[mask_c]
            #     x_d, y_d = x_d[mask_d], y_d[mask_d]

            # Voltage-time plots
            axs[i].plot(x_c, y_c, linestyle='dashed', label='Control', color='C1')
            axs[i].plot(x_d, y_d, label='Downsampled', color='C0', alpha=0.5)
            if xlim:
                axs[i].set_xlim(xlim)
            axs[i].set_ylim([min(y_c.min(), y_d.min()) * scaler, max(y_c.max(), y_d.max()) * scaler])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            axs[i].text(-0.1, 0.5, f'Channel {i}', va='center', ha='right', transform=axs[i].transAxes, fontsize=size)
            # axs[i, 0].spines['left'].set_visible(False)
            # axs[i, 0].xaxis.set_ticks_position('none')
            # axs[i, 0].xaxis.set_ticks_position('bottom')
            if i == 7:
                axs[i].set_xlabel('Time [s]', size=size)
            axs[i].legend()

        return resampled, data

    else: 
        fulldat_filt = butter_filter_sos(data, fs=fs_c)
        resampled_filt = butter_filter_sos(savgol_filt(resampled) if savgol else resampled, fs=fs_d)

        fig, axs = plt.subplots(8, 3, figsize=fig_spec)

        for i in range(7):
            for j in range(3):  
                axs[i, j].set_xticklabels([]) 
        # plt.subplots_adjust(wspace=.2, hspace=.2)

        # Adding channel labels on the left side of the first column
        for i, ax in enumerate(axs[:, 0]):
            ax.text(-0.1, 0.5, f'Channel {i}', va='center', ha='right', transform=ax.transAxes, fontsize=size)
        
        fig.text(0.695,.095, 'Magnitude', ha='center', va='center', rotation='vertical', fontsize=size)

        # Common y-labels
        fig.supylabel('Voltage [mV]', x=0.055, size=size)
        # fig.supylabel('Magnitude', x=0.375, size=16)

        titles = ['Voltage-Time', 'Spectrum (Downsampled)', 'Spectrum (Control)']
        for ax, title in zip(axs[0], titles):
            ax.set_title(title, size=size+2)

        max_y_vals = [0]*8

        for i in range(8):
            max_y = 0
            x_c = fulldat_filt['timestamps']
            y_c = fulldat_filt[f'Channel {i}']
            x_d = resampled_filt['timestamps']
            y_d = resampled_filt[f'Channel {i}']

            # if xlim:
            #     mask_c = (x_c >= xlim[0]) & (x_c <= xlim[1])
            #     mask_d = (x_d >= xlim[0]) & (x_d <= xlim[1])
            #     x_c, y_c = x_c[mask_c], y_c[mask_c]
            #     x_d, y_d = x_d[mask_d], y_d[mask_d]

            # Voltage-time plots
            axs[i, 0].plot(x_c, y_c, linestyle='dashed', label='Control', color='C1')
            axs[i, 0].plot(x_d, y_d, label='Downsampled', color='C0', alpha=0.5)
            if xlim:
                axs[i, 0].set_xlim(xlim)
            axs[i, 0].set_ylim([min(y_c.min(), y_d.min()) * scaler, max(y_c.max(), y_d.max()) * scaler])
            axs[i, 0].spines['right'].set_visible(False)
            axs[i, 0].spines['top'].set_visible(False)
            # axs[i, 0].spines['left'].set_visible(False)
            # axs[i, 0].xaxis.set_ticks_position('none')
            # axs[i, 0].xaxis.set_ticks_position('bottom')
            if i == 7:
                axs[i, 0].set_xlabel('Time [s]', size=size)
            axs[i, 0].legend()

            # Frequency spectrum analyses
            for j, (filt_data, fs, col) in enumerate([(resampled_filt, fs_d, 'C0'), (fulldat_filt, fs_c, 'C1')]):
                x = filt_data['timestamps']
                y = filt_data[f'Channel {i}']
                if xlim:
                    mask = (x >= xlim[0]) & (x <= xlim[1])
                    y = y[mask]
                    
                freq, mag = sig.periodogram(y, fs)
                freq *= 60  # Convert to cycles per minute (CPM)
                max_y = max(max_y, np.max(mag))
                d_idx = np.argmax(mag)  # Index of dominant frequency
                d_freq = freq[d_idx]  # Dominant frequency
                d_mag = mag[d_idx]  # Magnitude at dominant frequency
                axs[i, j+1].stem(freq, mag, linefmt=f'{col}-', markerfmt=f'{col}o', basefmt=" ")
                axs[i, j+1].plot(d_freq, d_mag, marker='*', markersize=8, color='red')
                axs[i, j+1].text(d_freq, d_mag*1.075, f'{d_freq:.2f} CPM', fontsize=16, color='red', ha='center', fontweight='bold')
                axs[i, j+1].set_xlim(freqlim)
                axs[i, j+1].ticklabel_format(axis='y', scilimits=(0,0))
                if i == 7:
                    axs[i, j+1].set_xlabel('Frequency [CPM]', size=size)
                axs[i, j+1].spines['right'].set_visible(False)
                axs[i, j+1].spines['top'].set_visible(False)            
            max_y_vals[i] = max_y

            for j in range(1, 3):
                axs[i, j].set_ylim(0, round(max_y_vals[i]*1.2, 1))
                axs[i, j].yaxis.get_offset_text().set_fontsize(size)
            
            for ax in axs[-1,:]:
                for label in ax.get_xticklabels():
                    label.set_fontsize(size)
            for ax in axs.flatten():  # Flatten the array to iterate over all subplots
                for label in ax.get_yticklabels():
                    label.set_fontsize(size)  # Set the y-tick label sizes to match the x-tick label sizes
            
            axs[i,2].set_yticklabels([])
            axs[i,2].set_yticks([])
            axs[i,2].spines['left'].set_visible(False)

        plt.tight_layout()
        plt.show()
        return resampled_filt, fulldat_filt

# Calculate the difference between voltage values downsampled/control, average per channel
def calc_diff_resample(resamp, data, freq=True, warp_plot=True, window=25,fs_d=.5, fs_c=62.5):
    
    merged = pd.merge_asof(resamp, data, on='timestamps', suffixes=('_d', '_c'))
    stats = []
    
    for i in range(8):
        chan = f'Channel {i}'
        chan_diff = f'Channel {i}_diff'
        chan_d = f'Channel {i}_d' #channels downsampled
        chan_c = f'Channel {i}_c' #channels control
        merged[chan_diff] = merged[chan_d] - merged[chan_c]
        MSE = mse(merged[chan_d], merged[chan_c])
        RMSE = np.sqrt(MSE)
        MAE = mae(merged[chan_d], merged[chan_c])
        s1 = np.array(merged[chan_d], dtype=np.double)
        s2 = np.array(merged[chan_c], dtype=np.double)
        dtw_dist = dtw.distance_fast(s1,s2)
        if warp_plot:
            s1_short = np.array(merged[chan_d][50:175], dtype=np.double)
            s2_short = np.array(merged[chan_c][50:175], dtype=np.double)
            d_short,paths = dtw.warping_paths(s1_short, s2_short, window=window, psi=2)
            best_path = dtw.best_path(paths)
            print(f'Channel: {chan} with warping path: {d_short}')
            dtwvis.plot_warpingpaths(s1_short, s2_short, paths, best_path)
            dtwvis.plot_warping(s1_short, s2_short, best_path)
        if freq:
            freq_d, mag_d = sig.periodogram(resamp[chan], fs=fs_d)
            freq_c, mag_c = sig.periodogram(data[chan], fs=fs_c)
            freq_d = freq_d*60
            freq_c = freq_c*60
            d_f_d = freq_d[np.argmax(mag_d)]
            d_f_c = freq_c[np.argmax(mag_c)]

            stats.append({
                'Channel': i,
                'MSE': MSE,
                'RMSE': RMSE,
                'MAE': MAE,
                'Freq_D': d_f_d,
                'Freq_C': d_f_c,
                'Diff D-Freq': np.abs(d_f_d-d_f_c),
                'D-Mag': np.abs(np.argmax(mag_d)-np.argmax(mag_c)),
                'DTW': dtw_dist
            })
        
        else:
            stats.append({
                'Channel': i,
                'MSE': MSE,
                'RMSE': RMSE,
                'MAE': MAE
            })

    diffs = merged[['timestamps']+[f'Channel {i}_diff' for i in range(8)]]
    
    chan_abs_avg = []
    for i in range(1,9):
        chan_abs_avg.append(np.mean(np.abs(diffs.iloc[:,i])))

    statz=pd.DataFrame(stats)

    return diffs, chan_abs_avg, statz

#%% MISC OLD FUNCTIONS 
# MOVING AVERAGE SMOOTHENING FUNCTION 
# def smooth_signal_moving_average(df, window_size=5):
#     df_smooth = df.copy()
#     window = np.ones(int(window_size))/float(window_size)
    
#     for column in df_smooth.columns:
#         if column.startswith('Channel'):
#             df_smooth[column] = np.convolve(df_smooth[column], window,'same')

#     return df_smooth
# OLD AVERAGING FUNCTION WITHOUT TAKING ANOTHER VALUE FROM THE GROUP TO REPRESENT ELAPSED TIMES ETC.
# def averaging_bursts(df, n_burst=5, sleep_ping=1):
#     df2=df.copy()
#     burst_length = n_burst+sleep_ping
#     for i in range(8):
#         channel = f'Channel {i}'
#         # Convert to numpy
#         data = df2[channel].values
#         # Calculate padding
#         remainder = len(data) % (n_burst+sleep_ping)
#         padding_size = 0 if remainder == 0 else ((n_burst+sleep_ping) - remainder)
#         # Pad with nan
#         padded_data = np.pad(data, (0, padding_size), constant_values=np.nan)
#         # Reshape the data to have n_burst+sleep_ping values per row
#         reshaped_data = padded_data.reshape(-1, (n_burst+sleep_ping))
#         # Compute the mean per row, ignoring nan's
#         means = np.nanmean(reshaped_data, axis=1)
#         # Repeat mean 6 times to get original shape back
#         repeated_means = np.repeat(means, (n_burst+sleep_ping))
#         # Trim back to old length
#         trimmed_means = repeated_means[:len(data)]
#         # Assign to the voltage channels
#         df2[channel] = trimmed_means

#     #Filter for 1st of burst only to shift averaging starting at t=0
#     df2 = df2[df2['packet_miss_idx'] % (n_burst+sleep_ping) == 0]
#     return df2

# OLD INTERPOLATION FUNCTIONS

# def interpolate_data(df, cycle_time, max_gap=14, rescale=False, time_ip=False, pchip=True):
#     start = time.time()
#     df2 = df.copy()
#     # Reset timestamps to start from zero if required
#     if rescale: 
#         df2['timestamps'] -= df2['timestamps'].iloc[0]

#     # Linearly interpolate timestamps if required (e.g., when downsampling)
#     if time_ip: 
#         df2['timestamps'] = df2['timestamps'].interpolate('linear')

#     # Define the maximum number of consecutive NaNs allowed based on cycle_time
#     max_consecutive_nans = int(max_gap / cycle_time)

#     # Interpolate each channel
#     for channel in df2.columns:
#         if 'Channel' in channel:  # Assuming channel data columns have 'Channel' in their names
#             # Convert the column to numeric and count consecutive NaNs
#             df2[channel] = pd.to_numeric(df2[channel], errors='coerce')
#             consec_nan_counts = df2[channel].isna().astype(int).groupby(df2[channel].notna().astype(int).cumsum()).sum()
#             print(consec_nan_counts)

#             # Prepare for interpolation
#             x = df2['timestamps'].values
#             y = df2[channel].values
#             non_nan_mask = ~np.isnan(y)
            
#             # Interpolate with Cubic Spline or linear depending on the gap size
#             for group, count in consec_nan_counts.items():
#                 if count <= max_consecutive_nans:
#                     # Cubic spline interpolation for small gaps
#                     cs = CubicSpline(x[non_nan_mask], y[non_nan_mask])
#                     df2.loc[non_nan_mask, channel] = cs(x[non_nan_mask])
#                 else:
#                     if pchip == True:
#                         # PCHIP interpolation for larger gaps
#                         pchip_interp = PchipInterpolator(x[non_nan_mask], y[non_nan_mask])
#                         nan_indices = df2[channel].isna()
#                         df2.loc[nan_indices, channel] = pchip_interp(x[nan_indices])
#                     # Linear interpolation for larger gaps
#                     else:
#                         linear_interp = interp1d(x[non_nan_mask], y[non_nan_mask], kind='linear', fill_value="extrapolate")
#                         nan_indices = df2[channel].isna()
#                         df2.loc[nan_indices, channel] = linear_interp(x[nan_indices])
#     end=time.time()
#     print(f'The function took {end-start:.2f} seconds to run')
#     return df2


#PANDA INTERPOLATION, NO SCIPY INVOLVED
# def interpolate_data_pandas(df, cycle_time, max_gap=14, rescale=False, time_ip=False, pchip=True):
#     start = time.time()
#     df2 = df.copy()

#     # Reset timestamps to start from zero if required
#     if rescale: 
#         df2['timestamps'] -= df2['timestamps'].iloc[0]

#     # Linearly interpolate timestamps if required (e.g., when downsampling)
#     if time_ip: 
#         df2['timestamps'] = df2['timestamps'].interpolate(method='linear')

#     # Define the maximum number of consecutive NaNs allowed based on cycle_time
#     max_consecutive_nans = int(max_gap / cycle_time)

#     # Interpolate each channel
#     for channel in df2.columns:
#         if 'Channel' in channel:  # Assuming channel data columns have 'Channel' in their names
#             # Convert the column to numeric and count consecutive NaNs
#             df2[channel] = pd.to_numeric(df2[channel], errors='coerce')
#             consec_nan_counts = df2[channel].isna().astype(int).groupby(df2[channel].notna().astype(int).cumsum()).sum()

#             # Prepare for interpolation
#             for group, count in consec_nan_counts.items():
#                 if count <= max_consecutive_nans:
#                     # Cubic spline interpolation for small gaps
#                     df2[channel] = df2[channel].interpolate(method='cubicspline')
#                 else:
#                     # PCHIP interpolation for larger gaps
#                     if pchip:
#                         df2[channel] = df2[channel].interpolate(method='pchip')
#                     else:
#                         # Linear interpolation for larger gaps
#                         df2[channel] = df2[channel].interpolate(method='linear')

#     end = time.time()
#     print(f'The function took {end - start:.2f} seconds to run')
#     return df2

# NEW METHOD FOR DOWNSAMPLING, NOT COMPLETELY WORKING
# def downsample_to_burst(file,time_str='2S',scale=150, date=None, round=False):
#     """
#     Sample down a 62.5 SPS recording to a predefined period

#     :param file: raw recording file (.txt file) 
#     :param time_str: The new period in down-sampling; '2S' is data every 2 seconds
#     :param scale: +- scale in mV 
#     :param date: The date of the recording, if None; it is gotten from the filename
#     :param round: Round off the first datetime object to the time_str variable
#     :return fulldat: Initial full dataframe, with pseudotime included
#     :return resampled_interp: The resampled/downsampled full dataset interpolated for nan values
#     : 
#     """
#     if date is None:
#         # get only the filename from the file path
#         file_path = pathlib.Path(file)
#         filename = file_path.name
#         # extract date from the filename
#         date = filename.split('_')[0]

#     # Creating datetime object
#     base_date = datetime.strptime(date, '%Y.%m.%d')
#     fulldat = read_egg_v3(file,scale=scale)
#     # base_date2 = pd.Timestamp('2023-09-21')
#     fulldat['realtime'] = fulldat['realtime'].str.strip()
#     fulldat['realtime'] = pd.to_timedelta(fulldat['realtime'])
#     if round == True:
#         base_time = base_date + fulldat['realtime'].iloc[0].round(time_str)  # Add first realtime value to base_date
#     else:
#         base_time = base_date + fulldat['realtime'].iloc[0]
#     fulldat['pseudo_time'] = pd.to_timedelta(fulldat['timestamps'], unit='s') + base_time  # Add to base_time
#     fulldat['pseudo_time'] = fulldat['pseudo_time'].dt.round('2S') # Example rounding operation
#     fulldat['timedelta'] = pd.to_timedelta(fulldat['timestamps'], unit='s')
#     fulldat.set_index('pseudo_time', inplace=True)
#     datcols = ['timestamps','timedelta']+[f'Channel {i}' for i in range(8)]
#     fulldat_short = fulldat[datcols]
#     # Resample and apply the custom function - burst resampler relies on n_burst=5 and rate=62.5
#     resampled_fulldat = fulldat_short.resample(time_str, label='left').apply(burst_resampler)
#     # Reset index to return pseudotime
#     fulldat.reset_index(inplace=True)
#     resampled_fulldat.reset_index(inplace=True)
#     resampled_interp = interpolate_egg_v3(resampled_fulldat, method='cubicspline',time=True)

#     return fulldat, resampled_interp