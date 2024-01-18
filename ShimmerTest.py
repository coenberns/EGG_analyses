# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 18:11:17 2023

@author: seany
"""
from pathlib import Path
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import fftpack
import numpy as np
import datetime as dt
from scipy.interpolate import CubicSpline
import re
from Plot_EGG import*
import sys

def read_shimmer(file,time_loc=0):
    data_temp=pd.read_csv(file,delimiter='\t',header=1,skiprows=[2])
    time=data_temp.columns[time_loc]
    data_temp['Time']=(data_temp[time]-data_temp[time][0])/1000
    data_temp['Datetime']=data_temp[time].apply(lambda x: dt.datetime.fromtimestamp(x/1000)) 
    return data_temp

def shimmer_get_columns(data,str_match='EMG_CH'):
    column_names=[]
    for name in data.columns:
        if 'EMG_CH' in name:
            column_names.append(name)
    return column_names

def filter_shimmer(data,freq,order=3,rate=50,ch=[],names=[]):
    sos=sig.butter(order,freq,btype='bandpass',output='sos',fs=rate)
    if len(ch) == 0:
        print("No Channels Found")
    for i,channel in enumerate(ch):
        if type(names)==str:
            name=names+'_'+str(i)
        elif len(names)==0 or len(names)!=len(ch): 
            name=channel+'_Filtered'
            print('Using Default Names')
        else: 
            name=names[i] #only assign names if there are the right nuber of names
        data[name]=sig.sosfiltfilt(sos,data[channel])
    return data

def get_sampling_rate(data,time='Synctime'):
    tlist=[]
    for i,ele in enumerate(data.Synctime):
        if i>1:
            tlist.append(data.Synctime[i]-data.Synctime[i-1])
    array=np.array(tlist)
    return 1/array.mean()

def settime_egg_v3(data,year=2023,month=8,day=2):
    basetime=dt.datetime.strptime(data['realtime'][0],'%H:%M:%S.%f ')
    basetime=basetime.replace(year=year,month=month,day=day)
    data['Datetime']=data['timestamps'].apply(lambda x: basetime+dt.timedelta(seconds=x))
    return data


def shimmer_migut_timesync(migut_data,shimmer_data):
    t0=min(migut_data['Datetime'][0],shimmer_data['Datetime'][0])
    migut_data['Synctime']=migut_data['Datetime'].apply(lambda x: ((x-t0).total_seconds()))
    shimmer_data['Synctime']=shimmer_data['Datetime'].apply(lambda x: ((x-t0).total_seconds()))
    return migut_data,shimmer_data

def migut_shimmer_merge_interpolate(migut_data,shimmer_data,m_rate=62.5,s_chan=[],new_chan='S'):
    t0=max(migut_data['Datetime'][0],shimmer_data['Datetime'][0])
    tend=min(migut_data['Datetime'].iloc[-1],shimmer_data['Datetime'].iloc[-1])
    t_total=(tend-t0).total_seconds()
    # Initalize the max overlapped recording time. 
    
    new_data=pd.DataFrame(np.arange(0,t_total,1/m_rate),columns=['Synctime'])
    
    migut_data['Synctime']=migut_data['Datetime'].apply(lambda x: ((x-t0).total_seconds()))
    shimmer_data['Synctime']=shimmer_data['Datetime'].apply(lambda x: ((x-t0).total_seconds()))
     
    migut_channels=[x for x in migut_data.columns if re.match('Channel*',x)] #Us regex to find Channel * column names
    for chan in migut_channels:
        f=CubicSpline(migut_data['Synctime'],migut_data[chan])
        interpolated_data=f(new_data['Synctime'])
        new_data[chan]=interpolated_data
    if len(s_chan)==0:
        s_chan=[x for x in shimmer_data.columns if re.search('EMG_CH',x)]
    
    for i,chan in enumerate(s_chan):
        f=CubicSpline(shimmer_data['Synctime'],shimmer_data[chan])
        interpolated_data=f(new_data['Synctime'])
#        plt.plot(new_data['Synctime'],interpolated_data) #only for debug
#        plt.plot(shimmer_data['Synctime'],shimmer_data[chan])
        new_data['Channel '+new_chan+str(i)]=interpolated_data
    new_data['Datetime']=new_data['Synctime'].apply(lambda x: t0+dt.timedelta(seconds=x))
    return new_data

def migut_shimmer_columns(data,col,n=[],invert=False):
    if col=='m':
        selected = [x for x in data.columns if re.match(r'\AC.* \d',x)]
        rtype='' #set the regex string in case we need to selecting channels
    elif col=='s':
        selected = [x for x in data.columns if re.match(r'\AC.* S\d',x)]
        rtype='S'
    else: 
        print('Set col to either m for migut or s for shimmer')
        selected = []
    
    if len(n) > 0:
        nums=list(map(lambda x:str(x),n))
        if invert:
            flip='^' # ^ is the character that flips the regex set to EXCEPT.
        else:
            flip='' # Otherwise don't include ^ as a character
        matching='.* '+rtype+'['+flip+''.join(nums)+']' # Create Regex Expression to return the strings of the column names desired
        # '.*' - Starting with anything
        # rtype - either '' for MiGUT or S for Shimmer channel
        # '[' ']' to access sets
        # flip - either '' for normal or '^' for except behavior
        # nums - array of string of channels. NOTE THIS ONLY SUPPORTS UP TO 0-9 channels, you will have to revise otherwise
        selected = [x for x in selected if re.match(matching,x)]
    return selected
#%%
# ch1='Shimmer_68ED_EMG_CH1_24BIT_CAL'
# ch2='Shimmer_68ED_EMG_CH2_24BIT_CAL'
# time='Shimmer_68ED_TimestampSync_Unix_CAL'


# #ch='Shimmer_7970_ECG_LA-RA_24BIT_CAL'

# script_path = Path(__file__).parent

# folderpath="2023-07-26_15.26.29_DefaultTrial_PC_Session3"




# file_pattern = '*.csv'
# file_paths = glob.glob(str(script_path / folderpath /file_pattern))
#%%
data=read_shimmer('2023.08.02_Shimmer.csv')

c_names=shimmer_get_columns(data)
data=filter_shimmer(data,[0.02,0.25],ch=c_names,names=['Slow_1','Slow_2'])
data=filter_shimmer(data,[5,24],ch=c_names,names=['EKG_1','EKG_2'])
data=filter_shimmer(data,[.25,5],ch=c_names,names=['Res_1','Res_2'])
data=filter_shimmer(data,[.005,24],ch=c_names,names=['Wide_1','Wide_2'])

#%%
data=read_shimmer('2023.08.02_Shimmer.csv')
mdata=read_egg_v3('2023.08.02_Full.txt',scale=300)
mdata=settime_egg_v3(mdata)
new_data=migut_shimmer_merge_interpolate(mdata,data)

#%%

signalplot(new_data,time='Synctime',xlim=[2000,2200],freq=[.02,.25],Normalize_channels=True)

egg_freq_heatplot_v2(new_data,xlim=[500,4000],freqlim=[.5,4],max_scale=1)

a,b,d=signalplot(new_data,time='Synctime',xlim=[3000,4000],freq=[5,24],output='PD')
a,b,e=signalplot(new_data,time='Synctime',xlim=[3000,4000],freq=[.25,1],output='PD')

egg_signalfreq(d,freqlim=[110,125])
egg_signalfreq(e,freqlim=[10,40])

egg_freq_heatplot_v2(new_data,xlim=[1500,3000],seg_length=200,n=10,freqlim=[1,4],freq=[0.01,.25],norm=True,max_scale=1)

#%%
# 68ED - Serosa
# 7970 0 Mucosa
data=read_egg_v3('2023.08_24_MiGUT.txt')
sdata_serosal=read_shimmer('2023.08.24_68ED.csv')
sdata_cutaneous=read_shimmer('2023.08.24_7970.csv')
data=settime_egg_v3(data,day=24)
new_data=migut_shimmer_merge_interpolate(data,sdata_serosal,new_chan='S')
new_data_all=migut_shimmer_merge_interpolate(new_data,sdata_cutaneous,new_chan='C')

#Respiration Matching
signalplot(new_data_all,time='Synctime',freq=[.02,.14],xlim=[2400,2660],Normalize_channels=True,skip_chan=['Channel 3','Channel 4','Channel 5','Channel 6','Channel 7'],figsize=(16,10),spacer=1.5, vline=[2425])

#
signalplot(new_data_all,time='Synctime',freq=[.02,.14],xlim=[2400,2660],Normalize_channels=True,skip_chan=['Channel 3','Channel 4','Channel 5','Channel 6','Channel 7'],figsize=(16,10),spacer=1.5)



#Compare slow wave and respiration:
egg_freq_heatplot_v2(new_data_all,xlim=[500,4500],freqlim=[1.5,20],freq=[0.02,5],seg_length=200,skip_chan=['Channel 3','Channel 4','Channel 5','Channel 6','Channel 7'])

signalplot(new_data_all,time='Synctime',freq=[.02,.25],xlim=[4200,4500],Normalize_channels=False,skip_chan=['Channel 3','Channel 4','Channel 5','Channel 6','Channel 7', 'Channel C1','Channel C0', 'Channel S0','Channel S1'],figsize=(16,10),spacer=1.5)

#%%

data_2=read_shimmer('2023.08.09_Shimmer.csv')
mdata_2=read_egg_v3('2023.08.09_Full.txt',scale=300)
mdata_2=settime_egg_v3(mdata_2,day=9)
new_data_2=migut_shimmer_merge_interpolate(mdata_2,data_2)

#%%
data=[]

#freq=[0.01,0.05]
freq=[0.01,.2]
for csv in file_paths:
    data_temp=pd.read_csv(csv,delimiter='\t',header=1,skiprows=[2])
    data_temp['Time']=(data_temp[time]-data_temp[time][0])/1000
    sos=sig.butter(3,freq,btype='bandpass',output='sos',fs=50)
#    filtered=sig.sosfiltfilt(sos,f(tfixed))
    data_temp['C1']=sig.sosfiltfilt(sos,data_temp[ch1])
    data_temp['C2']=sig.sosfiltfilt(sos,data_temp[ch2])
    data.append(data_temp)


#%%
##Plotting Both Channels with filtered Data
data[0].plot(x='Time',y='C1',figsize=(15,5),title='Ch 1 Filtered')
data[0].plot(x='Time',y='C2',figsize=(15,5),title='Ch 2 Filtered')
data[0].plot(x='Time',y='C1',figsize=(15,5),xlim=[2500,3000],title='Ch 2 Filtered',ylim=[-.4,0.4])
#data[0].plot(x='Time',y=ch1,figsize=(15,5),xlim=[1000,1400],ylim=[-7,-8],title='Ch 2 Filtered')

#%%

d1=data[0]['C1'].to_numpy()
d2=data[0]['C2'].to_numpy()
dtime=data[0]['Time'].to_numpy()
#d1=d1[(dtime > 200) & (dtime <2000)]
#d2=d2[(dtime > 200) & (dtime <2000)]


fftdat1=fftpack.fft(d1)
fftdat2=fftpack.fft(d2)
freqs=fftpack.fftfreq(len(d1),1/(50*60))

fig, ax = plt.subplots(nrows=2,figsize=(8,8))

loc= (freqs > 1.5) & (freqs <5)
ax[1].stem(freqs[loc], np.abs(fftdat2[loc]))
ax[1].set_xlabel('Frequency in 1/mins')
ax[1].set_ylabel('Channel 2 FFT - Near PEG Tube')

ax[0].stem(freqs[loc], np.abs(fftdat1[loc]))
#ax[0].set_xlabel('Frequency in 1/mins')
ax[0].set_ylabel('Channel 1 FFT - Underside')

# Can't see anything in the pig?


#%% Ambulation Analysis

dat=read_egg_v3_burst('C:/Users/LangerLab642/Dropbox (Personal)/Langer Lab/Data/EGG_Data_Repository/2023.08.28_Ambulation.txt')
