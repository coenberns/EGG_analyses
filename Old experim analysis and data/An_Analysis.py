# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:05:12 2023

@author: seany
"""

import sys
sys.path.append('C:/Users/seany/Dropbox/Langer Lab/Data/EGG_Data_Repository/')
from Old_Plot_EGG import * 
import scipy.signal as sig
import os
#%%

if not os.path.exists('Output'): os.makedirs('Output') # make output directory 

dat=read_egg_v3('2022.05.02_Combined.txt')
dat0114=read_egg_v3('2022.04.14_Terminal.txt')
dat0520=read_egg_v3('2022.05.20_Full_End_Clipped.txt')


dat0623=read_egg_v3('2022.06.23_Az.txt',scale=150)

dat0629=read_egg_v3('2022.06.29_Az.txt',scale=150)

dat0707=read_egg_v3('2022.07.07_Az.txt',scale=150)

#%%

f_all,an_all,c=heatplot(dat,freq=[0.0005,15],xlim=[500,5000])

#%%Propagation Speed

f,an,c=signalplot(dat,freq=[.02,.25],xlim=[700,850],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(15,10))
f,an,c=signalplot(dat,freq=[.25,5],xlim=[710,730],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(12,8))
f,an,c=signalplot(dat,freq=[8,30],xlim=[720,725],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(12,8))

#%%
f_slow,an_slow,c1=signalplot(dat,freq=[.01,.25],xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)
f_res,an_res,c2=signalplot(dat,freq=[.25,5],xlim=[700,730],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
f_ekg,an_ekg,c3=signalplot(dat,freq=[5,1000],xlim=[700,710],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),spacer=.15,hide_y=True)
f_raw,an_raw,c4=signalplot(dat,xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True, spacer=20)


f_res,an_res,c2=signalplot(dat,freq=[.25,1],xlim=[700,730],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)


signal_peakcalculate(c1,width=175,trim=[1000,1000])
signal_peakcalculate(c2,width=40,invert=True)
signal_peakcalculate(c3,width=1,invert=True,distance=1)




f_pylorus,an_pylorus,d=signalplot(dat,freq=[0.0005,15],xlim=[500,5000],skip_chan=[3,4,5,6,7],figsize=(10,7.5),spacer=40)

#%%
f_ekg,an_ekg,c3=signalplot(dat,freq=[5,1000],xlim=[704.5,705.5],skip_chan=[1,2,3,4,5,6,7],figsize=(4,4),spacer=.15,hide_y=True)

fig,ax=plt.subplots()
ax.plot(c3[:,0]-c3[:,0].min(),c3[:,1]*-1)

#%%

f_all.savefig('Output/HeatMap.svg',bbox_inches='tight',transparent=True)
f_slow.savefig('Output/SlowWave.svg',bbox_inches='tight',transparent=True)
f_res.savefig('Output/Res.svg',bbox_inches='tight',transparent=True)
f_ekg.savefig('Output/EKG.svg',bbox_inches='tight',transparent=True)
f_raw.savefig('Output/Raw.svg',bbox_inches='tight',transparent=True)
f_pylorus.savefig('Output/MMC.svg',bbox_inches='tight',transparent=True)


f_res,an_res,c=signalplot(dat,freq=[.25,5],xlim=[700,760],spacer=2)
f_ekg,an_ekg,c=signalplot(dat,freq=[5,1000],xlim=[700,710],spacer=.15)

f_ekg,an_ekg,c=signalplot(dat,freq=[5,1000],xlim=[500,5500],spacer=.20, skip_chan=[1,2,3,4,5,6,7],figsize=(10,4))


f_res,an_res,c=signalplot(dat,freq=[.25,5],xlim=[5000,5050],spacer=1)


#%% Measurement 2


f_ekg,an_ekg,c_0414_3=signalplot(dat0114,freq=[5,1000],xlim=[700,710],spacer=.15)
f_slow,an_slow,c_0414_1=signalplot(dat0114,freq=[.01,.25],xlim=[700,900],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)

f_res,an_res,c_0414_2=signalplot(dat0114,freq=[.25,5],xlim=[1200,1260],spacer=2)

signal_peakcalculate(c_0414_2,width=10,invert=False)
signal_peakcalculate(c_0414_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(c_0414_3,width=1,distance=1,threshold=0.005)



#%%  Measurement 3

f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[700,710],spacer=.15)
f_res,an_res,c_0520_2=signalplot(dat0520,freq=[.25,5],xlim=[700,760],spacer=3)
f_slow,an_slow,c_0520_1=signalplot(dat0520,freq=[.01,.25],xlim=[500,700],spacer=2)

signal_peakcalculate(c_0520_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(c_0520_2,width=10,invert=True)
signal_peakcalculate(c_0520_3,width=1,invert=True,distance=1,threshold=0.01)


#%% Measurement 4

f_slow,an_slow,c_0623_1=signalplot(dat0623,freq=[.01,.25],xlim=[800,1200])
signal_peakcalculate(c_0623_1,width=175,trim=[1000,1000],channel=5)

f_res,an_res,c_0623_2=signalplot(dat0623,freq=[.25,1],xlim=[700,760],spacer=3)
signal_peakcalculate(c_0623_2,width=15,invert=True,trim=[100,500],channel=3)

f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[800,810],spacer=.15)
signal_peakcalculate(c_0520_3,width=1,invert=True,distance=1,threshold=0.008)

#%% Measurement 
f_slow,an_slow,c_0629_1=signalplot(dat0629,freq=[.01,.25],xlim=[800,1200])

#%%  Measurement 5
f_slow,an_slow,c_0707_1=signalplot(dat0707,freq=[.01,.25],xlim=[1500,2000])
signal_peakcalculate(c_0707_1,width=400,trim=[1000,1000],channel=2)

f_res,an_res,c_0707_2=signalplot(dat0707,freq=[.25,5],xlim=[1500,1560],spacer=3)
signal_peakcalculate(c_0707_2,width=15,trim=[100,500],channel=2)

f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[800,810],spacer=.15)



#%%2022.07.12
dat0707=read_egg_v3('2022.07.07_Az.txt',scale=150)
dat0712=read_egg_v3('2022.07.12_Az.txt',scale=150)
#%%

data1s1_path='2023.08.02_Full.txt'
data1s1=read_egg_v3(data1s1_path,scale=300)
#%%
fig_size_aa = (15/2,5/2)

data_to_plot = data1s1 #'../data/20230802_Full.txt'
xlim_to_plot = [2160, 2160+60*13]

fig1, fig1_ax, filtered_data0 = signalplot(data_to_plot, freq=[0.01, 0.25], xlim=xlim_to_plot)
#time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
#fig1.savefig(out_path + 't1_20230802_r_' + time_string + output_type, bbox_inches='tight', transparent=True)
fig1, fig1_ax,freq_data0 = egg_signalfreq(filtered_data0, rate=60.7)
plt.title("Day 0")
plt.show()
# time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
# fig1.savefig(out_path + 't1_20230802_' + time_string + output_type, bbox_inches='tight', transparent=True)

#%%

f_ekg,an_ekg,c_0414_3=signalplot(data_to_plot,freq=[5,1000],xlim=[2160,2190],spacer=.15)
f_slow,an_slow,c_0414_1=signalplot(data_to_plot,freq=[.01,.25],xlim=[2160,2590],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
f_res,an_res,c_0414_2=signalplot(data_to_plot,freq=[.25,5],xlim=[2160,2190],spacer=2)

signal_peakcalculate(c_0414_1,width=175,trim=[1000,1000],channel=0) #slow
signal_peakcalculate(c_0414_2,width=10,invert=False) #res
signal_peakcalculate(c_0414_3,width=1,distance=1,threshold=0.005) #ekg