# Created by cberns 12/4/2023

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:05:12 2023

@author: coenberns
"""
#%%
import sys
# sys.path.append('C:/Users/seany/Dropbox/Langer Lab/Data/EGG_Data_Repository/')
from Old_Plot_EGG import * 
from functions_read_bursts import *
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

#%% Gap analysis files - depending on wanting full (1) previous recording or segmented parts (2)
gap1 = 'gap_df_48hrs_14sec.pkl'
gap2 = 'gap_df_segments_14sec.pkl'
#%%
dat,down= downsample_to_burst('2022.05.02_Combined.txt',scale=150, time_str='2S')
#%%
dat0114,down0114= downsample_to_burst('2022.04.14_Terminal.txt',scale=150, time_str='2S')
dat0520,down0520= downsample_to_burst('2022.05.20_Full_End_Clipped.txt',scale=150,time_str='2S')
#%%
dat0623,down0623= downsample_to_burst('2022.06.23_Az.txt',scale=150,time_str='2S')
dat0629,down0629= downsample_to_burst('2022.06.29_Az.txt',scale=150,time_str='2S')
dat0707,down0707= downsample_to_burst('2022.07.07_Az.txt',scale=150,time_str='2S')

#%%
down_gaps = distribute_gaps(down,gap_file=gap2,sec_gap=14,t_cycle=2)
down_gaps_ip = interpolate_egg_v3(down_gaps, time=True)
#%%
a1,b1,c1 = signalplot(dat,time='timestamps', output='PD', order=3, freq=[.02,.2], spacer=50)
a2,b2,c2 = signalplot(down_gaps_ip, time='timestamps', output='PD', rate=62.5, order=3,freq=[.02,.2], spacer=50)
#%%
dat_filt = butter_filter(dat, fs=62.5)
down_filt = butter_filter(down_gaps_ip, fs=.5)
#%%
merged_dats = pd.merge_asof(down_filt, dat_filt, direction='nearest', on='timestamps', suffixes=('_downsampled', '_actual'))
#%%
down0114_gaps = distribute_gaps(down0114,gap_file=gap2,sec_gap=14,t_cycle=2)
down0114_ip = interpolate_egg_v3(down0114_gaps)

down0520_gaps = distribute_gaps(down0520, gap_file=gap2 ,sec_gap=14,t_cycle=2)
down0520_ip = interpolate_egg_v3(down0520_gaps)
#%%
down0623_gaps = distribute_gaps(down0623,gap_file=gap2,sec_gap=14,t_cycle=2)
down0623_ip = interpolate_egg_v3(down0623_gaps)

down0629_gaps=distribute_gaps(down0629,gap_file=gap2,sec_gap=14,t_cycle=2)
down0629_ip=interpolate_egg_v3(down0629_gaps)

down0707_gaps=distribute_gaps(down0707,gap_file=gap2,sec_gap=14,t_cycle=2)
down0707_ip=interpolate_egg_v3(down0707_gaps)


#%%Heat plot

f_all,an_all,c=heatplot(dat,freq=[0.0005,15],xlim=[500,5000])
fd_all,and_all,cd=heatplot(down,freq=[0.0005,15], xlim=[500,5000])
fd_all,and_all,cd=heatplot(down_gaps_ip,freq=[0.0005,15], xlim=[500,5000])

#%%Propagation Speed

f,an,c=signalplot(dat,freq=[.02,.25],xlim=[700,850],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(15,10),spacer=5)
fd,an_d,cd=signalplot(down,freq=[.02,.25],xlim=[700,850],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(15,10),spacer=5)
fd,an_d,cd=signalplot(down_gaps_ip,freq=[.02,.25],xlim=[700,850],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(15,10),spacer=5)

#%%
# f,an,c=signalplot(dat,freq=[.25,5],xlim=[710,730],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(12,8))
# f,an,c=signalplot(dat,freq=[8,30],xlim=[720,725],skip_chan=['Channel 0','Channel 1','Channel 2','Channel 3','Channel 4'],figsize=(12,8))

#%%
f_slow,an_slow,c1=signalplot(dat,freq=[.01,.25],xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)
fd_slow,an_d_slow,c_d1=signalplot(down,freq=[.01,.25],xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)
fd_slow,an_d_slow,c_d12=signalplot(down_gaps_ip,freq=[.01,.25],xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)


f_raw,an_raw,c4=signalplot(dat,xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True, spacer=20)
fd_raw,an_d_raw,c_d4=signalplot(down,xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True, spacer=20)
fd_raw,an_d_raw,c_d4=signalplot(down_gaps_ip,xlim=[700,900],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True, spacer=20)

signal_peakcalculate(c1,width=175,trim=[1000,1000])
signal_peakcalculate(c_d1,width=175,trim=[1000,1000])
signal_peakcalculate(c_d12,width=175,trim=[1000,1000])


f_pylorus,an_pylorus,d=signalplot(dat,freq=[0.0005,15],xlim=[500,5000],skip_chan=[3,4,5,6,7],figsize=(10,7.5),spacer=40)
fd_pylorus,and_pylorus,dd=signalplot(down,freq=[0.0005,15],xlim=[500,5000],skip_chan=[3,4,5,6,7],figsize=(10,7.5),spacer=40)
fd_pylorus,and_pylorus,dd=signalplot(down_gaps_ip,freq=[0.0005,15],xlim=[500,5000],skip_chan=[3,4,5,6,7],figsize=(10,7.5),spacer=40)

# Not interesting for analysis
# f_res,an_res,c2=signalplot(dat,freq=[.25,5],xlim=[700,730],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
# f_ekg,an_ekg,c3=signalplot(dat,freq=[5,1000],xlim=[700,710],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),spacer=.15,hide_y=True)
# f_res,an_res,c2=signalplot(dat,freq=[.25,1],xlim=[700,730],skip_chan=[0,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
# signal_peakcalculate(c2,width=40,invert=True)
# signal_peakcalculate(c3,width=1,invert=True,distance=1)

# #%%
# f_ekg,an_ekg,c3=signalplot(dat,freq=[5,1000],xlim=[704.5,705.5],skip_chan=[1,2,3,4,5,6,7],figsize=(4,4),spacer=.15,hide_y=True)

# fig,ax=plt.subplots()
# ax.plot(c3[:,0]-c3[:,0].min(),c3[:,1]*-1)

#%%

f_all.savefig('Output/HeatMap.svg',bbox_inches='tight',transparent=True)
fd_all.savefig('Output/HeatMap_DS.svg',bbox_inches='tight',transparent=True)
f_slow.savefig('Output/SlowWave.svg',bbox_inches='tight',transparent=True)
fd_slow.savefig('Output/SlowWave_DS.svg',bbox_inches='tight',transparent=True)
f_raw.savefig('Output/Raw.svg',bbox_inches='tight',transparent=True)
fd_raw.savefig('Output/Raw_DC.svg',bbox_inches='tight',transparent=True)
f_pylorus.savefig('Output/MMC.svg',bbox_inches='tight',transparent=True)
fd_pylorus.savefig('Output/MMC_DC.svg',bbox_inches='tight',transparent=True)


# f_res.savefig('Output/Res.svg',bbox_inches='tight',transparent=True)
# f_ekg.savefig('Output/EKG.svg',bbox_inches='tight',transparent=True)

# f_res,an_res,c=signalplot(dat,freq=[.25,5],xlim=[700,760],spacer=2)
# f_ekg,an_ekg,c=signalplot(dat,freq=[5,1000],xlim=[700,710],spacer=.15)

# f_ekg,an_ekg,c=signalplot(dat,freq=[5,1000],xlim=[500,5500],spacer=.20, skip_chan=[1,2,3,4,5,6,7],figsize=(10,4))


# f_res,an_res,c=signalplot(dat,freq=[.25,5],xlim=[5000,5050],spacer=1)


#%% Measurement 2


# f_res,an_res,c_0414_2=signalplot(dat0114,freq=[.25,5],xlim=[1200,1260],spacer=2)
# f_ekg,an_ekg,c_0414_3=signalplot(dat0114,freq=[5,1000],xlim=[700,710],spacer=.15)
f_slow,an_slow,c_0414_1=signalplot(dat0114,freq=[.01,.25],xlim=[700,900],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
fd_slow,and_slow,cd_0414_1=signalplot(down0114,freq=[.01,.25],xlim=[700,900],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)
fd_slow,and_slow,cd_0414_12=signalplot(down0114_ip,freq=[.01,.25],xlim=[700,900],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=2)



# signal_peakcalculate(c_0414_2,width=10,invert=False)
signal_peakcalculate(c_0414_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(cd_0414_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(cd_0414_12,width=175,trim=[1000,1000],channel=0)

# signal_peakcalculate(c_0414_3,width=1,distance=1,threshold=0.005)



#%%  Measurement 3

# f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[700,710],spacer=.15)
# f_res,an_res,c_0520_2=signalplot(dat0520,freq=[.25,5],xlim=[700,760],spacer=3)
# signal_peakcalculate(c_0520_2,width=10,invert=True)
# signal_peakcalculate(c_0520_3,width=1,invert=True,distance=1,threshold=0.01)


f_slow,an_slow,c_0520_1=signalplot(dat0520,freq=[.01,.25],xlim=[500,700],spacer=2)
fd_slow,and_slow,cd_0520_1=signalplot(down0520,freq=[.01,.25],xlim=[500,700],spacer=2)
fd_slow,and_slow,cd_0520_12=signalplot(down0520_ip,freq=[.01,.25],xlim=[500,700],spacer=2)

signal_peakcalculate(c_0520_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(cd_0520_1,width=175,trim=[1000,1000],channel=0)
signal_peakcalculate(cd_0520_12,width=175,trim=[1000,1000],channel=0)





#%% Measurement 4

f_slow,an_slow,c_0623_1=signalplot(dat0623,freq=[.01,.25],xlim=[800,1200])
fd_slow, and_slow, cd_0623_1 = signalplot(down0623,freq=[.01,.25],xlim=[800,1200])
fd_slow,and_slow,cd_0623_12=signalplot(down0623_ip,freq=[.01,.25],xlim=[800,1200])

signal_peakcalculate(c_0623_1,width=175,trim=[1000,1000],channel=5)
signal_peakcalculate(cd_0623_1,width=175,trim=[1000,1000],channel=5)
signal_peakcalculate(cd_0623_12,width=175,trim=[1000,1000],channel=5)



# f_res,an_res,c_0623_2=signalplot(dat0623,freq=[.25,1],xlim=[700,760],spacer=3)
# signal_peakcalculate(c_0623_2,width=15,invert=True,trim=[100,500],channel=3)

# f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[800,810],spacer=.15)
# signal_peakcalculate(c_0520_3,width=1,invert=True,distance=1,threshold=0.008)

#%% Measurement 
f_slow,an_slow,c_0629_1=signalplot(dat0629,freq=[.01,.25],xlim=[800,1200])
fd_slow,and_slow,cd_0629_1=signalplot(down0629,freq=[.01,.25],xlim=[800,1200])
fd_slow,and_slow,cd_0629_1=signalplot(down0629_ip,freq=[.01,.25],xlim=[800,1200])



#%%  Measurement 5
f_slow,an_slow,c_0707_1=signalplot(dat0707,freq=[.01,.25],xlim=[1500,2000])
f_slow,and_slow,cd_0707_1=signalplot(down0707,freq=[.01,.25],xlim=[1500,2000])
f_slow,and_slow,cd_0707_12=signalplot(down0707_ip,freq=[.01,.25],xlim=[1500,2000])
signal_peakcalculate(c_0707_1,width=400,trim=[1000,1000],channel=2)
signal_peakcalculate(cd_0707_1,width=400,trim=[1000,1000],channel=2)
signal_peakcalculate(cd_0707_12,width=400,trim=[1000,1000],channel=2)


# f_res,an_res,c_0707_2=signalplot(dat0707,freq=[.25,5],xlim=[1500,1560],spacer=3)
# signal_peakcalculate(c_0707_2,width=15,trim=[100,500],channel=2)

# f_ekg,an_ekg,c_0520_3=signalplot(dat0520,freq=[5,1000],xlim=[800,810],spacer=.15)



#%%2022.07.12
# dat0712=read_egg_v3('2022.07.12_Az.txt',scale=150)
dat0712,down0712=downsample_to_burst('2022.07.12_Az.txt',scale=150, time_str='2S')
down0712_gaps=distribute_gaps(down0712, gap_file=gap2,t_cycle=2)
down0712_ip=interpolate_egg_v3(down0712_gaps)
#%%

data1s1_path='2023.08.02_Full.txt'
# data1s1=read_egg_v3(data1s1_path,scale=300)
data1s1,down1s1=downsample_to_burst('2023.08.02_Full.txt',scale=300, time_str='2S')
down1s1_gaps=distribute_gaps(down1s1, gap_file=gap2,t_cycle=2)
down1s1_ip=interpolate_egg_v3(down1s1_gaps)

#%%
fig_size_aa = (15/2,5/2)

data_to_plot = data1s1 #'../data/20230802_Full.txt'
down_to_plot = down1s1
down_to_plot_ip = down1s1_ip
xlim_to_plot = [2160, 2160+60*13]

fig1, fig1_ax, filtered_data0 = signalplot(data_to_plot, freq=[0.01, 0.25], xlim=xlim_to_plot)

fig2, fig2_ax, filtered_data1 = signalplot(down_to_plot, freq=[0.01,0.25], xlim=xlim_to_plot)
fig2, fig2_ax, filtered_data2 = signalplot(down_to_plot_ip, freq=[0.01,0.25], xlim=xlim_to_plot)
#time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
#fig1.savefig(out_path + 't1_20230802_r_' + time_string + output_type, bbox_inches='tight', transparent=True)
fig1, fig1_ax,freq_data0 = egg_signalfreq(filtered_data0, rate=60.7)
fig2,fig2_ax,freq_data1 = egg_signalfreq(filtered_data1, rate=60.7)
fig2,fig2_ax,freq_data1 = egg_signalfreq(filtered_data2, rate=60.7)
plt.title("Day 0")
plt.show()
# time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
# fig1.savefig(out_path + 't1_20230802_' + time_string + output_type, bbox_inches='tight', transparent=True)

#%%

# f_ekg,an_ekg,c_0414_3=signalplot(data_to_plot,freq=[5,1000],xlim=[2160,2190],spacer=.15)
# f_res,an_res,c_0414_2=signalplot(data_to_plot,freq=[.25,5],xlim=[2160,2190],spacer=2)
f_slow,an_slow,c_0414_1=signalplot(data_to_plot,freq=[.01,.25],xlim=[2160,2590],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)
fd_slow,and_slow,cd_0414_1=signalplot(down_to_plot,freq=[.01,.25],xlim=[2160,2590],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)
fd_slow,and_slow,cd_0414_12=signalplot(down_to_plot_ip,freq=[.01,.25],xlim=[2160,2590],skip_chan=[1,2,3,4,5,6,7],figsize=(10,4),hide_y=True,spacer=10)

signal_peakcalculate(c_0414_1,width=175,trim=[1000,1000],channel=0) #slow
signal_peakcalculate(cd_0414_1,width=175,trim=[1000,1000],channel=0) #slow
signal_peakcalculate(cd_0414_12,width=175,trim=[1000,1000],channel=0) #slow


# signal_peakcalculate(c_0414_2,width=10,invert=False) #res
# signal_peakcalculate(c_0414_3,width=1,distance=1,threshold=0.005) #ekg
# %%
