## Main script to run for data analysis of MiGUT data


## Imports
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math
import glob
import matplotlib as mpl
# from tqdm.notebook import tqdm
from datetime import datetime
from os.path import exists
from scipy import signal
from pathlib import Path
import scipy
import time
import datetime


# from Plot_EGG_20230924 import *
from AG_Analysis_Funcs import *

#%%

output_type = '.pdf'
out_path = './Output/fig5_new/'


#%%
path_d0='../data/2022.08.17_Full.txt'

data0=read_egg_v3(path_d0,scale=600, interp=1)
#%%
fig1,fig1_ax,filtered_data0=signalplot(data0,freq=[0.01, 0.25], vline=[120, 5100, 7620, 10020])
plt.show()
fig1.savefig('Output/fig5_new/fig5a_longmeas.pdf',bbox_inches='tight',transparent=True)

#%%
fig1,fig1_ax,filtered_data0=signalplot(data0,freq=[0.01, 0.25], vline=[120, 3720, 4320,5100, 7620, 10020, 11700])
plt.show()
fig1.savefig('Output/fig5_new/fig5a_longmeas.pdf',bbox_inches='tight',transparent=True)

#%%
final_time = data0["timestamps"].iloc[-1]

# f,a,d=egg_freq_heatplot(data0, xlim=[0,final_time-10], freqlim=[1,8], freq=[0.1,0.2])
f,a,d=egg_freq_heatplot(data0, xlim=[4000,6000], freqlim=[1,8], freq=[0.01,0.2], vrange=[0, 0.6])
plt.show()
f.savefig('Output/fig5_new/fig5b_freq.pdf',bbox_inches='tight',transparent=True)

#%%


path_d0 = '../data/2022.08.17_Full.txt'


data0 = read_egg_v3(path_d0, scale=600)


#%% Final day 0
fig_size_aa = None#(15,3)

# fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[8000,8250],  spacer=60)
fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[1750,2000],  spacer=50, skip_chan=[0,1,2,3,4,5,6], figsize=fig_size_aa)

# fig1, fig1_ax,freq_data0 = egg_signalfreq(filtered_data0, rate=60.7, freqlim=[1.5,10],chan_to_plot=[2], figsize_val=fig_size_aa)
# plt.title("Feeding")
plt.show()
time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
fig1.savefig(out_path + 'f5c_eating_notitle_' + time_string + output_type, bbox_inches='tight', transparent=True)


# fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[8000,8250],  spacer=60)
fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[8000,8250],  spacer=50, skip_chan=[0,1,2,3,4,5,6], figsize=fig_size_aa)

# fig1, fig1_ax,freq_data0 = egg_signalfreq(filtered_data0, rate=60.7, freqlim=[1.5,10],chan_to_plot=[2], figsize_val=fig_size_aa)
# plt.title("Sleeping")
plt.show()
time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
fig1.savefig(out_path + 'f5c_sleeping_notitle_' + time_string + output_type, bbox_inches='tight', transparent=True)


# fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[8000,8250],  spacer=60)
fig1, fig1_ax, filtered_data0 = signalplot(data0, freq=[0.01, 0.25], xlim=[6000,6250],  spacer=50, skip_chan=[0,1,2,3,4,5,6], figsize=fig_size_aa)

# fig1, fig1_ax,freq_data0 = egg_signalfreq(filtered_data0, rate=60.7, freqlim=[1.5,10],chan_to_plot=[2], figsize_val=fig_size_aa)
# plt.title("Ambulating")
plt.show()
time_string = datetime.datetime.now().strftime("%y%m%d-%H%S%f")
fig1.savefig(out_path + 'f5c_amb_notitle_' + time_string + output_type, bbox_inches='tight', transparent=True)


#%%

from Plot_EGG import *


#%%
final_time = data0["timestamps"].iloc[-1]

# f,a,d=egg_freq_heatplot(data0, xlim=[0,final_time-10], freqlim=[1,8], freq=[0.1,0.2])
f,a,d=egg_freq_heatplot(data0, xlim=[0,final_time-10], freqlim=[1,8], freq=[0.01,0.2], vrange=[0, 0.55], )
plt.show()
f.savefig('Output/fig5_new/fig5b_freq.pdf',bbox_inches='tight',transparent=True)
