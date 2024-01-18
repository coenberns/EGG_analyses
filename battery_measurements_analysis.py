
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, time
import pathlib 
from Plot_EGG import*
import time
import os
from functions_read_bursts import *
from pathlib import Path

#%% MEASUREMENTS MODE1 08/23
filepaths = []
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Temp raw mode 1\Full recs - no battmeas")
files=list(dir.glob('*'))
filepaths.extend(files)

elapsed_t = {}
times1={}
for i in range(len(filepaths)):
    file = filepaths[i]
    df, elapsed, secs = calculate_time(file)
    print("For file: ", file.stem, "the time was: ", elapsed)
    print("Time in seconds", secs)


# %% MEASUREMENTS MODE 1 11/07

filepaths = []
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 1 new")
files=list(dir.glob('*'))
filepaths.extend(files)

elapsed_t = {}
times1={}
for i in range(len(filepaths)):
    file = filepaths[i]
    df, elapsed, secs = calculate_time(file)
    print("For file: ", file.stem, "the time was: ", elapsed)
    print("Time in seconds", secs)

# %% MEASUREMENTS MODE 2  1.84s sleep - 10/23, 11/05
filepaths = []
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 2 - 1840ms sleep")
files=list(dir.glob('*'))
filepaths.extend(files)

elapsed_t = {}
times1={}
for i in range(len(filepaths)):
    file = filepaths[i]
    df, elapsed, secs = calculate_time(file)
    print("For file: ", file.stem, "the time was: ", elapsed)
    print("Time in seconds", secs)
# %% MEASUREMENTS MODE 2 1.84s sleep - 10/13
filepaths = []
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\Battery measurements\Mode 2 - no batt meas mcu")
files=list(dir.glob('*'))
filepaths.extend(files)

elapsed_t = {}
times1={}
for i in range(len(filepaths)):
    file = filepaths[i]
    df, elapsed, secs = calculate_time(file)
    print("For file: ", file.stem, "the time was: ", elapsed)
    print("Time in seconds", secs)



# %%
