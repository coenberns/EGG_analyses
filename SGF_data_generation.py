
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, timedelta
import os
from pathlib import Path
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from numpy.polynomial.polynomial import Polynomial
from functions_read_bursts import *

filepaths = []
#Initial analysis folder!!
dir = Path(r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\SGF Tests\Full SGF tests")
files=list(dir.glob('*'))
filepaths.extend(files)

elapsed_times = {}

for i in range(len(filepaths)):
    file = filepaths[i]
    _,elapsed_t,_ = calculate_time(file)

    elapsed_times[i]=elapsed_t

    print("For file:", file.stem)

print(elapsed_times)
# %%
