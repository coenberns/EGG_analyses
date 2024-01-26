# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:01:42 2023

@author: Coen
"""
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import scipy.stats as stats
from pathlib import Path
import re
plt.style.use('default')

#%%
def extract_packets_batch(file):
    match = re.search(r'(\d+)(samps)', file.stem) #stem for filename without extension
    if match:
        n_samples = str(match.group(1))
    else: 
        print(f"No match for file:{file}")
        return n_samples

def extract_info_from_filename(file):
    # Extract the number of samples
    samples_match = re.search(r'(\d+)samps', file.stem)
    n_samples = samples_match.group(1) if samples_match else None

    # Extract the sleep period
    sleep_match = re.search(r'(\d+)(?:_?)(\d*)sec', file.stem)
    sleep_period = None
    if sleep_match:
        seconds = sleep_match.group(1)
        fraction = sleep_match.group(2)
        sleep_period = float(f"{seconds}.{fraction}") if fraction else int(seconds)
    else:
        print(f"No match for sleep period in file: {file}")
    
    return n_samples, sleep_period


def plot_ppkcurrent_bar(folder):
    """Plot the mean currents and ci for all files in the measurement directory, plus retreive statistics"""
    start_time = datetime.now()

    dir_path = Path(folder)
    files = list(dir_path.glob('*'))  # list of all files in directory
    
    dfs = [pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current']) for file in files]

    labels = [f"{n_samples} TX\n{sleep_period}s" for file in files for n_samples, sleep_period in [extract_info_from_filename(file)]]
    filtered_data = pd.DataFrame(columns =["dataframe", "current"])
    
    # Initialize statistics DataFrame
    statistics = pd.DataFrame(columns =["Tx scheme", "mean", "std", "ci lower", "ci upper"])

    for i, df in enumerate(dfs):
        df['time'] = df['time'] / 1000
        df['current'] = df['current'] / 1000
        df_subset = df.sample(frac=1)
        df_filtered = df_subset.query('current < 30')
        
        mean_value = df_filtered['current'].mean()
        std_value = df_filtered['current'].std()
        sample_size = len(df_filtered['current'])
        SEM = std_value/np.sqrt(sample_size)
        DoF = sample_size - 1
        ci_lower, ci_upper = stats.t.interval(0.9999, DoF, loc=mean_value, scale=SEM)

        #Fix warnings and create large datataframe
        df_filtered = df_filtered.copy()  # Fix for warnings
        df_filtered['dataframe'] = labels[i]
        filtered_data = pd.concat([filtered_data, df_filtered[['dataframe', 'current']]], ignore_index=True)
        
        statistics.loc[i] = [labels[i], mean_value, std_value, ci_lower, ci_upper]
        print("Mean: {:.6f}, Confidence Interval (99.99%): [{:.6f}, {:.6f}]".format(mean_value, ci_lower, ci_upper))     
    
    #BAR PLOT WITH STATS
    #plt.style.use('ggplot')    
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.bar(statistics['Tx scheme'], statistics['mean'], 
            yerr=[statistics['mean'] - statistics['ci lower'], statistics['ci upper'] - statistics['mean']],
            capsize=15,  # Cap size for the error bar
            color='skyblue')
    
    #Add line plot for means
    plt.plot(statistics['Tx scheme'], statistics['mean'], marker='o', color='red')
    
    plt.title('Mean currents of varying transmissions per duty cycle', size=20)
    plt.xlabel('Transmitted samples per duty cycle (#/sleep [s])', size=16)
    plt.ylabel('Mean current [mA]', size = 16)
    
    plt.xticks(size = 12)
    plt.yticks(size=12)
    plt.show()
    
    end_time = datetime.now()
    print('Total Duration: {}'.format(end_time - start_time))
    
    return(filtered_data, statistics)

#%%
def plot_ppkcurrent_violin(folder):
    """Violin plots for current distribution for all files in the measurement directory, plus retreive statistics"""
    start_time = datetime.now()

    dir_path = Path(folder)
    files = list(dir_path.glob('*'))  # list of all files in directory
    
    dfs = [pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current']) for file in files]

    labels = [f"{extract_packets_batch(file)}" for file in files]

    filtered_data = pd.DataFrame(columns =["dataframe", "current"])
    
    # Initialize statistics DataFrame
    statistics = pd.DataFrame(columns =["Tx scheme", "mean", "std", "ci lower", "ci upper"])

    for i, df in enumerate(dfs):
        df['time'] = df['time'] / 1000
        df['current'] = df['current'] / 1000
        df_subset = df.sample(frac=1)
        df_filtered = df_subset[df_subset['current']<30]
        
        mean_value = df_filtered['current'].mean()
        std_value = df_filtered['current'].std()
        sample_size = len(df_filtered['current'])
        SEM = std_value/np.sqrt(sample_size)
        DoF = sample_size - 1
        ci_lower, ci_upper = stats.t.interval(0.99, DoF, loc=mean_value, scale=SEM)

        #Fix warnings and create large datataframe
        df_filtered = df_filtered.copy()  # Fix for warnings
        df_filtered['dataframe'] = labels[i]
        filtered_data = pd.concat([filtered_data, df_filtered[['dataframe', 'current']]], ignore_index=True)
        
        statistics.loc[i] = [labels[i], mean_value, std_value, ci_lower, ci_upper]
        print("Mean: {:.6f}, Confidence Interval (99%): [{:.6f}, {:.6f}]".format(mean_value, ci_lower, ci_upper)) 
    
    # VIOLIN PLOT WITH DISTRIBUTIONS 
    fig, ax1 = plt.subplots(figsize = (14,6)) 
    
    sns.violinplot(x="dataframe", 
                    y="current", 
                    data=filtered_data, 
                    width=0.5, 
                    inner = "box",
                    scale = 'area')
    #sns.barplot(x="dataframe", y="current", data=filtered_data, ci="sd")
    
    ax1.set_ylabel('Current (mA)')
    ax1.set_xlabel("Transmissions per cycle")
    
    # Creating ticks to suit the values
    major_ticks = np.arange(0, 23, 2) 
    
    ax1.set_yticks(major_ticks)
    
    # Creating custom tick labels
    major_ticklabels = [str(tick) for tick in major_ticks]
    major_ticklabels[0] = 1


    # ax1.set_yticklabels(major_ticklabels)
    ax1.set_yticklabels(major_ticklabels)
    
    ax1.grid(which='major', alpha=0.15)
        
    plt.title("Current measurements for different Tx schemes")
    plt.ylim(-1,22)    
    
    plt.show()
    
    
    end_time = datetime.now()
    print('Total Duration: {}'.format(end_time - start_time))
    
    return(filtered_data, statistics)
# %%
def plot_ppkcurrent(file):

    df = pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current'])
    df_frame = df[(df['time'] >= 2847) & (df['time'] <= 4000)]
    print(df_frame)
    df_frame_max = df_frame['current'].max()
    df_frame_mean = df_frame['current'].mean()
    df_frame_min = df_frame['current'].min()
    print(df_frame_max)
    x = df['time'] / 1000
    y = df['current'] / 1000
    
    plt.figure(figsize=(12,8))
    plt.plot(x,y)
    plt.axhline(y=df_frame_max/1000,linestyle='dashed', color='r', label=f"Max = {df_frame_max/1000:.2f} mA")
    plt.axhline(y=df_frame_mean/1000,linestyle='dashed', color='g', label=f"Mean = {df_frame_mean/1000:.3f} mA")
    plt.axhline(y=df_frame_min/1000,linestyle='dashed', color='orange', label=f"Min = {df_frame_min:.2f} uA")    
    plt.xlim(2.847,4)
    plt.ylim(-1,21)
    plt.title('A complete duty cycle with 5 samples transmitted', size=20)
    plt.xlabel('Time [s]', size=16)
    plt.ylabel('Current [mA]', size=16)
    plt.legend(loc=6)
    plt.xticks(size=14)
    plt.yticks(size=14)

    
# %%
def stats_ppkcurrent(file):

    df = pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current'])
    statistics = pd.DataFrame(columns =["Mean", "STD","Min", "Max","ci lower", "ci upper", "Q1", "Q2", "Q3"])
    
    mean_value = df['current'].mean()
    max_value = df['current'].max()
    min_value = df['current'].min()
    std_value = df['current'].std()
    sample_size = len(df['current'])
    SEM = std_value/np.sqrt(sample_size)
    DoF = sample_size - 1
    ci_lower, ci_upper = stats.t.interval(0.99, DoF, loc=mean_value, scale=SEM)
    q1 = df['current'].quantile(0.25)
    q2 = df['current'].quantile(0.5)
    q3 = df['current'].quantile(0.75)
    statistics.loc[0] = [mean_value, std_value, min_value, max_value, ci_lower, ci_upper, q1, q2, q3]

    return df, statistics


# %%
def boxplot_ppkcurrent(folder):
    dir_path = Path(folder)
    files = list(dir_path.glob('*'))  # list of all files in directory
    
    dfs = [pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current']) for file in files]

    labels = [f"{n_samples} TX\n{sleep_period}s" for file in files for n_samples, sleep_period in [extract_info_from_filename(file)]]

    filtered_data = pd.DataFrame(columns =["dataframe", "current"])
    
    # Initialize statistics DataFrame
    statistics = pd.DataFrame(columns =["Tx scheme", "mean", "std", "ci lower", "ci upper"])

    for i, df in enumerate(dfs):
        df['time'] = df['time'] / 1000
        df['current'] = df['current'] / 1000
        df_subset = df.sample(frac=1)
        df_filtered = df_subset[df_subset['current']<30]
        
        mean_value = df_filtered['current'].mean()
        std_value = df_filtered['current'].std()
        sample_size = len(df_filtered['current'])
        SEM = std_value/np.sqrt(sample_size)
        DoF = sample_size - 1
        ci_lower, ci_upper = stats.t.interval(0.9999, DoF, loc=mean_value, scale=SEM)

        #Fix warnings and create large datataframe
        df_filtered = df_filtered.copy()  # Fix for warnings
        df_filtered['dataframe'] = labels[i]
        filtered_data = pd.concat([filtered_data, df_filtered[['dataframe', 'current']]], ignore_index=True)
        
        statistics.loc[i] = [labels[i], mean_value, std_value, ci_lower, ci_upper]
        print("Mean: {:.6f}, Confidence Interval (99.99%): [{:.6f}, {:.6f}]".format(mean_value, ci_lower, ci_upper))

    plt.boxplot(x=filtered_data['dataframe'],y=filtered_data['current'], sym='', labels=labels)

    return filtered_data, statistics




# %%
