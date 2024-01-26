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
#plt.style.use('default')

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

    # Extract the skipped samples
    skipped_match = re.search(r'_(\d+)skipped', file.stem)
    skipped_samples = int(skipped_match.group(1)) if skipped_match else 0
    
    return n_samples, sleep_period, skipped_samples

def plot_ppkcurrent_bar(folder):
    start_time = datetime.now()
    # TX\n{sleep_period}s
    dir_path = Path(folder)
    files = list(dir_path.glob('*'))  # list of all files in directory
    dfs = [pd.read_csv(file, header=0, dtype=float, delimiter=',', names=['time', 'current']) for file in files]
    #labels = [f"{n_samples}" for file in files for n_samples, sleep_period,_ in [extract_info_from_filename(file)]]

    # Initialize combined DataFrame and statistics DataFrame
    combined_data = pd.DataFrame()
    statistics = pd.DataFrame(columns=["Tx scheme","Sleep", "mean", "std", "ci lower", "ci upper"])
    #color_map = {'1s': 'skyblue', '1.84s': 'orange'}

    for i, df in enumerate(dfs):
        df['time'] = df['time'] / 1000
        df['current'] = df['current'] / 1000
        df_filtered = df.query('current < 30')

        mean_value = df_filtered['current'].mean()
        std_value = df_filtered['current'].std()
        sample_size = len(df_filtered['current'])
        SEM = std_value / np.sqrt(sample_size)
        DoF = sample_size - 1
        ci_lower, ci_upper = stats.t.interval(0.9999, DoF, loc=mean_value, scale=SEM)

        # Add scheme, mean current, and sleep_period to combined DataFrame
        n_samples, sleep_period, skipped_samples = extract_info_from_filename(files[i])
        combined_cat = f"sleep: {sleep_period}, skip: {skipped_samples}"
        scheme_data = pd.DataFrame({
            'Tx scheme': n_samples,
            'mean_current': mean_value,
            #'category': combined_cat
            'sleep_period': sleep_period,  # Add sleep_period here
            # 'n_skipped': skipped_samples
        }, index=[0])
        combined_data = pd.concat([combined_data, scheme_data], ignore_index=True)
        print(combined_data)

        statistics.loc[i] = [n_samples, sleep_period, mean_value, std_value, ci_lower, ci_upper]
        #print(f"Mean: {mean_value:.4f}, Confidence Interval (99.99%): [{ci_lower:.4f}, {ci_upper:.4f}]")
        combined_data['sleep_period'] = combined_data['sleep_period'].astype(str)

    #palette = sns.color_palette(palette=['#0c2340', '#0076c2', '#00a6d6'])
    # Plotting with seaborn
    sns.set_palette('tab10')
    hue_order_sleeps=['1.84','1']
    hue_order_combined = ["sleep: 1.84, skip: 1","sleep: 1.84, skip: 3", "sleep: 1, skip: 3"]
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Tx scheme', 
                y='mean_current', 
                hue='sleep_period', 
                data=combined_data,
                palette='tab10',
                hue_order=hue_order_sleeps)
    
    # Group the data by 'Tx scheme' and 'combined_category' and calculate the mean for each group
    group_means = combined_data.groupby(['Tx scheme', 'sleep_period']).mean().reset_index()

    # Plot lines connecting the means of each category
    for category in hue_order_sleeps:
        subset = group_means[group_means['sleep_period'] == category]
        plt.plot(subset['Tx scheme'], 
                subset['mean_current'], 
                marker='o', 
                markersize=7, 
                linestyle='dashed')
    #plt.plot(statistics['Tx scheme'], statistics['mean'], marker='o', color='red')
    #plt.title('Mean currents of varying transmissions per duty cycle', size=20)
    plt.xlabel('Transmitted samples per duty cycle', size=20)
    plt.ylabel('Mean current [mA]', size=20)
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.legend(title='Sleep Period [s]', fontsize=16, title_fontsize=18)
    plt.show()

    end_time = datetime.now()
    print(f'Total Duration: {end_time - start_time}')

    return combined_data, statistics

#%% MISCELANEOUS PLT.BARPLOT

        # # Extract sleep period for color
        # _, sleep_period = extract_info_from_filename(files[i])
        # sleep_period_str = f"{sleep_period:.2f}s"  # Convert to string and format to 2 decimal places
        # color = color_map.get(sleep_period_str, 'skyblue')  # Default to 'blue' if not found in map

        # # Add scheme, mean current, sleep_period, and color to combined DataFrame
        # scheme_data = pd.DataFrame({
        #     'Tx scheme': labels[i],
        #     'mean_current': mean_value,
        #     'sleep_period': sleep_period_str,
        #     'color': color
        # }, index=[0])
        # combined_data = pd.concat([combined_data, scheme_data], ignore_index=True)

    # plt.bar(statistics['Tx scheme'], statistics['mean'], 
    #     yerr=[statistics['mean'] - statistics['ci lower'], statistics['ci upper'] - statistics['mean']],
    #     capsize=15,  # Cap size for the error bar
    #     color=combined_data['color'])

#%% VIOLIN PLOT
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
# %% PLOT CURRENT
def plot_ppkcurrent(file, frame = False, t_low=66, t_high=67.5):
    sns.set_palette('deep')

    df = pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current'])
    df['time'] = df['time'] / 1000
    df['current'] = df['current'] / 1000

    x = df['time']
    y = df['current']

    y_max= y.max()
    y_mean= y.mean()
    y_min = y.min()
    
    plt.figure(figsize=(10,7))
    plt.plot(x,y)
    plt.axhline(y=y_max,linestyle='dashed', color='r', label=f"Max = {y_max:.4f} mA")
    plt.axhline(y=y_mean,linestyle='dashed', color='g', label=f"Mean = {y_mean:.4f} mA")
    plt.axhline(y=y_min,linestyle='dashed', color='orange', label=f"Min = {y_min:.4f} mA")    
    #plt.xlim(t_low,t_high)
    #plt.ylim(-1,2.5)
    #plt.title('A complete duty cycle with 5 samples transmitted', size=20)
    plt.xlabel('Time [s]', size=20)
    plt.ylabel('Current [mA]', size=20)
    plt.legend(loc=6, fontsize=16)
    plt.xticks(size=16)
    plt.yticks(size=16)
    plt.show()


    if frame == True:  
        plt.figure(figsize=(10,7))
        plt.plot(x,y)
        # plt.axhline(y=y_max,linestyle='dashed', color='r', label=f"Max = {y_max:.4f} mA")
        # plt.axhline(y=y_mean,linestyle='dashed', color='g', label=f"Mean = {y_mean:.4f} mA")
        # plt.axhline(y=y_min,linestyle='dashed', color='orange', label=f"Min = {y_min:.4f} mA") 
        plt.xlim(t_low,t_high)
        plt.ylim(-1, 21.8)
        #plt.title('A complete duty cycle with 5 samples transmitted', size=20)
        plt.xlabel('Time [s]', size=16)
        plt.ylabel('Current [mA]', size=16)
        # plt.legend(loc=9)
        plt.xticks(size=14)
        plt.yticks(size=14)
        plt.show()

    
# %% STATISTICS NORMAL
def stats_ppkcurrent(file):

    df = pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current'])
    statistics = pd.DataFrame(columns =["Mean", "STD","Min", "Max","ci lower", "ci upper", "Q1", "Q2", "Q3"])
    
    # df = df[df['current']<10000]
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

# %% STATISTICS CURRENT
def stats_ppkcurrent(file):

    df = pd.read_csv(file, header=0, dtype = float, delimiter=',', names=['time','current','channel'])
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
# %% BOXPLOT
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

    sns.boxplot(data=filtered_data, x='dataframe', y='current',fliersize=0, orient='v', showmeans=True)
    plt.xlabel('TX Scheme')
    plt.ylabel('Current [mA]')
    plt.ylim(-0.2,1.2)
    plt.show()

    return filtered_data, statistics


