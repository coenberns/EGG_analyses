#File for determination of gap sizes in data, including CDF and weighted CDF plots
#%%
"""
@author: coenberns
11/29/2023 
"""
import sys
sys.path.append('/Users/coenberns/Documents_lab/Thesis/Coding/Ephys/Thesis Python')

import pandas as pd
import matplotlib.pyplot as plt
import datetime as datetime
import pathlib 
import seaborn as sns
from functions_read_bursts import*
from Old_Plot_EGG import*

#%%

def gap_distribution_plot(path, fontsize=18):
    plt.rcParams['font.size'] == fontsize
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

path_45 = r'/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 EDC1_2'
# path_week = r'/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/01042024_multiweek'
data_45, data_w_45 = gap_distribution_plot(path=path_45)
# data_week, data_w_week = gap_distribution_plot(path=path_week)
