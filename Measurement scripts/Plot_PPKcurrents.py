# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:01:42 2023

@author: Coen
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats


file_transfreq = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\6282023_Vary_packetfreqs_5minrec_2minsleep_ppkmeas"
file_lowP1 = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\6292023_PPKmeas_dcsupplyV_8samples_1srest_1.05mA_avg"
file_lowP2 = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\6302023_skip3_rec8_sleep1s_ppkmeas_file"
lowP_5rec = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\07122023_5samples1sec_current1hr_0_1msstep"
lowP_5rec2 = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\07122023_5samples1sec_current1hr_1msstep"
lowP_3rec = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\07132023_3samples1sec_30mincurrent_01msstep"
lowP_1rec = r"C:\Users\Coen\OneDrive - Mass General Brigham\Documents\Thesis\Measurements\PPKII\07132023_1samples1sec_30mincurrent_01msstep"

def plot_ppkcurrents(file1, file2, file3, file4, file5, file6, file7):
    
    start_time = datetime.now()

    df1 = pd.read_csv(file1, header=0, dtype = float, delimiter=',', names=['time','current', 'bliep'])
    df2 = pd.read_csv(file2, header=0, dtype = float, delimiter=',', names=['time','current', 'bloep'])
    df3 = pd.read_csv(file3, header=0, dtype = float, delimiter=',', names=['time','current'])
    df4 = pd.read_csv(file4, header=0, dtype = float, delimiter=',', names=['time','current'])
    df5 = pd.read_csv(file5, header=0, dtype = float, delimiter=',', names=['time','current'])
    df6 = pd.read_csv(file6, header=0, dtype = float, delimiter=',', names=['time','current'])
    df7 = pd.read_csv(file7, header=0, dtype = float, delimiter=',', names=['time','current'])                                                                           
    
    dfs = [df1, df2, df3, df4, df5, df6, df7]
    labels = ["Varying packet \nsend-frequency", 
              "Rapid S/W 7 recs \n[delta 0.1ms] (1)", 
              "Rapid S/W 7 recs \n[delta 0.1ms] (2)", 
              "Rapid S/W 5 recs \n[delta 0.1ms] (1)",
              "Rapid S/W 5 recs \n[delta 1ms] (2)",
              "Rapid S/W 3 recs \n[delta 0.1ms]",
              "Rapid S/W 1 recs \n[delta 0.1ms]"]
    
    filtered_data = pd.DataFrame(columns =["dataframe", "current"])
        
    for i, df in enumerate(dfs):
        
        df['time'] = df['time'] / 1000
        
        # Convert current to milliamperes (mA)
        df['current'] = df['current'] / 1000
        
        df_subset = df.sample(frac=1)
        if i == 0: 
            df_filtered = df_subset.query('time < 300 and current < 22') #otherwise sleeping period is also incorporated in mean etc.
            
        else:
            df_filtered = df_subset.query('current < 22') #current higher than 21 was caused by measurements made with multimeter
        #df_filtered_subs = df_filtered.sample(frac=1)
        
        
        
        # Statistical evaluation
        confidence = 0.99
        mean_value = df_filtered['current'].mean()
        std_value = df_filtered['current'].std()
        sample_size = len(df_filtered['current'])
        SEM = std_value/np.sqrt(sample_size)
        DoF = sample_size - 1
        ci_lower, ci_upper = stats.t.interval(confidence, DoF, loc=mean_value, scale=SEM)
        

        #Giving the correct labels to the dataframe column        
        df_filtered.loc[:, 'dataframe'] = labels[i]
        
        filtered_data = pd.concat([filtered_data, df_filtered[['dataframe', 'current']]], ignore_index=True)
        
        print("Mean: {:.6f}, Confidence Interval ({:.0%}): [{:.6f}, {:.6f}], standard deviation: {:.6f}".format(mean_value, confidence, ci_lower, ci_upper, std_value))  
        
    # fig, ax1 = plt.subplots(figsize = (14,6)) 
    
    # sns.violinplot(x="dataframe", 
    #                 y="current", 
    #                 data=filtered_data, 
    #                 width=0.5, 
    #                 inner = "box",
    #                 scale = 'area')
    # #sns.barplot(x="dataframe", y="current", data=filtered_data, ci="sd")
    
    # ax1.set_ylabel('Current (mA)')
    # ax1.set_xlabel("Transmission scheme")
    
    # # Creating ticks to suit the values
    # major_ticks = np.arange(0, 21, 2) 
    
    # ax1.set_yticks(major_ticks)
    
    # # Creating custom tick labels
    # major_ticklabels = [str(tick) for tick in major_ticks]
    # major_ticklabels[0] = 1


    # # ax1.set_yticklabels(major_ticklabels)
    # ax1.set_yticklabels(major_ticklabels)
    
    # ax1.grid(which='major', alpha=0.15)
        
    # # Adding title
    # plt.title("Current measurements for different Tx schemes")
    # plt.ylim(-1,21)    
    # # show plot
    # plt.show() 
    
    end_time = datetime.now()
    print('Total Duration: {}'.format(end_time - start_time))
    
    return(filtered_data)

#############################
#Statistical analysis


# ##ANOVA ONE-WAY TEST 
# filtered_data = plot_ppkcurrents(file_transfreq, file_lowP1, file_lowP2)

# #Get the current values for each group
# group1 = filtered_data[filtered_data['dataframe'] == 'Varying packet send-frequency']['current']
# group2 = filtered_data[filtered_data['dataframe'] == 'Rapid S/W 7 recs [delta 0.1ms] (1)']['current']
# group3 = filtered_data[filtered_data['dataframe'] == 'Rapid S/W 7 recs [delta 0.1ms] (2)']['current']

# # Perform one-way ANOVA
# f_val12, p_val12 = stats.f_oneway(group1, group2)
# f_val13, p_val13 = stats.f_oneway(group1, group3)
# f_val23, p_val23 = stats.f_oneway(group2, group3)

# print('F value between group 1&2, 1&3, and 2&3, respectively:', f_val12,',', f_val13,',', f_val23)
# print('p-value between group 1&2, 1&3, and 2&3, respectively:', p_val12,',', p_val13,',', p_val23)




#######################################
# OLD CODE    
    # fig, ax1 = plt.subplots(figsize = (12,8))
    
    # # Line plot
    # sns.lineplot(data=df_filtered, x='time', y='current', ax=ax1)
    # ax1.set_ylabel('Current (mA)')
    # ax1.set_xlabel("Time (s)")
    

    
    # fig.legend(loc='upper right')
    
    # # Annotating the values in the plot at the lines
    # ax2.annotate(f'{min_value:.4f} mA', xy=(0, min_value), xytext=(210, 7),
    #              textcoords='offset points', color='r', ha='left', va='center')
    # ax2.annotate(f'{max_value:.4f} mA', xy=(0, max_value), xytext=(530, -10),
    #              textcoords='offset points', color='g', ha='left', va='center')
    # ax2.annotate(f'{mean_value:.4f} mA', xy=(0, mean_value), xytext=(530, 7),
    #              textcoords='offset points', color='b', ha='left', va='center')
    
    # plt.show()
    
