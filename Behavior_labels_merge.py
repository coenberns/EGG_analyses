#%%
## Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import datetime
import glob
from os import listdir
from os.path import isfile, join
import scipy.stats as st
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn import preprocessing
# import mchmm as mc

import warnings
warnings.filterwarnings('ignore')

#%%
# names=['Time', 'Behavior', 'Category','status']
def read_behavior_data(folder, start_date, end_date, header=0, rate=14.93, scale=0, error=0, interp=0):
    beh = pd.DataFrame()
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)

    for file in folder:
        filename = os.path.basename(file)
        date = filename.split('_')[0]
        time = filename.split('_')[1]
        timestamp = pd.to_datetime(f'{date} {time[:2]}:{time[2:4]}:{time[4:]}')
        if (timestamp > start_date) & (timestamp < end_date):
            df = pd.read_csv(file, header=header, usecols=['Time', 'Behavior', 'Behavioral category','Behavior type'],
                                dtype={'Time': float, 'Behavior':str, 'Behavioral category':str, 'Behavior type': str})
            df['timestamps'] = timestamp + pd.to_timedelta(df['Time'], unit='s')
            #Filter out ambulation for now: 
            beh = pd.concat([beh, df], ignore_index=True)
        else:
            pass
    # print(beh['beh_label'].value_counts())
    # beh.set_index('beh_timestamp', inplace=True)
    return(beh)
    # for file in tqdm(behavior_folder):
    #     print(file)

#%%
folder_path_all=r"D:\Boris observations\all_boris_48hrs_rec"
folder_path_part1=r"D:\Boris observations\observations_part1_rec"
folder_path_part2=r"C:\Users\CoenBerns\OneDrive - Mass General Brigham\Documents\Thesis\Boris\Boris observations\08292023_part2rec_observations"
files=glob.glob(os.path.join(folder_path_part2,'*'))
print(files)
behavior = read_behavior_data(files, start_date='2023-08-28', end_date='2023-08-30')
# %%
def get_ethograms(df):
    ethograms = []
    # take the dataframe and convert to timestamps of the behaviors
    data_event = {}
    # filepath = df.iloc[1,1]
    # date = filepath.split('/')[-1].split('.')[0].split('_')[0]
    # time = filepath.split('/')[-1].split('.')[0].split('_')[1]
    # datetime = pd.to_datetime(f'{date} {time}')
    # df.time = df.time / 60 # time in mins
    for name, group in df.groupby('Behavior'):
        for index, row in group.iterrows():
            if row['Behavior type'] =='START':
                data_event['Behavior'] = row['Behavior']
                data_event['start_time'] = row['timestamps']
                data_event['Category'] = row['Behavioral category']
            elif 'START' in data_event and row['Behavior type'] == 'START':
                ethograms.append(data_event)
                data_event = {}
                data_event['Behavior'] = row['Behavior']
                data_event['start_time'] = row['timestamps']
                data_event['Category'] = row['Behavioral category']
            else:
                data_event['end_time'] = row['timestamps']
                ethograms.append(data_event)
                data_event = {}
    eth = pd.DataFrame.from_dict(ethograms)
    eth = eth.sort_values(by='start_time')
    # # CONVERT BEHAVIORS TO 4 CATEGORIES / simplifying the categories
    # # sleeping / resting
    # eth.loc[eth[‘behavior’] == ‘Resting’, ‘behavior’] = ‘Resting/Sleeping’
    # eth.loc[eth[‘behavior’] == ‘Sleeping’, ‘behavior’] = ‘Resting/Sleeping’
    # # feeding / drinking
    # eth.loc[eth[‘behavior’] == ‘Feeding: Pellets’, ‘behavior’] = ‘Feeding’
    # eth.loc[eth[‘behavior’] == ‘Drinking: Water Tap’, ‘behavior’] = ‘Drinking’
    # # active
    # eth.loc[eth[‘behavior’] == ‘Scratching: Cage’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Scratching: Ground’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Interacting with Others’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Feeding: Pedialyte’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Playing: Hanging Green Ball’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Playing: Hanging Gear 1’, ‘behavior’] = ‘Active’
    # eth.loc[eth[‘behavior’] == ‘Playing: Free Floating Green toy’, ‘behavior’] = ‘Active’
    return eth

eths = get_ethograms(behavior)


# %%
# def update_category(current, new):
#     if new == 'Feeding/Drinking':
#         return new
#     elif current == 'Inactive' and new == 'Active':
#         return new
#     else:
#         return current
    
def update_category(current, behavior, new_category):
    if pd.isna(behavior):
        # Return current category if behavior is NaN or None
        return current
    elif behavior == 'Resting':
        return 'Inactive'
    elif new_category == 'Feeding/Drinking':
        return new_category
    elif current is None and new_category == 'Active':
        return new_category
    else:
        return current if current is not None else 'Inactive'

def mergeDataAndEthograms(egg_data, behavior_df):
    #Postprocessing out the 'Ambulation' behavior, since it is wrong
    behavior_df.loc[behavior_df['Behavior'] == 'Ambulation', 'Behavior'] = None

    egg_data['Behavior'] = None
    egg_data['Category'] = None
    egg_data['beh_code'] = None
    egg_data['cat_code'] = None

    beh_code_map = {}
    cat_code_map = {}

    # iterate over the behavior_df to label the electroData
    for index, row in tqdm(behavior_df.iterrows(), total=behavior_df.shape[0]):
        # mask data points that fall within the behavior's time range from the ethograms created
        mask = (egg_data['datetime'] >= row['start_time']) & (egg_data['datetime'] <= row['end_time'])
        # multiple behavior instances are concetenated to a list
        egg_data.loc[mask, 'Behavior'] = egg_data.loc[mask, 'Behavior'].apply(lambda x: x if isinstance(x, list) else [])
        egg_data.loc[mask, 'Behavior'] = egg_data.loc[mask, 'Behavior'].apply(lambda x: x + [row['Behavior']])
        # egg_data.loc[mask, 'Category'] = egg_data.loc[mask, 'Category'].apply(
        #     lambda current: update_category(current, row['Category']))
        egg_data.loc[mask, 'Category'] = egg_data.loc[mask, 'Category'].apply(
            lambda current: update_category(current, row['Behavior'], row['Category']))
    # # # combine behaviors into a single string for each row
    # egg_data['Combined_Behavior'] = egg_data['Behavior'].apply(lambda x: ','.join(sorted(set(x))) if x else 'None')

    # Combine behaviors into a single string for each row, excluding empty entries based on 'Ambulation' 
    egg_data['Combined_Behavior'] = egg_data['Behavior'].apply(lambda x: ', '.join(sorted(set(filter(None, x)))) if x else 'No behavior')

    # Redefine only the '' as Ambulation, since then the behavior was JUST ambulation
    egg_data['Combined_Behavior'] = egg_data['Combined_Behavior'].replace('', 'Ambulation')

    #Replace the nan category of an '' behavior, so Ambulation with active category again
    egg_data.loc[egg_data['Combined_Behavior'] == 'Ambulation', 'Category'] = 'Active'

    #factorization combined beh
    codes, uniques = pd.factorize(egg_data['Combined_Behavior'])
    egg_data['beh_code'] = codes

    # get mapping of behaviors
    for unique, code in zip(uniques, range(len(uniques))):
        beh_code_map[unique] = code

    #factorize unique categories
    cat_codes, cat_uniques = pd.factorize(egg_data['Category'])
    egg_data['cat_code'] = cat_codes

    # get mapping of cats
    for unique, code in zip(cat_uniques, range(len(cat_uniques))):
        cat_code_map[unique] = code

    return egg_data, beh_code_map, cat_code_map

egg_beh_data_part2, beh_code_map, cat_code_map = mergeDataAndEthograms(egg_data_longer, eths)


# %%
behavior_codes_1D = egg_beh_data['beh_code']
cat_codes_1D = egg_beh_data['cat_code']

# %%
