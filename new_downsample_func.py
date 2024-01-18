#%%
def downsample_to_burst(file,time_str='2S',scale=150, date=None, round=False):
    """
    Sample down a 62.5 SPS recording to a predefined period
    :
    :param file: raw recording file (.txt file) 
    :param time_str: The new period in down-sampling; '2S' is data every 2 seconds
    :param scale: +- scale in mV 
    :param date: The date of the recording, if None; it is gotten from the filename
    :param round: Round off the first datetime object to the time_str variable
    :return fulldat: Initial full dataframe, with pseudotime included
    :return resampled_interp: The resampled/downsampled full dataset interpolated for nan values
    : 
    """
    if date is None:
        # get only the filename from the file path
        file_path = pathlib.Path(file)
        filename = file_path.name
        # extract date from the filename
        date = filename.split('_')[0]

    # Creating datetime object
    base_date = datetime.strptime(date, '%Y.%m.%d')
    fulldat = read_egg_v3(file,scale=scale)
    # base_date2 = pd.Timestamp('2023-09-21')
    fulldat['realtime'] = fulldat['realtime'].str.strip()
    fulldat['realtime'] = pd.to_timedelta(fulldat['realtime'])
    if round == True:
        base_time = base_date + fulldat['realtime'].iloc[0].round(time_str)  # Add first realtime value to base_date
    else:
        base_time = base_date + fulldat['realtime'].iloc[0]
    fulldat['date_time'] = pd.to_timedelta(fulldat['timestamps'], unit='s') + base_time  # Add to base_time
    fulldat['timedelta'] = pd.to_timedelta(fulldat['timestamps'], unit='s')
    fulldat.set_index('date_time', inplace=True)
    datcols = ['timestamps', 'timedelta']+[f'Channel {i}' for i in range(8)]
    fulldat_short = fulldat[datcols]

    new_index = pd.date_range(start=fulldat_short.index.min(), end=fulldat_short.index.max(), freq=time_str)
        # Reindex the DataFrame with the new index
    fulldat_reindexed = fulldat_short.reindex(new_index)

    # Perform the resampling
    # The burst resampler should handle NaNs properly as discussed previously
    resampled_fulldat = fulldat_reindexed.resample(time_str, label='left').apply(burst_resampler)

    # Reset the index if necessary
    resampled_fulldat.reset_index(inplace=True)

    
    # Resample and apply the custom function - burst resampler relies on n_burst=5 and rate=62.5
    # resampled_fulldat = fulldat_short.resample(time_str, label='left').apply(burst_resampler)
    # Reset index to return pseudotime
    # resampled_fulldat.reset_index(inplace=True)
    #resampled_fulldat['timestamps'] = fill_timestamps(resampled_fulldat['timestamps'])
    # resampled_fulldat['timestamps']=(resampled_fulldat['pseudo_time'] - resampled_fulldat['pseudo_time'].iloc[0]).dt.total_seconds()
    resampled_interp = interpolate_egg_v3(resampled_fulldat, method='cubicspline')

    return fulldat, resampled_interp
# %%
    if time == True: 
        df2['timestamps'] = df2['timestamps'].interpolate('linear')

