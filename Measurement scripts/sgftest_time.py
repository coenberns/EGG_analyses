#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from datetime import datetime, timedelta, time
import pathlib 
# %%
def calculate_time(file, date=None):

    if date is None:
        # get only the filename from the file path
        file_path = pathlib.Path(file)
        filename = file_path.name
        # extract date from the filename
        date = filename.split('_')[0]

    # Creating datetime object
    # which takes in "MMDDYYYY" like only US people write date order
    date = datetime.strptime(date, '%m%d%Y')
    dat = pd.read_csv(file, header=0, dtype=str, delimiter='|', names=[
        'realtime', 'misc', 'packet', 'msg', 'rssi'])

    dat = dat[~dat.rssi.str.contains('error')]
    dat = dat[dat.misc.str.contains('0a')]
    dat = dat.reset_index(drop=True)

    counter = dat.packet.astype(int)
    new_counter = [0]
    for j, ele in enumerate(counter[1:]):
        step = counter[j+1]-counter[j]
        if step > 0:
            new_counter.append(step+new_counter[j])
        else:
            new_counter.append(65536-counter[j]+counter[j+1]+new_counter[j])
            print('flip', step, 65536-counter[j]+counter[j+1])
    abscounterseries = pd.Series(new_counter, name='packet_re_idx')

    dat = pd.concat((dat, abscounterseries), axis=1)
    
    # Creating a datetime object from realtime, recalling it realtime (since it still is)
    dat["realtime"] = dat["realtime"].str.strip()
    dat["realtime"] = pd.to_datetime(dat["realtime"], format='%H:%M:%S.%f')
    dat["realtime"] = dat["realtime"].apply(
        lambda t: datetime.combine(date, t.time()))
    # Check for date rollover and increment the date if necessary, with additional glitch values excluded
    dat['time_diff'] = dat['realtime'].diff().dt.total_seconds()
    dat['rollover'] = dat['time_diff'] < 0
    dat['glitch'] = (dat['time_diff'] > -5) & (dat['rollover'])
    dat['correct_rollover'] = dat['rollover'] & ~dat['glitch'] 
    dat['days_to_add'] = dat['correct_rollover'].cumsum()
    dat['corrected_realtime'] = dat['realtime'] + pd.to_timedelta(dat['days_to_add'], unit='D')
    # dat['corrected_realtime'].interpolate(method='linear', inplace=True)

    # probably delete this if timestamps values at end are close to elapsed_s
    dat['elapsed_t'] = dat['corrected_realtime'] - dat['corrected_realtime'].iloc[0]
    elapsed_sgf = dat['elapsed_t'].max()
    print("The total time elapsed was: ", elapsed_sgf)

    return dat, elapsed_sgf

# %%
#Mac
# meas_path = pathlib.Path("/Users/coenberns/Library/CloudStorage/OneDrive-MassGeneralBrigham/Documents/Thesis/Measurements/Pig measurements/08282023 second - straight measurement mode 2")
# #Windows
sgf_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF Readings miniPC/Battery testing/Mode 2 recs - 1s840ms sleep")# # List all the files in the selected folder
in_folder = [f for f in sgf_path.iterdir() if f.is_file()]

# Print a list of available files
for i, f in enumerate(in_folder, start=1):
    print(f"{i}. {f.name}")

# Ask the user to choose a file
while True:
    try:
        choice = int(input("Enter the number of the file you want to use (1,2, etc.): "))
        if 1 <= choice <= len(in_folder):
            break
        else:
            print("Invalid choice. Please enter a valid number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# Assign the selected file to a variable
file = in_folder[choice - 1]

# Now you can work with the selected_file
print(f"File selected: {file.name}")

dat, elapsed = calculate_time(file)

# # %%
# sgf_path = pathlib.Path("C:/Users/CoenBerns/OneDrive - Mass General Brigham/Documents/Thesis/Measurements/RF Readings miniPC/SGF Readings")
# # # List all the files in the selected folder
# in_folder = [f for f in sgf_path.iterdir() if f.is_file()]

# for i, f in enumerate(in_folder, start=1):
#     sgf_dat, elapsed_sgf = calculate_time(file=f)
# # %%

# %%
