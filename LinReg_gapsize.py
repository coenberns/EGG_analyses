#%% Linear regression analysis on gap size for interpolation
import numpy as np
import pandas as pd
from scipy.fft import fft
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import datetime as datetime
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import seaborn as sns

from scipy.interpolate import UnivariateSpline as univsp
from scipy import signal
from functions_read_bursts import*
from Old_Plot_EGG import*

# %% IMPORT OLD MEASUREMENT - LOOK AT SETTINGS ETC
# All commented out are non-significant frequencies

file=r'/Users/coenberns/Documents_lab/Thesis/Coding/Ephys/Thesis Python/Interpolation and gap analysis/2022.08.17_Full.txt'
# dat = read_egg_v3(file, rate=62.5, scale=150)
# a,b,c_fulldat = signalplot(dat, output='PD')
# resampled_fulldat, _ = downsample_from_signalplot(c_fulldat)

resampled_fulldat, fulldat = downsample_to_burst(file,time_str='2S', scale=150, round=True)
resampled_fulldat = resampled_fulldat.drop(columns='pseudo_time', axis=1)

#%%
resampled_filt = butter_filter_sos(resampled_fulldat, fs=.5,freq=[0.02,0.2], order=3)

#%%
resampled_gaps = {}
resampled_ip = {}
for gap in range(2,43,4):
    name = f'gap {gap}'
    resampled_gaps[name] = distribute_gaps_weighted(resampled_filt, gap_file=None,max_gap=gap,t_cycle=2, weighted=False,rand_state=55)
    resampled_ip[name] = interpolate_data(resampled_gaps[name], cycle_time=2, max_gap=gap+2)

#%% Linear regression based on calc_diff_resample function
# Let's say that resamp is the variable for the resampled_ip and data is resampled

c_df = resampled_filt
gap_sizes = []
metrics_list = []

for i, gaps in enumerate(resampled_ip):
    gap_size = int(gaps.split(' ')[1])
    gap_df = resampled_ip[gaps]
    diffs, avg_chan_abs, stats = calc_diff_resample(gap_df, c_df, freq=True, warp_plot=False, fs_d=.5, fs_c=.5)

    # Calculate the average of the metrics across all channels if needed
    # stats['gap size'] = gap_size  # Add the gap size to the stats DataFrame
    stats_avg = stats.mean(axis=0)  # Take the average across channels (rows)
    stats_avg['gap size'] = int(gap_size)  # Ensure the gap size is included in the averaged stats

    # Append the stats to the list
    metrics_list.append(stats_avg)

# Convert the list of Series objects to a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Reorder the DataFrame to have 'Gap Size' as the first column if not already
cols = list(metrics_df)
cols.insert(0, cols.pop(cols.index('gap size')))
metrics_df = metrics_df.loc[:, cols]
metrics_df = metrics_df.drop(columns='Channel', axis=1)


#%% Linear regression on those values with gap size being the independent variable
model = LinearRegression()
plt.rcParams['font.family'] = 'sans-serif'

for metric in ['MSE', 'RMSE', 'MAE', 'DTW', 'D-Mag', 'Diff D-Freq']:
    X = metrics_df['gap size'].values.reshape(-1,1)
    y = metrics_df[metric].values

    model.fit(X,y)
    r2 = model.score(X, y)
    print(f"R-squared value: {r2}")
    print(f"{metric} - Coefficient: {model.coef_[0]}, Intercept: {model.intercept_}")
    # Plot
    col = sns.color_palette('deep')
    plt.figure(figsize=(7,5))
    plt.scatter(X, y, color= col[0],label='Observed', s=20, alpha=.8)
    plt.plot(X, model.predict(X),color = col[1], linestyle='--', label='Predicted', linewidth=2)
    plt.text(x=X.max()*.75, y=y.max()*.5, s=f'R² = {r2:.2f}', size=12, fontweight='bold')
    # plt.axvline(x=14, ymax=,  linestyle=':', linewidth=2, color='k', alpha=.6)
    plt.xlabel('Gap size [s]', size=16)
    plt.ylabel(f'{metric}', size=16)
    plt.legend(fontsize=14)
    plt.show()

#%% Poly-regression model with order 3 for cubic
degree_i = 3
poly_model_i = make_pipeline(PolynomialFeatures(degree_i), LinearRegression())

for metric in ['MSE', 'RMSE', 'MAE', 'DTW', 'D-Mag','Diff D-Freq']:
    X = metrics_df['gap size'].values.reshape(-1, 1)
    y = metrics_df[metric].values

    poly_model_i.fit(X, y)
    # For R-squared score
    r_squared = poly_model_i.score(X, y)

    # Access the LinearRegression object from the pipeline
    lin_reg_step = poly_model_i.named_steps['linearregression']
    coefficients = lin_reg_step.coef_
    intercept = lin_reg_step.intercept_

    # Predicting values
    X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = poly_model_i.predict(X_pred)

    # Plotting
    print(f"R-squared value for {metric} with Polynomial Model of degree {degree_i}: {r_squared}")
    print(f"{metric} - Coefficients: {coefficients}, Intercept: {intercept}")
    plt.scatter(X, y, color='blue', label='Observed Data')
    plt.plot(X_pred, y_pred, color='red', label='Polynomial Model')
    plt.xlabel('Gap Size')
    plt.ylabel(f'Metric: {metric}')
    plt.legend()
    plt.show()
    

#%% Poly-regression model with order 3 for cubic
degree = 4
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

X = metrics_df[['gap size']].values.reshape(-1, 1) 
y = metrics_df['Diff D-Freq'].values

poly_model.fit(X, y)
r_squared = poly_model.score(X, y)

X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred = poly_model.predict(X_pred)

# Plotting
plt.scatter(X, y, color='blue', label='Observed Data')
plt.plot(X_pred, y_pred, color='red', label='Polynomial Model')
plt.xlabel('Gap Size')
plt.ylabel('Frequency Difference')
plt.legend()
plt.show()

print(f"R-squared value for Polynomial Model: {r_squared}")

#%% Resampled (control) vs. gap+interpolated data

control = resampled_filt.to_numpy()
_,_,c_freq = egg_signalfreq(control, rate=.5,freqlim=[1,10])

#%%

gap_freqs = {}
for i, gaps in enumerate(resampled_ip):
    print(gaps)
    gap_df = resampled_ip[gaps]
    gap_np = gap_df.to_numpy()
    _,_,g_freq = egg_signalfreq(gap_np, rate=.5, freqlim=[1,10])
    gap_freqs[gaps] = g_freq


#%%
gap_sizes = []
metrics = []
for gap_key, gap_freq_array in gap_freqs.items():
    gap_size = int(gap_key.split(' ')[1])
    gap_sizes.append(gap_size)
    
    channel_diff_sums = np.sum(np.abs(gap_freq_array[1:] - c_freq[1:]), axis=1)
    
    avg_diff = np.mean(channel_diff_sums)
    metrics.append(avg_diff)

X = np.array(gap_sizes).reshape(-1, 1)
y = np.array(metrics)

model = LinearRegression()
model.fit(X, y)

plt.scatter(gap_sizes, metrics, color='blue', label='Observed Metrics')
plt.plot(gap_sizes, model.predict(X), color='red', linestyle='--', label='Predicted by Model')
plt.xlabel('Gap Size')
plt.ylabel('Average Frequency Difference')
plt.title('Linear Regression Analysis')
plt.legend()
plt.show()

# Output the coefficient and intercept
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")


# %%
