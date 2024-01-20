#  ╭──────────────────────────────────────────────────────────────────────────────╮
#  │ Sample script to plot the hourly time series.                                │
#  ╰──────────────────────────────────────────────────────────────────────────────╯

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#  ──────────────────────────────────────────────────────────────────────────
# Loading data

hourly_data = pd.read_csv('students_drahi_production_consumption_hourly.csv')

#  ──────────────────────────────────────────────────────────────────────────
# Plotting a handful of variables and a section of the time series

fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)

time = hourly_data['datetime'].astype(np.datetime64).values.astype('datetime64[s]')

# slicing out a few months of data
start = 450
some = slice(start, start + 1000)

for col in hourly_data.columns[1::]:

    # only plotting a few variables
    if col in ['AirTemp', 'rain', 'kw_total_zone2']:
        ax.plot(time[some], hourly_data[col].values[some], label=col)

ax.legend()

fig.tight_layout()

fig.savefig('sample_fig.png') # to save the figure to a png
# plt.show() # to show the data
