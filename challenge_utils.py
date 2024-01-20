#  ╭──────────────────────────────────────────────────────────────────────────────╮
#  │ Helper functions for the E4C Challenge 2024                                  │
#  ╰──────────────────────────────────────────────────────────────────────────────╯

import pandas as pd
import numpy as np
from skl2onnx import to_onnx
from onnxruntime import InferenceSession

#  ──────────────────────────────────────────────────────────────────────────

def save_onnx(model, fname, x_train):
    """Saves sklearn model to ONNX format."""

    onx = to_onnx(model, x_train[:1].astype(np.float32), target_opset=12)
    with open(fname, 'wb') as file:
        file.write(onx.SerializeToString())


def load_onnx(filename, x_test):
    """Loads and runs ONNX model from file."""

    sess = InferenceSession(filename, providers=["CPUExecutionProvider"])
    return sess.run(None, {"X": x_test.astype(np.float32)})[0][:, 0]


def build_training_data(hourly_data_path):
    """Build training data from hourly data function. Skips NaN'd days."""

    hourly_data = pd.read_csv(hourly_data_path)
    
    print('Loaded hourly data')

    # integrated power consumption

    dec = [] # daily energy consumption
    t_dec = []

    time = hourly_data['datetime'].astype(np.datetime64).values.astype('datetime64[s]')
    power_consumption = hourly_data['kw_total_zone2'].values

    for ti, t in enumerate(time):
        tmp_t = pd.Timestamp(t)

        if np.isclose(tmp_t.hour, 0) and np.isclose(tmp_t.minute, 0):

            day_end = np.datetime64(tmp_t + pd.Timedelta(days=1))
            ind = np.where((tmp_t < time) & (time < day_end), True, False)

            if len(time[ind]) > 0 and not np.isnan(power_consumption[ind]).any():
                t_dec.append(np.datetime64(tmp_t).astype('datetime64[s]'))
                dec.append(np.trapz(power_consumption[ind], time[ind].astype(int))/3600) # integrated kW to kJ then to kWh

    # time series of daily energy consumption
    t_dec = np.array(t_dec)
    dec = np.array(dec)

    print('Calculated daily energy consumption')

    # seperating predictors

    N = 7 # N days of predictors beforehand
    final_ind = []
    final_hourly = []

    predictor_window = pd.Timedelta(days=N)

    for ti, t in enumerate(t_dec):
        tmp_t = pd.Timestamp(t)
        ind = np.where((tmp_t - predictor_window <= time) & (time < tmp_t), True, False) # finding indices within the N prior days

        bad_ind = np.isnan(hourly_data.iloc[ind, 1::].values)
        if len(time[ind]) >= 24 * N and not bad_ind.any(): # rejecting any data with NaNs; useful for the student dataset
            final_ind.append(ti)
            final_hourly.append(hourly_data.iloc[ind, 1::].values) # dropping datetime column

    # getting targets and predictors
    target_time = t_dec[final_ind]
    targets = dec[final_ind]
    predictors = np.array(final_hourly)

    print('Calculated predictor window')

    return target_time, targets, predictors


def train_test_split(predictors, targets, test_ind=[]):
    """Make training/test data for sklearn from targets and predictors."""

    ntot = len(targets)
    train_ind = np.arange(ntot)
    train_ind = np.delete(train_ind, test_ind)
    n = len(train_ind)
    m = len(test_ind)

    x_train = predictors[train_ind].reshape(n, -1)
    y_train = targets[train_ind]

    if len(test_ind) > 0:
        x_test = predictors[test_ind].reshape(m, -1)
        y_test = targets[test_ind]

    else:
        x_test = None
        y_test = None

    return x_train, y_train, x_test, y_test


def relative_squared_error(y_pred, y_true):
    """Relative squared error (RSE; also called relative mean square error). < 1 is good, = 1 is bad, > 1 really bad."""
    return np.mean((y_pred - y_true)**2)/np.mean((y_true - y_true.mean())**2)
