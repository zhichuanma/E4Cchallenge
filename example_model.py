#  ╭──────────────────────────────────────────────────────────────────────────────╮
#  │ Sample script to build the train/test dataset with the                       │
#  │  `build_training_data` function.                                             │
#  ╰──────────────────────────────────────────────────────────────────────────────╯

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#  ──────────────────────────────────────────────────────────────────────────
# Local import
# Make sure `challenge_utils.py` is in the same directory as this script.

from challenge_utils import build_training_data, relative_squared_error, train_test_split, save_onnx, load_onnx

#  ──────────────────────────────────────────────────────────────────────────
# Building training/testing set data
# Make sure the `students_drahi_production_consumption_hourly.csv` file is in the same directory as this script.

student_data_path = 'students_drahi_production_consumption_hourly.csv'

target_time, targets, predictors = build_training_data(student_data_path)

ntot = len(targets)
x_all = predictors.reshape(ntot, -1)
y_all = targets

# separating train/test sets
n = 250
test_ind = np.arange(n, len(targets))
x_train, y_train, x_test, y_test = train_test_split(predictors, targets, test_ind)
# considering looking into https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn-model-selection-train-test-split

#  ──────────────────────────────────────────────────────────────────────────
# Simple linear regression model
# https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification

reg = Ridge(alpha=1e8)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

RelativeMSE = relative_squared_error(y_pred, y_test)
print('Simple trained linear model RSE:', RelativeMSE)

#  ──────────────────────────────────────────────────────────────────────────
# Searching for optimal coefficients (validation)

param_grid = {'alpha': np.logspace(-2, 10, 21)}

grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')

grid.fit(x_all, y_all) # using the entire dataset for validation

best_alpha = grid.best_params_['alpha']
best_rse = -grid.best_score_/np.mean((y_all-y_all.mean())**2)
print('Best alpha from validation:', best_alpha)

best_estimator = grid.best_estimator_
y_pred = best_estimator.predict(x_all)
RelativeMSE = relative_squared_error(y_pred, y_all)
print('Best linear model RSE:', RelativeMSE)

#  ──────────────────────────────────────────────────────────────────────────
# Save best version of simple model

print('Saving in ONNX format')
save_onnx(best_estimator, 'linear_model.onnx', x_train)

#  ──────────────────────────────────────────────────────────────────────────
# Load and run saved simple model (the function combines the actions)

y_pred_onnx = load_onnx('linear_model.onnx', x_all)
RelativeMSE = relative_squared_error(y_pred_onnx, y_all)
print('Loaded from ONNX file RSE:', RelativeMSE)
