#  ╭──────────────────────────────────────────────────────────────────────────────╮
#  │ Script to check the model for proper loading and return of predictions.      │
#  ╰──────────────────────────────────────────────────────────────────────────────╯

import numpy as np
import sys
from pathlib import Path

#  ──────────────────────────────────────────────────────────────────────────
# Local import
# Make sure `challenge_utils.py` is in the same directory as this script.

from challenge_utils import load_onnx

#  ──────────────────────────────────────────────────────────────────────────
# Load and run saved simple model to test that it works

D = 100 # can be variable
x = np.ones((D, 4536)) # your model must accept a predictor array where the second dimension is 4704 (the first one can be variable)

try:
    y = load_onnx(Path(sys.argv[1]), x)
    assert y.shape == np.zeros(D).shape, "Output shape does not match first dimension of input shape"
    print('Model works!')
except:
    print('Model does not load and run given an input with proper size')


