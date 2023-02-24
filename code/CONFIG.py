import os
from pathlib import Path

#pre-trained models
LOGISTIC_REGRESSION_MODEL = Path.cwd().parent / 'models' / 'LogisticRegression_model.sav'
VECTORS_MODEL = Path.cwd().parent / 'models' / 'vectors_model.pkl'

#data
DATA_STREAM = Path.cwd().parent / 'data' / 'data.csv'
SPY_STREAM = Path.cwd().parent / 'data' / 'SPY.csv'