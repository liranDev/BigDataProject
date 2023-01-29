import os
from pathlib import Path

#pre-trained models
LOGISTIC_REGRESSION_MODEL = Path.cwd() / 'models' / 'LogisticRegression_model.sav'
VECTORS_MODEL = Path.cwd() / 'models' / 'vectors_model.pkl'

#data
DATA_STREAM = Path.cwd() / 'data' / 'data.csv'