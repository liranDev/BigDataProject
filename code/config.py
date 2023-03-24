import os
from pathlib import Path

# pre-trained models
LOGISTIC_REGRESSION_MODEL = Path.cwd().parent / 'models' / 'LogisticRegression_model.sav'
VECTORS_MODEL = Path.cwd().parent / 'models' / 'vectors_model.pkl'

# portfolio
PORTFOLIO = Path.cwd().parent / 'models' / 'account.json'
# parameters
PARAMETERS = Path.cwd().parent / 'models' / 'parameters.json'

# data
DATA_STREAM = Path.cwd().parent / 'data' / 'data.csv'
SPY_STREAM = Path.cwd().parent / 'data' / 'SPY.csv'


class KafkaConfig:
    FINANCE_TOPIC = 'finance'
    HOST = 'localhost:9092'


class TwitterConfig:
    CONSUMER_TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    CONSUMER_TWITTER_API_KEY_SECRET = os.getenv('TWITTER_API_KEY_SECRET')
    BEARER_TOKEN = os.getenv('BEARER_TOKEN')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_TOKEN_SECRET = os.getenv('TWITTER_TOKEN_SECRET')
