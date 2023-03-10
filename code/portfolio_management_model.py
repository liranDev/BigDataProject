import os
import json
from config import PORTFOLIO

def implement(dec):
    if dec[0] == 0:
        return
    # get currant account status
    # checks if file exists
    if os.path.isfile(PORTFOLIO) and os.access(PORTFOLIO, os.R_OK):
        # load account status
        with open(PORTFOLIO, 'r') as fp:
            account = json.load(fp)
        if dec[0] == 1:
            account['position'] = 'long'
            account['amount'] = 50
        if dec[0] == -1:
            account['position'] = 'short'
            account['amount'] = 50
        with open(PORTFOLIO, "w") as outfile:
            json.dump(account, outfile)
    else:
        if dec[0] == 1:
            pos = 'long'
            dictionary = {"position": pos, "amount": 50}
            with open(PORTFOLIO, "w") as outfile:
                json.dump(dictionary, outfile)
        if dec[0] == -1:
            pos = 'short'
            dictionary = {"position": pos, "amount": 50}
            with open(PORTFOLIO, "w") as outfile:
                json.dump(dictionary, outfile)

def get_results(dec,actual_price):
    pred = dec[5]
    diff = (actual_price - pred[50])/actual_price
    return diff
