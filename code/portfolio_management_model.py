import json
from config import PORTFOLIO

def implement(dec):
    if dec[0] == 0:
        return
    if dec[0] == 1:
        pos = 'long'
        dictionary = {"position": pos, "amount": 50}
        with open(PORTFOLIO, "w") as outfile:
            json.dump(dictionary, outfile)