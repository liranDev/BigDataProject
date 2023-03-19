import os
import json
import pandas as pd
import fft_model
import sentiment_model
from config import PARAMETERS
from sklearn import svm
import matplotlib.pyplot as plt

def get_fft_factor(pred):
    return (pred[50]-pred[49])/pred[49]


def analyze(prices_50, sent_50, sent_fac, fft_fac, results):
    a = 0.8
    b = 0.2
    fft_pred = fft_model.predict(prices_50)
    sent_factor = sentiment_model.predict(sent_50)
    fft_factor = get_fft_factor(fft_pred)
    if len(results) > 100:
        clf = svm.SVC(kernel='linear')  # Linear Kernel
        X_train = pd.DataFrame({'fft': fft_fac, 'sent': sent_fac})
        clf.fit(X_train.tail(50), pd.Series(results).tail(50))
        #colors = {0: 'red', 1: 'blue'}
        #plt.scatter(X_train['fft'].tail(50), X_train['sent'].tail(50), c=pd.Series(results).tail(50).map(colors))
        #plt.show()
        decision = clf.predict(pd.DataFrame({'fft': fft_factor, 'sent': sent_factor}, index=[0]))[0]
        return decision, a, b, fft_factor, sent_factor, fft_pred
    else:
        decision = 1 if a*fft_factor+b*sent_factor > 1 else 0
        return decision, a, b, fft_factor, sent_factor, fft_pred


def update_parameters(a, b):
    # get currant parameters
    # checks if file exists
    if os.path.isfile(PARAMETERS) and os.access(PARAMETERS, os.R_OK):
        # load parameters file
        with open(PARAMETERS, 'r') as fp:
            params = json.load(fp)
        params['a'] = a
        params['b'] = b
        with open(PARAMETERS, "w") as outfile:
            json.dump(params, outfile)
    else:
        dictionary = {"a": a, "b": b}
        with open(PARAMETERS, "w") as outfile:
            json.dump(dictionary, outfile)


def optimize(res, a, b):
    update_parameters(a, b)
