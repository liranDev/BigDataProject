import fft_model
import sentiment_model
from config import PARAMETERS

def get_fft_factor(pred):
    return (pred[50]-pred[49])/pred[49]


def analyze(prices_50,sent_50, a = 0.8, b = 0.2):
    fft_pred = fft_model.predict(prices_50)
    sent_factor = sentiment_model.predict(sent_50)
    fft_factor = get_fft_factor(fft_pred)
    decision = 1 if a*fft_factor+b*sent_factor > 1 else 0
    return decision, a, b, fft_factor, sent_factor, fft_pred


def update_parameters(a,b):
    # get currant parameters
    # checks if file exists
    if os.path.isfile(PARAMETERS) and os.access(PARAMETERS, os.R_OK):
        # load parameters file
        with open(PARAMETERS, 'r') as fp:
            params = json.load(fp)
        params['a'] = a
        params['b'] = b
        with open(PORTFOLIO, "w") as outfile:
            json.dump(params, outfile)
    else:
        dictionary = {"a": a, "b": b}
        with open(PARAMETERS, "w") as outfile:
            json.dump(dictionary, outfile)


def optimize(res, a, b):
    update_parameters(a, b)
