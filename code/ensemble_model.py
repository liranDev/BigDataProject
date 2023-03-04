import fft_model
import sentiment_model

def get_fft_factor(pred):
    return pred[50]-pred[49]
def analyze(prices_50,sent_50, a = 0.8, b = 0.2):
    fft_pred = fft_model.predict(prices_50)
    sent_pred = sentiment_model.predict(sent_50)
    fft_factor = get_fft_factor(fft_pred)
    decision = 1 if a*fft_factor+b*sent_pred > 1 else 0
    return decision, a, b