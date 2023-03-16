import json


with open('sample_data.json') as f:
    data = json.load(f)

for data_obj in data:
    print(data_obj['tweet_timestap'], data_obj['sentence'])

