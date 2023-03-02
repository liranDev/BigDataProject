import twitter

from config import TwitterConfig
from kafka_api import KafkaProducerAPI, KafkaConsumerAPI, KafkaTopicManager

# Fill in your Twitter API credentials here
consumer_key = TwitterConfig.CONSUMER_TWITTER_API_KEY
consumer_secret = TwitterConfig.CONSUMER_TWITTER_API_KEY_SECRET
access_token = TwitterConfig.TWITTER_ACCESS_TOKEN
access_token_secret = TwitterConfig.TWITTER_TOKEN_SECRET

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret)

user = api.VerifyCredentials()


query = 'finance AND "S&P 500"'

# Search for tweets containing the search query
results = api.GetSearch(raw_query="q={}&result_type=recent&count=10".format(query), lang='en')

# Print out the text of each tweet
TOPIC = 'finance'

topic_manager = KafkaTopicManager('localhost:9092')
topic_manager.create_topic('test-topic')
consumer = KafkaConsumerAPI('test-group', [TOPIC], 'localhost:9092')

producer = KafkaProducerAPI('localhost:9092')

for tweet in results:
    producer.produce(TOPIC, tweet.text)

consumer.consume()
