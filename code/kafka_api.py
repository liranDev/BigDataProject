from confluent_kafka.admin import AdminClient, NewTopic

from confluent_kafka import Producer, Consumer, KafkaError


class KafkaConsumerAPI:
    def __init__(self, group_id, topics, brokers):
        self.consumer = Consumer({
            'bootstrap.servers': brokers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
        self.topics = topics

    def consume(self):
        self.consumer.subscribe(self.topics)

        while True:
            msg = self.consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print('End of partition reached')
                else:
                    print('Error while consuming message: {}'.format(msg.error()))
            else:
                print('Received message: {}'.format(msg.value()))

    def close(self):
        self.consumer.close()


class KafkaProducerAPI:
    def __init__(self, brokers):
        self.producer = Producer({'bootstrap.servers': brokers})

    def produce(self, topic, value):
        self.producer.produce(topic)


class KafkaTopicManager:
    def __init__(self, brokers):
        self.admin_client = AdminClient({'bootstrap.servers': brokers})

    def create_topic(self, topic_name, num_partitions=1, replication_factor=1):
        topic = NewTopic(topic_name, num_partitions, replication_factor)
        futures = self.admin_client.create_topics([topic])
        for _, future in futures.items():
            try:
                future.result()
                print("Topic {} created successfully".format(topic_name))
            except Exception as e:
                print("Failed to create topic {}: {}".format(topic_name, e))
