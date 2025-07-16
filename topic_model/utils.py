def print_topic_labels(topic_model):
    for topic in topic_model.get_topics().keys():
        print(f"Topic {topic}: {topic_model.get_topic(topic)}")
