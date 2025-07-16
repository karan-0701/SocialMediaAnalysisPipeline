from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def build_topic_model(docs: list):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True)

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    return topic_model, topics, probs, embeddings
