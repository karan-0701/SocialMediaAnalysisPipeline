import argparse
from .data_loader import load_youtube_titles
from .model import build_topic_model
from .visualize import plot_dendrogram
from .utils import print_topic_labels

def main():
    parser = argparse.ArgumentParser(description="Run BERTopic topic modeling.")
    parser.add_argument("csv_path", help="Path to the YouTube CSV file")
    parser.add_argument("--visualize", action="store_true", help="Show dendrogram of topics")
    parser.add_argument("--print-topics", action="store_true", help="Print topic keywords")

    args = parser.parse_args()
    
    docs, df = load_youtube_titles(args.csv_path)
    topic_model, topics, probs, embeddings = build_topic_model(docs)

    if args.visualize:
        plot_dendrogram(topic_model.topic_embeddings_)

    if args.print_topics:
        print_topic_labels(topic_model)

if __name__ == "__main__":
    main()
